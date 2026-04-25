"""Face tracking control loop.

Strategy
--------
We keep the arm end-effector at a fixed "look pose" (x, y, z) and modulate two
degrees of freedom only:

    yaw   – rotate base around Z axis: (x, y) → R_z(yaw) · (x0, y0)
    pitch – tilt end-effector via the IK `pitch` parameter

Per loop iteration we measure the pixel offset of the chosen face from image
centre, convert it to an angular error using the camera FOV, apply a P
controller with optional EMA smoothing and per-step clamps, then dispatch a
short IK move via the persistent `arm_bridge` HTTP API.

The loop runs in its own daemon thread and shares a single `CameraManager`
with the WebUI so MJPEG streaming and tracking can read frames in parallel.
"""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from common import clamp
from face_detector import FaceDetector, pick_face

_BRIDGE_URL = "http://localhost:9091"


def _bridge_post(path: str, data: dict[str, Any], timeout: float = 4.0) -> dict | None:
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{_BRIDGE_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError):
        return None


def _bridge_healthy() -> bool:
    try:
        with urllib.request.urlopen(f"{_BRIDGE_URL}/health", timeout=1.5) as r:
            return r.status == 200
    except Exception:
        return False


@dataclass
class FaceTracker:
    detector: FaceDetector
    read_frame: Callable[[], tuple[bool, Any]]
    cfg: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._state_lock = threading.Lock()
        self._state: dict[str, Any] = {
            "running": False,
            "backend": self.detector.effective_backend,
            "yaw_deg": 0.0,
            "pitch_deg": float(self.cfg.get("home_pose", {}).get("pitch", -10.0)),
            "face_found": False,
            "lost_frames": 0,
            "fps": 0.0,
            "last_error": None,
            "last_face": None,
            "img_size": None,
            "last_move_pulses": None,
            "use_arm": False,
        }

    def get_state(self) -> dict[str, Any]:
        with self._state_lock:
            return dict(self._state)

    def _set(self, **kw: Any) -> None:
        with self._state_lock:
            self._state.update(kw)

    def update_cfg(self, cfg: dict[str, Any]) -> None:
        self.cfg = dict(cfg)

    # ── main loop ────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        use_arm = _bridge_healthy()
        self._set(
            running=True,
            face_found=False,
            lost_frames=0,
            last_error=None if use_arm else "arm_bridge 未就绪，仅做检测，不发送 IK",
            use_arm=use_arm,
        )
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=2.0)
        self._set(running=False)

    def _loop(self) -> None:
        last_t = time.monotonic()
        last_seen = time.monotonic()
        ema_yaw_err: float | None = None
        ema_pitch_err: float | None = None
        homed = False
        try:
            while not self._stop_evt.is_set():
                cfg = self.cfg
                hz = max(1.0, float(cfg.get("loop_hz", 10)))
                period = 1.0 / hz

                ok, frame = self.read_frame()
                if not ok or frame is None:
                    self._set(last_error="读取相机帧失败")
                    if self._stop_evt.wait(0.2):
                        break
                    continue

                h, w = frame.shape[:2]
                faces = self.detector.detect(frame)
                primary = pick_face(faces, w, h, strategy=str(cfg.get("pick", "largest")))

                now = time.monotonic()
                fps = 1.0 / max(now - last_t, 1e-3)
                last_t = now

                if primary is None:
                    with self._state_lock:
                        self._state["lost_frames"] = self._state.get("lost_frames", 0) + 1
                        lost_n = self._state["lost_frames"]
                    self._set(
                        face_found=False,
                        last_face=None,
                        img_size=[w, h],
                        fps=round(fps, 1),
                    )
                    threshold_n = int(cfg.get("lost_frames_to_home", 30))
                    threshold_t = float(cfg.get("lost_seconds_to_home", 3.0))
                    if not homed and (
                        lost_n >= threshold_n or (now - last_seen) >= threshold_t
                    ):
                        self._home(cfg)
                        homed = True
                        ema_yaw_err = ema_pitch_err = None
                    if self._stop_evt.wait(period):
                        break
                    continue

                last_seen = now
                homed = False
                self._set(
                    face_found=True,
                    lost_frames=0,
                    img_size=[w, h],
                    last_face=primary,
                    fps=round(fps, 1),
                )

                cx, cy = primary["center"]
                err_x = cx - w / 2.0
                err_y = cy - h / 2.0
                deadzone = float(cfg.get("deadzone_px", 25))
                if abs(err_x) < deadzone and abs(err_y) < deadzone:
                    if self._stop_evt.wait(period):
                        break
                    continue

                alpha = clamp(float(cfg.get("smoothing", 0.5)), 0.0, 1.0)
                if alpha > 0 and ema_yaw_err is not None:
                    err_x = alpha * err_x + (1 - alpha) * ema_yaw_err
                    err_y = alpha * err_y + (1 - alpha) * ema_pitch_err
                ema_yaw_err, ema_pitch_err = err_x, err_y

                # pixel error → desired angle delta
                fov_h = float(cfg.get("fov_h_deg", 60))
                fov_v = float(cfg.get("fov_v_deg", 45))
                yaw_kp = float(cfg.get("yaw_kp", 0.45))
                pitch_kp = float(cfg.get("pitch_kp", 0.45))

                d_yaw = -(err_x / (w / 2.0)) * (fov_h / 2.0) * yaw_kp
                d_pitch = -(err_y / (h / 2.0)) * (fov_v / 2.0) * pitch_kp

                d_yaw = clamp(
                    d_yaw,
                    -float(cfg.get("max_yaw_step_deg", 8)),
                    float(cfg.get("max_yaw_step_deg", 8)),
                )
                d_pitch = clamp(
                    d_pitch,
                    -float(cfg.get("max_pitch_step_deg", 6)),
                    float(cfg.get("max_pitch_step_deg", 6)),
                )

                with self._state_lock:
                    new_yaw = clamp(
                        self._state["yaw_deg"] + d_yaw,
                        float(cfg.get("yaw_min_deg", -75)),
                        float(cfg.get("yaw_max_deg", 75)),
                    )
                    new_pitch = clamp(
                        self._state["pitch_deg"] + d_pitch,
                        float(cfg.get("pitch_min_deg", -45)),
                        float(cfg.get("pitch_max_deg", 35)),
                    )
                    self._state["yaw_deg"] = round(new_yaw, 2)
                    self._state["pitch_deg"] = round(new_pitch, 2)

                self._send_pose(new_yaw, new_pitch, cfg)
                if self._stop_evt.wait(period):
                    break
        except Exception as e:
            self._set(last_error=f"tracker loop crashed: {e}")
        finally:
            self._set(running=False)

    # ── motion dispatch ──────────────────────────────────────────
    def _send_pose(self, yaw_deg: float, pitch_deg: float, cfg: dict[str, Any]) -> None:
        if not self._state.get("use_arm"):
            if not _bridge_healthy():
                return
            self._set(use_arm=True, last_error=None)

        home = cfg.get("home_pose", {}) or {}
        x0 = float(home.get("x", 0.18))
        y0 = float(home.get("y", 0.0))
        z0 = float(home.get("z", 0.10))

        yaw_r = math.radians(yaw_deg)
        c, s = math.cos(yaw_r), math.sin(yaw_r)
        x = x0 * c - y0 * s
        y = x0 * s + y0 * c
        dur = float(cfg.get("move_duration", 0.12))

        r = _bridge_post(
            "/ik_move",
            {
                "x": round(x, 4),
                "y": round(y, 4),
                "z": round(z0, 4),
                "pitch": round(float(pitch_deg), 2),
                "pitch_range": [
                    float(cfg.get("pitch_min_deg", -45)),
                    float(cfg.get("pitch_max_deg", 35)),
                ],
                "duration": dur,
            },
            timeout=max(2.0, dur + 1.5),
        )
        if r is None:
            self._set(last_error="arm_bridge 不可达，本周期未发送")
            self._set(use_arm=False)
            return
        if not r.get("ok"):
            self._set(last_error=f"IK 失败: {r.get('error', 'unknown')}")
            return
        self._set(last_error=None, last_move_pulses=r.get("pulses"))

    def _home(self, cfg: dict[str, Any]) -> None:
        home = cfg.get("home_pose", {}) or {}
        with self._state_lock:
            self._state["yaw_deg"] = 0.0
            self._state["pitch_deg"] = float(home.get("pitch", -10.0))
        self._send_pose(0.0, float(home.get("pitch", -10.0)), cfg)
