"""Face tracking control loop.

Strategy
--------
Per loop iteration we measure the pixel offset of the chosen face from image
centre, convert it to an angular error using the camera FOV, apply a P
controller with optional EMA smoothing and per-step clamps, then dispatch the
target (yaw, pitch) to the arm via one of two modes:

* ``joints`` (default) – On start we solve IK once for the home pose to lock
  the shoulder/elbow/wrist baseline pulses; subsequent updates only rotate the
  base servo (yaw) and the wrist-pitch servo (pitch). Each update is just a
  ROS topic publish (~5 ms), so the bridge can never become the bottleneck.

* ``ik`` – Each update calls ``/ik_move`` with ``(x, y, z, pitch)``, where
  ``(x, y) = R_z(yaw) · (x0, y0)``. Useful when you need full Cartesian
  control, but ``pitch`` is constrained by the chosen ``z`` and the IK service
  may reject extreme angles.

The loop runs in its own daemon thread and shares a single ``CameraManager``
with the WebUI so MJPEG streaming and tracking can read frames in parallel.
A separate dispatcher thread serialises bridge calls and silently drops any
target that arrives while a previous one is still in-flight.
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

# ArmPi Ultra: pulse 500 = 0°, 1° ≈ 5.556 pulse (range 0~1000 = ±90°)
_PULSE_PER_DEG = 1.0 / 0.18


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
        self._dispatcher: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._state_lock = threading.Lock()
        # Single-slot mailbox: dispatcher always sends the *latest* target,
        # silently dropping any older one that hasn't been sent yet.
        self._target_lock = threading.Lock()
        self._target_evt = threading.Event()
        self._target_pose: tuple[float, float] | None = None
        # Cached baseline pulses for joints-mode (set on start()).
        self._home_pulses: dict[int, int] | None = None
        self._state: dict[str, Any] = {
            "running": False,
            "backend": self.detector.effective_backend,
            "yaw_deg": 0.0,
            "pitch_deg": float(self.cfg.get("home_pose", {}).get("pitch", 0.0)),
            "face_found": False,
            "lost_frames": 0,
            "fps": 0.0,
            "last_error": None,
            "last_face": None,
            "img_size": None,
            "last_move_pulses": None,
            "use_arm": False,
            "ik_dropped": 0,
            "ik_inflight": 0,
            "ik_failed": 0,
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
        self._target_evt.clear()
        with self._target_lock:
            self._target_pose = None
        use_arm = _bridge_healthy()
        # Reset cumulative angles to home so the first IK send is a known pose,
        # not "(假设 yaw=last, pitch=last) + 大 delta".
        home_pitch = float(self.cfg.get("home_pose", {}).get("pitch", 0.0))
        self._set(
            running=True,
            face_found=False,
            lost_frames=0,
            last_error=None if use_arm else "arm_bridge 未就绪，仅做检测，不发送 IK",
            use_arm=use_arm,
            ik_dropped=0,
            ik_inflight=0,
            ik_failed=0,
            yaw_deg=0.0,
            pitch_deg=home_pitch,
        )
        # Sync arm physically with our assumed state BEFORE accepting any deltas,
        # using a deliberately slow duration so the user does not see a slam.
        if use_arm:
            home_dur = float(self.cfg.get("start_home_duration", 1.5))
            self._home_pulses = None  # force re-solve in joints mode
            self._send_pose_blocking(
                0.0, home_pitch, self.cfg, override_duration=home_dur
            )
            # In joints mode we additionally pre-cache the baseline servo
            # pulses by asking the bridge to solve IK once (no movement).
            if str(self.cfg.get("mode", "joints")).lower() == "joints":
                self._cache_home_pulses(self.cfg)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._dispatcher = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._dispatcher.start()

    def _cache_home_pulses(self, cfg: dict[str, Any]) -> None:
        """Solve IK once for home pose to obtain baseline servo pulses; the
        joints-mode dispatcher then just adds yaw/pitch deltas to these."""
        home = cfg.get("home_pose", {}) or {}
        body = {
            "x": float(home.get("x", 0.18)),
            "y": float(home.get("y", 0.0)),
            "z": float(home.get("z", 0.15)),
            "pitch": float(home.get("pitch", 0.0)),
            "pitch_range": [
                float(cfg.get("pitch_min_deg", -45)),
                float(cfg.get("pitch_max_deg", 35)),
            ],
            "duration": 0.001,
        }
        r = _bridge_post("/ik", body, timeout=4.0)
        if r is None or not r.get("ok"):
            self._set(
                last_error=(
                    "joints 模式：home 位姿 IK 求解失败，回退到 ik 模式。"
                    "请调小 home.x 或抬高 home.z"
                )
            )
            self._home_pulses = None
            return
        pulses = r.get("pulses") or []
        # SERVO_JOINT_MAP from arm_bridge: pulse[0]→6, [1]→5, [2]→4, [3]→3, [4]→2
        mapping: dict[int, int] = {}
        if len(pulses) >= 4:
            mapping = {6: int(pulses[0]), 5: int(pulses[1]), 4: int(pulses[2]), 3: int(pulses[3])}
        if len(pulses) >= 5:
            mapping[2] = int(pulses[4])
        self._home_pulses = mapping or None

    def stop(self) -> None:
        self._stop_evt.set()
        # Wake the dispatcher so it can exit promptly.
        self._target_evt.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=2.0)
        d = self._dispatcher
        if d and d.is_alive():
            d.join(timeout=2.0)
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

    # ── motion dispatch (async) ──────────────────────────────────
    def _send_pose(self, yaw_deg: float, pitch_deg: float, cfg: dict[str, Any]) -> None:
        """Non-blocking: drop the new (yaw, pitch) into the single-slot mailbox
        and signal the dispatcher. Older un-sent target is silently overwritten,
        so the loop never waits on the bridge's IK service."""
        with self._target_lock:
            if self._target_pose is not None:
                # An older target was queued and not yet sent → it's stale, drop it.
                with self._state_lock:
                    self._state["ik_dropped"] = self._state.get("ik_dropped", 0) + 1
            self._target_pose = (float(yaw_deg), float(pitch_deg))
        self._target_evt.set()

    def _dispatch_loop(self) -> None:
        """Drain the latest target and POST to the bridge; serial so the
        bridge never sees overlapping requests but the producer side is freed."""
        while not self._stop_evt.is_set():
            self._target_evt.wait(timeout=0.5)
            if self._stop_evt.is_set():
                break
            with self._target_lock:
                target = self._target_pose
                self._target_pose = None
                self._target_evt.clear()
            if target is None:
                continue
            yaw_deg, pitch_deg = target
            self._set(ik_inflight=1)
            try:
                mode = str(self.cfg.get("mode", "joints")).lower()
                if mode == "joints" and self._home_pulses:
                    self._send_pose_joints(yaw_deg, pitch_deg, self.cfg)
                else:
                    self._send_pose_blocking(yaw_deg, pitch_deg, self.cfg)
            finally:
                self._set(ik_inflight=0)

    def _send_pose_joints(
        self,
        yaw_deg: float,
        pitch_deg: float,
        cfg: dict[str, Any],
        override_duration: float | None = None,
    ) -> None:
        """Joints mode: just rotate the base servo for yaw and the wrist-pitch
        servo for pitch, leaving shoulder/elbow/wrist-roll at their cached
        baseline. Bypasses /ik_move entirely → ~5ms per command."""
        if not self._state.get("use_arm"):
            if not _bridge_healthy():
                return
            self._set(use_arm=True, last_error=None)

        if self._home_pulses is None:
            self._set(last_error="joints 模式：未缓存基线脉宽，回退到 ik")
            self._send_pose_blocking(yaw_deg, pitch_deg, cfg, override_duration)
            return

        joints_cfg = cfg.get("joints", {}) or {}
        base_id = int(joints_cfg.get("base_servo", 6))
        pitch_id = int(joints_cfg.get("pitch_servo", 3))
        yaw_ppd = float(joints_cfg.get("yaw_pulse_per_deg", _PULSE_PER_DEG))
        pitch_ppd = float(joints_cfg.get("pitch_pulse_per_deg", _PULSE_PER_DEG))

        positions: dict[int, int] = dict(self._home_pulses)
        if base_id in positions:
            positions[base_id] = int(round(positions[base_id] + yaw_deg * yaw_ppd))
        if pitch_id in positions:
            positions[pitch_id] = int(round(positions[pitch_id] + pitch_deg * pitch_ppd))

        # Apply per-servo invert (pulse → 1000 - pulse) as configured globally.
        # IMPORTANT: The cached _home_pulses came from /ik which already applies
        # the bridge's own invert before publish. So we should NOT double-invert
        # here. The arm.servo_invert config in default.yaml is consumed inside
        # arm_demo's CLI fallback only. The bridge does not re-invert /servo
        # input either. So pass positions as-is.

        for sid, p in list(positions.items()):
            positions[sid] = int(clamp(p, 0, 1000))

        dur = (
            float(override_duration)
            if override_duration is not None
            else float(cfg.get("move_duration", 0.08))
        )
        r = _bridge_post(
            "/servo",
            {
                "positions": {str(k): v for k, v in positions.items()},
                "duration": dur,
            },
            timeout=max(2.0, dur + 1.0),
        )
        if r is None:
            self._set(last_error="arm_bridge 不可达，本周期未发送", use_arm=False)
            return
        if not r.get("ok"):
            with self._state_lock:
                self._state["ik_failed"] = self._state.get("ik_failed", 0) + 1
                self._state["last_error"] = f"servo 发送失败: {r.get('error', 'unknown')}"
            return
        self._set(last_error=None, last_move_pulses=positions)

    def _send_pose_blocking(
        self,
        yaw_deg: float,
        pitch_deg: float,
        cfg: dict[str, Any],
        override_duration: float | None = None,
    ) -> None:
        if not self._state.get("use_arm"):
            if not _bridge_healthy():
                return
            self._set(use_arm=True, last_error=None)

        home = cfg.get("home_pose", {}) or {}
        x0 = float(home.get("x", 0.18))
        y0 = float(home.get("y", 0.0))
        z0 = float(home.get("z", 0.15))

        yaw_r = math.radians(yaw_deg)
        c, s = math.cos(yaw_r), math.sin(yaw_r)
        x = x0 * c - y0 * s
        y = x0 * s + y0 * c
        dur = (
            float(override_duration)
            if override_duration is not None
            else float(cfg.get("move_duration", 0.12))
        )

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
            self._set(last_error="arm_bridge 不可达，本周期未发送", use_arm=False)
            return
        if not r.get("ok"):
            with self._state_lock:
                self._state["ik_failed"] = self._state.get("ik_failed", 0) + 1
                self._state["last_error"] = (
                    f"IK 失败 (pitch={pitch_deg:.1f}° z={z0:.2f}m): "
                    f"{r.get('error', 'unknown')}"
                )
            return
        self._set(last_error=None, last_move_pulses=r.get("pulses"))

    def _home(self, cfg: dict[str, Any]) -> None:
        home = cfg.get("home_pose", {}) or {}
        home_pitch = float(home.get("pitch", 0.0))
        with self._state_lock:
            self._state["yaw_deg"] = 0.0
            self._state["pitch_deg"] = home_pitch
        self._send_pose(0.0, home_pitch, cfg)
