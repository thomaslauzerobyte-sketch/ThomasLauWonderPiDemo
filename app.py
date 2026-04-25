#!/usr/bin/env python3
"""WonderPi 人脸追踪 — Flask WebUI.

只保留最小骨架：相机 + 人脸检测 + 机械臂 yaw/pitch 跟随。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from common import load_config, save_config  # noqa: E402
from face_detector import FaceDetector, draw_faces, pick_face  # noqa: E402
from face_tracker import FaceTracker, _bridge_healthy  # noqa: E402

app = Flask(__name__, static_folder=str(ROOT / "web" / "static"), template_folder=str(ROOT / "web"))

# ── Config (file + runtime overrides) ─────────────────────────────
_runtime_overrides: dict[str, Any] = {}
_cfg_lock = threading.Lock()


def _cfg() -> dict[str, Any]:
    cfg = load_config()
    for top, sub in _runtime_overrides.items():
        if isinstance(sub, dict) and isinstance(cfg.get(top), dict):
            cfg[top].update(sub)
        else:
            cfg[top] = sub
    return cfg


# ── Camera Manager (ros_web / opencv / picamera2 / deptrum) ───────
def _http_get_bytes(url: str, timeout_sec: float) -> bytes | None:
    timeout_sec = max(1.0, float(timeout_sec))
    curl = shutil.which("curl")
    if curl:
        try:
            r = subprocess.run(
                [curl, "-sS", "--connect-timeout", "2",
                 "--max-time", str(int(timeout_sec + 0.99)), "-o", "-", url],
                capture_output=True, timeout=timeout_sec + 1.5,
            )
            if r.returncode == 0 and len(r.stdout) > 100:
                return r.stdout
        except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
            pass
    import socket
    old = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(timeout_sec)
        with urllib.request.urlopen(url) as resp:
            if getattr(resp, "status", 200) != 200:
                return None
            return resp.read()
    except Exception:
        return None
    finally:
        socket.setdefaulttimeout(old)


def _ros_discover_snapshot_urls(ros_base: str, timeout_sec: float) -> list[str]:
    raw = _http_get_bytes(ros_base.rstrip("/") + "/", min(4.0, float(timeout_sec)))
    if not raw:
        return []
    html = raw.decode("utf-8", errors="ignore")
    topics = re.findall(r"/snapshot\?topic=([^\"'>\s]+)", html)
    seen: set[str] = set()
    base = ros_base.rstrip("/")
    out: list[str] = []
    for t in topics:
        t = unquote(t)
        if t in seen:
            continue
        seen.add(t)
        out.append(f"{base}/snapshot?topic={quote(t, safe='/')}")
    return out


class CameraManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cap: Any = None
        self._backend: str = ""
        self._running = False
        self._ros_rgb_url: str = ""
        self._ros_stream_url: str = ""
        self._ros_http_timeout: float = 4.0
        self._cached_frame: np.ndarray | None = None
        self._cached_t: float = 0.0
        # 80ms ≈ 12 FPS：人脸追踪不需要更高，足以让 MJPEG 流与
        # tracker 循环大多数情况下共用同一帧，避免互相抢锁。
        self._cache_ttl: float = 0.08

    def _open(self, cfg: dict[str, Any]) -> None:
        cam = cfg["camera"]
        backend = cam.get("backend", "auto").lower()
        index = int(cam.get("index", 0))
        ros_base = cam.get("ros_web_video_url", "http://localhost:8080")

        if backend in ("ros_web", "auto"):
            ros_to = float(cam.get("ros_web_timeout_sec", 4.0))
            base = ros_base.rstrip("/")
            topics = cam.get("ros_snapshot_topics") or ["/depth_cam/rgb/image_raw"]
            snap_urls: list[str] = []
            for topic in topics:
                t = str(topic).strip()
                if not t.startswith("/"):
                    t = "/" + t
                snap_urls.append(f"{base}/snapshot?topic={quote(t, safe='/')}")
            for u in _ros_discover_snapshot_urls(ros_base, ros_to):
                if u not in snap_urls:
                    snap_urls.append(u)
            for snap_url in snap_urls:
                body = _http_get_bytes(snap_url, ros_to)
                if body and len(body) > 100:
                    m = re.search(r"[?&]topic=([^&]+)", snap_url)
                    topic = unquote(m.group(1)) if m else "/depth_cam/rgb/image_raw"
                    self._ros_rgb_url = snap_url
                    self._ros_stream_url = (
                        f"{base}/stream?topic={quote(topic, safe='/')}&type=mjpeg&quality=70"
                    )
                    self._ros_http_timeout = ros_to
                    self._backend = "ros_web"
                    self._running = True
                    return

        if backend in ("opencv", "auto"):
            indices = cam.get("opencv_indices") or [index]
            indices = [int(x) for x in indices]
            w, h = int(cam.get("width", 640)), int(cam.get("height", 480))
            fps = float(cam.get("fps", 30))
            for idx in indices:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                    cap.set(cv2.CAP_PROP_FPS, fps)
                    self._cap = cap
                    self._backend = "opencv"
                    self._running = True
                    return
                cap.release()

        if backend in ("deptrum",):
            try:
                from deptrum_camera import DeptrumCamera
                dc = DeptrumCamera(resolution_mode_index=int(cam.get("resolution_mode_index", 2)))
                dc.open(); dc.start()
                self._cap = dc
                self._backend = "deptrum"
                self._running = True
                return
            except Exception:
                pass

        if backend in ("picamera2",):
            try:
                from picamera2 import Picamera2
                pc = Picamera2()
                pcfg = pc.create_video_configuration(
                    main={"size": (int(cam.get("width", 640)), int(cam.get("height", 480)))}
                )
                pc.configure(pcfg); pc.start()
                time.sleep(0.3)
                self._cap = pc
                self._backend = "picamera2"
                self._running = True
                return
            except Exception:
                pass

        raise RuntimeError(
            "无法打开任何相机。请检查 config/default.yaml 的 camera.backend 设置；"
            "若相机在 Docker 内：确保 ROS2 web_video_server (http://localhost:8080) 有图像话题。"
        )

    def ensure_open(self) -> None:
        with self._lock:
            if self._running:
                return
            self._open(_cfg())

    def _read_raw(self) -> tuple[bool, np.ndarray | None]:
        if self._backend == "ros_web":
            data = _http_get_bytes(self._ros_rgb_url, self._ros_http_timeout)
            if not data or len(data) < 10:
                return False, None
            arr = np.frombuffer(data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return (True, bgr) if bgr is not None else (False, None)
        if self._backend == "picamera2":
            arr = self._cap.capture_array()
            if arr.ndim == 2:
                return True, cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            if arr.shape[2] == 4:
                return True, cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            return True, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if self._cap is None:
            return False, None
        return self._cap.read()

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if not self._running:
                return False, None
            now = time.monotonic()
            if self._cached_frame is not None and (now - self._cached_t) < self._cache_ttl:
                return True, self._cached_frame.copy()
            ok, bgr = self._read_raw()
            if ok and bgr is not None:
                self._cached_frame = bgr
                self._cached_t = now
            return ok, bgr

    @property
    def ros_stream_url(self) -> str:
        return self._ros_stream_url

    def release(self) -> None:
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    try:
                        self._cap.stop(); self._cap.close()
                    except Exception:
                        pass
                self._cap = None
            self._running = False
            self._backend = ""
            self._ros_rgb_url = ""
            self._ros_stream_url = ""
            self._cached_frame = None

    @property
    def is_open(self) -> bool:
        return self._running

    @property
    def backend_name(self) -> str:
        return self._backend


camera = CameraManager()

# ── Face detector + tracker singletons ────────────────────────────
# Two separate locks; never hold one while waiting for the other.
_detector_lock = threading.Lock()
_tracker_lock = threading.Lock()
_detector: FaceDetector | None = None
_tracker: FaceTracker | None = None


def _get_detector(force_reload: bool = False) -> FaceDetector:
    global _detector
    with _detector_lock:
        if _detector is None or force_reload:
            _detector = FaceDetector(_cfg().get("face_detect", {}))
        return _detector


def _get_tracker() -> FaceTracker:
    global _tracker
    detector = _get_detector()
    cfg = _cfg().get("face_tracking", {})
    with _tracker_lock:
        if _tracker is None:
            _tracker = FaceTracker(
                detector=detector,
                read_frame=camera.read_frame,
                cfg=cfg,
            )
        else:
            _tracker.update_cfg(cfg)
        return _tracker


# ── Routes: page ──────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(str(ROOT / "web"), "index.html")


# ── Routes: camera ────────────────────────────────────────────────
@app.route("/api/camera/open", methods=["POST"])
def api_camera_open():
    try:
        camera.ensure_open()
        return jsonify(ok=True, backend=camera.backend_name)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.route("/api/camera/close", methods=["POST"])
def api_camera_close():
    if _tracker is not None and _tracker.get_state().get("running"):
        _tracker.stop()
    camera.release()
    return jsonify(ok=True)


@app.route("/api/camera/status")
def api_camera_status():
    return jsonify(open=camera.is_open, backend=camera.backend_name)


def _proxy_ros_stream(url: str):
    try:
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req, timeout=30)
        while True:
            chunk = resp.read(8192)
            if not chunk:
                break
            yield chunk
    except GeneratorExit:
        pass
    except Exception:
        pass


def _annotated_mjpeg_gen():
    last_t = 0.0
    while camera.is_open:
        now = time.monotonic()
        if now - last_t < 0.05:
            time.sleep(0.01)
            continue
        last_t = now
        ok, frame = camera.read_frame()
        if not ok or frame is None:
            time.sleep(0.05)
            continue
        try:
            faces = _get_detector().detect(frame)
            primary = pick_face(
                faces, frame.shape[1], frame.shape[0],
                strategy=str(_cfg().get("face_tracking", {}).get("pick", "largest")),
            )
            frame = draw_faces(frame, faces, primary=primary)
        except Exception:
            pass
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


@app.route("/api/stream")
def api_stream():
    if not camera.is_open:
        try:
            camera.ensure_open()
        except Exception:
            return "相机未打开", 503
    annotate = request.args.get("annotate", "1") == "1"
    if not annotate and camera.backend_name == "ros_web" and camera.ros_stream_url:
        return Response(
            _proxy_ros_stream(camera.ros_stream_url),
            mimetype="multipart/x-mixed-replace;boundary=boundarydonotcross",
        )
    return Response(_annotated_mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/snapshot")
def api_snapshot():
    if not camera.is_open:
        try:
            camera.ensure_open()
        except Exception as e:
            return str(e), 503
    ok, frame = camera.read_frame()
    if not ok or frame is None:
        return "无法读取帧", 500
    annotate = request.args.get("annotate", "1") == "1"
    dets: list[dict[str, Any]] = []
    if annotate:
        dets = _get_detector().detect(frame)
        primary = pick_face(
            dets, frame.shape[1], frame.shape[0],
            strategy=str(_cfg().get("face_tracking", {}).get("pick", "largest")),
        )
        frame = draw_faces(frame, dets, primary=primary)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    resp = Response(buf.tobytes(), mimetype="image/jpeg")
    resp.headers["X-Faces"] = json.dumps(dets, ensure_ascii=False)
    return resp


# ── Routes: face detect (one-shot) ────────────────────────────────
@app.route("/api/face/detect")
def api_face_detect():
    if not camera.is_open:
        try:
            camera.ensure_open()
        except Exception as e:
            return jsonify(ok=False, error=str(e)), 503
    ok, frame = camera.read_frame()
    if not ok or frame is None:
        return jsonify(ok=False, error="无法读取帧"), 500
    det = _get_detector()
    faces = det.detect(frame)
    h, w = frame.shape[:2]
    primary = pick_face(
        faces, w, h, strategy=str(_cfg().get("face_tracking", {}).get("pick", "largest"))
    )
    return jsonify(
        ok=True,
        backend=det.effective_backend,
        img_size=[w, h],
        faces=faces,
        primary=primary,
    )


# ── Routes: tracker control ───────────────────────────────────────
@app.route("/api/track/start", methods=["POST"])
def api_track_start():
    try:
        camera.ensure_open()
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 503
    tr = _get_tracker()
    tr.start()
    return jsonify(ok=True, state=tr.get_state(), arm_bridge=_bridge_healthy())


@app.route("/api/track/stop", methods=["POST"])
def api_track_stop():
    if _tracker is None:
        return jsonify(ok=True, state={"running": False})
    _tracker.stop()
    return jsonify(ok=True, state=_tracker.get_state())


@app.route("/api/track/status")
def api_track_status():
    if _tracker is None:
        return jsonify(
            running=False,
            backend=_get_detector().effective_backend,
            arm_bridge=_bridge_healthy(),
        )
    s = _tracker.get_state()
    s["arm_bridge"] = _bridge_healthy()
    return jsonify(s)


@app.route("/api/arm/home", methods=["POST"])
def api_arm_home():
    if not _bridge_healthy():
        return jsonify(ok=False, error="arm_bridge 不可达 (http://localhost:9091)"), 503
    cfg = _cfg().get("face_tracking", {})
    home = cfg.get("home_pose", {}) or {}
    body = json.dumps({
        "x": float(home.get("x", 0.18)),
        "y": float(home.get("y", 0.0)),
        "z": float(home.get("z", 0.10)),
        "pitch": float(home.get("pitch", -10.0)),
        "pitch_range": [
            float(cfg.get("pitch_min_deg", -45)),
            float(cfg.get("pitch_max_deg", 35)),
        ],
        "duration": 0.6,
    }).encode()
    req = urllib.request.Request(
        "http://localhost:9091/ik_move", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=4) as r:
            payload = json.loads(r.read())
        if _tracker is not None:
            _tracker._set(yaw_deg=0.0, pitch_deg=float(home.get("pitch", -10.0)))
        return jsonify(ok=True, **payload)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.route("/api/arm/status")
def api_arm_status():
    return jsonify(arm_bridge=_bridge_healthy())


# ── Routes: config ────────────────────────────────────────────────
@app.route("/api/config")
def api_config_get():
    return jsonify(_cfg())


@app.route("/api/config", methods=["POST"])
def api_config_post():
    data = request.get_json(silent=True) or {}
    persist = bool(data.get("persist", False))
    patch = data.get("patch") or {}
    if not isinstance(patch, dict):
        return jsonify(ok=False, error="patch 必须是对象"), 400

    with _cfg_lock:
        for top, sub in patch.items():
            if isinstance(sub, dict) and isinstance(_runtime_overrides.get(top), dict):
                _runtime_overrides[top].update(sub)
            else:
                _runtime_overrides[top] = sub

        new_cfg = _cfg()

        if persist:
            try:
                save_config(new_cfg)
            except Exception as e:
                return jsonify(ok=False, error=f"保存失败: {e}"), 500

    if "face_detect" in patch:
        _get_detector(force_reload=True)
    if "face_tracking" in patch and _tracker is not None:
        _tracker.update_cfg(new_cfg.get("face_tracking", {}))

    return jsonify(ok=True, applied=patch, persisted=persist, config=new_cfg)


# ── Routes: health ────────────────────────────────────────────────
@app.route("/api/health")
def api_health():
    return jsonify(
        ok=True,
        camera_open=camera.is_open,
        camera_backend=camera.backend_name,
        tracker_running=bool(_tracker and _tracker.get_state().get("running")),
        arm_bridge=_bridge_healthy(),
        face_backend=_get_detector().effective_backend,
    )


if __name__ == "__main__":
    port = int(__import__("os").environ.get("PORT", "5000"))
    print(f"WonderPi 人脸追踪 WebUI starting on http://0.0.0.0:{port}", flush=True)
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
