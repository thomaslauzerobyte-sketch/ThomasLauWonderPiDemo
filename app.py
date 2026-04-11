#!/usr/bin/env python3
"""WonderPi Demo Web UI — Flask app integrating camera, recording, frame extraction, inference, arm control."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from common import load_config, resolve_path  # noqa: E402

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(ROOT / "web" / "static"), template_folder=str(ROOT / "web"))


# ── Config ────────────────────────────────────────────────────────
_runtime_overrides: dict[str, Any] = {}


def _cfg() -> dict[str, Any]:
    cfg = load_config()
    cfg.update(_runtime_overrides)
    return cfg


# ── Camera Singleton ──────────────────────────────────────────────
import urllib.request
from urllib.parse import quote, unquote


def _http_get_bytes(url: str, timeout_sec: float) -> bytes | None:
    """HTTP GET with hard wall-clock limit (ROS snapshot can hang without this)."""
    timeout_sec = max(1.0, float(timeout_sec))
    curl = shutil.which("curl")
    if curl:
        try:
            r = subprocess.run(
                [
                    curl,
                    "-sS",
                    "--connect-timeout",
                    "2",
                    "--max-time",
                    str(int(timeout_sec + 0.99)),
                    "-o",
                    "-",
                    url,
                ],
                capture_output=True,
                timeout=timeout_sec + 1.5,
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
    """Parse web_video_server index HTML for /snapshot?topic=... links."""
    idx_url = ros_base.rstrip("/") + "/"
    raw = _http_get_bytes(idx_url, min(4.0, float(timeout_sec)))
    if not raw:
        return []
    html = raw.decode("utf-8", errors="ignore")
    topics = re.findall(r"/snapshot\?topic=([^\"'>\s]+)", html)
    seen: set[str] = set()
    out: list[str] = []
    base = ros_base.rstrip("/")
    for t in topics:
        t = unquote(t)
        if t in seen:
            continue
        seen.add(t)
        out.append(f"{base}/snapshot?topic={quote(t, safe='/')}")
    return out


class CameraManager:
    """Thread-safe camera manager.

    Supports multiple backends:
      - ros_web: read frames from ROS2 web_video_server (default when Docker
        container occupies the physical camera)
      - opencv / deptrum / picamera2: direct hardware access (fallback)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cap: Any = None
        self._backend: str = ""
        self._running = False
        self._ros_rgb_url: str = ""
        self._ros_depth_url: str = ""
        self._ros_stream_url: str = ""
        self._ros_stereo_right_url: str = ""
        self._stereo: dict[str, Any] = {}
        self._ros_http_timeout: float = 5.0

    @staticmethod
    def _depth_roi_median_ok(depth: np.ndarray | None, min_valid: int = 50) -> bool:
        if depth is None or depth.size == 0:
            return False
        h, w = depth.shape[:2]
        cy, cx = h // 2, w // 2
        patch = depth[max(0, cy - 24) : cy + 24, max(0, cx - 24) : cx + 24]
        v = patch[patch > min_valid]
        return v.size >= 20

    def _open(self, cfg: dict[str, Any]) -> None:
        cam = cfg["camera"]
        backend = cam.get("backend", "auto").lower()
        index = int(cam.get("index", 0))
        ros_base = cam.get("ros_web_video_url", "http://localhost:8080")

        if backend in ("ros_web", "auto"):
            ros_to = float(cam.get("ros_web_timeout_sec", 4.0))
            base = ros_base.rstrip("/")
            topics = cam.get("ros_snapshot_topics") or [
                "/depth_cam/rgb/image_raw",
                "/calibration/image_result",
            ]
            snap_urls: list[str] = []
            for topic in topics:
                topic = str(topic).strip()
                if not topic.startswith("/"):
                    topic = "/" + topic
                snap_urls.append(f"{base}/snapshot?topic={quote(topic, safe='/')}")
            for u in _ros_discover_snapshot_urls(ros_base, ros_to):
                if u not in snap_urls:
                    snap_urls.append(u)
            for snap_url in snap_urls:
                body = _http_get_bytes(snap_url, ros_to)
                if body and len(body) > 100:
                    self._ros_rgb_url = snap_url
                    m = re.search(r"[?&]topic=([^&]+)", snap_url)
                    topic = unquote(m.group(1)) if m else ""
                    self._ros_depth_url = (
                        f"{base}/snapshot?topic=/depth_cam/depth/image_raw"
                        if topic and "depth_cam" in topic
                        else ""
                    )
                    st = topic or "/depth_cam/rgb/image_raw"
                    self._ros_stream_url = (
                        f"{base}/stream?topic={quote(st, safe='/')}&type=mjpeg&quality=70"
                    )
                    self._ros_http_timeout = ros_to
                    self._backend = "ros_web"
                    self._running = True
                    st = cam.get("stereo") or {}
                    self._ros_stereo_right_url = ""
                    self._stereo = {}
                    if st.get("enabled"):
                        rt = str(st.get("right_topic", "")).strip()
                        if rt:
                            if not rt.startswith("/"):
                                rt = "/" + rt
                            self._ros_stereo_right_url = (
                                f"{base}/snapshot?topic={quote(rt, safe='/')}"
                            )
                            self._stereo = dict(st)
                    return

        if backend in ("opencv", "auto"):
            indices = cam.get("opencv_indices")
            if not indices:
                indices = [index]
            else:
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
                dc.open()
                dc.start()
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
                pc.configure(pcfg)
                pc.start()
                time.sleep(0.3)
                self._cap = pc
                self._backend = "picamera2"
                self._running = True
                return
            except Exception:
                pass

        raise RuntimeError(
            "无法打开任何相机。"
            "若相机在 Docker 内：请启动深度相机 ROS 节点（如 aurora930_node），"
            "使 http://127.0.0.1:8080 上至少一个图像话题有数据；"
            "或暂时将 camera.backend 改为 opencv 并指定可用的 opencv_indices。"
            "详见 config/default.yaml 中 camera 段。"
        )

    def ensure_open(self) -> None:
        with self._lock:
            if self._running:
                return
            self._open(_cfg())

    def _fetch_snapshot(self, url: str) -> np.ndarray | None:
        if not url:
            return None
        try:
            data = _http_get_bytes(url, self._ros_http_timeout)
            if not data or len(data) < 10:
                return None
            arr = np.frombuffer(data, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        except Exception:
            return None

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if not self._running:
                return False, None
            if self._backend == "ros_web":
                bgr = self._fetch_snapshot(self._ros_rgb_url)
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

    def read_frame_with_depth(self) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        """Read BGR + depth."""
        with self._lock:
            if not self._running:
                return False, None, None
            if self._backend == "ros_web":
                bgr = self._fetch_snapshot(self._ros_rgb_url)
                if bgr is None:
                    return False, None, None
                depth_img = (
                    self._fetch_snapshot(self._ros_depth_url)
                    if self._ros_depth_url
                    else None
                )
                depth_16 = None
                if depth_img is not None:
                    if depth_img.ndim == 3:
                        depth_16 = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY).astype(
                            np.uint16
                        )
                    else:
                        depth_16 = depth_img.astype(np.uint16)
                use_stereo = bool(self._stereo.get("enabled")) and bool(
                    self._ros_stereo_right_url
                )
                if use_stereo and (
                    depth_16 is None or not self._depth_roi_median_ok(depth_16)
                ):
                    from stereo_depth import depth_from_stereo_pair, effective_focal_px

                    rbgr = self._fetch_snapshot(self._ros_stereo_right_url)
                    if rbgr is not None:
                        w = int(bgr.shape[1])
                        fp = effective_focal_px(self._stereo, w)
                        b_mm = float(self._stereo.get("baseline_mm", 50))
                        try:
                            depth_16 = depth_from_stereo_pair(
                                bgr,
                                rbgr,
                                baseline_mm=b_mm,
                                focal_px=fp,
                                min_disparity=int(self._stereo.get("min_disparity", 0)),
                                num_disparities=int(self._stereo.get("num_disparities", 128)),
                                block_size=int(self._stereo.get("block_size", 5)),
                            )
                        except Exception:
                            pass
                return True, bgr, depth_16
            if self._backend == "deptrum" and self._cap and hasattr(self._cap, "read_all"):
                ok, bgr, depth, _ir = self._cap.read_all()
                return ok, bgr, depth
            if self._backend == "picamera2":
                arr = self._cap.capture_array()
                if arr.ndim == 2:
                    return True, cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR), None
                if arr.shape[2] == 4:
                    return True, cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR), None
                return True, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), None
            if self._cap is None:
                return False, None, None
            ok, bgr = self._cap.read()
            return ok, bgr, None

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
                        self._cap.stop()
                        self._cap.close()
                    except Exception:
                        pass
                self._cap = None
            self._running = False
            self._backend = ""
            self._ros_rgb_url = ""
            self._ros_depth_url = ""
            self._ros_stream_url = ""
            self._ros_stereo_right_url = ""
            self._stereo = {}

    @property
    def is_open(self) -> bool:
        return self._running

    @property
    def backend_name(self) -> str:
        return self._backend


camera = CameraManager()
_BRIDGE_URL = "http://localhost:9091"


# ── Inference helpers (reuse scripts/infer.py logic) ──────────────
def _infer_bgr(bgr: np.ndarray, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    from infer import detect_haar, classify_onnx, _load_class_names
    inf = cfg.get("inference", {})
    method = inf.get("method", "haar")
    paths = cfg.get("paths", {})

    if method == "haar":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return detect_haar(gray)
    elif method == "onnx":
        model_path = resolve_path(paths.get("model_path", "models/model.onnx"))
        sz = inf.get("onnx_input_size", [224, 224])
        names_path = paths.get("class_names")
        names = _load_class_names(resolve_path(names_path) if names_path else None)
        return classify_onnx(bgr, model_path, (int(sz[0]), int(sz[1])), bool(inf.get("normalize", True)), names)
    return []


def _draw_detections(bgr: np.ndarray, dets: list[dict[str, Any]]) -> np.ndarray:
    out = bgr.copy()
    for d in dets:
        bb = d.get("bbox")
        label = d.get("label", "?")
        score = d.get("score", 0)
        if bb and len(bb) == 4:
            x, y, w, h = [int(v) for v in bb]
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(out, f"{label} {score:.2f}", (x, max(y - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


# ── Arm driver singleton ─────────────────────────────────────────
_arm_driver_lock = threading.Lock()
_arm_driver: Any = None


def _get_arm_driver():
    global _arm_driver
    with _arm_driver_lock:
        if _arm_driver is None:
            from arm_demo import make_driver
            _arm_driver = make_driver(_cfg().get("arm", {}))
        return _arm_driver


def _reset_arm_driver():
    global _arm_driver
    with _arm_driver_lock:
        _arm_driver = None


# ── Arm helper ────────────────────────────────────────────────────
def _run_arm_from_dets(dets: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    from arm_demo import _pick_detection, pixel_from_detection, to_workspace
    arm_cfg = cfg.get("arm", {})
    img_w = int(arm_cfg.get("image_width", 640))
    img_h = int(arm_cfg.get("image_height", 480))
    payload = {"detections": dets}
    det = _pick_detection(payload)
    if det is None:
        return {"status": "skip", "reason": "无检测结果"}
    pix = pixel_from_detection(det, img_w, img_h)
    if pix is None:
        return {"status": "skip", "reason": "无法提取像素坐标"}
    wx, wy = to_workspace(pix[0], pix[1], img_w, img_h)
    driver = _get_arm_driver()
    result = {
        "status": "ok",
        "workspace": {"x": round(wx, 4), "y": round(wy, 4)},
        "pixel": {"x": round(pix[0], 1), "y": round(pix[1], 1)},
        "detection": det,
        "arm_mode": arm_cfg.get("driver", "mock"),
    }
    try:
        driver.move_to_workspace(wx, wy, {"label": det.get("label"), "score": det.get("score")})
        result["sent"] = True
    except Exception as e:
        result["sent"] = False
        result["error"] = str(e)
    return result


# ── Routes: Pages ─────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(str(ROOT / "web"), "index.html")


# ── Routes: Camera ────────────────────────────────────────────────
@app.route("/api/camera/open", methods=["POST"])
def api_camera_open():
    try:
        camera.ensure_open()
        return jsonify(ok=True, backend=camera.backend_name)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.route("/api/camera/close", methods=["POST"])
def api_camera_close():
    camera.release()
    return jsonify(ok=True)


@app.route("/api/camera/status")
def api_camera_status():
    return jsonify(open=camera.is_open, backend=camera.backend_name)


def _mjpeg_gen():
    while camera.is_open:
        ok, frame = camera.read_frame()
        if not ok or frame is None:
            time.sleep(0.05)
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.033)


def _proxy_ros_stream(url: str):
    """Proxy ROS2 web_video_server MJPEG stream, preserving its boundary."""
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


@app.route("/api/stream")
def api_stream():
    if not camera.is_open:
        try:
            camera.ensure_open()
        except Exception:
            return "相机未打开", 503
    if camera.backend_name == "ros_web" and camera.ros_stream_url:
        return Response(
            _proxy_ros_stream(camera.ros_stream_url),
            mimetype="multipart/x-mixed-replace;boundary=boundarydonotcross",
        )
    return Response(_mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Routes: Snapshot + live infer ─────────────────────────────────
@app.route("/api/snapshot")
def api_snapshot():
    """Grab one frame, optionally run inference, return JPEG with detections drawn."""
    if not camera.is_open:
        return "相机未打开", 503
    ok, frame = camera.read_frame()
    if not ok or frame is None:
        return "无法读取帧", 500
    run_infer = request.args.get("infer", "0") == "1"
    dets: list[dict[str, Any]] = []
    if run_infer:
        dets = _infer_bgr(frame, _cfg())
        frame = _draw_detections(frame, dets)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    resp = Response(buf.tobytes(), mimetype="image/jpeg")
    if dets:
        resp.headers["X-Detections"] = json.dumps(dets, ensure_ascii=False)
    return resp


# ── Routes: Record ────────────────────────────────────────────────
_recording_lock = threading.Lock()
_recording = False


@app.route("/api/record", methods=["POST"])
def api_record():
    global _recording
    with _recording_lock:
        if _recording:
            return jsonify(ok=False, error="正在录制中"), 409
        _recording = True

    data = request.get_json(silent=True) or {}
    seconds = float(data.get("seconds", 10))
    cfg = _cfg()
    cam_cfg = cfg["camera"]
    rec_cfg = cfg["recording"]
    paths_cfg = cfg["paths"]
    videos_dir = resolve_path(paths_cfg["videos_dir"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = rec_cfg.get("filename_prefix", "capture")
    out_path = videos_dir / f"{prefix}_{stamp}.mp4"
    width = int(cam_cfg.get("width", 640))
    height = int(cam_cfg.get("height", 480))
    fps = float(cam_cfg.get("fps", 30))
    fourcc_name = str(rec_cfg.get("fourcc", "mp4v"))

    def do_record():
        global _recording
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*(fourcc_name.lower()))
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                return

            if not camera.is_open:
                camera.ensure_open()

            t_end = time.monotonic() + seconds
            count = 0
            while time.monotonic() < t_end and camera.is_open:
                ok, frame = camera.read_frame()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
                count += 1
                time.sleep(max(0, (1.0 / fps) * 0.1))
            writer.release()
            if count == 0:
                out_path.unlink(missing_ok=True)
        finally:
            with _recording_lock:
                _recording = False

    threading.Thread(target=do_record, daemon=True).start()
    return jsonify(ok=True, file=out_path.name, seconds=seconds)


@app.route("/api/record/status")
def api_record_status():
    return jsonify(recording=_recording)


# ── Routes: Videos ────────────────────────────────────────────────
@app.route("/api/videos")
def api_videos():
    cfg = _cfg()
    vdir = resolve_path(cfg["paths"]["videos_dir"])
    if not vdir.is_dir():
        return jsonify(videos=[])
    exts = {".mp4", ".avi", ".mkv"}
    files = sorted(
        ({"name": p.name, "size_kb": round(p.stat().st_size / 1024, 1)}
         for p in vdir.iterdir() if p.suffix.lower() in exts),
        key=lambda x: x["name"], reverse=True,
    )
    return jsonify(videos=files)


@app.route("/api/video/<name>")
def api_video_file(name: str):
    cfg = _cfg()
    vdir = resolve_path(cfg["paths"]["videos_dir"])
    p = vdir / name
    if not p.is_file():
        return "Not found", 404
    return send_file(p, mimetype="video/mp4")


# ── Routes: Extract Frames ────────────────────────────────────────
@app.route("/api/extract", methods=["POST"])
def api_extract():
    data = request.get_json(silent=True) or {}
    video_name = data.get("video")
    step = int(data.get("step", 1))
    if not video_name:
        return jsonify(ok=False, error="缺少 video 参数"), 400

    cfg = _cfg()
    vdir = resolve_path(cfg["paths"]["videos_dir"])
    video_path = vdir / video_name
    if not video_path.is_file():
        return jsonify(ok=False, error=f"视频不存在: {video_name}"), 404

    ex = cfg["extract"]
    frames_root = resolve_path(cfg["paths"]["frames_dir"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = ex.get("subdir_prefix", "run")
    out_dir = frames_root / f"{prefix}_{stamp}"

    from extract_frames import extract
    try:
        n = extract(
            video_path=video_path,
            output_dir=out_dir,
            frame_step=step,
            image_format=str(ex.get("image_format", "jpg")),
            jpeg_quality=int(ex.get("jpeg_quality", 92)),
        )
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500

    return jsonify(ok=True, count=n, dir=out_dir.name)


# ── Routes: Frames listing ────────────────────────────────────────
@app.route("/api/frames")
def api_frames_dirs():
    cfg = _cfg()
    fdir = resolve_path(cfg["paths"]["frames_dir"])
    if not fdir.is_dir():
        return jsonify(dirs=[])
    dirs = sorted(
        ({"name": d.name, "count": sum(1 for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))}
         for d in fdir.iterdir() if d.is_dir()),
        key=lambda x: x["name"], reverse=True,
    )
    return jsonify(dirs=dirs)


@app.route("/api/frames/<dirname>")
def api_frames_list(dirname: str):
    cfg = _cfg()
    fdir = resolve_path(cfg["paths"]["frames_dir"]) / dirname
    if not fdir.is_dir():
        return jsonify(files=[]), 404
    exts = {".jpg", ".jpeg", ".png"}
    files = sorted(p.name for p in fdir.iterdir() if p.suffix.lower() in exts)
    return jsonify(dir=dirname, files=files)


@app.route("/api/frame/<dirname>/<filename>")
def api_frame_file(dirname: str, filename: str):
    cfg = _cfg()
    fdir = resolve_path(cfg["paths"]["frames_dir"]) / dirname
    p = fdir / filename
    if not p.is_file():
        return "Not found", 404
    return send_file(p)


# ── Routes: Inference ─────────────────────────────────────────────
@app.route("/api/infer", methods=["POST"])
def api_infer():
    data = request.get_json(silent=True) or {}
    frame_dir = data.get("dir")
    frame_file = data.get("file")
    cfg = _cfg()
    froot = resolve_path(cfg["paths"]["frames_dir"])

    results = []
    if frame_dir:
        d = froot / frame_dir
        if not d.is_dir():
            return jsonify(ok=False, error="目录不存在"), 404
        exts = {".jpg", ".jpeg", ".png"}
        imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in exts)
        for p in imgs[:50]:
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue
            dets = _infer_bgr(bgr, cfg)
            results.append({"file": p.name, "detections": dets})
    elif frame_file and "/" in frame_file:
        parts = frame_file.split("/", 1)
        p = froot / parts[0] / parts[1]
        if not p.is_file():
            return jsonify(ok=False, error="文件不存在"), 404
        bgr = cv2.imread(str(p))
        if bgr is None:
            return jsonify(ok=False, error="无法读取图像"), 500
        dets = _infer_bgr(bgr, cfg)
        results.append({"file": p.name, "detections": dets})
    else:
        return jsonify(ok=False, error="缺少 dir 或 file 参数"), 400

    total_dets = sum(len(r["detections"]) for r in results)
    return jsonify(ok=True, results=results, total_detections=total_dets)


@app.route("/api/infer/annotated/<dirname>/<filename>")
def api_infer_annotated(dirname: str, filename: str):
    """Return a JPEG with detection boxes drawn."""
    cfg = _cfg()
    p = resolve_path(cfg["paths"]["frames_dir"]) / dirname / filename
    if not p.is_file():
        return "Not found", 404
    bgr = cv2.imread(str(p))
    if bgr is None:
        return "读取失败", 500
    dets = _infer_bgr(bgr, cfg)
    out = _draw_detections(bgr, dets)
    _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(buf.tobytes(), mimetype="image/jpeg")


# ── Routes: Arm Control ──────────────────────────────────────────
@app.route("/api/arm", methods=["POST"])
def api_arm():
    data = request.get_json(silent=True) or {}
    dets = data.get("detections", [])
    if not dets:
        return jsonify(ok=False, error="无检测结果"), 400
    cfg = _cfg()
    result = _run_arm_from_dets(dets, cfg)
    return jsonify(ok=True, **result)


# ── Routes: Pipeline ─────────────────────────────────────────────
@app.route("/api/pipeline", methods=["POST"])
def api_pipeline():
    """Snapshot → infer → arm (all in one call)."""
    if not camera.is_open:
        return jsonify(ok=False, error="相机未打开"), 503
    ok, frame = camera.read_frame()
    if not ok or frame is None:
        return jsonify(ok=False, error="无法读取帧"), 500

    cfg = _cfg()
    dets = _infer_bgr(frame, cfg)
    annotated = _draw_detections(frame, dets)
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

    arm_result = _run_arm_from_dets(dets, cfg) if dets else {"status": "skip", "reason": "无检测结果"}

    import base64
    img_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    return jsonify(
        ok=True,
        detections=dets,
        arm=arm_result,
        image="data:image/jpeg;base64," + img_b64,
    )


# ── Routes: Arm Direct Control ────────────────────────────────────
@app.route("/api/arm/status")
def api_arm_status():
    cfg = _cfg()
    arm_cfg = cfg.get("arm", {})
    driver_type = arm_cfg.get("driver", "mock")
    return jsonify(ok=True, driver=driver_type, mock=arm_cfg.get("mock", True))


@app.route("/api/arm/state")
def api_arm_state():
    """Fetch live arm joint state from arm_bridge for 3D preview."""
    try:
        with urllib.request.urlopen(f"{_BRIDGE_URL}/state", timeout=2.0) as r:
            data = json.loads(r.read().decode("utf-8"))
        return jsonify(data)
    except Exception as e:
        return jsonify(ok=False, error=f"arm_bridge state unavailable: {e}"), 503


@app.route("/api/arm/home", methods=["POST"])
def api_arm_home():
    try:
        d = _get_arm_driver()
        d.home()
        log = d.action_log[-1] if hasattr(d, "action_log") and d.action_log else {}
        return jsonify(ok=True, action="home", log=log)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.route("/api/arm/grip", methods=["POST"])
def api_arm_grip():
    data = request.get_json(silent=True) or {}
    action = data.get("action", "open")
    try:
        d = _get_arm_driver()
        if action == "close":
            d.grip_close()
        else:
            d.grip_open()
        log = d.action_log[-1] if hasattr(d, "action_log") and d.action_log else {}
        return jsonify(ok=True, action=f"grip_{action}", log=log)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.route("/api/arm/move", methods=["POST"])
def api_arm_move():
    """Move arm to x,y,z (meters) using IK."""
    data = request.get_json(silent=True) or {}
    x = data.get("x")
    y = data.get("y")
    z = data.get("z")
    if x is None or y is None or z is None:
        return jsonify(ok=False, error="缺少 x/y/z 参数"), 400
    try:
        d = _get_arm_driver()
        d.move_to(float(x), float(y), float(z))
        log = d.action_log[-1] if hasattr(d, "action_log") and d.action_log else {}
        return jsonify(ok=True, action="move_to", x=x, y=y, z=z, log=log)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.route("/api/arm/servos", methods=["POST"])
def api_arm_servos():
    """Directly set servo positions (for HiWonder driver only)."""
    data = request.get_json(silent=True) or {}
    positions = data.get("positions", {})
    duration = float(data.get("duration", 1.0))
    if not positions:
        return jsonify(ok=False, error="缺少 positions 参数"), 400
    try:
        d = _get_arm_driver()
        if hasattr(d, "set_servos"):
            pos_int = {int(k): int(v) for k, v in positions.items()}
            d.set_servos(pos_int, duration=duration)
            return jsonify(ok=True, action="set_servos", positions=pos_int)
        else:
            return jsonify(ok=False, error="当前驱动不支持直接舵机控制"), 400
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.route("/api/arm/buzzer", methods=["POST"])
def api_arm_buzzer():
    try:
        d = _get_arm_driver()
        if hasattr(d, "buzzer"):
            d.buzzer()
            return jsonify(ok=True, action="buzzer")
        return jsonify(ok=False, error="当前驱动不支持蜂鸣器"), 400
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


# ── Routes: Detection Config ──────────────────────────────────────
@app.route("/api/config/detect_method", methods=["GET", "POST"])
def api_detect_method():
    if request.method == "GET":
        cfg = _cfg()
        method = cfg.get("detect_method", "blue")
        cb = cfg.get("checkerboard", {})
        return jsonify(ok=True, method=method,
                       checkerboard={"cols": cb.get("cols", 5), "rows": cb.get("rows", 4)})
    data = request.get_json(silent=True) or {}
    method = data.get("method", "blue")
    if method not in ("blue", "checkerboard", "tag"):
        return jsonify(ok=False, error="method 仅支持 blue / checkerboard / tag"), 400
    _runtime_overrides["detect_method"] = method
    if "cols" in data or "rows" in data:
        cb = dict(_cfg().get("checkerboard", {}))
        if "cols" in data:
            cb["cols"] = int(data["cols"])
        if "rows" in data:
            cb["rows"] = int(data["rows"])
        _runtime_overrides["checkerboard"] = cb
    return jsonify(ok=True, method=method)


# ── Routes: Object Detect ────────────────────────────────────────
@app.route("/api/detect/blue", methods=["POST"])
def api_detect_blue():
    """Grab one frame, detect object, return annotated JPEG + JSON."""
    if not camera.is_open:
        return jsonify(ok=False, error="相机未打开"), 503
    from blue_detector import detect_from_config, draw_detection
    ok, bgr, depth = camera.read_frame_with_depth()
    if not ok or bgr is None:
        return jsonify(ok=False, error="无法读取帧"), 500
    cfg = _cfg()
    det = detect_from_config(bgr, cfg, depth=depth)
    annotated = draw_detection(bgr, det)
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    import base64
    img_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return jsonify(
        ok=True,
        detection=det,
        image="data:image/jpeg;base64," + img_b64,
    )


def _mjpeg_detect_gen():
    from blue_detector import detect_from_config, draw_detection
    cfg = _cfg()
    while camera.is_open:
        ok, bgr, depth = camera.read_frame_with_depth()
        if not ok or bgr is None:
            time.sleep(0.05)
            continue
        det = detect_from_config(bgr, cfg, depth=depth)
        annotated = draw_detection(bgr, det)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.066)


@app.route("/api/stream/detect")
def api_stream_detect():
    if not camera.is_open:
        return "相机未打开", 503
    return Response(_mjpeg_detect_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Routes: Calibration ──────────────────────────────────────────
_calib_points: list[dict[str, Any]] = []


@app.route("/api/calibrate/status")
def api_calib_status():
    from calibration import load_calibration
    from calibration3d import load_calibration3d
    calib = load_calibration()
    calib3d = load_calibration3d()
    return jsonify(
        ok=True,
        points=_calib_points,
        calibrated=calib is not None and "H" in (calib or {}),
        calibrated3d=calib3d is not None and "A" in (calib3d or {}),
        saved_points=calib.get("points", []) if calib else [],
        saved_points3d=calib3d.get("points", []) if calib3d else [],
    )


@app.route("/api/calibrate/point", methods=["POST"])
def api_calib_point():
    """Record one calibration point: detect blue block pixel/depth + user-supplied arm coords.

    数据结构（同时服务于 2D Homography 与 depth-aware 3D 标定）:
        {
          "pixel": [u, v],
          "arm": [x, y],
          "depth_mm": d  # 若有深度，则记录
        }
    """
    data = request.get_json(silent=True) or {}
    arm_x = data.get("arm_x")
    arm_y = data.get("arm_y")
    if arm_x is None or arm_y is None:
        return jsonify(ok=False, error="缺少 arm_x / arm_y"), 400

    if not camera.is_open:
        return jsonify(ok=False, error="相机未打开"), 503
    from blue_detector import detect_from_config
    ok, bgr, depth = camera.read_frame_with_depth()
    if not ok or bgr is None:
        return jsonify(ok=False, error="无法读取帧"), 500
    cfg = _cfg()
    det = detect_from_config(bgr, cfg, depth=depth)
    if det is None:
        method = cfg.get("detect_method", "blue")
        if method == "checkerboard":
            hint = "棋盘格标定板"
        elif method == "tag":
            hint = "黑白标记板(TAG)"
        else:
            hint = "蓝色物体"
        return jsonify(ok=False, error=f"未检测到{hint}，请将目标放入视野"), 400

    depth_mm = det.get("depth_mm")

    point = {
        "pixel": list(det["center_px"]),
        "arm": [float(arm_x), float(arm_y)],
        "depth_mm": int(depth_mm) if depth_mm is not None else None,
    }
    _calib_points.append(point)
    return jsonify(ok=True, point=point, total=len(_calib_points))


@app.route("/api/calibrate/board3d", methods=["POST"])
def api_calib_board3d():
    """一次拍摄棋盘格，按格距生成多组 (pixel, depth_mm, arm) 写入标定点列表。"""
    data = request.get_json(silent=True) or {}
    if data.get("arm_x_mm") is not None and data.get("arm_y_mm") is not None:
        ox = float(data["arm_x_mm"]) / 1000.0
        oy = float(data["arm_y_mm"]) / 1000.0
    elif data.get("arm_x") is not None and data.get("arm_y") is not None:
        ox = float(data["arm_x"])
        oy = float(data["arm_y"])
    else:
        return jsonify(
            ok=False,
            error="缺少坐标：请传 arm_x_mm/arm_y_mm（毫米）或 arm_x/arm_y（米）",
        ), 400

    if not camera.is_open:
        return jsonify(ok=False, error="相机未打开"), 503
    ok, bgr, depth = camera.read_frame_with_depth()
    if not ok or bgr is None:
        return jsonify(ok=False, error="无法读取帧"), 500
    if depth is None:
        return jsonify(
            ok=False,
            error=(
                "无深度图：请保证深度话题有数据，或在 config/default.yaml 中设置 "
                "camera.stereo.enabled=true，并正确填写 right_topic、baseline_mm、"
                "focal_px（或 focal_ratio_of_width）"
            ),
        ), 400

    cfg = _cfg()
    bc = cfg.get("calibration_board", {})
    cols = int(data.get("cols") or bc.get("cols", 5))
    rows = int(data.get("rows") or bc.get("rows", 4))
    square_mm = float(data.get("square_size_mm") or bc.get("square_size_mm", 25))
    yaw = float(data.get("board_yaw_deg", bc.get("board_yaw_deg", 0)))

    from board_calib3d import (
        board_corners_to_calib_points,
        draw_chessboard_overlay,
        find_chessboard_corners,
    )

    corners = find_chessboard_corners(bgr, cols, rows)
    if corners is None:
        return jsonify(
            ok=False,
            error=f"未检测到 {cols}×{rows} 棋盘格内角点，请调整光照/距离或修改 calibration_board.cols/rows",
        ), 400

    try:
        new_points, meta = board_corners_to_calib_points(
            corners,
            depth,
            pattern_cols=cols,
            pattern_rows=rows,
            square_size_mm=square_mm,
            origin_arm_x_m=ox,
            origin_arm_y_m=oy,
            board_yaw_deg=yaw,
        )
    except ValueError as e:
        return jsonify(ok=False, error=str(e)), 400

    if len(new_points) < 4:
        return jsonify(
            ok=False,
            error=f"有效深度角点不足 4 个（当前 {len(new_points)}），请对准深度有效的区域或检查标定板",
            meta=meta,
        ), 400

    for p in new_points:
        _calib_points.append(
            {
                "pixel": p["pixel"],
                "arm": p["arm"],
                "depth_mm": p["depth_mm"],
            }
        )

    resp: dict[str, Any] = dict(
        ok=True,
        added=len(new_points),
        total=len(_calib_points),
        meta=meta,
    )
    if data.get("include_preview"):
        vis = draw_chessboard_overlay(bgr, corners, cols, rows)
        _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        import base64

        resp["preview"] = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(
            "ascii"
        )
    return jsonify(resp)


@app.route("/api/calibrate/reset", methods=["POST"])
def api_calib_reset():
    _calib_points.clear()
    return jsonify(ok=True)


@app.route("/api/calibrate/compute", methods=["POST"])
def api_calib_compute():
    from calibration import calibrate_and_save
    from calibration3d import calibrate3d_and_save

    points = _calib_points[:]
    if len(points) < 4:
        return jsonify(ok=False, error=f"需要至少 4 个标定点，当前 {len(points)} 个"), 400

    try:
        data2d = calibrate_and_save(points)
    except Exception as e:
        return jsonify(ok=False, error=f"2D 标定失败: {e}"), 500

    data3d: dict | None = None
    # 仅当存在 depth_mm 时才尝试 3D 标定
    if any(p.get("depth_mm") not in (None, 0) for p in points):
        try:
            data3d = calibrate3d_and_save(points)
        except Exception:
            data3d = None

    return jsonify(
        ok=True,
        points=data2d["points"],
        H=data2d["H"],
        points3d=data3d["points"] if data3d else None,
        A=data3d["A"] if data3d else None,
    )


# ── Routes: Teach ─────────────────────────────────────────────────
_teach_state: dict[str, Any] = {}


def _validate_ik_coords(x: float, y: float) -> str | None:
    """Check if (x, y) are within ArmPi Ultra IK workspace (meters)."""
    if abs(x) > 0.35 or abs(y) > 0.35:
        return (f"坐标 ({x*1000:.0f}, {y*1000:.0f}) mm 超出机械臂工作范围。"
                f"典型范围: X=50~300, Y=-200~200 mm")
    return None


@app.route("/api/teach/pick", methods=["POST"])
def api_teach_pick():
    """Detect blue block pixel position; optionally accept manual IK coords."""
    data = request.get_json(silent=True) or {}
    manual_x = data.get("arm_x")
    manual_y = data.get("arm_y")

    pixel = None
    depth_mm = None

    if camera.is_open:
        from blue_detector import detect_from_config
        ok, bgr, depth = camera.read_frame_with_depth()
        if ok and bgr is not None:
            cfg = _cfg()
            det = detect_from_config(bgr, cfg, depth=depth)
            if det is not None:
                pixel = list(det["center_px"])
                depth_mm = det.get("depth_mm")

    arm_pos = None
    warning = None
    if manual_x is not None and manual_y is not None:
        mx, my = round(float(manual_x), 4), round(float(manual_y), 4)
        err = _validate_ik_coords(mx, my)
        if err:
            return jsonify(ok=False, error=f"手动坐标无效: {err}"), 400
        arm_pos = [mx, my]
    elif pixel is not None:
        # 优先使用 depth-aware 3D 标定，将 (u, v, depth_mm) → (x, y)
        ax = ay = None
        if depth_mm is not None:
            try:
                from calibration3d import pixel_depth_to_arm_xy, load_calibration3d

                calib3d = load_calibration3d()
                if calib3d is not None:
                    ax3, ay3 = pixel_depth_to_arm_xy(tuple(pixel), depth_mm, calib3d)
                    err3 = _validate_ik_coords(ax3, ay3)
                    if err3 is None:
                        ax, ay = ax3, ay3
                    else:
                        warning = f"3D 标定坐标超范围({ax3*1000:.0f}, {ay3*1000:.0f}) mm，建议在示教中手动填写 X/Y"
            except Exception:
                # 回退到 2D Homography
                pass

        if ax is None or ay is None:
            from calibration import get_homography_matrix, pixel_to_arm

            H = get_homography_matrix()
            if H is not None:
                try:
                    ax2, ay2 = pixel_to_arm(tuple(pixel), H)
                    err2 = _validate_ik_coords(ax2, ay2)
                    if err2 is None:
                        ax, ay = ax2, ay2
                    else:
                        w = f"2D 标定坐标超范围({ax2*1000:.0f}, {ay2*1000:.0f}) mm，建议在示教中手动填写 X/Y"
                        warning = warning or w
                except Exception:
                    w = "标定矩阵映射失败，请在示教中手动填写 X/Y"
                    warning = warning or w
            else:
                warning = warning or "尚未完成标定，请在示教中手动填写 X/Y"

        if ax is not None and ay is not None:
            arm_pos = [round(ax, 4), round(ay, 4)]

    if pixel is None and arm_pos is None:
        return jsonify(ok=False, error="未检测到目标物体，且未提供手动坐标"), 400

    cfg = _cfg()
    tc = cfg.get("teach", {})
    z_pick = float(tc.get("pick_z", -0.02))
    z_move = float(tc.get("move_z", 0.05))

    _teach_state["pick"] = {
        "pixel": pixel,
        "arm": arm_pos,
        "depth_mm": depth_mm,
        "z_pick": z_pick,
        "z_move": z_move,
    }
    resp = dict(ok=True, pick=_teach_state["pick"])
    if warning:
        resp["warning"] = warning
    return jsonify(resp)


@app.route("/api/teach/place", methods=["POST"])
def api_teach_place():
    """Detect blue block pixel position; optionally accept manual IK coords."""
    data = request.get_json(silent=True) or {}
    manual_x = data.get("arm_x")
    manual_y = data.get("arm_y")

    pixel = None
    depth_mm = None

    if camera.is_open:
        from blue_detector import detect_from_config
        ok, bgr, depth = camera.read_frame_with_depth()
        if ok and bgr is not None:
            cfg = _cfg()
            det = detect_from_config(bgr, cfg, depth=depth)
            if det is not None:
                pixel = list(det["center_px"])
                depth_mm = det.get("depth_mm")

    arm_pos = None
    warning = None
    if manual_x is not None and manual_y is not None:
        mx, my = round(float(manual_x), 4), round(float(manual_y), 4)
        err = _validate_ik_coords(mx, my)
        if err:
            return jsonify(ok=False, error=f"手动坐标无效: {err}"), 400
        arm_pos = [mx, my]
    elif pixel is not None:
        # depth-aware 3D 标定优先
        ax = ay = None
        if depth_mm is not None:
            try:
                from calibration3d import pixel_depth_to_arm_xy, load_calibration3d

                calib3d = load_calibration3d()
                if calib3d is not None:
                    ax3, ay3 = pixel_depth_to_arm_xy(tuple(pixel), depth_mm, calib3d)
                    err3 = _validate_ik_coords(ax3, ay3)
                    if err3 is None:
                        ax, ay = ax3, ay3
                    else:
                        warning = f"3D 标定坐标超范围({ax3*1000:.0f}, {ay3*1000:.0f}) mm，建议在示教中手动填写 X/Y"
            except Exception:
                pass

        if ax is None or ay is None:
            from calibration import get_homography_matrix, pixel_to_arm

            H = get_homography_matrix()
            if H is not None:
                try:
                    ax2, ay2 = pixel_to_arm(tuple(pixel), H)
                    err2 = _validate_ik_coords(ax2, ay2)
                    if err2 is None:
                        ax, ay = ax2, ay2
                    else:
                        w = f"2D 标定坐标超范围({ax2*1000:.0f}, {ay2*1000:.0f}) mm，建议在示教中手动填写 X/Y"
                        warning = warning or w
                except Exception:
                    w = "标定矩阵映射失败，请在示教中手动填写 X/Y"
                    warning = warning or w
            else:
                warning = warning or "尚未完成标定，请在示教中手动填写 X/Y"

        if ax is not None and ay is not None:
            arm_pos = [round(ax, 4), round(ay, 4)]

    if pixel is None and arm_pos is None:
        return jsonify(ok=False, error="未检测到目标物体，且未提供手动坐标"), 400

    cfg = _cfg()
    tc = cfg.get("teach", {})
    z_pick = float(tc.get("pick_z", -0.02))
    z_move = float(tc.get("move_z", 0.05))

    _teach_state["place"] = {
        "pixel": pixel,
        "arm": arm_pos,
        "depth_mm": depth_mm,
        "z_pick": z_pick,
        "z_move": z_move,
    }
    resp = dict(ok=True, place=_teach_state["place"])
    if warning:
        resp["warning"] = warning
    return jsonify(resp)


@app.route("/api/teach/save", methods=["POST"])
def api_teach_save():
    from teach import save_task
    data = request.get_json(silent=True) or {}
    name = data.get("name", "pick_blue_block")
    pick = _teach_state.get("pick")
    place = _teach_state.get("place")
    if not pick or not place:
        return jsonify(ok=False, error="请先记录拾取点和放置点"), 400

    pick_arm = tuple(pick["arm"]) if pick.get("arm") else None
    place_arm = tuple(place["arm"]) if place.get("arm") else None
    missing = []
    if pick_arm is None:
        missing.append("拾取点")
    if place_arm is None:
        missing.append("放置点")
    if missing:
        return jsonify(ok=False, error=f"{'和'.join(missing)}缺少 IK 坐标。"
                       "标定映射可能失效，请在记录时手动填写 X/Y 坐标"
                       "（典型范围 X=50~300, Y=-200~200 mm）。"), 400

    err = _validate_ik_coords(pick_arm[0], pick_arm[1])
    if err:
        return jsonify(ok=False, error=f"拾取点{err}"), 400
    err = _validate_ik_coords(place_arm[0], place_arm[1])
    if err:
        return jsonify(ok=False, error=f"放置点{err}"), 400

    cfg = _cfg()
    tc = cfg.get("teach", {})
    task = save_task(
        name=name,
        pick_pixel=tuple(pick["pixel"]) if pick.get("pixel") else (0, 0),
        pick_arm=pick_arm,
        place_pixel=tuple(place["pixel"]) if place.get("pixel") else (0, 0),
        place_arm=place_arm,
        pick_z=float(tc.get("pick_z", -0.02)),
        move_z=float(tc.get("move_z", 0.05)),
    )
    _teach_state.clear()
    return jsonify(ok=True, task=task)


@app.route("/api/teach/state")
def api_teach_state():
    return jsonify(ok=True, state=_teach_state)


# ── Routes: Tasks ─────────────────────────────────────────────────
@app.route("/api/tasks")
def api_tasks():
    from teach import list_tasks
    return jsonify(ok=True, tasks=list_tasks())


# ── Routes: Execute ───────────────────────────────────────────────
@app.route("/api/execute", methods=["POST"])
def api_execute():
    data = request.get_json(silent=True) or {}
    task_file = data.get("task")
    if not task_file:
        return jsonify(ok=False, error="缺少 task 参数"), 400

    from teach import load_task
    from executor import execute_pick_place
    from blue_detector import detect_from_config
    from calibration import get_homography_matrix, pixel_to_arm

    try:
        task = load_task(task_file)
    except FileNotFoundError as e:
        return jsonify(ok=False, error=str(e)), 404

    cfg = _cfg()
    arm_cfg = cfg.get("arm", {})

    pick_arm = None
    if camera.is_open:
        ok, bgr, depth = camera.read_frame_with_depth()
        if ok and bgr is not None:
            det = detect_from_config(bgr, cfg, depth=depth)
            if det is not None:
                px = tuple(det["center_px"])
                depth_mm = det.get("depth_mm")

                # 1) depth-aware 3D 标定优先
                if depth_mm is not None:
                    try:
                        from calibration3d import pixel_depth_to_arm_xy, load_calibration3d

                        calib3d = load_calibration3d()
                        if calib3d is not None:
                            ax3, ay3 = pixel_depth_to_arm_xy(px, depth_mm, calib3d)
                            if _validate_ik_coords(ax3, ay3) is None:
                                pick_arm = (ax3, ay3)
                    except Exception:
                        pick_arm = None

                # 2) 回退到 2D Homography
                if pick_arm is None:
                    H = get_homography_matrix()
                    if H is not None:
                        try:
                            ax2, ay2 = pixel_to_arm(px, H)
                            if _validate_ik_coords(ax2, ay2) is None:
                                pick_arm = (ax2, ay2)
                        except Exception:
                            pass

    try:
        d = _get_arm_driver()
        log = execute_pick_place(task, arm_cfg, pick_arm=pick_arm, driver=d)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500

    return jsonify(ok=True, action_log=log, used_live_detect=pick_arm is not None)


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(f"WonderPi Web UI: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
