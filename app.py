#!/usr/bin/env python3
"""WonderPi Demo Web UI — Flask app integrating camera, recording, frame extraction, inference, arm control."""

from __future__ import annotations

import json
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
def _cfg() -> dict[str, Any]:
    return load_config()


# ── Camera Singleton ──────────────────────────────────────────────
import urllib.request


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

    def _open(self, cfg: dict[str, Any]) -> None:
        cam = cfg["camera"]
        backend = cam.get("backend", "auto").lower()
        index = int(cam.get("index", 0))
        ros_base = cam.get("ros_web_video_url", "http://localhost:8080")

        if backend in ("ros_web", "auto"):
            snap_url = f"{ros_base}/snapshot?topic=/depth_cam/rgb/image_raw"
            try:
                resp = urllib.request.urlopen(snap_url, timeout=3)
                if resp.status == 200 and len(resp.read()) > 100:
                    self._ros_rgb_url = snap_url
                    self._ros_depth_url = f"{ros_base}/snapshot?topic=/depth_cam/depth/image_raw"
                    self._ros_stream_url = f"{ros_base}/stream?topic=/depth_cam/rgb/image_raw&type=mjpeg&quality=70"
                    self._backend = "ros_web"
                    self._running = True
                    return
            except Exception:
                pass

        if backend in ("opencv", "auto"):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam.get("width", 640)))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam.get("height", 480)))
                cap.set(cv2.CAP_PROP_FPS, float(cam.get("fps", 30)))
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

        raise RuntimeError("无法打开任何相机，请检查 config/default.yaml 与硬件连接。")

    def ensure_open(self) -> None:
        with self._lock:
            if self._running:
                return
            self._open(_cfg())

    def _fetch_snapshot(self, url: str) -> np.ndarray | None:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            data = resp.read()
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
                depth_img = self._fetch_snapshot(self._ros_depth_url)
                depth_16 = None
                if depth_img is not None:
                    if depth_img.ndim == 3:
                        depth_16 = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY).astype(np.uint16)
                    else:
                        depth_16 = depth_img.astype(np.uint16)
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

    @property
    def is_open(self) -> bool:
        return self._running

    @property
    def backend_name(self) -> str:
        return self._backend


camera = CameraManager()


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


# ── Routes: Blue Detect ───────────────────────────────────────────
@app.route("/api/detect/blue", methods=["POST"])
def api_detect_blue():
    """Grab one frame, detect blue object, return annotated JPEG + JSON."""
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
    calib = load_calibration()
    return jsonify(
        ok=True,
        points=_calib_points,
        calibrated=calib is not None and "H" in (calib or {}),
        saved_points=calib.get("points", []) if calib else [],
    )


@app.route("/api/calibrate/point", methods=["POST"])
def api_calib_point():
    """Record one calibration point: detect blue block pixel + user-supplied arm coords."""
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
        return jsonify(ok=False, error="未检测到蓝色物体，请将积木放入视野"), 400

    point = {
        "pixel": list(det["center_px"]),
        "arm": [float(arm_x), float(arm_y)],
    }
    _calib_points.append(point)
    return jsonify(ok=True, point=point, total=len(_calib_points))


@app.route("/api/calibrate/reset", methods=["POST"])
def api_calib_reset():
    _calib_points.clear()
    return jsonify(ok=True)


@app.route("/api/calibrate/compute", methods=["POST"])
def api_calib_compute():
    from calibration import calibrate_and_save
    points = _calib_points[:]
    if len(points) < 4:
        return jsonify(ok=False, error=f"需要至少 4 个标定点，当前 {len(points)} 个"), 400
    try:
        data = calibrate_and_save(points)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500
    return jsonify(ok=True, points=data["points"], H=data["H"])


# ── Routes: Teach ─────────────────────────────────────────────────
_teach_state: dict[str, Any] = {}


def _validate_ik_coords(x: float, y: float) -> str | None:
    """Check if (x, y) are within ArmPi Ultra IK workspace (meters)."""
    if abs(x) > 0.35 or abs(y) > 0.35:
        return (f"坐标 ({x:.4f}, {y:.4f}) 超出机械臂工作范围。"
                f"IK 坐标单位为米，典型范围: x=0.05~0.30, y=-0.20~0.20")
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
        from calibration import get_homography_matrix, pixel_to_arm
        H = get_homography_matrix()
        if H is not None:
            try:
                ax, ay = pixel_to_arm(tuple(pixel), H)
                err = _validate_ik_coords(ax, ay)
                if err is None:
                    arm_pos = [round(ax, 4), round(ay, 4)]
                else:
                    warning = f"标定映射坐标超范围({ax:.2f}, {ay:.2f})，请手动填写 IK 坐标(米)"
            except Exception:
                warning = "标定矩阵映射失败，请手动填写 IK 坐标(米)"
        else:
            warning = "未完成标定，请手动填写 IK 坐标(米)"

    if pixel is None and arm_pos is None:
        return jsonify(ok=False, error="未检测到蓝色积木，且未提供手动坐标"), 400

    _teach_state["pick"] = {
        "pixel": pixel,
        "arm": arm_pos,
        "depth_mm": depth_mm,
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
        from calibration import get_homography_matrix, pixel_to_arm
        H = get_homography_matrix()
        if H is not None:
            try:
                ax, ay = pixel_to_arm(tuple(pixel), H)
                err = _validate_ik_coords(ax, ay)
                if err is None:
                    arm_pos = [round(ax, 4), round(ay, 4)]
                else:
                    warning = f"标定映射坐标超范围({ax:.2f}, {ay:.2f})，请手动填写 IK 坐标(米)"
            except Exception:
                warning = "标定矩阵映射失败，请手动填写 IK 坐标(米)"
        else:
            warning = "未完成标定，请手动填写 IK 坐标(米)"

    if pixel is None and arm_pos is None:
        return jsonify(ok=False, error="未检测到蓝色积木，且未提供手动坐标"), 400

    _teach_state["place"] = {
        "pixel": pixel,
        "arm": arm_pos,
        "depth_mm": depth_mm,
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
                       "标定映射可能失效，请在记录时手动填写 X/Y 坐标（单位: 米，"
                       "典型范围 X=0.05~0.30, Y=-0.20~0.20）。"), 400

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
                H = get_homography_matrix()
                if H is not None:
                    try:
                        ax, ay = pixel_to_arm(det["center_px"], H)
                        if _validate_ik_coords(ax, ay) is None:
                            pick_arm = (ax, ay)
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
