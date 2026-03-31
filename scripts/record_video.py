#!/usr/bin/env python3
"""从摄像头录制视频到 data/videos（或指定路径）。

树莓派（libcamera）上常无 /dev/video0，OpenCV 的 index=0 会失败。
支持 backend: opencv | rpicam | picamera2 | auto（依次回退）。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from common import load_config, resolve_path


def _fourcc_code(name: str) -> int:
    name = (name or "mp4v").strip().lower()
    if len(name) != 4:
        raise ValueError("fourcc 必须为 4 个字符，例如 mp4v、MJPG")
    return cv2.VideoWriter_fourcc(*name)


def record_opencv(
    *,
    camera_index: int,
    width: int,
    height: int,
    fps: float,
    duration_sec: float,
    fourcc_name: str,
    output_path: Path,
) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 OpenCV index={camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = _fourcc_code(fourcc_name)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"无法创建视频文件 {output_path}（编码器 {fourcc_name} 可能不支持）。"
            "可尝试将 config 中 recording.fourcc 改为 MJPG 或 XVID。"
        )

    frame_interval = 1.0 / max(fps, 1e-3)
    t_end = time.monotonic() + duration_sec
    count = 0
    try:
        while time.monotonic() < t_end:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            count += 1
            time.sleep(max(0, frame_interval * 0.1))
    finally:
        writer.release()
        cap.release()

    if count == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("未读到任何帧，录制失败。")


def record_rpicam(
    *,
    duration_sec: float,
    width: int,
    height: int,
    output_path: Path,
) -> None:
    exe = shutil.which("rpicam-vid")
    if not exe:
        raise RuntimeError("未找到 rpicam-vid。请: sudo apt install libcamera-apps")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 无单位时 rpicam-vid 默认按毫秒解析
    t_ms = max(1, int(round(float(duration_sec) * 1000)))

    cmd = [
        exe,
        "-n",
        "--timeout",
        str(t_ms),
        "-o",
        str(output_path),
    ]
    if width > 0:
        cmd.extend(["--width", str(width)])
    if height > 0:
        cmd.extend(["--height", str(height)])

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()
        raise RuntimeError(f"rpicam-vid 失败 (exit {r.returncode}): {err or '无输出'}")

    if not output_path.is_file() or output_path.stat().st_size < 100:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            "rpicam-vid 未生成有效视频文件。请运行: python3 scripts/list_cameras.py "
            "或 rpicam-vid --list-cameras 确认已连接并启用 CSI 相机。"
        )


def record_picamera2(
    *,
    width: int,
    height: int,
    fps: float,
    duration_sec: float,
    fourcc_name: str,
    output_path: Path,
) -> None:
    try:
        from picamera2 import Picamera2
    except ImportError as e:
        raise RuntimeError(
            "未安装 picamera2。系统 Python 可能已有 python3-picamera2；虚拟环境请执行: pip install picamera2"
        ) from e

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = _fourcc_code(fourcc_name)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            f"无法创建视频文件 {output_path}（编码器 {fourcc_name} 可能不支持）。"
            "可尝试将 recording.fourcc 改为 MJPG。"
        )

    picam2 = Picamera2()
    cfg = picam2.create_video_configuration(main={"size": (width, height)})
    picam2.configure(cfg)
    picam2.start()
    time.sleep(0.3)

    frame_interval = 1.0 / max(fps, 1e-3)
    t_end = time.monotonic() + duration_sec
    count = 0
    try:
        while time.monotonic() < t_end:
            arr = picam2.capture_array()
            if arr.ndim == 2:
                frame = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            elif arr.shape[2] == 4:
                frame = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            else:
                frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            count += 1
            time.sleep(max(0, frame_interval * 0.05))
    finally:
        writer.release()
        picam2.stop()
        picam2.close()

    if count == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("picamera2 未采集到任何帧。")


def record_deptrum(
    *,
    width: int,
    height: int,
    fps: float,
    duration_sec: float,
    fourcc_name: str,
    output_path: Path,
    resolution_mode_index: int = 2,
) -> None:
    from deptrum_camera import DeptrumCamera

    cam = DeptrumCamera(resolution_mode_index=resolution_mode_index)
    cam.open()
    cam.start()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = _fourcc_code(fourcc_name)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cam.release()
        raise RuntimeError(
            f"无法创建视频文件 {output_path}（编码器 {fourcc_name} 可能不支持）。"
        )

    frame_interval = 1.0 / max(fps, 1e-3)
    t_end = time.monotonic() + duration_sec
    count = 0
    try:
        while time.monotonic() < t_end:
            ok, frame = cam.read()
            if not ok or frame is None:
                continue
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            count += 1
            time.sleep(max(0, frame_interval * 0.1))
    finally:
        writer.release()
        cam.release()

    if count == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("Deptrum 相机未读到任何帧，录制失败。")


def record_auto(
    *,
    camera_index: int,
    width: int,
    height: int,
    fps: float,
    duration_sec: float,
    fourcc_name: str,
    output_path: Path,
) -> None:
    errors: list[str] = []

    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        cap.release()
        try:
            record_opencv(
                camera_index=camera_index,
                width=width,
                height=height,
                fps=fps,
                duration_sec=duration_sec,
                fourcc_name=fourcc_name,
                output_path=output_path,
            )
            return
        except Exception as e:
            errors.append(f"OpenCV(index={camera_index}): {e}")
    else:
        cap.release()
        errors.append(f"OpenCV(index={camera_index}): 无法打开")

    if shutil.which("rpicam-vid"):
        try:
            record_rpicam(
                duration_sec=duration_sec,
                width=width,
                height=height,
                output_path=output_path,
            )
            return
        except Exception as e:
            errors.append(f"rpicam-vid: {e}")

    try:
        record_picamera2(
            width=width,
            height=height,
            fps=fps,
            duration_sec=duration_sec,
            fourcc_name=fourcc_name,
            output_path=output_path,
        )
        return
    except Exception as e:
        errors.append(f"picamera2: {e}")

    try:
        record_deptrum(
            width=width,
            height=height,
            fps=fps,
            duration_sec=duration_sec,
            fourcc_name=fourcc_name,
            output_path=output_path,
        )
        return
    except Exception as e:
        errors.append(f"deptrum: {e}")

    hint = (
        "1) USB 摄像头: 运行 python3 scripts/list_cameras.py 找到可用 index，config 设 camera.backend: opencv 与 camera.index\n"
        "2) CSI 相机: 确认排线接好且已启用（sudo raspi-config / config.txt），再设 camera.backend: rpicam\n"
        "3) 虚拟环境: pip install picamera2 后可用 backend: picamera2 或 auto\n"
        "4) Deptrum 深度相机: 设 camera.backend: deptrum\n"
    )
    raise RuntimeError("所有采集方式均失败:\n- " + "\n- ".join(errors) + "\n\n" + hint)


def main() -> int:
    parser = argparse.ArgumentParser(description="WonderPi Demo：摄像头录制视频")
    parser.add_argument("--config", type=Path, default=None, help="YAML 配置路径")
    parser.add_argument("--seconds", type=float, default=None, help="录制时长（秒）")
    parser.add_argument("-o", "--output", type=Path, default=None, help="输出文件路径")
    parser.add_argument(
        "--backend",
        choices=("auto", "opencv", "rpicam", "picamera2", "deptrum"),
        default=None,
        help="覆盖配置中的 camera.backend",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cam = cfg["camera"]
    rec = cfg["recording"]
    paths = cfg["paths"]

    backend = (args.backend or cam.get("backend", "opencv")).lower()
    duration = float(args.seconds if args.seconds is not None else rec["default_duration_sec"])
    videos_dir = resolve_path(paths["videos_dir"])
    prefix = rec.get("filename_prefix", "capture")
    if args.output:
        out = args.output.resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = videos_dir / f"{prefix}_{stamp}.mp4"

    width = int(cam["width"])
    height = int(cam["height"])
    fps = float(cam.get("fps", 30))
    fourcc = str(rec.get("fourcc", "mp4v"))
    index = int(cam["index"])

    try:
        if backend == "opencv":
            record_opencv(
                camera_index=index,
                width=width,
                height=height,
                fps=fps,
                duration_sec=duration,
                fourcc_name=fourcc,
                output_path=out,
            )
        elif backend == "rpicam":
            record_rpicam(duration_sec=duration, width=width, height=height, output_path=out)
        elif backend == "picamera2":
            record_picamera2(
                width=width,
                height=height,
                fps=fps,
                duration_sec=duration,
                fourcc_name=fourcc,
                output_path=out,
            )
        elif backend == "deptrum":
            record_deptrum(
                width=width,
                height=height,
                fps=fps,
                duration_sec=duration,
                fourcc_name=fourcc,
                output_path=out,
                resolution_mode_index=int(cam.get("resolution_mode_index", 2)),
            )
        else:
            record_auto(
                camera_index=index,
                width=width,
                height=height,
                fps=fps,
                duration_sec=duration,
                fourcc_name=fourcc,
                output_path=out,
            )
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1

    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
