#!/usr/bin/env python3
"""列出本机可用相机：V4L2 设备树、libcamera、OpenCV index。"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys


def main() -> int:
    print("=== V4L2（v4l2-ctl --list-devices）===", flush=True)
    v4l2 = shutil.which("v4l2-ctl")
    if not v4l2:
        print("未找到 v4l2-ctl。可安装: sudo apt install v4l-utils")
        nodes = sorted(glob.glob("/dev/video*"))
        if nodes:
            print("当前 /dev/video* 节点:")
            for p in nodes:
                print(f"  {p}")
    else:
        r = subprocess.run([v4l2, "--list-devices"], capture_output=True, text=True)
        combined = f"{r.stdout or ''}{r.stderr or ''}".strip()
        lines = [ln for ln in combined.splitlines() if ln.strip()]
        # 部分系统无 /dev/video0 时 v4l2-ctl 会先打一行误报，仍会继续列出其它节点
        skip_prefix = "Cannot open device /dev/video0"
        lines = [ln for ln in lines if not ln.startswith(skip_prefix)]
        if lines:
            print("\n".join(lines))
        else:
            print("(无输出)")
        if r.returncode != 0 and lines:
            print(f"(v4l2-ctl 退出码 {r.returncode}，已忽略与 video0 相关的提示)", flush=True)
    print(
        "\n提示: 上表中 pispbe / rpivid 多为板载 ISP 或硬件解码节点，"
        "一般不能当作 OpenCV 的「普通摄像头」用；USB 摄像头通常会显示为独立设备名。"
    )

    print("\n=== libcamera / rpicam-vid ===", flush=True)
    exe = shutil.which("rpicam-vid")
    if not exe:
        print("未找到 rpicam-vid（树莓派 CSI 相机常用）。可安装: sudo apt install libcamera-apps")
    else:
        r = subprocess.run([exe, "--list-cameras"], capture_output=True, text=True)
        sys.stdout.write(r.stdout or "")
        sys.stderr.write(r.stderr or "")

    print("\n=== OpenCV VideoCapture（按 index 0–9 快速探测）===", flush=True)
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    import cv2

    for i in range(10):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened()
        if ok:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ret, frame = cap.read()
            cap.release()
            shape = None if not ret or frame is None else frame.shape
            print(f"  index {i}: 已打开, 分辨率约 {w}x{h}, 试读一帧: {shape}")
        else:
            cap.release()
            print(f"  index {i}: 不可用")

    print("\n=== Deptrum 深度相机（Aurora 930 等）===", flush=True)
    try:
        from deptrum_camera import device_count, sdk_version
        n = device_count()
        if n > 0:
            print(f"  检测到 {n} 台 Deptrum 深度相机（SDK {sdk_version()}）")
            print("  可在 config 中设 camera.backend: deptrum")
        else:
            print("  未检测到 Deptrum 深度相机（设备数=0）")
    except FileNotFoundError:
        print("  bridge 库未编译。请执行: bash sdk/deptrum/build.sh")
    except Exception as e:
        print(f"  检测失败: {e}")

    print(
        "\n说明: CSI 相机以 libcamera 列表为准；USB 摄像头可看 V4L2 段设备名，"
        "并在 config 中设 camera.backend: opencv 与合适的 camera.index。"
        "\n      Deptrum 深度相机请设 camera.backend: deptrum。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
