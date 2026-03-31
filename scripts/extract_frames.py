#!/usr/bin/env python3
"""将视频拆成有序图像序列。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2

from common import load_config, resolve_path


def extract(
    *,
    video_path: Path,
    output_dir: Path,
    frame_step: int,
    image_format: str,
    jpeg_quality: int,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    ext = image_format.lower().lstrip(".")
    if ext not in ("jpg", "jpeg", "png"):
        raise ValueError("image_format 仅支持 jpg / jpeg / png")

    step = max(int(frame_step), 1)
    idx_saved = 0
    frame_idx = 0
    params = []
    if ext in ("jpg", "jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_idx % step == 0:
                name = f"frame_{idx_saved:06d}.{ext}"
                out_file = output_dir / name
                cv2.imwrite(str(out_file), frame, params)
                idx_saved += 1
            frame_idx += 1
    finally:
        cap.release()

    if idx_saved == 0:
        raise RuntimeError("未导出任何帧，请检查视频是否可读或 frame_step 是否过大。")
    return idx_saved


def main() -> int:
    parser = argparse.ArgumentParser(description="WonderPi Demo：视频拆帧")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--video", type=Path, required=True, help="输入视频路径")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录；默认在 data/frames 下按时间戳建子目录",
    )
    parser.add_argument("--step", type=int, default=None, help="每隔多少帧保存一张（覆盖配置）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ex = cfg["extract"]
    paths = cfg["paths"]

    step = int(args.step if args.step is not None else ex.get("frame_step", 1))
    if args.output_dir:
        out_dir = args.output_dir.resolve()
    else:
        frames_root = resolve_path(paths["frames_dir"])
        prefix = ex.get("subdir_prefix", "run")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = frames_root / f"{prefix}_{stamp}"

    try:
        n = extract(
            video_path=args.video.resolve(),
            output_dir=out_dir,
            frame_step=step,
            image_format=str(ex.get("image_format", "jpg")),
            jpeg_quality=int(ex.get("jpeg_quality", 92)),
        )
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1

    print(f"已保存 {n} 帧 -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
