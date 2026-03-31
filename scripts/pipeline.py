#!/usr/bin/env python3
"""串联：录制 -> 拆帧 -> 推理 -> 臂控演示（可选）。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import ROOT, load_config, resolve_path


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="WonderPi Demo：一键流水线")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seconds", type=float, default=None, help="录制时长")
    parser.add_argument("--skip-arm", action="store_true", help="不执行 arm_demo")
    args = parser.parse_args()

    py = sys.executable
    scripts = ROOT / "scripts"

    cmd_rec = [py, str(scripts / "record_video.py")]
    if args.config:
        cmd_rec += ["--config", str(args.config)]
    if args.seconds is not None:
        cmd_rec += ["--seconds", str(args.seconds)]

    r = _run(cmd_rec)
    if r.returncode != 0:
        print(r.stderr or r.stdout, file=sys.stderr)
        return r.returncode
    video_path = (r.stdout or "").strip().splitlines()[-1].strip()
    if not video_path:
        print("录制未返回视频路径。", file=sys.stderr)
        return 1

    cmd_ext = [py, str(scripts / "extract_frames.py"), "--video", video_path]
    if args.config:
        cmd_ext += ["--config", str(args.config)]
    r2 = _run(cmd_ext)
    if r2.returncode != 0:
        print(r2.stderr or r2.stdout, file=sys.stderr)
        return r2.returncode
    # 输出形如: 已保存 N 帧 -> /path
    out_line = (r2.stdout or "").strip().splitlines()[-1]
    frames_dir = out_line.split("->")[-1].strip()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    first_frame = None
    fd = Path(frames_dir)
    for p in sorted(fd.iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            first_frame = p
            break
    if first_frame is None:
        print("拆帧目录为空。", file=sys.stderr)
        return 1

    infer_json = resolve_path(paths.get("frames_dir", "data/frames")) / "_last_infer.json"
    cmd_inf = [
        py,
        str(scripts / "infer.py"),
        "--source",
        str(first_frame),
        "--json-out",
        str(infer_json),
    ]
    if args.config:
        cmd_inf += ["--config", str(args.config)]
    r3 = _run(cmd_inf)
    if r3.returncode != 0:
        print(r3.stderr or r3.stdout, file=sys.stderr)
        return r3.returncode
    print(r3.stdout or "")

    if args.skip_arm:
        print("已跳过臂控。")
        return 0

    cmd_arm = [py, str(scripts / "arm_demo.py"), "--json", str(infer_json)]
    if args.config:
        cmd_arm += ["--config", str(args.config)]
    r4 = _run(cmd_arm)
    sys.stdout.write(r4.stdout or "")
    sys.stderr.write(r4.stderr or "")
    return r4.returncode


if __name__ == "__main__":
    raise SystemExit(main())
