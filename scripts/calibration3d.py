"""Depth-aware calibration: (u, v, depth_mm) -> arm (x, y).

相比纯 2D Homography，本模块把深度信息一起用于拟合一个
二维仿射映射：

    [x, y]^T = A @ [u, v, d, 1]^T

其中 A 为 2x4 矩阵，d 为深度（mm）。

标定数据保存在 config/calibration3d.json:
{
  "points": [
    {"pixel": [u, v], "depth_mm": d, "arm": [x, y]},
    ...
  ],
  "A": [[a00, a01, a02, a03],
        [a10, a11, a12, a13]]
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from common import ROOT


CALIB3D_FILE = ROOT / "config" / "calibration3d.json"


def load_calibration3d(path: Path | None = None) -> dict[str, Any] | None:
    p = path or CALIB3D_FILE
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save_calibration3d(data: dict[str, Any], path: Path | None = None) -> Path:
    p = path or CALIB3D_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def compute_affine_xy(points: list[dict[str, Any]]) -> np.ndarray:
    """Least squares fit: [x, y]^T = A @ [u, v, d, 1]^T."""
    if len(points) < 4:
        raise ValueError("calibration3d 至少需要 4 个点")

    rows = []
    targets = []
    for p in points:
        u, v = p["pixel"]
        d = float(p.get("depth_mm", 0.0))
        x, y = p["arm"]
        rows.append([u, v, d, 1.0])
        targets.append([x, y])

    X = np.asarray(rows, dtype=np.float64)  # (N,4)
    Y = np.asarray(targets, dtype=np.float64)  # (N,2)

    # Solve X @ A.T ≈ Y  -> A.T = lstsq(X, Y)
    A_t, *_ = np.linalg.lstsq(X, Y, rcond=None)
    A = A_t.T  # (2,4)
    return A


def pixel_depth_to_arm_xy(
    px: tuple[float, float],
    depth_mm: float,
    calib: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """Map (u, v, depth_mm) to arm (x, y) using fitted A matrix."""
    if calib is None:
        calib = load_calibration3d()
    if calib is None or "A" not in calib:
        raise RuntimeError("尚未完成 3D 标定（缺少 calibration3d.json）")

    A = np.asarray(calib["A"], dtype=np.float64)  # (2,4)
    u, v = px
    d = float(depth_mm)
    vec = np.array([u, v, d, 1.0], dtype=np.float64)
    out = A @ vec  # (2,)
    return float(out[0]), float(out[1])


def calibrate3d_and_save(points: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit A from points and save to calibration3d.json."""
    A = compute_affine_xy(points)
    data = {
        "points": points,
        "A": A.tolist(),
    }
    save_calibration3d(data)
    return data

