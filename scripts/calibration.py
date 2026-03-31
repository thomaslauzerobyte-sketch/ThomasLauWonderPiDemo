"""4 点 Homography 标定：像素坐标 ↔ 机械臂工作区坐标。

适用于固定相机俯视平面工作台的场景（3-DOF 臂 + 桌面积木）。

标定数据保存在 config/calibration.json:
{
  "points": [
    {"pixel": [u, v], "arm": [x, y]},
    ...
  ],
  "H": [[h00, h01, h02], [h10, h11, h12], [h20, h21, h22]]
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import ROOT

CALIB_FILE = ROOT / "config" / "calibration.json"


def load_calibration(path: Path | None = None) -> dict[str, Any] | None:
    p = path or CALIB_FILE
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save_calibration(data: dict[str, Any], path: Path | None = None) -> Path:
    p = path or CALIB_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def compute_homography(points: list[dict[str, Any]]) -> np.ndarray:
    if len(points) < 4:
        raise ValueError("至少需要 4 个标定点")
    src = np.array([p["pixel"] for p in points], dtype=np.float32)
    dst = np.array([p["arm"] for p in points], dtype=np.float32)
    H, status = cv2.findHomography(src, dst)
    if H is None:
        raise RuntimeError("Homography 计算失败，请检查标定点是否共线")
    return H


def pixel_to_arm(px: tuple[float, float], H: np.ndarray) -> tuple[float, float]:
    pt = np.array([px[0], px[1], 1.0], dtype=np.float64)
    out = H @ pt
    if abs(out[2]) < 1e-12:
        raise RuntimeError("Homography 投影奇异（w≈0）")
    return float(out[0] / out[2]), float(out[1] / out[2])


def get_homography_matrix(calib: dict[str, Any] | None = None) -> np.ndarray | None:
    if calib is None:
        calib = load_calibration()
    if calib is None or "H" not in calib:
        return None
    return np.array(calib["H"], dtype=np.float64)


def calibrate_and_save(points: list[dict[str, Any]]) -> dict[str, Any]:
    H = compute_homography(points)
    data = {
        "points": points,
        "H": H.tolist(),
    }
    save_calibration(data)
    return data
