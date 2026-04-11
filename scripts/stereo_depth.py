"""由左右目灰度图估计视差并换算深度（毫米），用于无 ToF 深度图时的 3D 标定。

公式（水平基线、已校正立体对）: Z_mm = baseline_mm * focal_px / disparity_px
其中 focal_px 为左相机水平焦距（像素），需与标定或近似值一致。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def depth_from_stereo_pair(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    *,
    baseline_mm: float,
    focal_px: float,
    min_disparity: int = 0,
    num_disparities: int = 128,
    block_size: int = 5,
    uniqueness_ratio: int = 10,
    speckle_window_size: int = 100,
    speckle_range: int = 2,
) -> np.ndarray:
    """返回与 left_bgr 同尺寸的 uint16 深度图（毫米），无效处为 0。"""
    if baseline_mm <= 0 or focal_px <= 0:
        raise ValueError("baseline_mm 与 focal_px 必须为正")

    lg = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    h, w = lg.shape[:2]
    if rg.shape[:2] != (h, w):
        rg = cv2.resize(rg, (w, h), interpolation=cv2.INTER_LINEAR)

    num_disparities = max(16, (num_disparities // 16) * 16)
    block_size = max(3, block_size | 1)

    p1 = 8 * 3 * block_size**2
    p2 = 32 * 3 * block_size**2
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=1,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = sgbm.compute(lg, rg).astype(np.float32) / 16.0
    disp = np.maximum(disp, 0.01)

    z = (float(baseline_mm) * float(focal_px)) / disp
    z = np.clip(z, 0, 65535.0)
    out = z.astype(np.uint16)
    out[disp < 0.5] = 0
    return out


def effective_focal_px(stereo_cfg: dict[str, Any], image_width: int) -> float:
    fp = float(stereo_cfg.get("focal_px", 0) or 0)
    if fp > 0:
        return fp
    ratio = float(stereo_cfg.get("focal_ratio_of_width", 0.75))
    return max(100.0, ratio * float(image_width))
