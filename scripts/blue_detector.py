"""蓝色物体 HSV 检测 + 可选深度读取。

返回的 Detection 字典：
  center_px: (cx, cy)    像素中心
  bbox:      (x, y, w, h)
  area:      轮廓面积（像素²）
  depth_mm:  深度值（mm），无深度时为 None
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def detect_blue(
    bgr: np.ndarray,
    *,
    hsv_lower: tuple[int, int, int] = (100, 80, 50),
    hsv_upper: tuple[int, int, int] = (130, 255, 255),
    min_area: int = 500,
    blur_kernel: int = 5,
    depth: np.ndarray | None = None,
) -> dict[str, Any] | None:
    blurred = cv2.GaussianBlur(bgr, (blur_kernel | 1, blur_kernel | 1), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower, np.uint8), np.array(hsv_upper, np.uint8))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < min_area:
        return None

    x, y, w, h = cv2.boundingRect(best)
    M = cv2.moments(best)
    if M["m00"] == 0:
        cx, cy = x + w // 2, y + h // 2
    else:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

    depth_mm: int | None = None
    if depth is not None and 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
        region = depth[max(0, cy - 2):cy + 3, max(0, cx - 2):cx + 3]
        valid = region[region > 0]
        if valid.size > 0:
            depth_mm = int(np.median(valid))

    return {
        "center_px": (cx, cy),
        "bbox": (x, y, w, h),
        "area": float(area),
        "depth_mm": depth_mm,
    }


def draw_detection(bgr: np.ndarray, det: dict[str, Any] | None) -> np.ndarray:
    out = bgr.copy()
    if det is None:
        cv2.putText(out, "No blue object", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return out

    x, y, w, h = det["bbox"]
    cx, cy = det["center_px"]
    cv2.rectangle(out, (x, y), (x + w, y + h), (255, 180, 0), 2)
    cv2.drawMarker(out, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)

    label = f"({cx},{cy})"
    if det["depth_mm"] is not None:
        label += f" d={det['depth_mm']}mm"
    cv2.putText(out, label, (x, max(y - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1)
    return out


def detect_from_config(bgr: np.ndarray, cfg: dict[str, Any],
                       depth: np.ndarray | None = None) -> dict[str, Any] | None:
    bd = cfg.get("blue_detect", {})
    return detect_blue(
        bgr,
        hsv_lower=tuple(bd.get("hsv_lower", [100, 80, 50])),
        hsv_upper=tuple(bd.get("hsv_upper", [130, 255, 255])),
        min_area=int(bd.get("min_area", 500)),
        blur_kernel=int(bd.get("blur_kernel", 5)),
        depth=depth,
    )
