"""目标检测：支持蓝色 HSV / 棋盘格 / 黑白标记板(TAG)。

detect_from_config 根据 cfg["detect_method"] 自动选择检测器：
  - "blue"         : HSV 蓝色物体检测（默认）
  - "checkerboard" : 黑白棋盘格检测（亚像素精度）
  - "tag"          : 黑白方形标记板（如 TAG36H11）

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


# ── 深度采样 ──────────────────────────────────────────────────────
def _sample_depth(depth: np.ndarray | None, cx: int, cy: int, radius: int = 3) -> int | None:
    if depth is None or not (0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]):
        return None
    r = radius
    region = depth[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1]
    valid = region[region > 0]
    if valid.size > 0:
        return int(np.median(valid))
    return None


# ── 蓝色 HSV 检测 ────────────────────────────────────────────────
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

    return {
        "center_px": (cx, cy),
        "bbox": (x, y, w, h),
        "area": float(area),
        "depth_mm": _sample_depth(depth, cx, cy),
    }


# ── 棋盘格检测 ───────────────────────────────────────────────────
def detect_checkerboard(
    bgr: np.ndarray,
    *,
    pattern_size: tuple[int, int] = (5, 4),
    depth: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """检测棋盘格标定板，返回棋盘格中心点（亚像素精度）。

    pattern_size: 棋盘格内角点数 (cols, rows)，例如 6×5 格的板 → (5, 4)。
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found or corners is None:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

    pts = corners.reshape(-1, 2)
    cx_f = float(pts[:, 0].mean())
    cy_f = float(pts[:, 1].mean())
    cx, cy = int(round(cx_f)), int(round(cy_f))

    xs, ys = pts[:, 0], pts[:, 1]
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    w, h = x_max - x_min, y_max - y_min

    return {
        "center_px": (cx, cy),
        "bbox": (x_min, y_min, w, h),
        "area": float(w * h),
        "depth_mm": _sample_depth(depth, cx, cy),
        "corners": corners,
    }


# ── 黑白标记板(TAG) 检测 ──────────────────────────────────────────
def detect_tag(
    bgr: np.ndarray,
    *,
    min_area: int = 800,
    depth: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """检测类似 TAG36H11 的黑白方形标记板，返回外接矩形中心。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 黑底白框符号：先二值化再取外轮廓
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr_inv = cv2.bitwise_not(thr)

    contours, _ = cv2.findContours(thr_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if w <= 0 or h <= 0:
            continue
        ratio = w / float(h)
        if ratio < 0.6 or ratio > 1.4:
            continue
        if area > best_area:
            best = (approx, x, y, w, h, area)
            best_area = area

    if best is None:
        return None

    approx, x, y, w, h, area = best
    M = cv2.moments(approx)
    if M["m00"] == 0:
        cx, cy = x + w // 2, y + h // 2
    else:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

    return {
        "center_px": (cx, cy),
        "bbox": (x, y, w, h),
        "area": float(area),
        "depth_mm": _sample_depth(depth, cx, cy),
    }


# ── 绘制检测结果 ─────────────────────────────────────────────────
def draw_detection(bgr: np.ndarray, det: dict[str, Any] | None) -> np.ndarray:
    out = bgr.copy()
    if det is None:
        cv2.putText(out, "No target", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return out

    x, y, w, h = det["bbox"]
    cx, cy = det["center_px"]

    if "corners" in det:
        cv2.drawChessboardCorners(out, _guess_pattern(det["corners"]),
                                  det["corners"], True)
    else:
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 180, 0), 2)

    cv2.drawMarker(out, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

    label = f"({cx},{cy})"
    if det["depth_mm"] is not None:
        label += f" d={det['depth_mm']}mm"
    cv2.putText(out, label, (x, max(y - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1)
    return out


def _guess_pattern(corners: np.ndarray) -> tuple[int, int]:
    n = corners.shape[0]
    for r in range(3, 15):
        if n % r == 0:
            c = n // r
            if c >= 3:
                return (c, r)
    return (n, 1)


# ── 统一入口 ─────────────────────────────────────────────────────
def detect_from_config(bgr: np.ndarray, cfg: dict[str, Any],
                       depth: np.ndarray | None = None) -> dict[str, Any] | None:
    method = cfg.get("detect_method", "blue")

    if method == "checkerboard":
        cb = cfg.get("checkerboard", {})
        cols = int(cb.get("cols", 5))
        rows = int(cb.get("rows", 4))
        return detect_checkerboard(
            bgr,
            pattern_size=(cols, rows),
            depth=depth,
        )

    if method == "tag":
        td = cfg.get("tag_detect", {})
        return detect_tag(
            bgr,
            min_area=int(td.get("min_area", 800)),
            depth=depth,
        )

    bd = cfg.get("blue_detect", {})
    return detect_blue(
        bgr,
        hsv_lower=tuple(bd.get("hsv_lower", [100, 80, 50])),
        hsv_upper=tuple(bd.get("hsv_upper", [130, 255, 255])),
        min_area=int(bd.get("min_area", 500)),
        blur_kernel=int(bd.get("blur_kernel", 5)),
        depth=depth,
    )
