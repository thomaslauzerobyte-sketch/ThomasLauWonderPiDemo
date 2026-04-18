"""棋盘格黑白标定板 → 批量生成 3D 标定点 (u, v, depth_mm) → 臂 (x, y)。

OpenCV 角点顺序：按行扫描，第 0 个角点对应棋盘「物体坐标系」原点 (0,0)，
X 沿列方向、Y 沿行方向，格距 square_size_mm。

用法：将标定板平放在工作台上，用机械臂末端（或探针）对准 **第 0 个内角点**
在真实世界中的位置，将该点的臂坐标 (origin_x, origin_y)（米）传入；
其余角点的臂坐标由格距与可选平面旋转 board_yaw_deg 推算。

与 calibration3d.compute_affine_xy 配合：一次拍照可得到 N>>4 个约束，标定更稳。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _sample_depth_mm(depth: np.ndarray | None, cx: int, cy: int, radius: int = 2) -> int | None:
    if depth is None or not (0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]):
        return None
    r = radius
    region = depth[max(0, cy - r) : cy + r + 1, max(0, cx - r) : cx + r + 1]
    valid = region[region > 0]
    if valid.size > 0:
        return int(np.median(valid))
    return None


def _find_chessboard_corners_gray(
    gray: np.ndarray,
    pattern_cols: int,
    pattern_rows: int,
) -> np.ndarray | None:
    """单种内角点尺寸；成功返回亚像素角点，失败 None。"""
    size = (pattern_cols, pattern_rows)
    flags_fast = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    flags_full = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, size, flags_fast)
    if not found or corners is None:
        found, corners = cv2.findChessboardCorners(gray, size, flags_full)
    if not found or corners is None:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    return corners


def find_chessboard_corners(
    bgr: np.ndarray,
    pattern_cols: int,
    pattern_rows: int,
) -> np.ndarray | None:
    """返回 shape (N, 1, 2) float32 亚像素角点；失败返回 None。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return _find_chessboard_corners_gray(gray, pattern_cols, pattern_rows)


def find_chessboard_corners_autotry(
    bgr: np.ndarray,
    *,
    hint_cols: int = 5,
    hint_rows: int = 4,
    min_inner: int = 3,
    max_inner: int = 16,
) -> tuple[np.ndarray, int, int, int] | None:
    """按优先级自动尝试多组内角点 (宽×高)，与 OpenCV findChessboardCorners 一致。

    先试 hint（及行列互换），再在 [min_inner, max_inner] 内二维枚举。
    返回 (corners, pattern_cols, pattern_rows, attempts_used)；全失败返回 None。
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    candidates: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add(w: int, h: int) -> None:
        if min_inner <= w <= max_inner and min_inner <= h <= max_inner and (w, h) not in seen:
            seen.add((w, h))
            candidates.append((w, h))

    add(hint_cols, hint_rows)
    add(hint_rows, hint_cols)
    dist_pairs: list[tuple[int, tuple[int, int]]] = []
    for w in range(min_inner, max_inner + 1):
        for h in range(min_inner, max_inner + 1):
            if (w, h) in seen:
                continue
            dist = abs(w - hint_cols) + abs(h - hint_rows)
            dist_pairs.append((dist, (w, h)))
    dist_pairs.sort(key=lambda x: (x[0], x[1][0] * x[1][1]))
    for _, (w, h) in dist_pairs:
        add(w, h)

    attempts = 0
    for cols, rows in candidates:
        attempts += 1
        corners = _find_chessboard_corners_gray(gray, cols, rows)
        if corners is not None:
            return corners, cols, rows, attempts
    return None


def board_corners_to_calib_points(
    corners: np.ndarray,
    depth: np.ndarray | None,
    *,
    pattern_cols: int,
    pattern_rows: int,
    square_size_mm: float,
    origin_arm_x_m: float,
    origin_arm_y_m: float,
    board_yaw_deg: float = 0.0,
    min_depth_mm: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """将棋盘角点转为 calibration3d 用的点列表。

    角点索引 k：row = k // pattern_cols, col = k % pattern_cols
    物体平面坐标（毫米）：board_x = col * square_size_mm, board_y = row * square_size_mm
    """
    n_expected = pattern_cols * pattern_rows
    pts = corners.reshape(-1, 2)
    if pts.shape[0] != n_expected:
        raise ValueError(f"角点数量 {pts.shape[0]} 与 pattern {pattern_cols}x{pattern_rows} 不符")

    theta = np.radians(board_yaw_deg)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    sm = float(square_size_mm)

    out: list[dict[str, Any]] = []
    skipped_no_depth = 0
    for k in range(n_expected):
        row = k // pattern_cols
        col = k % pattern_cols
        bx_mm = col * sm
        by_mm = row * sm
        bx_m = bx_mm * 1e-3
        by_m = by_mm * 1e-3
        arm_x = origin_arm_x_m + bx_m * c - by_m * s
        arm_y = origin_arm_y_m + bx_m * s + by_m * c

        u_f, v_f = float(pts[k, 0]), float(pts[k, 1])
        ui, vi = int(round(u_f)), int(round(v_f))
        d_mm = _sample_depth_mm(depth, ui, vi)
        if d_mm is None or d_mm < min_depth_mm:
            skipped_no_depth += 1
            continue

        out.append(
            {
                "pixel": [round(u_f, 3), round(v_f, 3)],
                "arm": [round(arm_x, 6), round(arm_y, 6)],
                "depth_mm": d_mm,
                "board": {"row": row, "col": col, "k": k},
            }
        )

    meta = {
        "pattern_cols": pattern_cols,
        "pattern_rows": pattern_rows,
        "square_size_mm": sm,
        "origin_arm_m": [origin_arm_x_m, origin_arm_y_m],
        "board_yaw_deg": board_yaw_deg,
        "corners_total": n_expected,
        "points_with_depth": len(out),
        "skipped_no_depth": skipped_no_depth,
    }
    return out, meta


def draw_chessboard_overlay(bgr: np.ndarray, corners: np.ndarray, pattern_cols: int, pattern_rows: int) -> np.ndarray:
    vis = bgr.copy()
    cv2.drawChessboardCorners(vis, (pattern_cols, pattern_rows), corners, True)
    return vis
