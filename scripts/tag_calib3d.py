"""AprilTag / ArUco 单标记 3D 标定点生成（一次检测 → 4 个角点 + 深度）。

适用于双目深度相机：只需画面中 **一个** 打印好的 AprilTag（如 36h11），
无需整板棋盘格。四个角点在标记平面上的几何位置由 tag 外沿边长 tag_edge_mm 确定，
用户输入 **标记几何中心** 在机械臂坐标系下的 (x, y)（米）。

OpenCV 角点顺序：左上 → 右上 → 右下 → 左下（图像坐标系）。
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


def _get_aruco_dict(family: str):
    """family: e.g. DICT_APRILTAG_36h11, DICT_APRILTAG_36h10, DICT_ARUCO_ORIGINAL"""
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("当前 OpenCV 未编译 aruco 模块，无法检测 AprilTag")
    dict_id = getattr(cv2.aruco, family, None)
    if dict_id is None:
        dict_id = cv2.aruco.DICT_APRILTAG_36h11
    return cv2.aruco.getPredefinedDictionary(dict_id)


def detect_apriltag_quad(
    bgr: np.ndarray,
    *,
    family: str = "DICT_APRILTAG_36h11",
    tag_id: int | None = None,
) -> tuple[np.ndarray, int] | None:
    """检测单个 AprilTag，返回 (4,2) float32 像素坐标、marker id。未检测到返回 None。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    aruco_dict = _get_aruco_dict(family)
    params = cv2.aruco.DetectorParameters()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    if ids is None or len(ids) == 0 or len(corners) == 0:
        return None

    if tag_id is not None:
        for i, mid in enumerate(ids.flatten()):
            if int(mid) == int(tag_id):
                quad = corners[i].reshape(4, 2).astype(np.float32)
                return quad, int(mid)
        return None

    quad = corners[0].reshape(4, 2).astype(np.float32)
    mid = int(ids.flatten()[0])
    return quad, mid


def tag_quad_to_calib_points(
    quad_px: np.ndarray,
    depth: np.ndarray | None,
    *,
    tag_edge_mm: float,
    center_arm_x_m: float,
    center_arm_y_m: float,
    yaw_deg: float = 0.0,
    min_depth_mm: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """四像素角点 + 深度 → calibration3d 点列表。

    标记平面：中心为原点，+X 向右、+Y 向下（与 OpenCV 角点顺序一致的外框方向）。
    角点相对中心（毫米）：TL(-h,-h), TR(h,-h), BR(h,h), BL(-h,h)，h = tag_edge_mm/2。
    """
    if quad_px.shape != (4, 2):
        raise ValueError("quad 必须为 (4,2)")

    S = float(tag_edge_mm)
    h = S / 2.0
    # 与 OpenCV 顺序一致：TL, TR, BR, BL
    offsets_mm = [
        (-h, -h),
        (h, -h),
        (h, h),
        (-h, h),
    ]

    theta = np.radians(yaw_deg)
    c, s = float(np.cos(theta)), float(np.sin(theta))

    out: list[dict[str, Any]] = []
    skipped = 0
    for k in range(4):
        bx_m = offsets_mm[k][0] * 1e-3
        by_m = offsets_mm[k][1] * 1e-3
        arm_x = center_arm_x_m + bx_m * c - by_m * s
        arm_y = center_arm_y_m + bx_m * s + by_m * c

        u_f, v_f = float(quad_px[k, 0]), float(quad_px[k, 1])
        ui, vi = int(round(u_f)), int(round(v_f))
        d_mm = _sample_depth_mm(depth, ui, vi)
        if d_mm is None or d_mm < min_depth_mm:
            skipped += 1
            continue
        out.append(
            {
                "pixel": [round(u_f, 3), round(v_f, 3)],
                "arm": [round(arm_x, 6), round(arm_y, 6)],
                "depth_mm": d_mm,
                "tag_corner": k,
            }
        )

    meta = {
        "tag_edge_mm": S,
        "corners_total": 4,
        "points_with_depth": len(out),
        "skipped_no_depth": skipped,
        "center_arm_m": [center_arm_x_m, center_arm_y_m],
        "yaw_deg": yaw_deg,
    }
    return out, meta


def draw_tag_overlay(bgr: np.ndarray, quad_px: np.ndarray, marker_id: int) -> np.ndarray:
    vis = bgr.copy()
    pts = quad_px.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
    cx, cy = int(quad_px[:, 0].mean()), int(quad_px[:, 1].mean())
    cv2.drawMarker(vis, (cx, cy), (0, 200, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(
        vis,
        f"AprilTag id={marker_id}",
        (max(cx - 80, 5), max(cy - 15, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 180),
        1,
        cv2.LINE_AA,
    )
    return vis
