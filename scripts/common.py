"""项目根路径、配置加载，以及与官方 `app/utils/common.py` 对齐的几何/图像工具。

齐次变换与欧拉角使用内嵌 `_tfs_lite`（与 transforms3d 等价），无需 pip 安装 transforms3d。
无 ROS 时，`rpy2qua` 返回 `types.SimpleNamespace(w,x,y,z)`，便于与 `geometry_msgs` 用法对照。
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import yaml

from _tfs_lite import decompose, euler2mat, mat2euler, quat2mat

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or DEFAULT_CONFIG
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (ROOT / p).resolve()


# ── 与官方 get_yaml_data / save_yaml_data 等价（使用 safe_load / safe_dump）────────────────


def get_yaml_data(yaml_file: str | Path) -> Any:
    p = Path(yaml_file)
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml_data(data: Any, yaml_file: str | Path) -> None:
    p = Path(yaml_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True)


# ── 以下与官方 app/utils/common.py 同名函数保持同一实现（transforms3d）────────────────


def loginfo(msg: str) -> None:
    """占位：官方依赖 rclpy Node；Demo 中仅打印。"""
    print(f"[common] {msg}")


def val_map(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def empty_func(img=None):
    return img


def set_range(x: float, x_min: float, x_max: float) -> float:
    tmp = x if x > x_min else x_min
    tmp = tmp if tmp < x_max else x_max
    return tmp


def distance(point_1, point_2) -> float:
    return math.sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)


def box_center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def point_remapped(point, now, new, data_type=float):
    x, y = point
    now_w, now_h = now
    new_w, new_h = new
    new_x = x * new_w / now_w
    new_y = y * new_h / now_h
    return data_type(new_x), data_type(new_y)


def vector_2d_angle(v1, v2):
    d_v1_v2 = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos = v1.dot(v2) / (d_v1_v2)
    sin = np.cross(v1, v2) / (d_v1_v2)
    angle = np.degrees(np.arctan2(sin, cos))
    return angle


def warp_affine(image, points, scale=1.0):
    w, h = image.shape[:2]
    dy = points[1][1] - points[0][1]
    dx = points[1][0] - points[0][0]
    angle = cv2.fastAtan2(dy, dx)
    rot = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, scale=scale)
    return cv2.warpAffine(image, rot, dsize=(h, w))


def perspective_transform(img, src, dst, debug=False):
    img_size = (img.shape[1], img.shape[0])
    m = cv2.getPerspectiveTransform(src, dst)
    if debug:
        m_inv = cv2.getPerspectiveTransform(dst, src)
    else:
        m_inv = None
    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
    return warped, m, m_inv


def pixels_to_world(pixels, K, T):
    invK = K.I
    t, r, _, _ = decompose(np.asarray(T, dtype=np.float64))
    invR = np.matrix(r).I
    R_inv_T = np.dot(invR, np.matrix(t).T)
    world_points = []
    for p in pixels:
        coords = np.float64([p[0], p[1], 1.0]).reshape(3, 1)
        cam_point = np.dot(invK, coords)
        world_point = np.dot(invR, cam_point)
        scale = R_inv_T[2][0] / world_point[2][0]
        scale_world = np.multiply(scale, world_point)
        world_point = np.array((np.asmatrix(scale_world) - np.asmatrix(R_inv_T))).reshape(-1,)
        world_points.append(world_point)
    return world_points


def world_to_pixels(world_points, K, T):
    pixel_points = []
    for wp in world_points:
        world_homo = np.append(wp, 1).reshape(4, 1)
        camera_point = np.dot(T, world_homo)
        pixel_homo = np.dot(K, camera_point[:3])
        pixel = (pixel_homo / pixel_homo[2])[:2].reshape(-1)
        pixel_points.append(pixel)
    return pixel_points


def extristric_plane_shift(tvec, rmat, delta_z):
    delta_t = np.array([[0], [0], [delta_z]])
    tvec_new = tvec + np.dot(rmat, delta_t)
    return tvec_new, rmat


pixel_to_world = pixels_to_world


def ros_pose_to_list(pose) -> tuple[np.ndarray, np.ndarray]:
    """geometry_msgs/Pose → (t, q) 其中 q 为 [w,x,y,z]。"""
    t = np.asarray([pose.position.x, pose.position.y, pose.position.z])
    q = np.asarray([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
    return t, q


def qua2rpy(qua):
    """与官方一致：Quaternion 消息或序列 [x,y,z,w]（非 ROS 分支下标顺序）。"""
    try:
        from geometry_msgs.msg import Quaternion as _Q

        if isinstance(qua, _Q):
            x, y, z, w = qua.x, qua.y, qua.z, qua.w
        else:
            x, y, z, w = qua[0], qua[1], qua[2], qua[3]
    except ImportError:
        x, y, z, w = qua[0], qua[1], qua[2], qua[3]
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(np.clip(2 * (w * y - x * z), -1.0, 1.0))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw


def rpy2qua(roll, pitch, yaw):
    """官方返回 geometry_msgs Pose.orientation；无 ROS 时返回 SimpleNamespace(w,x,y,z)。"""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    try:
        from geometry_msgs.msg import Pose

        q = Pose()
        q.orientation.w = w
        q.orientation.x = x
        q.orientation.y = y
        q.orientation.z = z
        return q.orientation
    except ImportError:
        return SimpleNamespace(w=w, x=x, y=y, z=z)


def xyz_quat_to_mat(xyz, quat):
    R = quat2mat(np.asarray(quat))
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = np.squeeze(np.asarray(xyz))
    return M


def xyz_rot_to_mat(xyz, rot):
    return np.row_stack((np.column_stack((rot, xyz)), np.array([[0, 0, 0, 1]])))


def xyz_euler_to_mat(xyz, euler, degrees=True):
    if degrees:
        ai, aj, ak = math.radians(euler[0]), math.radians(euler[1]), math.radians(euler[2])
    else:
        ai, aj, ak = euler[0], euler[1], euler[2]
    R = euler2mat(ai, aj, ak, "sxyz")
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = np.squeeze(np.asarray(xyz))
    return M


def mat_to_xyz_euler(mat, degrees=True):
    t, r, _, _ = decompose(np.asarray(mat, dtype=np.float64))
    ax, ay, az = mat2euler(r, "sxyz")
    euler = np.array([ax, ay, az], dtype=np.float64)
    if degrees:
        euler = np.degrees(euler)
    return t, euler


def load_camera_info_hand2cam(
    path: Path | str,
    *,
    depth_link_to_color_optical: list | np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """从 camera_info.yaml 读 hand2cam_tf_matrix；可选左乘 depth_link_to_color_optical（与官方 CalibrationNode 一致）。"""
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "hand2cam_tf_matrix" not in raw:
        raise KeyError(f"{p} 中缺少 hand2cam_tf_matrix")

    h = np.asarray(raw["hand2cam_tf_matrix"], dtype=np.float64)
    if h.shape != (4, 4):
        raise ValueError("hand2cam_tf_matrix 须为 4×4")

    t_extra = depth_link_to_color_optical
    if t_extra is None and "depth_link_to_color_optical" in raw:
        t_extra = raw["depth_link_to_color_optical"]
    if t_extra is not None:
        t_mat = np.asarray(t_extra, dtype=np.float64).reshape(4, 4)
        h = t_mat @ h

    return h, raw
