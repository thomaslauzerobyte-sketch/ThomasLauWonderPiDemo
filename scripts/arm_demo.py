#!/usr/bin/env python3
"""机械臂驱动接口与辅助函数。

提供 ArmDriver protocol 与 MockArmDriver / HiWonderArmDriver 实现。
HiWonderArmDriver 优先通过 arm_bridge HTTP API 控制机械臂（延迟 ~100ms），
若 bridge 未启动则自动启动并回退到 docker exec（延迟 ~3s/命令）。
MockArmDriver 记录所有动作到 action_log 列表，便于 Web UI 回显。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from common import load_config


# ── Driver Protocol ───────────────────────────────────────────────
class ArmDriver(Protocol):
    def move_to_workspace(self, x: float, y: float, meta: dict[str, Any]) -> None: ...
    def move_to(self, x: float, y: float, z: float) -> None: ...
    def grip_open(self) -> None: ...
    def grip_close(self) -> None: ...
    def home(self) -> None: ...


# ── Mock Driver ───────────────────────────────────────────────────
@dataclass
class MockArmDriver:
    """无硬件时用于联调：记录每步动作到 action_log 并打印。"""

    action_log: list[dict[str, Any]] = field(default_factory=list)

    def _log(self, action: str, **kw: Any) -> None:
        entry = {"action": action, "t": time.time(), **kw}
        self.action_log.append(entry)
        print(f"[mock-arm] {action} {kw}")

    def move_to_workspace(self, x: float, y: float, meta: dict[str, Any]) -> None:
        self._log("move_to_workspace", x=round(x, 4), y=round(y, 4), meta=meta)

    def move_to(self, x: float, y: float, z: float) -> None:
        self._log("move_to", x=round(x, 4), y=round(y, 4), z=round(z, 4))

    def grip_open(self) -> None:
        self._log("grip_open")

    def grip_close(self) -> None:
        self._log("grip_close")

    def home(self) -> None:
        self._log("home")


# ── HiWonder ArmPi Ultra Driver ──────────────────────────────────
_ROS2_ENV = (
    "source ~/.zshrc 2>/dev/null; "
    "export ROS_DOMAIN_ID=100 && "
    "export CYCLONEDDS_URI=file:///etc/cyclonedds/config.xml && "
    "source /opt/ros/humble/setup.bash && "
    "source /home/ubuntu/ros2_ws/install/setup.bash"
)

_SERVO_HOME = {6: 500, 5: 500, 4: 500, 3: 500, 2: 500, 1: 700}

_BRIDGE_URL = "http://localhost:9091"
_BRIDGE_SCRIPT = "/home/ubuntu/arm_bridge.py"
_BRIDGE_SRC = Path(__file__).with_name("arm_bridge.py")


def _bridge_post(path: str, data: dict, timeout: float = 12.0) -> dict | None:
    """Send a POST request to the arm_bridge HTTP API."""
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{_BRIDGE_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _bridge_healthy() -> bool:
    try:
        with urllib.request.urlopen(f"{_BRIDGE_URL}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _ensure_bridge(container: str) -> bool:
    """Make sure arm_bridge is running inside Docker; start it if needed."""
    if _bridge_healthy():
        return True

    if _BRIDGE_SRC.exists():
        subprocess.run(
            f"docker cp {_BRIDGE_SRC} {container}:{_BRIDGE_SCRIPT}",
            shell=True, capture_output=True, timeout=10,
        )

    subprocess.run(
        f"docker exec -d -u ubuntu {container} bash -c "
        f"'{_ROS2_ENV} && python3 {_BRIDGE_SCRIPT}'",
        shell=True, capture_output=True, timeout=15,
    )

    for _ in range(20):
        time.sleep(0.5)
        if _bridge_healthy():
            print("[arm_bridge] started successfully")
            return True

    print("[arm_bridge] failed to start, falling back to docker exec")
    return False


@dataclass
class HiWonderArmDriver:
    """控制 ArmPi Ultra 机械臂。

    优先使用 arm_bridge HTTP API（毫秒级延迟），若不可用则
    回退到 docker exec + ros2 CLI（秒级延迟）。
    """

    container: str = "ArmPiUltra"
    pitch: float = -90.0
    pitch_range: tuple[float, float] = (-90.0, 90.0)
    grip_open_pulse: int = 700
    grip_close_pulse: int = 100
    duration: float = 1.0
    ik_timeout: int = 15
    action_log: list[dict[str, Any]] = field(default_factory=list)
    _use_bridge: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        self._use_bridge = _ensure_bridge(self.container)

    def _log(self, action: str, **kw: Any) -> None:
        entry = {"action": action, "t": time.time(), **kw}
        self.action_log.append(entry)
        print(f"[hiwonder-arm] {action} {kw}")

    # ── docker exec fallback ──────────────────────────────────────
    def _docker_exec(self, cmd: str, timeout: int | None = None) -> subprocess.CompletedProcess:
        full = f"docker exec -u ubuntu {self.container} bash -c '{_ROS2_ENV} && {cmd}'"
        return subprocess.run(
            full, shell=True, capture_output=True, text=True,
            timeout=timeout or self.ik_timeout,
        )

    def _pub_servos_cli(self, positions: dict[int, int], duration: float) -> None:
        pos_str = ", ".join(
            f"{{id: {sid}, position: {pos}}}" for sid, pos in sorted(positions.items())
        )
        cmd = (
            f'ros2 topic pub --once /servo_controller '
            f'servo_controller_msgs/msg/ServosPosition '
            f'"{{duration: {duration}, position_unit: \\"pulse\\", position: [{pos_str}]}}"'
        )
        r = self._docker_exec(cmd)
        if r.returncode != 0:
            raise RuntimeError(f"发布舵机指令失败: {r.stderr[:200]}")

    def _call_ik_cli(self, x: float, y: float, z: float) -> list[int] | None:
        cmd = (
            f'timeout {self.ik_timeout} ros2 service call '
            f'/kinematics/set_pose_target kinematics_msgs/srv/SetRobotPose '
            f'"{{position: [{x}, {y}, {z}], pitch: {self.pitch}, '
            f'pitch_range: [{self.pitch_range[0]}, {self.pitch_range[1]}], '
            f'resolution: 1.0, duration: {self.duration}}}"'
        )
        r = self._docker_exec(cmd, timeout=self.ik_timeout + 5)
        if r.returncode != 0:
            self._log("ik_error", x=x, y=y, z=z, stderr=r.stderr[:200])
            return None
        match = re.search(r"pulse=array\('H',\s*\[([^\]]+)\]\)", r.stdout)
        if not match:
            match = re.search(r"pulse=\[([^\]]+)\]", r.stdout)
        if not match:
            self._log("ik_parse_error", stdout=r.stdout[:300])
            return None
        try:
            return [int(v.strip()) for v in match.group(1).split(",")]
        except (ValueError, IndexError):
            return None

    # ── bridge-aware commands ─────────────────────────────────────
    def _pub_servos(self, positions: dict[int, int], duration: float | None = None) -> None:
        dur = duration if duration is not None else self.duration
        if self._use_bridge:
            r = _bridge_post("/servo", {
                "positions": {str(k): v for k, v in positions.items()},
                "duration": dur,
            })
            if r and r.get("ok"):
                return
            self._use_bridge = False
            print("[arm] bridge servo failed, falling back to docker exec")
        self._pub_servos_cli(positions, dur)

    def move_to(self, x: float, y: float, z: float) -> None:
        self._log("move_to", x=round(x, 4), y=round(y, 4), z=round(z, 4))

        if self._use_bridge:
            r = _bridge_post("/ik_move", {
                "x": x, "y": y, "z": z,
                "pitch": self.pitch,
                "pitch_range": list(self.pitch_range),
                "duration": self.duration,
            })
            if r and r.get("ok"):
                self._log("move_to_done", pulses=r.get("pulses"))
                return
            if r and not r.get("ok"):
                err = r.get("error", "unknown")
                self._log("bridge_ik_failed", error=err)
                raise RuntimeError(f"IK 求解失败: ({x}, {y}, {z}) - {err}")
            self._use_bridge = False
            print("[arm] bridge ik_move failed, falling back to docker exec")

        pulses = self._call_ik_cli(x, y, z)
        if pulses is None:
            self._log("ik_failed", x=x, y=y, z=z)
            raise RuntimeError(f"IK 求解失败: ({x}, {y}, {z})")
        if len(pulses) < 4:
            raise RuntimeError(f"IK 返回脉宽数量不足: {pulses}")

        servos = {6: pulses[0], 5: pulses[1], 4: pulses[2], 3: pulses[3]}
        if len(pulses) >= 5:
            servos[2] = pulses[4]
        self._pub_servos_cli(servos, self.duration)
        self._log("move_to_done", pulses=pulses)

    def move_to_workspace(self, x: float, y: float, meta: dict[str, Any]) -> None:
        self.move_to(x, y, 0.15)

    def grip_open(self) -> None:
        self._log("grip_open", pulse=self.grip_open_pulse)
        self._pub_servos({1: self.grip_open_pulse}, duration=0.5)

    def grip_close(self) -> None:
        self._log("grip_close", pulse=self.grip_close_pulse)
        self._pub_servos({1: self.grip_close_pulse}, duration=0.5)

    def home(self) -> None:
        self._log("home")
        self._pub_servos(_SERVO_HOME, duration=1.0)

    def set_servos(self, positions: dict[int, int], duration: float = 1.0) -> None:
        self._log("set_servos", positions=positions, duration=duration)
        self._pub_servos(positions, duration=duration)

    def buzzer(self, freq: int = 1900, on_time: float = 0.1) -> None:
        if self._use_bridge:
            r = _bridge_post("/buzzer", {"freq": freq, "on_time": on_time})
            if r and r.get("ok"):
                return
        cmd = (
            f'ros2 topic pub --once /ros_robot_controller/set_buzzer '
            f'ros_robot_controller_msgs/msg/Buzzer '
            f'"{{freq: {freq}, on_time: {on_time}, off_time: 0.01, repeat: 1}}"'
        )
        try:
            self._docker_exec(cmd, timeout=10)
        except Exception:
            pass


# ── Factory ───────────────────────────────────────────────────────
def make_driver(arm_cfg: dict[str, Any]) -> ArmDriver:
    mode = arm_cfg.get("driver", "mock" if arm_cfg.get("mock", True) else "serial")

    if mode == "hiwonder":
        return HiWonderArmDriver(
            container=arm_cfg.get("docker_container", "ArmPiUltra"),
            pitch=float(arm_cfg.get("ik_pitch", -90.0)),
            grip_open_pulse=int(arm_cfg.get("grip_open_pulse", 700)),
            grip_close_pulse=int(arm_cfg.get("grip_close_pulse", 100)),
            duration=float(arm_cfg.get("move_duration", 1.5)),
        )

    if mode == "mock" or arm_cfg.get("mock", True):
        return MockArmDriver()

    raise RuntimeError(f"未知的 arm.driver 类型: {mode}")


# ── Legacy helpers (used by existing app.py routes) ───────────────
def _bbox_area(d: dict[str, Any]) -> float:
    bb = d.get("bbox")
    if not bb or len(bb) != 4:
        return 0.0
    return float(max(bb[2], 0) * max(bb[3], 0))


def _pick_detection(payload: dict[str, Any]) -> dict[str, Any] | None:
    if "results" in payload:
        dets: list[dict[str, Any]] = []
        for r in payload["results"]:
            dets.extend(r.get("detections") or [])
    else:
        dets = payload.get("detections") or []
    if not dets:
        return None
    with_bbox = [d for d in dets if d.get("bbox")]
    pool = with_bbox if with_bbox else dets
    return max(pool, key=_bbox_area)


def pixel_from_detection(det: dict[str, Any], img_w: int, img_h: int) -> tuple[float, float] | None:
    bb = det.get("bbox")
    if bb and len(bb) == 4:
        x, y, w, h = bb
        return x + w / 2.0, y + h / 2.0
    if det.get("bbox") is None and det.get("score") is not None:
        return img_w / 2.0, img_h / 2.0
    return None


def to_workspace(cx: float, cy: float, img_w: int, img_h: int) -> tuple[float, float]:
    nx = (cx / max(img_w, 1) - 0.5) * 2.0
    ny = (0.5 - cy / max(img_h, 1)) * 2.0
    return nx, ny


def run_arm(
    payload: dict[str, Any],
    arm_cfg: dict[str, Any],
    driver: ArmDriver | None = None,
) -> None:
    img_w = int(arm_cfg.get("image_width", 640))
    img_h = int(arm_cfg.get("image_height", 480))
    det = _pick_detection(payload)
    if det is None:
        print("无检测结果，跳过臂控。", file=sys.stderr)
        return

    pix = pixel_from_detection(det, img_w, img_h)
    if pix is None:
        print("无法从检测结果得到像素坐标。", file=sys.stderr)
        return

    wx, wy = to_workspace(pix[0], pix[1], img_w, img_h)
    d = driver or make_driver(arm_cfg)
    d.move_to_workspace(
        wx, wy,
        {"label": det.get("label"), "score": det.get("score"), "pixel": list(pix)},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="WonderPi Demo：根据识别结果驱动机械臂（mock/串口占位）")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--json", type=Path, required=True, help="infer.py 输出的 JSON 文件")
    args = parser.parse_args()

    cfg = load_config(args.config)
    arm_cfg = cfg.get("arm", {})

    try:
        payload = json.loads(args.json.read_text(encoding="utf-8"))
        run_arm(payload, arm_cfg)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
