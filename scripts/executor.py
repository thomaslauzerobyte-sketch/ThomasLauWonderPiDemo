"""任务执行器：按示教任务驱动机械臂完成拾取 → 放置序列。

执行返回 action_log 列表，每步含 action / 坐标 / 时间戳，供 Web UI 实时展示。
"""

from __future__ import annotations

import time
from typing import Any

from arm_demo import ArmDriver, make_driver


def execute_pick_place(
    task: dict[str, Any],
    arm_cfg: dict[str, Any],
    *,
    pick_arm: tuple[float, float] | None = None,
    driver: ArmDriver | None = None,
) -> list[dict[str, Any]]:
    """执行一次完整拾取-放置。

    Args:
        task: 示教任务 JSON（含 pick / place / pick_z / move_z）。
        arm_cfg: config 中的 arm 段。
        pick_arm: 若提供，使用此实时检测坐标作为实际拾取点；否则用 task 中记录的。
        driver: 臂控驱动，None 时按 arm_cfg 自动创建。
    """
    d = driver or make_driver(arm_cfg)
    if hasattr(d, "action_log"):
        d.action_log.clear()

    pick = pick_arm or tuple(task["pick"]["arm"])
    place = tuple(task["place"]["arm"])
    pick_z = float(task.get("pick_z", -0.02))
    move_z = float(task.get("move_z", 0.05))
    move_dur = float(arm_cfg.get("move_duration", 0.8))
    step_delay = float(arm_cfg.get("step_delay_sec", 0.5))
    grip_delay = min(step_delay, 0.6)

    # 1. 张开夹爪
    d.grip_open()
    time.sleep(grip_delay)

    # 2. 移动到抓取位置（先到安全高度，再到抓取高度）
    d.move_to(pick[0], pick[1], move_z)
    time.sleep(move_dur + 0.1)
    d.move_to(pick[0], pick[1], pick_z)
    time.sleep(move_dur + 0.1)

    # 3. 闭合夹爪
    d.grip_close()
    time.sleep(grip_delay)

    # 4. 移动到放置位置（先抬起再平移）
    d.move_to(pick[0], pick[1], move_z)
    time.sleep(move_dur + 0.1)
    d.move_to(place[0], place[1], move_z)
    time.sleep(move_dur + 0.1)
    d.move_to(place[0], place[1], pick_z)
    time.sleep(move_dur + 0.1)

    # 5. 张开夹爪
    d.grip_open()
    time.sleep(grip_delay)

    d.move_to(place[0], place[1], move_z)
    time.sleep(move_dur + 0.1)

    if hasattr(d, "action_log"):
        return d.action_log
    return [{"action": "done"}]
