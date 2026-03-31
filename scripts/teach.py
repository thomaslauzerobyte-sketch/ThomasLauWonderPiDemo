"""示教录制：记录蓝色积木的拾取 / 放置位置，保存为任务 JSON。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from common import ROOT

TASKS_DIR = ROOT / "data" / "tasks"


def save_task(
    name: str,
    pick_pixel: tuple[float, float],
    pick_arm: tuple[float, float],
    place_pixel: tuple[float, float],
    place_arm: tuple[float, float],
    pick_z: float = -0.02,
    move_z: float = 0.05,
) -> dict[str, Any]:
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    task = {
        "name": name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "pick": {
            "pixel": list(pick_pixel),
            "arm": [round(pick_arm[0], 4), round(pick_arm[1], 4)],
        },
        "place": {
            "pixel": list(place_pixel),
            "arm": [round(place_arm[0], 4), round(place_arm[1], 4)],
        },
        "pick_z": pick_z,
        "move_z": move_z,
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    fname = f"{safe_name}_{stamp}.json"
    path = TASKS_DIR / fname
    path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
    task["_file"] = fname
    return task


def list_tasks() -> list[dict[str, Any]]:
    if not TASKS_DIR.is_dir():
        return []
    tasks = []
    for p in sorted(TASKS_DIR.glob("*.json"), reverse=True):
        try:
            t = json.loads(p.read_text(encoding="utf-8"))
            t["_file"] = p.name
            tasks.append(t)
        except Exception:
            continue
    return tasks


def load_task(filename: str) -> dict[str, Any]:
    p = TASKS_DIR / filename
    if not p.is_file():
        raise FileNotFoundError(f"任务文件不存在: {filename}")
    return json.loads(p.read_text(encoding="utf-8"))
