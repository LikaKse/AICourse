"""Загрузка задач из tasks.json и привязка функций проверки из validators.py."""

import json
from pathlib import Path

from validators import CHECKERS

TASKS_PATH = Path(__file__).parent / "tasks.json"

REQUIRED_FIELDS = ("id", "category", "question", "expected", "checker", "source")


def load_tasks(path=TASKS_PATH):
    """Список задач. Каждая задача — dict из tasks.json плюс поле "check" (функция проверки)."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    tasks, seen = [], set()
    for item in raw:
        missing = [k for k in REQUIRED_FIELDS if k not in item]
        if missing:
            raise ValueError(f"Задача {item.get('id')!r}: нет полей {missing}")
        if item["id"] in seen:
            raise ValueError(f"Повторяется id задачи: {item['id']}")
        if item["checker"] not in CHECKERS:
            raise ValueError(f"Задача {item['id']}: неизвестная проверка {item['checker']}")
        seen.add(item["id"])
        tasks.append({**item, "check": CHECKERS[item["checker"]]})
    return tasks


TASKS = load_tasks()
TASKS_BY_ID = {t["id"]: t for t in TASKS}
