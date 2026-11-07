"""Две задачи эксперимента."""

import config
from validators import EXPECTED_DAYS, evaluate_generation, evaluate_single

TASKS = [
    {
        "id": "days_count",
        "kind": "single_answer",
        "title": "Один правильный ответ: дни с 3 марта по 17 июня 2025 включительно",
        "prompt": config.SINGLE_ANSWER_PROMPT,
        "expected": f"{EXPECTED_DAYS}",
        "evaluate": evaluate_single,
    },
    {
        "id": "cafe_names",
        "kind": "generation",
        "title": "Генерация: 5 названий кофейни рядом с офисом IT-компании",
        "prompt": config.GENERATION_PROMPT,
        "expected": None,                 # правильного ответа нет
        "evaluate": evaluate_generation,
    },
]

TASKS_BY_ID = {task["id"]: task for task in TASKS}
