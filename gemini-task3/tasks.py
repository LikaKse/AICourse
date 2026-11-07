"""Набор из пяти задач с эталонными ответами и функциями проверки."""

from validators import (
    check_bat_ball,
    check_code_total,
    check_combinatorics,
    check_order,
    check_yes,
)

# id — стабильный идентификатор задачи (входит в имя решения и в отчёт)
# question — текст задачи, который получает модель
# expected — эталонный ответ (для отчёта и записи expected_answer)
# check — функция программной проверки ответа: check(answer) -> (bool, reason)
# leak_patterns — regex-шаблоны эталона для эвристики «сгенерированный промпт уже содержит ответ»
#                 (None — утечку для этой задачи определить нельзя, например «да/нет»)

TASKS = [
    {
        "id": "math_bat_ball",
        "category": "Математика",
        "question": "Бита и мяч вместе стоят $1,10. Бита на $1 дороже мяча. Сколько стоит мяч?",
        "expected": "0.05 доллара",
        "check": check_bat_ball,
        "leak_patterns": [r"0[.,]05", r"\b5\s*(центов|цента|цент|cents?)"],
    },
    {
        "id": "logic_syllogism",
        "category": "Логика",
        "question": "Все A являются B. Все B являются C. Следует ли, что все A являются C?",
        "expected": "да",
        "check": check_yes,
        "leak_patterns": None,
    },
    {
        "id": "combinatorics_digits",
        "category": "Комбинаторика",
        "question": "Сколько трёхзначных чисел можно составить из цифр 1, 2, 3, 4 без повторений?",
        "expected": "24",
        "check": check_combinatorics,
        "leak_patterns": [r"\b24\b"],
    },
    {
        "id": "algorithm_loop",
        "category": "Алгоритмы",
        "question": (
            "Выполняется код:\n"
            "total = 0\n"
            "for i in range(1, 4):\n"
            "    total += i * i\n"
            "Чему равен total?"
        ),
        "expected": "14",
        "check": check_code_total,
        "leak_patterns": [r"\b14\b"],
    },
    {
        "id": "order_logic",
        "category": "Логика порядка",
        "question": (
            "Четыре человека D, A, B, C должны выступить. D выступает раньше A, A раньше B, "
            "B раньше C. Каков порядок?"
        ),
        "expected": "D, A, B, C",
        "check": check_order,
        # Сама задача перечисляет D, A, B, C именно в этом порядке, поэтому повтор этой
        # строки в промпте не считается утечкой; ищем только явные формулировки ответа.
        "leak_patterns": [r"(ответ|порядок)\s*[:—-]\s*d\s*[,→>-]\s*a\s*[,→>-]\s*b\s*[,→>-]\s*c"],
    },
]

TASKS_BY_ID = {task["id"]: task for task in TASKS}
