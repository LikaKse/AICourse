"""Инструменты агента: обычные Python-функции, их описания для Gemini и диспетчер.

Модель никогда не выполняет код сама. Она только присылает имя функции и аргументы,
а функцию вызывает dispatch() в Python.

Каждый инструмент возвращает dict (его можно сериализовать в JSON). Ошибку в аргументах
инструмент сообщает через ToolError, а диспетчер превращает её в {"error": "..."}, чтобы
модель увидела ошибку и смогла исправиться.
"""

from __future__ import annotations

import ast
import inspect
import math
import operator
from datetime import datetime
from zoneinfo import ZoneInfo

from google.genai import types

import config


class ToolError(Exception):
    """Ожидаемая ошибка инструмента (неверные аргументы и т. п.). Её текст увидит модель."""


# =====================================================================
# 1. get_current_time
# =====================================================================

WEEKDAYS_RU = ("понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье")


def get_current_time(now: datetime | None = None) -> dict:
    """Текущие дата и время в часовом поясе Asia/Ho_Chi_Minh.

    `now` нужен только тестам: модель этот параметр не видит (его нет в схеме).
    """
    tz = ZoneInfo(config.TIMEZONE)
    current = now.astimezone(tz) if now else datetime.now(tz)
    return {
        "timezone": config.TIMEZONE,
        "iso": current.isoformat(timespec="seconds"),
        "date": current.strftime("%Y-%m-%d"),
        "time": current.strftime("%H:%M:%S"),
        "weekday": WEEKDAYS_RU[current.weekday()],
        "utc_offset": current.strftime("%z"),
    }


# =====================================================================
# 2. calculate — безопасный калькулятор на ast (без eval)
# =====================================================================

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> int | float:
    """Рекурсивно вычисляет узел AST. Разрешены только числа и арифметика из белого списка."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        # bool — подкласс int, его тоже запрещаем: True + True не арифметика.
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ToolError(f"Недопустимое значение: {node.value!r}. Разрешены только числа.")

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > config.CALC_MAX_POWER_EXPONENT:
                raise ToolError(f"Слишком большая степень: {right}. Максимум {config.CALC_MAX_POWER_EXPONENT}.")
            if left == 0 and right < 0:
                raise ToolError("Ноль нельзя возводить в отрицательную степень.")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ToolError("Деление на ноль.")
        try:
            result = _BINARY_OPS[type(node.op)](left, right)
        except OverflowError:
            raise ToolError("Результат слишком большой.") from None
        if isinstance(result, complex):
            raise ToolError("Результат не является вещественным числом.")
        if abs(result) > config.CALC_MAX_ABS_RESULT:
            raise ToolError("Результат слишком большой.")
        return result

    raise ToolError(
        f"Недопустимая конструкция: {type(node).__name__}. "
        "Разрешены числа, скобки и операции + - * / // % **."
    )


def _normalize_number(value: int | float) -> int | float:
    """Убирает шум плавающей точки: 3750000 * 1.1 = 4125000.0000000005 → 4125000.

    Округляем до 12 значащих цифр: этого с запасом хватает для денег и бытовой арифметики.
    """
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ToolError("Результат не является конечным числом.")
        value = float(f"{value:.12g}")
        if value.is_integer():
            return int(value)
    return value


def calculate(expression: str) -> dict:
    """Вычисляет арифметическое выражение, например "(150 * 25000) * 1.1"."""
    if not isinstance(expression, str) or not expression.strip():
        raise ToolError("Параметр expression должен быть непустой строкой.")
    if len(expression) > config.CALC_MAX_EXPRESSION_LENGTH:
        raise ToolError(f"Выражение длиннее {config.CALC_MAX_EXPRESSION_LENGTH} символов.")

    # Модели иногда пишут знаки ×, ÷, − или пробелы-разделители тысяч. Приводим к Python-синтаксису.
    cleaned = (
        expression.replace("×", "*").replace("÷", "/").replace("−", "-")
        .replace(" ", "").replace("_", "")
    )
    try:
        tree = ast.parse(cleaned.strip(), mode="eval")
    except SyntaxError:
        raise ToolError(f"Не удалось разобрать выражение: {expression!r}.") from None

    return {"expression": expression, "result": _normalize_number(_eval_node(tree))}


# =====================================================================
# 3. convert_currency — USD ↔ VND по фиксированному демонстрационному курсу
# =====================================================================

_RATE_TEXT = f"{config.USD_TO_VND_RATE:,}".replace(",", " ")  # 25000 → "25 000"
RATE_NOTE = (
    f"Используется фиксированный демонстрационный курс 1 USD = {_RATE_TEXT} VND. "
    "Это НЕ актуальный рыночный курс."
)


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Конвертирует сумму между USD и VND по фиксированному курсу."""
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        raise ToolError("Параметр amount должен быть числом.")
    if not math.isfinite(amount) or amount < 0:
        raise ToolError("Параметр amount должен быть неотрицательным конечным числом.")

    src = str(from_currency).strip().upper()
    dst = str(to_currency).strip().upper()
    for code in (src, dst):
        if code not in config.SUPPORTED_CURRENCIES:
            raise ToolError(f"Валюта {code!r} не поддерживается. Доступны: USD, VND.")
    if src == dst:
        raise ToolError("Исходная и целевая валюты совпадают.")

    rate = config.USD_TO_VND_RATE
    converted = amount * rate if src == "USD" else amount / rate
    return {
        "amount": _normalize_number(amount),
        "from_currency": src,
        "to_currency": dst,
        "rate": f"1 USD = {rate} VND",
        "converted_amount": _normalize_number(round(converted, 2)),
        "rate_is_fixed_demo": True,
        "note": RATE_NOTE,
    }


# =====================================================================
# Описания для модели (FunctionDeclaration + JSON Schema)
# =====================================================================

FUNCTION_DECLARATIONS = [
    types.FunctionDeclaration(
        name="get_current_time",
        description=(
            f"Возвращает текущие дату, время и день недели в часовом поясе {config.TIMEZONE} (Вьетнам). "
            "Используй, когда пользователь спрашивает, который час, какое сегодня число или день недели."
        ),
        # Параметров нет, поэтому схема параметров не передаётся.
    ),
    types.FunctionDeclaration(
        name="calculate",
        description=(
            "Безопасно вычисляет арифметическое выражение и возвращает число. "
            "Поддерживает числа, скобки и операции + - * / // % **. "
            "Переменные, функции и знак % как «процент» не поддерживаются: "
            "10 процентов от X записывай как X * 10 / 100, увеличение на 10% — как X * 1.1."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Арифметическое выражение, например \"3750000 * 1.1\" или \"(2 + 3) * 4\".",
                },
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    ),
    types.FunctionDeclaration(
        name="convert_currency",
        description=(
            "Конвертирует сумму из долларов США (USD) во вьетнамские донги (VND) и обратно. "
            f"Курс фиксированный демонстрационный: 1 USD = {config.USD_TO_VND_RATE} VND, "
            "он не является актуальным рыночным курсом."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Сумма в исходной валюте, неотрицательное число."},
                "from_currency": {
                    "type": "string", "enum": list(config.SUPPORTED_CURRENCIES),
                    "description": "Исходная валюта: USD или VND.",
                },
                "to_currency": {
                    "type": "string", "enum": list(config.SUPPORTED_CURRENCIES),
                    "description": "Целевая валюта: USD или VND.",
                },
            },
            "required": ["amount", "from_currency", "to_currency"],
            "additionalProperties": False,
        },
    ),
]

# Реестр: имя инструмента → Python-функция. Имена совпадают с FUNCTION_DECLARATIONS.
TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "convert_currency": convert_currency,
}

# Аргументы, которые модель может передать каждой функции (только из схемы; `now` у
# get_current_time служебный и модели недоступен).
_ALLOWED_ARGS = {
    decl.name: set((decl.parameters_json_schema or {}).get("properties", {}))
    for decl in FUNCTION_DECLARATIONS
}


def dispatch(name: str, args: dict | None) -> dict:
    """Находит инструмент по имени, проверяет аргументы и вызывает функцию.

    Никогда не бросает исключений: любая проблема превращается в {"error": ...},
    чтобы агент мог передать её модели как FunctionResponse.
    Успех: {"result": <dict, который вернула функция>}.
    """
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"Неизвестный инструмент {name!r}. Доступны: {', '.join(TOOL_FUNCTIONS)}."}

    args = dict(args or {})
    unexpected = set(args) - _ALLOWED_ARGS[name]
    if unexpected:
        return {"error": f"Инструмент {name} не принимает аргументы: {', '.join(sorted(unexpected))}."}
    try:
        inspect.signature(func).bind(**args)
    except TypeError as exc:
        return {"error": f"Неверные аргументы для {name}: {exc}."}

    try:
        return {"result": func(**args)}
    except ToolError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # непредвиденный сбой внутри инструмента не должен ронять агента
        return {"error": f"Внутренняя ошибка инструмента {name}: {type(exc).__name__}: {exc}"}
