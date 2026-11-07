"""Разбор ответа модели и программная проверка правильности.

Скопировано из gemini-task3/validators.py без изменений логики; в конец добавлена
проверка задачи из gemini-task4 (107 дней) и реестр CHECKERS.

Модели не доверяем: правильность определяется только функциями проверки ниже,
отдельной для каждой задачи. Самооценка модели (explanation) не учитывается.
"""

import json
import re
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, ConfigDict, Field, ValidationError


# ---------- Схема ответа ----------
# ApiSolution передаётся в response_schema (без extra="forbid": Gemini API не принимает
# additionalProperties — урок задачи №2). StrictSolution используется для проверки.

class ApiSolution(BaseModel):
    """Формат финального ответа модели (схема для API)."""

    answer: str = Field(description="Короткий ответ")
    explanation: str = Field(description="Краткое объяснение")


class StrictSolution(ApiSolution):
    """Строгая проверка: те же поля, лишние поля запрещены, ответ непустой."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, description="Короткий ответ")


def parse_solution(text):
    """Строгий разбор JSON-ответа: json.loads + StrictSolution.

    Возвращает dict: parse_status (ok | empty | json_error), parse_error,
    validation_status (ok | validation_error | skipped), validation_errors, answer, explanation.
    """
    result = {
        "parse_status": None,
        "parse_error": None,
        "validation_status": None,
        "validation_errors": None,
        "answer": None,
        "explanation": None,
    }
    if text is None or not text.strip():
        result.update(parse_status="empty", parse_error="Пустой ответ", validation_status="skipped")
        return result
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError as e:
        result.update(parse_status="json_error",
                      parse_error=f"{e.msg} (строка {e.lineno}, позиция {e.colno})",
                      validation_status="skipped")
        return result
    result["parse_status"] = "ok"
    try:
        solution = StrictSolution.model_validate(data)
    except ValidationError as e:
        result["validation_status"] = "validation_error"
        result["validation_errors"] = [
            {"loc": ".".join(str(p) for p in err["loc"]), "type": err["type"], "msg": err["msg"]}
            for err in e.errors()
        ]
        return result
    result.update(validation_status="ok", answer=solution.answer, explanation=solution.explanation)
    return result


# ---------- Нормализация ----------

_EDGE_PUNCT = " \t\n\r.,;:!?«»\"'()[]{}"


def normalize_text(text):
    """Нижний регистр, неразрывные пробелы → обычные, схлопывание пробелов, без пунктуации по краям."""
    if text is None:
        return ""
    text = str(text).replace(" ", " ").replace(" ", " ")
    text = re.sub(r"\s+", " ", text.lower())
    return text.strip(_EDGE_PUNCT)


# Число: целое или дробное, дробная часть через точку или запятую («0,05», «0.05»).
# Запятая с пробелом после неё («1, 2») — разделитель списка, а не дробная часть.
_NUMBER_RE = re.compile(r"(?<![\d.,])[-−]?\d+(?:[.,]\d+)?(?![\d])")


def extract_numbers(text):
    """Все числа из текста как Decimal (с позициями конца числа). Запятая в дробях → точка."""
    numbers = []
    for match in _NUMBER_RE.finditer(text):
        raw = match.group().replace("−", "-").replace(",", ".")
        try:
            numbers.append((Decimal(raw), match.end()))
        except InvalidOperation:
            continue
    return numbers


# ---------- Проверки по задачам ----------
# Каждая функция: check_*(answer: str) -> (is_correct: bool, reason: str)

_CENT_RE = re.compile(r"^\s*(центов|цента|цент|cents|cent|¢)")


def check_bat_ball(answer):
    """Эталон: 0.05 доллара. Допускаются «$0.05», «0,05», «5 центов», «5 cents».

    Каждое число переводится в доллары (если сразу после него «цент/cent/¢» — делится на 100).
    Верно, только если все найденные значения равны ровно 0.05.
    """
    text = normalize_text(answer)
    numbers = extract_numbers(text)
    if not numbers:
        return False, "в ответе нет числа"
    values = set()
    for value, end in numbers:
        if _CENT_RE.match(text[end:]):
            value = value / 100
        values.add(value)
    if values == {Decimal("0.05")}:
        return True, "0.05 доллара"
    return False, f"найдено значений: {sorted(str(v) for v in values)}"


_YES_RE = re.compile(r"\b(да|yes|true|верно)\b")
_NO_RE = re.compile(r"\b(нет|no|false|неверно)\b|\bне\s+следует\b")
_FOLLOWS_RE = re.compile(r"(?<!не )\bследует\b")


def check_yes(answer):
    """Эталон: «да». Верно, если есть утверждение и нет отрицания («да и нет» — неверно)."""
    text = normalize_text(answer)
    has_yes = bool(_YES_RE.search(text) or _FOLLOWS_RE.search(text))
    has_no = bool(_NO_RE.search(text))
    if has_yes and not has_no:
        return True, "да"
    if has_yes and has_no:
        return False, "неоднозначный ответ: и «да», и «нет»"
    if has_no:
        return False, "ответ «нет»"
    return False, "нет ни «да», ни «нет»"


def make_integer_checker(expected):
    """Проверка целого числа: в ответе ровно одно различное число, и оно равно expected."""

    def check(answer):
        text = normalize_text(answer)
        values = {value for value, _ in extract_numbers(text)}
        if not values:
            return False, "в ответе нет числа"
        if len(values) > 1:
            return False, f"неоднозначно, несколько чисел: {sorted(str(v) for v in values)}"
        (value,) = values
        if value == expected:
            return True, str(expected)
        return False, f"получено {value}"

    check.__name__ = f"check_integer_{expected}"
    return check


check_combinatorics = make_integer_checker(24)
check_code_total = make_integer_checker(14)

# Кириллические буквы, похожие на латинские A, B, C (только заглавные: строчные «а», «в», «с» —
# это обычные русские слова и предлоги).
_CYRILLIC_TO_LATIN = {"А": "A", "В": "B", "С": "C"}
_ORDER_LETTERS = set("ABCD")


def extract_order(answer):
    """Последовательность букв A–D из ответа: «D, A, B, C», «D → A → B → C», «DABC», «d-a-b-c».

    Берутся только «слова», целиком состоящие из букв A–D (латиница любого регистра
    или заглавные кириллические А, В, С). Остальные слова («Порядок», «затем») пропускаются.
    """
    letters = []
    for word in re.findall(r"[^\W\d_]+", answer or ""):
        if word.isascii():
            candidate = word.upper()                       # латиница любого регистра
        elif all(ch in _CYRILLIC_TO_LATIN or ch in _ORDER_LETTERS for ch in word):
            candidate = "".join(_CYRILLIC_TO_LATIN.get(ch, ch) for ch in word)
        else:
            continue                                       # обычное русское слово
        if set(candidate) <= _ORDER_LETTERS:
            letters.extend(candidate)
    return letters


def check_order(answer):
    """Эталон: D, A, B, C — точное совпадение последовательности."""
    order = extract_order(answer)
    if order == ["D", "A", "B", "C"]:
        return True, "D, A, B, C"
    if not order:
        return False, "в ответе нет последовательности букв A–D"
    return False, f"получено {', '.join(order)}"


# ---------- Утечка ответа в сгенерированный промпт ----------

def contains_any(text, patterns):
    """True, если в тексте (после нормализации) встречается хотя бы один из regex-шаблонов."""
    text = normalize_text(text)
    return any(re.search(p, text) for p in patterns)


# ---------- Задача из Дня 4 ----------
# В gemini-task4 эталон EXPECTED_DAYS = 107 проверялся так же: «в ответе ровно одно различное
# число, и оно равно 107». Логика та же, что у make_integer_checker.

EXPECTED_DAYS = 107
check_days = make_integer_checker(EXPECTED_DAYS)


# Реестр проверок: tasks.json ссылается на функцию по имени (поле "checker").
CHECKERS = {
    "check_bat_ball": check_bat_ball,
    "check_yes": check_yes,
    "check_combinatorics": check_combinatorics,
    "check_code_total": check_code_total,
    "check_order": check_order,
    "check_days": check_days,
}
