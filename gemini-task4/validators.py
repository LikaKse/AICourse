"""Разбор ответа и проверки: формат JSON, правильность ответа, структура списка названий.

Формат и правильность оцениваются раздельно:
  * format_status — соблюдён ли формат «только JSON» буквально (строгий json.loads);
  * если строгий разбор не удался, JSON пытаемся восстановить (из ```-блока или по первым { … }),
    чтобы всё равно оценить содержание — но формат при этом считается нарушенным;
  * обрезанный ответ (finish_reason = MAX_TOKENS) отмечается отдельно и автоматически
    ошибкой правильности не считается.
"""

import json
import re
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, ConfigDict, ValidationError

# ---------- Разбор JSON ----------

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*(.*?)\s*```$", re.S)


def parse_json_response(text):
    """Возвращает dict: format_status (ok | fenced | extra_text | json_error | empty),
    format_error, data (dict/list или None), recovered (JSON извлечён не строгим разбором)."""
    result = {"format_status": None, "format_error": None, "data": None, "recovered": False}
    if text is None or not text.strip():
        result.update(format_status="empty", format_error="Пустой ответ")
        return result
    stripped = text.strip()
    try:
        result.update(format_status="ok", data=json.loads(stripped))
        return result
    except json.JSONDecodeError as e:
        strict_error = f"{e.msg} (строка {e.lineno}, позиция {e.colno})"

    fence = _FENCE_RE.match(stripped)
    if fence:
        try:
            result.update(format_status="fenced", format_error="JSON обёрнут в ```-блок",
                          data=json.loads(fence.group(1)), recovered=True)
            return result
        except json.JSONDecodeError:
            pass
    start, end = stripped.find("{"), stripped.rfind("}")
    if 0 <= start < end:
        try:
            result.update(format_status="extra_text", format_error="Текст вне JSON",
                          data=json.loads(stripped[start:end + 1]), recovered=True)
            return result
        except json.JSONDecodeError:
            pass
    result.update(format_status="json_error", format_error=strict_error)
    return result


# ---------- Схемы (только для проверки, в API не передаются) ----------

class SingleAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    explanation: str


class NameList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    names: list[str]


def _schema_errors(model, data):
    try:
        model.model_validate(data)
        return None
    except ValidationError as e:
        return [{"loc": ".".join(str(p) for p in err["loc"]), "type": err["type"], "msg": err["msg"]}
                for err in e.errors()]


# ---------- Нормализация ----------

def normalize_text(text):
    if text is None:
        return ""
    text = str(text).replace(" ", " ").replace("ё", "е").replace("Ё", "Е")
    text = re.sub(r"\s+", " ", text.lower())
    return text.strip(" \t\n.,;:!?«»\"'“”„()[]{}—–-")


def normalize_name(name):
    """Название для сравнения: регистр, ё→е, кавычки и пунктуация по краям, пробелы."""
    return normalize_text(name)


_NUMBER_RE = re.compile(r"(?<![\d.,])[-−]?\d+(?:[.,]\d+)?(?!\d)")


def extract_numbers(text):
    values = []
    for match in _NUMBER_RE.finditer(text or ""):
        try:
            values.append(Decimal(match.group().replace("−", "-").replace(",", ".")))
        except InvalidOperation:
            continue
    return values


# ---------- Задача с одним ответом ----------

EXPECTED_DAYS = 107


def check_days(answer):
    """Эталон 107: в ответе ровно одно различное число, и оно равно 107."""
    values = set(extract_numbers(normalize_text(answer)))
    if not values:
        return False, "в ответе нет числа"
    if len(values) > 1:
        return False, f"неоднозначно, несколько чисел: {sorted(str(v) for v in values)}"
    (value,) = values
    return (value == EXPECTED_DAYS), (f"{EXPECTED_DAYS}" if value == EXPECTED_DAYS else f"получено {value}")


def evaluate_single(text, finish_reason):
    """Оценка ответа на задачу с одним правильным ответом."""
    parsed = parse_json_response(text)
    out = {
        "format_status": parsed["format_status"],
        "format_error": parsed["format_error"],
        "recovered": parsed["recovered"],
        "schema_errors": None,
        "format_ok": False,
        "truncated": finish_reason == "MAX_TOKENS",
        "answer": None,
        "explanation": None,
        "answer_key": None,          # нормализованный ответ для сравнения прогонов
        "is_correct": None,          # None — не оценено (нет ответа API / формат не разобран)
        "check_reason": None,
    }
    data = parsed["data"]
    if data is None:
        out["check_reason"] = ("ответ обрезан (MAX_TOKENS), JSON не разобран — правильность не оценена"
                               if out["truncated"] else "формат не разобран — правильность не оценена")
        return out
    out["schema_errors"] = _schema_errors(SingleAnswer, data)
    out["format_ok"] = parsed["format_status"] == "ok" and out["schema_errors"] is None
    answer = data.get("answer") if isinstance(data, dict) else None
    if answer is None or (isinstance(answer, str) and not answer.strip()):
        out["check_reason"] = "нет поля answer — правильность не оценена"
        return out
    out["answer"] = str(answer)
    explanation = data.get("explanation") if isinstance(data, dict) else None
    out["explanation"] = None if explanation is None else str(explanation)
    numbers = sorted(set(extract_numbers(normalize_text(answer))))
    out["answer_key"] = str(numbers[0]) if len(numbers) == 1 else normalize_text(answer)
    out["is_correct"], out["check_reason"] = check_days(answer)
    return out


# ---------- Задача на генерацию ----------

EXPECTED_NAMES = 5


def evaluate_generation(text, finish_reason):
    """Оценка списка названий: формат, структура (5 непустых строк без повторов)."""
    parsed = parse_json_response(text)
    out = {
        "format_status": parsed["format_status"],
        "format_error": parsed["format_error"],
        "recovered": parsed["recovered"],
        "schema_errors": None,
        "format_ok": False,
        "truncated": finish_reason == "MAX_TOKENS",
        "names": None,
        "names_normalized": None,
        "duplicates_within": None,
        "structure_issues": None,
        "structure_ok": None,        # None — не оценено
    }
    data = parsed["data"]
    if data is None:
        out["structure_issues"] = ["ответ обрезан (MAX_TOKENS), JSON не разобран" if out["truncated"]
                                   else "формат не разобран"]
        return out
    out["schema_errors"] = _schema_errors(NameList, data)
    out["format_ok"] = parsed["format_status"] == "ok" and out["schema_errors"] is None
    names = data.get("names") if isinstance(data, dict) else None
    if not isinstance(names, list):
        out["structure_ok"] = False
        out["structure_issues"] = ["нет списка names"]
        return out
    issues = []
    strings = [n for n in names if isinstance(n, str)]
    if len(strings) != len(names):
        issues.append(f"не строки: {len(names) - len(strings)}")
    cleaned = [s.strip() for s in strings]
    if any(not s for s in cleaned):
        issues.append("есть пустые названия")
    cleaned = [s for s in cleaned if s]
    normalized = [normalize_name(s) for s in cleaned]
    seen, dups = set(), []
    for n in normalized:
        if n in seen and n not in dups:
            dups.append(n)
        seen.add(n)
    if dups:
        issues.append(f"повторы внутри списка: {dups}")
    if len(names) != EXPECTED_NAMES:
        issues.append(f"названий {len(names)} вместо {EXPECTED_NAMES}")
    out.update(names=cleaned, names_normalized=normalized, duplicates_within=dups,
               structure_issues=issues or None, structure_ok=not issues)
    return out
