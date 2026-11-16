"""ЯВНЫЕ правила записи и чтения трёх слоёв памяти.

Таблицы WRITE_RULES / READ_RULES — человекочитаемое описание (печатается в чате командой /rules
и в отчёте эксперимента). Функции check_* — исполняемая форма тех же правил записи: модель только
ПРЕДЛАГАЕТ операции, а записывает их код и только после проверки. Каждое отклонение получает причину.

Схема операции (одинакова для обоих слоёв, которые пишет модель):

    {"layer": "working" | "long_term", "op": "set" | "delete", "key": ..., "value": ...,
     "kind": "user" | "accepted_proposal" | "assistant_proposal", "category": ...,
     "evidence": ..., "proposal_evidence": ..., "reason": ...}

Краткосрочную память модель не пишет вообще — её ведёт код (MemoryManager.record_exchange).
"""

from __future__ import annotations

import json
import re

from memory.evidence import numbers_supported, quote_in, quote_is_question, same_value

LAYERS = ("short_term", "working", "long_term")
MODEL_WRITABLE_LAYERS = ("working", "long_term")
LAYER_TITLES = {
    "short_term": "Краткосрочная память",
    "working": "Рабочая память",
    "long_term": "Долговременная память",
}

# ---------- Виды записей (Sticky Facts, День 10) ----------
KINDS = ("user", "accepted_proposal", "assistant_proposal")
DECISION_KINDS = ("user", "accepted_proposal")

# ---------- Рабочая память: фиксированный набор полей задачи ----------
SCALAR_FIELDS = ("goal", "deadline", "budget", "status", "next_step")   # значение заменяется
LIST_FIELDS = ("constraints", "results")                               # элементы добавляются
WORKING_FIELDS = SCALAR_FIELDS + LIST_FIELDS
FIELD_ALIASES = {
    "constraint": "constraints", "result": "results", "intermediate_result": "results",
    "intermediate_results": "results", "next": "next_step", "nextstep": "next_step",
    "due": "deadline", "due_date": "deadline", "task_goal": "goal", "task_status": "status",
}
TASK_STATUSES = ("planning", "in_progress", "blocked", "done")
STATUS_ALIASES = {
    "in progress": "in_progress", "inprogress": "in_progress", "в работе": "in_progress",
    "в процессе": "in_progress", "планирование": "planning", "planned": "planning",
    "заблокирована": "blocked", "blocked_by": "blocked", "завершена": "done", "готово": "done",
    "completed": "done", "complete": "done", "finished": "done",
}

# ---------- Долговременная память: разрешённые категории ----------
LONG_TERM_CATEGORIES = ("name", "location", "environment", "constraint", "preference", "decision",
                        "knowledge", "reason", "alternative", "other")
# Категории данных задачи: в long_term запрещены (их место — working).
TASK_CATEGORIES = ("goal", "deadline", "budget", "status", "result", "next_step", "task", "plan")
ALL_CATEGORIES = LONG_TERM_CATEGORIES + TASK_CATEGORIES

WRITE_RULES = {
    "short_term": (
        "Пишет только код, без вызова LLM: каждая пара «сообщение пользователя — ответ ассистента» как есть.",
        "Только если ответ получен успешно (status=ok) и сообщение не контрольное (remember=True).",
        "Ошибки API, пустые ответы и контрольные вопросы не записываются.",
        "Хранится вся история текущего диалога; размер окна влияет только на чтение.",
        "Операции модели с layer=short_term отклоняются.",
    ),
    "working": (
        "Только операции layer=working из вызова менеджера памяти (1 вызов на сообщение пользователя).",
        "key — одно из полей задачи: goal, deadline, budget, status, next_step (значение заменяется, "
        "прежнее уходит в history) или constraints, results (добавляется элемент, дубликаты не пишутся).",
        "status — только planning | in_progress | blocked | done.",
        "Цитата evidence — в сообщении пользователя (kind=user) или согласие пользователя + цитата "
        "ассистента (kind=accepted_proposal); числа значения должны быть в исходном тексте.",
        "Неподтверждённые предложения ассистента (kind=assistant_proposal) в рабочую память не пишутся.",
        "После status=done задача не меняется; новая задача начинается с goal, завершённая уходит в archived.",
        "Ничего не переносится в долговременную память — ни автоматически, ни при завершении или очистке.",
    ),
    "long_term": (
        "Только операции layer=long_term (правила Sticky Facts из Дня 10).",
        "Три вида записи: user, accepted_proposal, assistant_proposal — с проверкой цитат в нужных сообщениях "
        "и чисел значения.",
        "Категории: name, location, environment, constraint, preference, decision, knowledge, reason, "
        "alternative, other.",
        "Данные текущей задачи (категория или ключ goal, deadline, budget, status, result, next_step, "
        "task, plan…) отклоняются: их место — рабочая память.",
        "Вопрос пользователя — не факт: цитата из вопросительного предложения отклоняется.",
        "assistant_proposal: если цитата ассистента пришла в proposal_evidence вместо evidence, она "
        "перекладывается в evidence (только если дословно есть в сообщении ассистента).",
        "Предложение ассистента не может перезаписать решение пользователя; изменение значения сохраняет "
        "прежнее в history; delete — только с причиной и цитатой пользователя.",
    ),
}

READ_RULES = {
    "long_term": (
        "Первым блоком промпта, если есть хотя бы один факт.",
        "Факты сгруппированы по kind; неподтверждённые предложения помечены «НЕ решение пользователя» "
        "(флаг INCLUDE_UNCONFIRMED_PROPOSALS=False скрывает их).",
        "С цитатами-источниками, если RENDER_EVIDENCE=True.",
    ),
    "working": (
        "Вторым блоком — только если есть текущая задача и её status ≠ done.",
        "Цель, срок, бюджет, ограничения, промежуточные результаты, следующий шаг и статус — без цитат.",
    ),
    "short_term": (
        "После блоков памяти — последние SHORT_TERM_WINDOW сообщений как обычные реплики диалога.",
        "Если окно начинается с ответа ассистента, он отбрасывается, чтобы роли чередовались.",
        "window=0 — история в промпт не попадает.",
    ),
}

#: JSON Schema ответа менеджера памяти (response_json_schema). Все поля обязательны — проще проверять.
OPERATIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "layer": {"type": "string", "enum": list(MODEL_WRITABLE_LAYERS)},
                    "op": {"type": "string", "enum": ["set", "delete"]},
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                    "kind": {"type": "string", "enum": list(KINDS)},
                    "category": {"type": "string", "enum": list(ALL_CATEGORIES)},
                    "evidence": {"type": "string"},
                    "proposal_evidence": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["layer", "op", "key", "value", "kind", "category", "evidence",
                             "proposal_evidence", "reason"],
            },
        },
    },
    "required": ["operations"],
}


class MemoryFormatError(Exception):
    """Ответ модели не является корректным JSON со списком операций."""


def parse_operations(text: str) -> list[dict]:
    """JSON модели → список операций. Допускается обёртка ```json … ```."""
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MemoryFormatError(f"ответ не JSON: {exc.msg}") from exc
    ops = data.get("operations") if isinstance(data, dict) else None
    if not isinstance(ops, list) or not all(isinstance(o, dict) for o in ops):
        raise MemoryFormatError("в ответе нет списка 'operations'")
    return ops


def normalize_key(key: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]", "_", (key or "").strip().lower())).strip("_")


def normalize_field(key: str) -> str:
    k = normalize_key(key)
    return FIELD_ALIASES.get(k, k)


# ---------------------------------------------------------------------
# Проверка источника (общая для working и long_term)
# ---------------------------------------------------------------------

def check_evidence(op: dict, user_text: str, assistant_text: str | None) -> str | None:
    """Цитаты и числа в нужных сообщениях по kind. → причина отклонения или None."""
    kind = op.get("kind") or "user"
    evidence = op.get("evidence") or ""
    value = op.get("value") or ""
    if kind not in KINDS:
        return f"неизвестный kind {kind!r}"
    if kind == "user":
        if not quote_in(evidence, user_text):
            return "цитаты нет в сообщении пользователя"
        if quote_is_question(evidence, user_text):
            return "вопрос пользователя — не факт"
        if not numbers_supported(value, user_text):
            return "число в значении не встречается в сообщении пользователя"
    elif kind == "accepted_proposal":
        if not quote_in(evidence, user_text):
            return "нет цитаты согласия пользователя"
        if not quote_in(op.get("proposal_evidence") or "", assistant_text):
            return "нет цитаты предложения ассистента"
        if not numbers_supported(value, user_text, assistant_text):
            return "число в значении не встречается в сообщениях"
    else:
        if not quote_in(evidence, assistant_text):
            return "цитаты нет в сообщении ассистента"
        if not numbers_supported(value, assistant_text):
            return "число в значении не встречается в сообщении ассистента"
    return None


# ---------------------------------------------------------------------
# Правила записи: рабочая память
# ---------------------------------------------------------------------

def check_working_op(op: dict, task, user_text: str, assistant_text: str | None) -> tuple[str, str, str | None]:
    """→ (поле, нормализованное значение, причина отклонения или None).

    task — текущая TaskState или None (утиная типизация: .values, .items, .active).
    """
    fld = normalize_field(op.get("key", ""))
    value = (op.get("value") or "").strip()
    if fld not in WORKING_FIELDS:
        return fld, value, f"поле {fld!r} не входит в рабочую память (допустимо: {', '.join(WORKING_FIELDS)})"
    action = op.get("op")
    if action == "delete":
        if task is None:
            return fld, value, "нет текущей задачи"
        if not (op.get("reason") or "").strip():
            return fld, value, "удаление без причины"
        if not quote_in(op.get("evidence") or "", user_text):
            return fld, value, "удаление без цитаты пользователя"
        if fld in SCALAR_FIELDS and fld not in task.values:
            return fld, value, "удаление пустого поля"
        if fld in LIST_FIELDS and not any(same_value(i["value"], value) for i in task.items.get(fld, [])):
            return fld, value, "удаление несуществующего элемента"
        return fld, value, None
    if action != "set":
        return fld, value, f"неизвестная операция {action!r}"
    if not value:
        return fld, value, "пустое значение"
    if (op.get("kind") or "user") == "assistant_proposal":
        return fld, value, "неподтверждённое предложение ассистента — не данные задачи"
    if fld == "status":
        value = STATUS_ALIASES.get(value.lower(), value.lower())
        if value not in TASK_STATUSES:
            return fld, value, f"недопустимый статус (допустимо: {', '.join(TASK_STATUSES)})"
    problem = check_evidence({**op, "value": value}, user_text, assistant_text)
    if problem:
        return fld, value, problem
    if task is not None and not task.active and fld != "goal":
        return fld, value, "текущая задача завершена: новая задача начинается с goal"
    return fld, value, None


# ---------------------------------------------------------------------
# Правила записи: долговременная память
# ---------------------------------------------------------------------

def normalize_proposal_evidence(op: dict, assistant_text: str | None) -> dict:
    """assistant_proposal: цитата ассистента может прийти в proposal_evidence вместо evidence.

    Так модель делает по аналогии с accepted_proposal (evidence — ответ пользователя, proposal_evidence —
    цитата ассистента). Реальный прогон v2: запись про launchd была отклонена именно из-за этого.
    Перекладываем цитату, только если её нет в evidence и она ДОСЛОВНО есть в сообщении ассистента —
    проверка источника остаётся прежней.
    """
    if ((op.get("kind") or "user") == "assistant_proposal"
            and not quote_in(op.get("evidence") or "", assistant_text)
            and quote_in(op.get("proposal_evidence") or "", assistant_text)):
        return {**op, "evidence": op["proposal_evidence"], "proposal_evidence": "",
                "normalized": "evidence ← proposal_evidence"}
    return op


def is_task_data(key: str, category: str) -> bool:
    return category in TASK_CATEGORIES or key in WORKING_FIELDS or key in FIELD_ALIASES


def check_long_term_op(op: dict, state, user_text: str, assistant_text: str | None) -> tuple[str, str | None]:
    """→ (нормализованный ключ, причина отклонения или None). state — FactsState."""
    key = normalize_key(op.get("key", ""))
    if not key:
        return key, "пустой ключ"
    action = op.get("op")
    if action == "delete":
        if key not in state.facts:
            return key, "удаление несуществующего факта"
        if not (op.get("reason") or "").strip():
            return key, "удаление без причины"
        if not quote_in(op.get("evidence") or "", user_text):
            return key, "удаление без цитаты пользователя"
        return key, None
    if action != "set":
        return key, f"неизвестная операция {action!r}"
    category = op.get("category") or "other"
    if is_task_data(key, category):
        return key, "данные текущей задачи — место в рабочей памяти, не в долговременной"
    if category not in LONG_TERM_CATEGORIES:
        return key, f"категория {category!r} не разрешена для долговременной памяти"
    if not (op.get("value") or "").strip():
        return key, "пустое значение"
    problem = check_evidence(op, user_text, assistant_text)
    if problem:
        return key, problem
    old = state.facts.get(key)
    if (op.get("kind") or "user") == "assistant_proposal" and old and old.kind in DECISION_KINDS:
        return key, "предложение ассистента не может заменить решение пользователя"
    return key, None
