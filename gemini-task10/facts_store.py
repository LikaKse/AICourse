"""Хранилище Sticky Facts: key-value память с историей изменений и проверкой операций.

Файл ``sticky_facts.json`` (отдельно от полной истории):

    {"format_version": 1, "version": N, "facts": {key: Fact}, "deleted": {key: Fact},
     "locked": {key: причина}, "journal": [...]}

Три вида записей (kind) — чтобы не путать предложения ассистента и решения пользователя:

* ``user``               — пользователь сам сообщил или решил;
* ``accepted_proposal``  — предложение ассистента, ЯВНО принятое пользователем;
* ``assistant_proposal`` — предложение ассистента без подтверждения (НЕ решение).

Модель только ПРЕДЛАГАЕТ операции; применяет их код, и только после проверки:

* цитата evidence действительно есть в нужном сообщении (user — у пользователя,
  assistant_proposal — у ассистента, accepted_proposal — согласие у пользователя
  и proposal_evidence у ассистента);
* все числа значения встречаются в исходном тексте (защита от выдуманных чисел);
* предложение ассистента не может перезаписать решение пользователя;
* delete — только с причиной и цитатой пользователя;
* при изменении значения прежнее уходит в history (ничего не теряется);
* locked-ключи (изменены в ветке) не перезаписываются унаследованными сообщениями — конфликт.

Отклонённые операции сохраняются в журнале вместе с причиной.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from history_store import FORMAT_VERSION, HistoryLoadError, _read_json
from utils import atomic_write_text, normalize_text, redact_secrets

KINDS = ("user", "accepted_proposal", "assistant_proposal")
DECISION_KINDS = ("user", "accepted_proposal")
CATEGORIES = ("name", "location", "goal", "budget", "deadline", "environment", "constraint", "preference",
              "decision", "plan", "schedule", "alternative", "reason", "other")

KIND_TITLES = {
    "user": "Данные и решения пользователя (сообщил или решил сам):",
    "accepted_proposal": "Предложения ассистента, ЯВНО принятые пользователем:",
    "assistant_proposal": "Предложения ассистента (НЕ решения пользователя, пользователь их не подтверждал):",
}

#: JSON Schema ответа модели (response_json_schema). Все поля обязательны — проще проверять.
OPERATIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["set", "delete"]},
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                    "kind": {"type": "string", "enum": list(KINDS)},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "evidence": {"type": "string"},
                    "proposal_evidence": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["op", "key", "value", "kind", "category", "evidence", "proposal_evidence", "reason"],
            },
        },
    },
    "required": ["operations"],
}


class FactsFormatError(Exception):
    """Ответ модели не является корректным JSON со списком операций."""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class Fact:
    key: str
    value: str
    kind: str
    category: str
    evidence: str                    # цитата, подтверждающая факт
    source_message: int              # номер сообщения (с 1) в полной истории
    proposal_evidence: str = ""      # для accepted_proposal / assistant_proposal — цитата ассистента
    history: list[dict] = field(default_factory=list)   # прежние значения

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class FactsState:
    facts: dict[str, Fact] = field(default_factory=dict)
    deleted: dict[str, Fact] = field(default_factory=dict)
    locked: dict[str, str] = field(default_factory=dict)
    version: int = 0
    journal: list[dict] = field(default_factory=list)
    updated_at: str | None = None

    def to_dict(self) -> dict:
        return {"format_version": FORMAT_VERSION, "version": self.version, "updated_at": self.updated_at,
                "facts": {k: f.to_dict() for k, f in self.facts.items()},
                "deleted": {k: f.to_dict() for k, f in self.deleted.items()},
                "locked": dict(self.locked), "journal": self.journal}

    @classmethod
    def from_dict(cls, payload: object) -> "FactsState":
        if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
            raise HistoryLoadError("sticky_facts: неверный формат или версия")
        try:
            return cls(facts={k: Fact.from_dict(v) for k, v in (payload.get("facts") or {}).items()},
                       deleted={k: Fact.from_dict(v) for k, v in (payload.get("deleted") or {}).items()},
                       locked=dict(payload.get("locked") or {}), version=int(payload.get("version", 0)),
                       journal=list(payload.get("journal") or []), updated_at=payload.get("updated_at"))
        except (TypeError, ValueError) as exc:
            raise HistoryLoadError(f"sticky_facts: повреждённая запись: {exc}") from exc

    def public_view(self) -> dict:
        """То, что видит модель-извлекатель: текущие факты без журнала."""
        return {k: {"value": f.value, "kind": f.kind, "category": f.category} for k, f in self.facts.items()}


class FactsStore:
    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)

    def load(self) -> FactsState:
        if not self.path.is_file():
            return FactsState()
        return FactsState.from_dict(_read_json(self.path))

    def save(self, state: FactsState) -> None:
        state.updated_at = _now()
        atomic_write_text(self.path, redact_secrets(json.dumps(state.to_dict(), ensure_ascii=False, indent=2)))


# ---------------------------------------------------------------------
# Разбор ответа модели
# ---------------------------------------------------------------------

def parse_operations(text: str) -> list[dict]:
    """JSON модели → список операций. Допускается обёртка ```json … ```."""
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FactsFormatError(f"ответ не JSON: {exc.msg}") from exc
    ops = data.get("operations") if isinstance(data, dict) else None
    if not isinstance(ops, list) or not all(isinstance(o, dict) for o in ops):
        raise FactsFormatError("в ответе нет списка 'operations'")
    return ops


# ---------------------------------------------------------------------
# Проверка и применение операций
# ---------------------------------------------------------------------

def normalize_key(key: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]", "_", (key or "").strip().lower())).strip("_")


def _plain(text: str) -> str:
    """Для сравнения цитат: нормализация + без пунктуации и кавычек."""
    text = re.sub(r"[^\w\s:/+.-]", " ", normalize_text(text))
    text = re.sub(r"(?<!\w)[:/+.-]+|[:/+.-]+(?!\w)", " ", text)     # знаки только внутри слов
    return re.sub(r"\s+", " ", text).strip()


def quote_in(quote: str, text: str | None) -> bool:
    q = _plain(quote)
    return bool(text) and len(q) >= 3 and q in _plain(text)


def numbers_supported(value: str, *sources: str | None) -> bool:
    """Каждое число значения встречается в источнике (после нормализации «18 500 000» → «18500000»)."""
    source = " ".join(normalize_text(s) for s in sources if s)
    return all(n in source for n in re.findall(r"\d+", normalize_text(value)))


def check_operation(op: dict, state: FactsState, user_text: str, assistant_text: str | None) -> tuple[str, str | None]:
    """→ (нормализованный ключ, причина отклонения или None)."""
    key = normalize_key(op.get("key", ""))
    kind = op.get("kind", "user")
    evidence = op.get("evidence") or ""
    value = (op.get("value") or "").strip()
    if not key:
        return key, "пустой ключ"
    if op.get("op") == "delete":
        if key not in state.facts:
            return key, "удаление несуществующего факта"
        if not (op.get("reason") or "").strip():
            return key, "удаление без причины"
        if not quote_in(evidence, user_text):
            return key, "удаление без цитаты пользователя"
        return key, None
    if op.get("op") != "set":
        return key, f"неизвестная операция {op.get('op')!r}"
    if kind not in KINDS:
        return key, f"неизвестный kind {kind!r}"
    if not value:
        return key, "пустое значение"
    if kind == "user":
        if not quote_in(evidence, user_text):
            return key, "цитаты нет в сообщении пользователя"
        if not numbers_supported(value, user_text):
            return key, "число в значении не встречается в сообщении пользователя"
    elif kind == "accepted_proposal":
        if not quote_in(evidence, user_text):
            return key, "нет цитаты согласия пользователя"
        if not quote_in(op.get("proposal_evidence") or "", assistant_text):
            return key, "нет цитаты предложения ассистента"
        if not numbers_supported(value, user_text, assistant_text):
            return key, "число в значении не встречается в сообщениях"
    else:  # assistant_proposal
        if not quote_in(evidence, assistant_text):
            return key, "цитаты нет в сообщении ассистента"
        if not numbers_supported(value, assistant_text):
            return key, "число в значении не встречается в сообщении ассистента"
        old = state.facts.get(key)
        if old and old.kind in DECISION_KINDS:
            return key, "предложение ассистента не может заменить решение пользователя"
    return key, None


def apply_operations(
    state: FactsState,
    ops: list[dict],
    *,
    user_text: str,
    assistant_text: str | None,
    message_no: int,
    inherited: bool = False,
) -> dict:
    """Проверить и применить операции. message_no — номер сообщения пользователя (с 1).

    inherited=True — сообщение унаследовано из исходного диалога после checkpoint (ветки):
    операции над locked-ключами не применяются и записываются как конфликт.
    → {"applied": [...], "rejected": [...], "conflicts": [...], "unchanged": [...]}
    """
    out = {"applied": [], "rejected": [], "conflicts": [], "unchanged": []}
    for op in ops:
        key, problem = check_operation(op, state, user_text, assistant_text)
        rec = {**op, "key": key}
        if problem:
            out["rejected"].append({**rec, "rejection": problem})
            continue
        if inherited and key in state.locked:
            out["conflicts"].append({**rec, "locked_reason": state.locked[key],
                                     "kept_value": state.facts[key].value if key in state.facts else None,
                                     "resolution": "оставлено значение ветки"})
            continue
        old = state.facts.get(key)
        if op["op"] == "delete":
            gone = state.facts.pop(key)
            gone.history.append({"value": gone.value, "kind": gone.kind, "source_message": gone.source_message,
                                 "deleted_at_message": message_no, "reason": op.get("reason")})
            state.deleted[key] = gone
            out["applied"].append(rec)
            continue
        value = op["value"].strip()
        kind = op.get("kind", "user")
        source = message_no - 1 if kind == "assistant_proposal" else message_no
        if old and old.value == value and old.kind == kind:
            out["unchanged"].append(rec)
            continue
        fact = Fact(key=key, value=value, kind=kind, category=op.get("category") or "other",
                    evidence=op.get("evidence", "").strip(), source_message=source,
                    proposal_evidence=(op.get("proposal_evidence") or "").strip(),
                    history=list(old.history) if old else [])
        if old:
            fact.history.append({"value": old.value, "kind": old.kind, "source_message": old.source_message,
                                 "replaced_at_message": message_no})
        state.facts[key] = fact
        state.deleted.pop(key, None)
        out["applied"].append(rec)
    if out["applied"]:
        state.version += 1
    return out


def lock(state: FactsState, keys: list[str], reason: str) -> None:
    for k in keys:
        state.locked[normalize_key(k)] = reason


# ---------------------------------------------------------------------
# Представление для модели
# ---------------------------------------------------------------------

def render_facts(state: FactsState, *, with_evidence: bool = True) -> str:
    """Блок памяти для запроса. Одна строка — один факт (так работает построчный аудит)."""
    if not state.facts:
        return ""
    lines = []
    for kind in KINDS:
        items = [f for f in state.facts.values() if f.kind == kind]
        if not items:
            continue
        lines.append(KIND_TITLES[kind])
        for f in items:
            line = f"- {f.key}: {f.value}"
            if f.history:
                line += " (ранее: " + "; ".join(h["value"] for h in f.history) + ")"
            if kind == "assistant_proposal":
                line += " — предложение ассистента, не подтверждено"
            if with_evidence:
                src = "ассистент" if kind == "assistant_proposal" else "пользователь"
                line += f" [{src}, сообщение {f.source_message}: «{f.evidence}»]"
                if kind == "accepted_proposal" and f.proposal_evidence:
                    line += f" [предложил ассистент: «{f.proposal_evidence}»]"
            lines.append(line)
    return "\n".join(lines)
