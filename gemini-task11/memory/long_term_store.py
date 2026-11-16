"""Данные долговременной памяти — Sticky Facts Дня 10 (facts_store.py), перенесённые без ветвления.

Что сохранено из Дня 10 без изменений: Fact (key, value, kind, category, evidence, source_message,
proposal_evidence, history), три вида записей, проверка цитат и чисел, запрет предложению
ассистента перезаписывать решение пользователя, история прежних значений, журнал отклонений.

Что изменено: убраны locked / inherited (нужны были только для веток Дня 10); правила проверки
вынесены в memory/rules.py, где к ним добавлены правила слоя — «данные задачи → working»
и «вопрос пользователя — не факт».
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from memory.rules import check_long_term_op, normalize_proposal_evidence
from memory.storage import MemoryLoadError


@dataclass
class Fact:
    key: str
    value: str
    kind: str
    category: str
    evidence: str                    # цитата, подтверждающая факт
    source_message: int              # номер сообщения (с 1) в диалоге
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
    version: int = 0
    journal: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"version": self.version,
                "facts": {k: f.to_dict() for k, f in self.facts.items()},
                "deleted": {k: f.to_dict() for k, f in self.deleted.items()},
                "journal": self.journal}

    @classmethod
    def from_dict(cls, payload: dict) -> "FactsState":
        try:
            return cls(facts={k: Fact.from_dict(v) for k, v in (payload.get("facts") or {}).items()},
                       deleted={k: Fact.from_dict(v) for k, v in (payload.get("deleted") or {}).items()},
                       version=int(payload.get("version", 0)), journal=list(payload.get("journal") or []))
        except (TypeError, ValueError) as exc:
            raise MemoryLoadError(f"long_term: повреждённая запись: {exc}") from exc

    def public_view(self) -> dict:
        """То, что видит менеджер памяти: текущие факты без журнала."""
        return {k: {"value": f.value, "kind": f.kind, "category": f.category} for k, f in self.facts.items()}


def apply_long_term_operations(state: FactsState, ops: list[dict], *, user_text: str,
                               assistant_text: str | None, message_no: int) -> dict:
    """Проверить и применить операции layer=long_term. message_no — номер сообщения пользователя.

    → {"applied": [...], "rejected": [...], "unchanged": [...]}
    """
    out: dict[str, list] = {"applied": [], "rejected": [], "unchanged": []}
    for op in ops:
        op = normalize_proposal_evidence(op, assistant_text)
        key, problem = check_long_term_op(op, state, user_text, assistant_text)
        rec = {**op, "key": key}
        if problem:
            out["rejected"].append({**rec, "rejection": problem})
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
        kind = op.get("kind") or "user"
        source = message_no - 1 if kind == "assistant_proposal" else message_no
        if old and old.value == value and old.kind == kind:
            out["unchanged"].append(rec)
            continue
        fact = Fact(key=key, value=value, kind=kind, category=op.get("category") or "other",
                    evidence=(op.get("evidence") or "").strip(), source_message=source,
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
