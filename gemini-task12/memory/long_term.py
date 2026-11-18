"""Долговременная память: факты, решения и знания → state/long_term.json.

Основа — Sticky Facts Дня 10 (memory/long_term_store.py). Запись — операции layer=long_term,
прошедшие rules.check_long_term_op. Настройки ответа (профиль) живут отдельно; старые
preference/профильные constraint в промпт не выводятся, но остаются в файле.
"""

from __future__ import annotations

import os
from collections import Counter

from google.genai import types

import config
from memory.base import MemoryLayer
from memory.long_term_store import FactsState, apply_long_term_operations
from memory.rules import KINDS, is_profile_style_fact
from memory.storage import text_content

KIND_TITLES = {
    "user": "Данные и решения пользователя (сообщил или решил сам):",
    "accepted_proposal": "Предложения ассистента, ЯВНО принятые пользователем:",
    "assistant_proposal": "Предложения ассистента (НЕ решения пользователя, пользователь их не подтверждал):",
}


def render_facts(state: FactsState, *, with_evidence: bool = True, include_unconfirmed: bool = True) -> str:
    """Блок памяти для запроса (как в Дне 10). Одна строка — один факт."""
    lines = []
    for kind in KINDS:
        if kind == "assistant_proposal" and not include_unconfirmed:
            continue
        items = [f for f in state.facts.values()
                 if f.kind == kind and not is_profile_style_fact(f.category, f.key, f.value, f.evidence)]
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


class LongTermMemory(MemoryLayer):
    name = "long_term"

    def __init__(self, path: str | os.PathLike | None = None, *, render_evidence: bool = config.RENDER_EVIDENCE,
                 include_unconfirmed: bool = config.INCLUDE_UNCONFIRMED_PROPOSALS) -> None:
        super().__init__(path)
        self.render_evidence = render_evidence
        self.include_unconfirmed = include_unconfirmed
        self.state = FactsState()
        self.load()

    # ----- хранилище -----

    def load(self) -> None:
        payload = self._read()
        if payload is not None:
            self.state = FactsState.from_dict(payload)

    def reset_state(self) -> None:
        self.state = FactsState()

    def to_dict(self) -> dict:
        return self.state.to_dict()

    # ----- запись -----

    def apply_operations(self, ops: list[dict], *, user_text: str, assistant_text: str | None,
                         message_no: int) -> dict:
        out = apply_long_term_operations(self.state, ops, user_text=user_text, assistant_text=assistant_text,
                                         message_no=message_no)
        if ops:
            self.state.journal.append({"message": message_no, **out})
            self.save()
        return out

    # ----- чтение -----

    def public_view(self) -> dict:
        """То, что видит менеджер памяти: без legacy preference/профильных constraint и ключей."""
        return {
            key: {"value": fact.value, "kind": fact.kind, "category": fact.category}
            for key, fact in self.state.facts.items()
            if not is_profile_style_fact(fact.category, fact.key, fact.value, fact.evidence)
        }

    def storage_view(self) -> dict:
        """Полное содержимое файла для dump/диагностики, включая скрытые legacy-факты."""
        return self.state.public_view()

    def render(self) -> str:
        return render_facts(self.state, with_evidence=self.render_evidence,
                            include_unconfirmed=self.include_unconfirmed)

    def prompt_contents(self) -> list[types.Content]:
        text = self.render()
        if not text:
            return []
        return [text_content("user", f"{config.LONG_TERM_HEADER}\n{text}"),
                text_content("model", config.LONG_TERM_ACK)]

    def is_empty(self) -> bool:
        return not self.state.facts and not self.state.deleted

    def stats(self) -> dict:
        facts = self.state.facts.values()
        return {"facts": len(self.state.facts),
                "by_kind": dict(Counter(f.kind for f in facts)),
                "by_category": dict(Counter(f.category for f in facts)),
                "deleted": len(self.state.deleted), "version": self.state.version,
                "in_prompt": bool(self.prompt_contents())}

    def dump(self) -> dict:
        # dump не фильтруется: это диагностическое представление полного хранилища.
        return {"facts": self.storage_view()}
