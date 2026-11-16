"""Стратегия 2: Sticky Facts — key-value память + последние N сообщений.

После КАЖДОГО сообщения пользователя (до ответа ассистента) — один вызов Gemini:
текущие facts + предыдущее сообщение ассистента + новое сообщение пользователя →
JSON-операции set/delete. Код проверяет операции (facts_store.check_operation) и применяет их.

Запрос к модели: [блок facts (user) + подтверждение (model)] + последние N сообщений + вопрос.

Стоимость: +1 вызов generate_content (и +1 бесплатный count_tokens) на сообщение пользователя.
Эти вызовы возвращаются как CallRecord(kind="facts_update") и учитываются отдельно.
Контрольные вопросы (remember=False) память не обновляют.
"""

from __future__ import annotations

import logging
import os

from google.genai import types

import config
from context_manager import content_text, text_content
from facts_store import FactsState, FactsStore, apply_operations, render_facts
from strategies.base import CallRecord, ContextStrategy
from token_stats import Usage

logger = logging.getLogger("facts")


class FactsExtractionError(Exception):
    """Не удалось получить операции от модели (ошибка API, пустой ответ, не JSON)."""

    def __init__(self, message: str, meta: dict | None = None) -> None:
        super().__init__(message)
        self.meta = meta or {}


class StickyFactsStrategy(ContextStrategy):
    name = "facts"
    state_files = (config.FULL_HISTORY_FILE, config.STICKY_FACTS_FILE)

    def __init__(
        self,
        state_dir: str | os.PathLike | None = None,
        *,
        window_size: int = config.WINDOW_SIZE,
        render_evidence: bool = config.FACTS_RENDER_EVIDENCE,
    ) -> None:
        if window_size < 0:
            raise ValueError("window_size должен быть ≥ 0")
        super().__init__(state_dir)
        self.window_size = window_size
        self.render_evidence = render_evidence
        self.facts_store = FactsStore(self.state_dir / config.STICKY_FACTS_FILE) if self.state_dir else None
        self.facts: FactsState = self.facts_store.load() if self.facts_store else FactsState()
        #: True — сообщения унаследованы из исходного диалога после checkpoint (locked-ключи не трогаются)
        self.inherited_mode = False

    @property
    def system_instruction(self) -> str:
        return config.FACTS_ANSWER_SYSTEM_INSTRUCTION

    # ----- обновление памяти -----

    def before_user_message(self, user: types.Content) -> list[CallRecord]:
        if self.agent is None:
            raise RuntimeError("Стратегия facts не подключена к агенту (нужен вызов Gemini)")
        message_no = len(self.full_history) + 1
        prev = self.full_history[-1] if self.full_history and self.full_history[-1].role == "model" else None
        assistant_text = content_text(prev) if prev else None
        user_text = content_text(user)
        entry = {"message": message_no, "inherited": self.inherited_mode}
        try:
            ops, meta = self.agent.extract_facts(self.facts.public_view(), assistant_text, user_text, message_no)
        except FactsExtractionError as exc:
            logger.warning("Facts не обновлены (сообщение %d): %s", message_no, exc)
            entry.update(ok=False, error=str(exc), call=exc.meta)
            self.facts.journal.append(entry)
            self._save_facts()
            return [self._record(False, exc.meta, str(exc), message_no)]
        result = apply_operations(self.facts, ops, user_text=user_text, assistant_text=assistant_text,
                                  message_no=message_no, inherited=self.inherited_mode)
        for r in result["rejected"]:
            logger.info("Операция отклонена (%s): %s", r["rejection"], r.get("key"))
        for c in result["conflicts"]:
            logger.warning("Конфликт с веткой: %s — %s", c["key"], c["resolution"])
        entry.update(ok=True, proposed=len(ops), call=meta, **result)
        self.facts.journal.append(entry)
        self._save_facts()
        return [self._record(True, meta, None, message_no, result)]

    @staticmethod
    def _record(ok: bool, meta: dict, error: str | None, message_no: int, result: dict | None = None) -> CallRecord:
        u = {k: v for k, v in (meta.get("usage") or {}).items() if k in Usage.__dataclass_fields__}
        counts = {k: len(v) for k, v in (result or {}).items()}
        return CallRecord(kind="facts_update", ok=ok, usage=Usage(**u), request_tokens=meta.get("request_tokens"),
                          error=error, meta={"message": message_no, **counts,
                                             **{k: v for k, v in meta.items() if k != "usage"}})

    # ----- запрос -----

    def facts_contents(self) -> list[types.Content]:
        text = render_facts(self.facts, with_evidence=self.render_evidence)
        if not text:
            return []
        return [text_content("user", f"{config.FACTS_HEADER}\n{text}"), text_content("model", config.FACTS_ACK)]

    def window(self) -> list[types.Content]:
        return list(self.full_history[-self.window_size:]) if self.window_size else []

    def memory_contents(self) -> list[types.Content]:
        return self.facts_contents() + self.window()

    # ----- состояние -----

    def _save_facts(self) -> None:
        if self.facts_store:
            self.facts_store.save(self.facts)

    def save(self) -> None:
        self._save_history()
        self._save_facts()

    def reset(self) -> None:
        super().reset()
        self.facts = FactsState()
        self._save_facts()

    def settings(self) -> dict:
        return {"strategy": self.name, "window_size": self.window_size, "render_evidence": self.render_evidence}

    def describe(self) -> str:
        return (f"facts: {len(self.facts.facts)} фактов (v{self.facts.version}), "
                f"{len(self.full_history)} сообщ. в полной истории, в запрос — последние {len(self.window())}")
