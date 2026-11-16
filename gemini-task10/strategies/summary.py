"""Стратегия 3: Summary — адаптер над ContextManager из Дня 9 (механизм не переписан).

Логика SUMMARY_EVERY / KEEP_LAST, хранение summary.json, отложенное сжатие при ошибке —
всё в context_manager.py без изменений. Адаптер только переводит CompressionEvent
в CallRecord (для общего учёта служебных вызовов) и подключает Agent.summarize.
"""

from __future__ import annotations

import os

from google.genai import types

import config
from context_manager import CompressionEvent, ContextManager
from history_store import HistoryStore, SummaryStore
from strategies.base import CallRecord, ContextStrategy
from token_stats import Usage


def event_to_call(event: CompressionEvent) -> CallRecord:
    meta = event.meta or {}
    u = {k: v for k, v in (meta.get("usage") or {}).items() if k in Usage.__dataclass_fields__}
    return CallRecord(kind="summary", ok=event.ok, usage=Usage(**u), request_tokens=meta.get("request_tokens"),
                      error=event.error, meta={"event": event.to_dict()})


class SummaryStrategy(ContextStrategy):
    name = "summary"
    state_files = (config.FULL_HISTORY_FILE, config.SUMMARY_FILE)

    def __init__(
        self,
        state_dir: str | os.PathLike | None = None,
        *,
        summary_every: int = config.SUMMARY_EVERY,
        keep_last: int = config.KEEP_LAST,
        compression_enabled: bool = True,
        context: ContextManager | None = None,
    ) -> None:
        super().__init__(state_dir)
        if context is None:
            context = ContextManager(
                history_store=HistoryStore(self.state_dir / config.FULL_HISTORY_FILE) if self.state_dir else None,
                summary_store=SummaryStore(self.state_dir / config.SUMMARY_FILE) if self.state_dir else None,
                summary_every=summary_every, keep_last=keep_last, compression_enabled=compression_enabled,
            )
        self.context = context
        self.history_store = context.history_store
        #: последние события сжатия (для AgentResult.compression — совместимость с Днём 9)
        self.last_events: list[CompressionEvent] = []

    # Полная история — та же, что у ContextManager (один список, без копий).
    @property
    def full_history(self) -> list[types.Content]:
        return self.context.full_history

    @full_history.setter
    def full_history(self, value: list[types.Content]) -> None:
        if hasattr(self, "context"):
            self.context.full_history = value

    def bind(self, agent) -> None:
        super().bind(agent)
        if self.context.summarizer is None:
            self.context.summarizer = agent.summarize

    def _calls(self, events: list[CompressionEvent]) -> list[CallRecord]:
        self.last_events.extend(events)
        return [event_to_call(e) for e in events]

    def prepare(self) -> list[CallRecord]:
        self.last_events = []
        event = self.context.maybe_compress()
        return self._calls([event] if event else [])

    def add_exchange(self, user: types.Content, model: types.Content) -> list[CallRecord]:
        return self._calls(self.context.add_exchange(user, model))

    def memory_contents(self) -> list[types.Content]:
        return self.context.build_contents()

    def save(self) -> None:
        self.context._save_history()
        self.context._save_summary()

    def reset(self) -> None:
        self.context.reset()

    def settings(self) -> dict:
        return {"strategy": self.name, "summary_every": self.context.summary_every,
                "keep_last": self.context.keep_last, "compression_enabled": self.context.compression_enabled}

    def describe(self) -> str:
        s = self.context.summary
        return (f"summary: {len(self.full_history)} сообщ. в полной истории, сжато {s.summarized_upto} "
                f"(summary v{s.version}), активных {len(self.context.active)}")
