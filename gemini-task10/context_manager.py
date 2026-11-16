"""Управление контекстом: активное окно, summary и сжатие старых сообщений.

Схема хранения:

    full_history:  [m0 m1 ... m9 | m10 ... m19 | m20 ... m25]   ← ничего не удаляется
                    └── summary v1 ┘└─ summary v2 ┘ └ активные ┘
    summary.summarized_upto = 20  →  active = full_history[20:]

Правило сжатия (SUMMARY_EVERY=10, KEEP_LAST=6):
    если len(active) >= KEEP_LAST + SUMMARY_EVERY  (16):
        chunk = active[:-KEEP_LAST]                      (обычно ровно 10 сообщений)
        summary = summarizer(предыдущее summary, chunk)  (один вызов Gemini)
        summarized_upto += len(chunk)                    (указатель только растёт)
    → после сжатия в active ровно KEEP_LAST сообщений, и следующий вызов maybe_compress()
      ничего не делает, пока не накопится ещё SUMMARY_EVERY сообщений.

Запрос к модели: [summary (user) + подтверждение (model)] + active + новый вопрос.
Полная история в обычный запрос после сжатия не передаётся.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from google.genai import types

import config
from history_store import HistoryStore, SummaryState, SummaryStore

logger = logging.getLogger("context")


class SummarizationError(Exception):
    """Не удалось получить summary от модели (ошибка API, пустой ответ)."""


class StateError(Exception):
    """Файлы истории и summary не согласованы между собой."""


@dataclass
class SummaryOutput:
    """Что возвращает summarizer: новый текст summary и сведения о вызове (токены и т. п.)."""
    text: str
    meta: dict = field(default_factory=dict)


Summarizer = Callable[[str, list[types.Content]], SummaryOutput]


@dataclass
class CompressionEvent:
    """Одна попытка сжатия: успешная или нет."""
    ok: bool
    start: int                      # [start, end) — индексы сжимаемых сообщений полной истории
    end: int
    messages: int
    version: int                    # версия summary после попытки (при ошибке — прежняя)
    error: str | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def text_content(role: str, text: str) -> types.Content:
    return types.Content(role=role, parts=[types.Part.from_text(text=text)])


def content_text(content: types.Content) -> str:
    """Текст сообщения без «мыслей» модели."""
    return "\n".join(p.text for p in content.parts or [] if p.text and not p.thought).strip()


class ContextManager:
    """Полная история + summary + активное окно последних сообщений."""

    def __init__(
        self,
        *,
        history_store: HistoryStore | None = None,
        summary_store: SummaryStore | None = None,
        summarizer: Summarizer | None = None,
        summary_every: int = config.SUMMARY_EVERY,
        keep_last: int = config.KEEP_LAST,
        compression_enabled: bool = True,
    ) -> None:
        if summary_every < 1:
            raise ValueError("summary_every должен быть ≥ 1")
        if keep_last < 0:
            raise ValueError("keep_last должен быть ≥ 0")
        self.history_store = history_store
        self.summary_store = summary_store
        self.summarizer = summarizer
        self.summary_every = summary_every
        self.keep_last = keep_last
        self.compression_enabled = compression_enabled

        self.full_history: list[types.Content] = history_store.load() if history_store else []
        self.summary: SummaryState = summary_store.load() if summary_store else SummaryState()
        if self.summary.summarized_upto > len(self.full_history):
            raise StateError(
                f"summary покрывает {self.summary.summarized_upto} сообщений, а в полной истории их "
                f"только {len(self.full_history)}. Файлы истории и summary не согласованы — "
                "удалите папку состояния или восстановите full_history.json."
            )

    @classmethod
    def from_dir(cls, state_dir: str | os.PathLike, **kwargs) -> "ContextManager":
        state_dir = Path(state_dir)
        return cls(
            history_store=HistoryStore(state_dir / config.FULL_HISTORY_FILE),
            summary_store=SummaryStore(state_dir / config.SUMMARY_FILE),
            **kwargs,
        )

    # ----- состояние -----

    @property
    def active(self) -> list[types.Content]:
        """Несжатые сообщения — то, что передаётся модели как есть."""
        return self.full_history[self.summary.summarized_upto:]

    def _save_history(self) -> None:
        if self.history_store:
            self.history_store.save(self.full_history)

    def _save_summary(self) -> None:
        if self.summary_store:
            self.summary_store.save(self.summary)

    def reset(self) -> None:
        self.full_history = []
        self.summary = SummaryState()
        self._save_history()
        self._save_summary()

    # ----- добавление сообщений -----

    def add_exchange(self, user: types.Content, model: types.Content) -> list[CompressionEvent]:
        """Добавить пару «вопрос — ответ» в полную историю, сохранить и при необходимости сжать."""
        self.full_history.extend([user, model])
        self._save_history()
        event = self.maybe_compress()
        return [event] if event else []

    # ----- сжатие -----

    def needs_compression(self) -> bool:
        return self.compression_enabled and len(self.active) >= self.keep_last + self.summary_every

    def maybe_compress(self) -> CompressionEvent | None:
        """Не больше ОДНОЙ попытки сжатия за вызов; без порога — ничего не делает."""
        if not self.needs_compression():
            return None
        return self.compress()

    def compress(self) -> CompressionEvent | None:
        """Сжать всё активное, кроме последних keep_last сообщений.

        При ошибке summarizer состояние не меняется: сообщения остаются активными
        и будут сжаты при следующей попытке.
        """
        if self.summarizer is None:
            raise RuntimeError("Для сжатия нужен summarizer (функция создания summary)")
        start = self.summary.summarized_upto
        end = len(self.full_history) - self.keep_last
        if end <= start:
            return None
        chunk = self.full_history[start:end]
        logger.info("Сжатие сообщений [%d, %d) → summary v%d", start, end, self.summary.version + 1)
        try:
            output = self.summarizer(self.summary.text, chunk)
        except SummarizationError as exc:
            logger.warning("Сжатие не удалось, история не изменена: %s", exc)
            return CompressionEvent(False, start, end, len(chunk), self.summary.version, error=str(exc),
                                    meta=getattr(exc, "meta", {}) or {})
        text = (output.text or "").strip()
        if not text:
            return CompressionEvent(False, start, end, len(chunk), self.summary.version,
                                    error="summarizer вернул пустой текст", meta=output.meta)

        new_version = self.summary.version + 1
        self.summary = SummaryState(
            text=text,
            summarized_upto=end,
            version=new_version,
            updates=self.summary.updates + [{
                "version": new_version, "start": start, "end": end, "messages": len(chunk),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                # Полный текст версии — чтобы офлайн-аудит видел, на каком шаге теряются детали.
                "text": text,
            }],
        )
        self._save_summary()
        return CompressionEvent(True, start, end, len(chunk), new_version, meta=output.meta)

    # ----- сборка запроса -----

    def summary_contents(self) -> list[types.Content]:
        """Summary в виде пары сообщений (пусто, если summary ещё нет)."""
        if not self.summary.text:
            return []
        return [
            text_content("user", f"{config.SUMMARY_HEADER}\n{self.summary.text}"),
            text_content("model", config.SUMMARY_ACK),
        ]

    def build_contents(self, new_message: types.Content | None = None) -> list[types.Content]:
        """Что уходит в generate_content: summary + активные сообщения (+ новый вопрос)."""
        contents = self.summary_contents() + list(self.active)
        if new_message is not None:
            contents.append(new_message)
        return contents
