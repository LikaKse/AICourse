"""Общий интерфейс стратегий управления контекстом.

Каждая стратегия хранит ПОЛНУЮ историю диалога (full_history.json) и решает только одно:
что из неё (и из своей дополнительной памяти) попадёт в запрос модели.

Жизненный цикл одного сообщения пользователя (Agent.ask или проигрывание сценария):

    strategy.prepare()                  ← summary: отложенное сжатие (как в Дне 9)
    strategy.before_user_message(user)  ← facts: вызов Gemini для обновления памяти
    strategy.build_contents(user)       ← запрос к модели
    strategy.add_exchange(user, model)  ← пара в полную историю; summary: сжатие

Служебные вызовы Gemini (summary, извлечение facts) возвращаются как CallRecord,
чтобы их токены учитывались отдельно от ответов и не терялись в сравнении.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from google.genai import types

import config
from history_store import HistoryStore
from token_stats import Usage


@dataclass
class CallRecord:
    """Один служебный вызов generate_content (не ответ пользователю)."""
    kind: str                           # "summary" | "facts_update"
    ok: bool
    usage: Usage = field(default_factory=Usage)
    request_tokens: int | None = None   # count_tokens запроса до отправки
    error: str | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["usage"] = self.usage.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CallRecord":
        u = {k: v for k, v in (d.get("usage") or {}).items() if k in Usage.__dataclass_fields__}
        return cls(kind=d["kind"], ok=d["ok"], usage=Usage(**u), request_tokens=d.get("request_tokens"),
                   error=d.get("error"), meta=d.get("meta") or {})


class ContextStrategy(ABC):
    """Базовая стратегия: полная история в отдельном файле + правило сборки запроса."""

    name: str = "base"
    #: файлы состояния в state_dir (копируются целиком при checkpoint / ветвлении)
    state_files: tuple[str, ...] = (config.FULL_HISTORY_FILE,)

    def __init__(self, state_dir: str | os.PathLike | None = None) -> None:
        self.state_dir = Path(state_dir) if state_dir is not None else None
        self.history_store = HistoryStore(self.state_dir / config.FULL_HISTORY_FILE) if self.state_dir else None
        self.full_history: list[types.Content] = self.history_store.load() if self.history_store else []
        self.agent: Any = None

    # ----- связь с агентом (служебные вызовы Gemini делает агент) -----

    def bind(self, agent: Any) -> None:
        self.agent = agent

    @property
    def system_instruction(self) -> str:
        return config.SYSTEM_INSTRUCTION

    # ----- жизненный цикл сообщения -----

    def prepare(self) -> list[CallRecord]:
        return []

    def before_user_message(self, user: types.Content) -> list[CallRecord]:
        return []

    def add_exchange(self, user: types.Content, model: types.Content) -> list[CallRecord]:
        self.full_history.extend([user, model])
        self._save_history()
        return []

    @abstractmethod
    def memory_contents(self) -> list[types.Content]:
        """Всё, что уходит в модель, КРОМЕ нового сообщения."""

    def build_contents(self, new_message: types.Content | None = None) -> list[types.Content]:
        contents = list(self.memory_contents())
        if new_message is not None:
            contents.append(new_message)
        return contents

    def memory_text(self) -> str:
        """Текст контекста (без вопроса) — по нему офлайн-аудит ищет 36 деталей.

        Каждое сообщение — отдельная строка «Роль: текст», чтобы построчные правила аудита работали.
        """
        from agent import ROLE_NAMES
        from context_manager import content_text
        return "\n".join(f"{ROLE_NAMES.get(c.role, c.role)}: {content_text(c)}" for c in self.memory_contents())

    # ----- состояние -----

    def _save_history(self) -> None:
        if self.history_store:
            self.history_store.save(self.full_history)

    def save(self) -> None:
        """Записать все файлы состояния (используется перед checkpoint/переключением ветки)."""
        self._save_history()

    def reset(self) -> None:
        self.full_history = []
        self._save_history()

    def settings(self) -> dict:
        return {"strategy": self.name}

    def describe(self) -> str:
        return f"{self.name}: {len(self.full_history)} сообщ. в полной истории"
