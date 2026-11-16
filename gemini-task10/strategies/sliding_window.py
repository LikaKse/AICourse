"""Стратегия 1: Sliding Window — в запрос только последние N сообщений.

Полная история хранится целиком (full_history.json), но в модель уходит только
full_history[-N:]. Старые сообщения в запрос не попадают совсем; дополнительных
вызовов Gemini нет.
"""

from __future__ import annotations

import os

from google.genai import types

import config
from strategies.base import ContextStrategy


class SlidingWindowStrategy(ContextStrategy):
    name = "sliding"

    def __init__(self, state_dir: str | os.PathLike | None = None, *, window_size: int = config.WINDOW_SIZE) -> None:
        if window_size < 0:
            raise ValueError("window_size должен быть ≥ 0")
        super().__init__(state_dir)
        self.window_size = window_size

    def window(self) -> list[types.Content]:
        if self.window_size == 0:
            return []
        return list(self.full_history[-self.window_size:])

    def memory_contents(self) -> list[types.Content]:
        return self.window()

    def settings(self) -> dict:
        return {"strategy": self.name, "window_size": self.window_size}

    def describe(self) -> str:
        return (f"sliding: {len(self.full_history)} сообщ. в полной истории, "
                f"в запрос — последние {len(self.window())} (N={self.window_size})")
