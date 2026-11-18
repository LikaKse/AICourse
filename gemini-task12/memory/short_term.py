"""Краткосрочная память: история текущего диалога → state/short_term.json.

Запись (только код, без LLM): успешная пара «вопрос — ответ» целиком.
Чтение: последние ``window`` сообщений (хранится вся история).

``message_counter`` — сквозной номер сообщений; не сбрасывается при очистке, чтобы номера
источников в долговременной памяти (source_message) не совпали с номерами новых сообщений.
"""

from __future__ import annotations

import os

from google.genai import types

import config
from memory.base import MemoryLayer
from memory.storage import MemoryLoadError, content_text, dump_contents, load_contents


class ShortTermMemory(MemoryLayer):
    name = "short_term"

    def __init__(self, path: str | os.PathLike | None = None, *, window: int = config.SHORT_TERM_WINDOW) -> None:
        if window < 0:
            raise ValueError("window должен быть ≥ 0")
        super().__init__(path)
        self.window = window
        self.message_counter = 0
        self.messages: list[types.Content] = []
        self.load()

    # ----- хранилище -----

    def load(self) -> None:
        payload = self._read()
        if payload is None:
            return
        self.messages = load_contents(payload.get("messages"), self.name)
        counter = payload.get("message_counter", len(self.messages))
        if not isinstance(counter, int) or isinstance(counter, bool) or counter < len(self.messages):
            raise MemoryLoadError(f"{self.name}: некорректный message_counter {counter!r}")
        self.message_counter = counter

    def reset_state(self) -> None:
        self.messages = []

    def to_dict(self) -> dict:
        return {"message_counter": self.message_counter, "messages": dump_contents(self.messages)}

    # ----- запись -----

    def add_exchange(self, user: types.Content, model: types.Content) -> None:
        if user.role != "user" or model.role != "model":
            raise ValueError("ожидается пара (user, model)")
        self.messages.extend([user, model])
        self.message_counter += 2
        self.save()

    def next_message_no(self) -> int:
        return self.message_counter + 1

    def last_assistant_text(self) -> str | None:
        if self.messages and self.messages[-1].role == "model":
            return content_text(self.messages[-1])
        return None

    # ----- чтение -----

    def recent(self) -> list[types.Content]:
        if not self.window:
            return []
        window = list(self.messages[-self.window:])
        if window and window[0].role == "model":     # роли должны чередоваться после блоков памяти
            window = window[1:]
        return window

    def prompt_contents(self) -> list[types.Content]:
        return self.recent()

    def is_empty(self) -> bool:
        return not self.messages

    def numbered(self) -> list[tuple[int, str, str]]:
        first = self.message_counter - len(self.messages) + 1
        return [(first + i, c.role, content_text(c)) for i, c in enumerate(self.messages)]

    def stats(self) -> dict:
        return {"stored_messages": len(self.messages), "window": self.window,
                "in_prompt_messages": len(self.recent()), "message_counter": self.message_counter}

    def dump(self) -> dict:
        return {"messages": [{"no": n, "role": r, "text": t} for n, r, t in self.numbered()]}
