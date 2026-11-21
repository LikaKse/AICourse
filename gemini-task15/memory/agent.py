"""Минимальная интеграция памяти с агентом.

Агент получает только MemoryReader и ModelClient. Write-API памяти ему
недоступен. История изменений в prompt не добавляется.
"""
from __future__ import annotations

from typing import Protocol

from .reader import MemoryReader

SYSTEM_PROMPT = (
    "Ты — ассистент. Используй только актуальные факты из памяти. "
    "Не добавляй в ответ неактуальные или удалённые данные."
)


class ModelClient(Protocol):
    def generate(self, prompt: str) -> str: ...


class MemoryAwareAgent:
    """Собирает prompt из активной памяти и вопроса пользователя."""

    __slots__ = ("_reader", "_client", "_system_prompt")

    def __init__(
        self,
        reader: MemoryReader,
        client: ModelClient,
        *,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self._reader = reader
        self._client = client
        self._system_prompt = system_prompt

    def ask(
        self,
        user_text: str,
        *,
        task_id: str | None = None,
    ) -> str:
        if not isinstance(user_text, str):
            raise ValueError("user_text должен быть строкой")
        context = self._reader.prompt_context(task_id)
        prompt = (
            f"{self._system_prompt}\n\n"
            f"{context}\n\n"
            f"[USER]\n{user_text}"
        )
        return self._client.generate(prompt)


__all__ = ["MemoryAwareAgent", "ModelClient", "SYSTEM_PROMPT"]
