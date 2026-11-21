"""Read-only доступ агента к актуальной памяти.

Reader использует штатный MemoryStore.load(): блокировка, чтение файла и
строгая валидация snapshot остаются на стороне store. Прямого чтения
memory.json здесь нет.
"""
from __future__ import annotations

from .context import ContextBuilder, MemoryContext
from .models import Fact
from .store import MemoryStore


class MemoryReader:
    """Только чтение активных фактов и prompt-контекста."""

    __slots__ = ("_store", "_builder")

    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        self._builder = ContextBuilder()

    def context(self, task_id: str | None = None) -> MemoryContext:
        snapshot = self._store.load()
        return self._builder.build(snapshot, task_id)

    def prompt_context(self, task_id: str | None = None) -> str:
        return self.context(task_id).render()

    def active_facts(self, task_id: str | None = None) -> tuple[Fact, ...]:
        return self._store.load().active_facts(task_id)


__all__ = ["MemoryReader"]
