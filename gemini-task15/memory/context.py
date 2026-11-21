"""Детерминированный контекст агента из актуальной памяти.

ContextBuilder читает только active-секции MemorySnapshot. Journal,
tombstones, old/new значения и отклонённые попытки сюда не попадают.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .errors import MemoryValidationError
from .models import Fact, FactKind, MemorySnapshot, _validate_task_id


def _format_value(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _format_time(value) -> str:
    return value.isoformat(timespec="microseconds")


@dataclass(frozen=True)
class MemoryContext:
    """Только активные факты, сгруппированные по секциям prompt."""

    task_id: str | None = None
    long_term: tuple[Fact, ...] = ()
    decisions: tuple[Fact, ...] = ()
    temporary: tuple[Fact, ...] = ()

    def __post_init__(self) -> None:
        if self.task_id is not None:
            object.__setattr__(
                self, "task_id",
                _validate_task_id(self.task_id, "MemoryContext.task_id"))

        long_term = tuple(self.long_term)
        decisions = tuple(self.decisions)
        temporary = tuple(self.temporary)
        for fact in long_term:
            if not isinstance(fact, Fact) or fact.kind is not FactKind.MUTABLE:
                raise MemoryValidationError(
                    "MemoryContext.long_term: ожидаются mutable Fact")
        for fact in decisions:
            if not isinstance(fact, Fact) or fact.kind is not FactKind.CUMULATIVE:
                raise MemoryValidationError(
                    "MemoryContext.decisions: ожидаются cumulative Fact")
        for fact in temporary:
            if not isinstance(fact, Fact) or fact.kind is not FactKind.TEMPORARY:
                raise MemoryValidationError(
                    "MemoryContext.temporary: ожидаются temporary Fact")
            if self.task_id is None or fact.task_id != self.task_id:
                raise MemoryValidationError(
                    "MemoryContext.temporary: несовпадение task_id")

        object.__setattr__(
            self, "long_term",
            tuple(sorted(long_term, key=lambda item: item.key or "")))
        object.__setattr__(
            self, "decisions",
            tuple(sorted(
                decisions,
                key=lambda item: (item.recorded_at, item.entry_id or ""))))
        object.__setattr__(
            self, "temporary",
            tuple(sorted(temporary, key=lambda item: item.key or "")))

    @property
    def facts(self) -> tuple[Fact, ...]:
        return self.long_term + self.decisions + self.temporary

    def render(self) -> str:
        lines = ["[ACTIVE MEMORY]"]
        if not self.facts:
            lines.append("(empty)")
            return "\n".join(lines)

        if self.long_term:
            lines.append("long_term:")
            for fact in self.long_term:
                lines.append(
                    f"  {fact.key} = {_format_value(fact.value)} | "
                    f"recorded_at={_format_time(fact.recorded_at)} | "
                    f"source={fact.source}")

        if self.decisions:
            lines.append("decisions:")
            for fact in self.decisions:
                lines.append(
                    f"  - {_format_value(fact.value)} | "
                    f"recorded_at={_format_time(fact.recorded_at)} | "
                    f"source={fact.source}")

        if self.temporary:
            lines.append(f"task {self.task_id}:")
            for fact in self.temporary:
                lines.append(
                    f"  {fact.key} = {_format_value(fact.value)} | "
                    f"recorded_at={_format_time(fact.recorded_at)} | "
                    f"source={fact.source}")

        return "\n".join(lines)

    def to_prompt(self) -> str:
        return self.render()


class ContextBuilder:
    """Строит MemoryContext из снимка активной памяти."""

    def build(
        self,
        snapshot: MemorySnapshot,
        task_id: str | None = None,
    ) -> MemoryContext:
        if not isinstance(snapshot, MemorySnapshot):
            raise MemoryValidationError(
                "ContextBuilder.build ожидает MemorySnapshot")
        if task_id is not None:
            _validate_task_id(task_id, "ContextBuilder.task_id")

        long_term = tuple(sorted(
            snapshot.mutable.values(), key=lambda item: item.key or ""))
        decisions = tuple(sorted(
            snapshot.cumulative,
            key=lambda item: (item.recorded_at, item.entry_id or "")))

        temporary: tuple[Fact, ...] = ()
        if task_id is not None and task_id not in snapshot.completed_tasks:
            task_facts = snapshot.temporary.get(task_id, {})
            temporary = tuple(sorted(
                task_facts.values(), key=lambda item: item.key or ""))

        return MemoryContext(
            task_id=task_id,
            long_term=long_term,
            decisions=decisions,
            temporary=temporary,
        )

    def render(
        self,
        snapshot: MemorySnapshot,
        task_id: str | None = None,
    ) -> str:
        return self.build(snapshot, task_id).render()


__all__ = ["ContextBuilder", "MemoryContext"]
