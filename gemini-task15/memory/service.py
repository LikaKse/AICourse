"""Сервис памяти Дня 15: транзакции, конфликты, tombstones, завершение задач."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Protocol

from .errors import MemoryValidationError, MemoryWriteError
from .models import (
    EventResult,
    Fact,
    FactKind,
    MemoryAction,
    MemoryEvent,
    MemorySnapshot,
    TaskCompletion,
    Tombstone,
    _validate_datetime,
    _validate_key,
    _validate_reason,
    _validate_source,
    _validate_task_id,
)
from .store import MemoryStore


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


@dataclass(frozen=True)
class MemoryResult:
    status: str
    fact: Fact | None = None
    event: MemoryEvent | None = None
    events: tuple[MemoryEvent, ...] = ()
    completion: TaskCompletion | None = None
    cleared_count: int = 0

    @property
    def applied(self) -> bool:
        return self.status in {"applied", "confirmed"}


class MemoryService:
    """Единственный API изменения памяти. Все операции идут под одним lock."""

    def __init__(self, store: MemoryStore, *, clock: Clock | None = None) -> None:
        self._store = store
        self._clock = clock if clock is not None else SystemClock()

    # ----- helpers -----

    def _now(self) -> datetime:
        return _validate_datetime(self._clock.now(), "clock.now()")

    def _event(
        self,
        revision: int,
        *,
        action: MemoryAction,
        at: datetime,
        source: str,
        reason: str,
        kind: FactKind | None = None,
        key: str | None = None,
        task_id: str | None = None,
        entry_id: str | None = None,
        old_present: bool = False,
        old_value=None,
        old_recorded_at: datetime | None = None,
        new_present: bool = False,
        new_value=None,
        new_recorded_at: datetime | None = None,
        result: EventResult = EventResult.APPLIED,
    ) -> MemoryEvent:
        return MemoryEvent(
            event_id=f"evt-{revision:08d}",
            revision=revision,
            action=action,
            at=at,
            source=source,
            reason=reason,
            kind=kind,
            key=key,
            task_id=task_id,
            entry_id=entry_id,
            old_present=old_present,
            old_value=old_value if old_present else None,
            old_recorded_at=old_recorded_at if old_present else None,
            new_present=new_present,
            new_value=new_value if new_present else None,
            new_recorded_at=new_recorded_at,
            result=result,
        )

    def _commit(
        self,
        snapshot: MemorySnapshot,
        events: tuple[MemoryEvent, ...],
        **changes,
    ) -> MemorySnapshot:
        if not events:
            raise MemoryWriteError("commit без событий")
        expected = snapshot.revision + 1
        for event in events:
            if event.revision != expected:
                raise MemoryWriteError(
                    "события должны иметь последовательные revision")
            expected += 1
        new_snapshot = replace(
            snapshot,
            revision=snapshot.revision + len(events),
            updated_at=self._now(),
            journal=snapshot.journal + events,
            **changes,
        )
        self._store._save_unlocked(new_snapshot)
        return new_snapshot

    @staticmethod
    def _fact_version(current: Fact | None) -> int:
        return 1 if current is None else current.version + 1

    # ----- long-term -----

    def set_long_term(
        self,
        key: str,
        value,
        *,
        recorded_at: datetime | None = None,
        source: str,
        reason: str = "",
    ) -> MemoryResult:
        key = _validate_key(key)
        new_fact = Fact(
            kind=FactKind.MUTABLE,
            key=key,
            value=value,
            recorded_at=recorded_at if recorded_at is not None else self._now(),
            source=source,
            reason=reason,
        )
        event_at = self._now()

        with self._store.locked():
            snapshot = self._store._load_unlocked()
            current = snapshot.mutable.get(key)
            tombstone = snapshot.mutable_tombstones.get(key)

            if current is None:
                if tombstone is not None and new_fact.recorded_at <= tombstone.deleted_at:
                    result_kind = (
                        EventResult.REJECTED_EQUAL
                        if new_fact.recorded_at == tombstone.deleted_at
                        else EventResult.REJECTED_STALE
                    )
                    event = self._event(
                        snapshot.revision + 1,
                        action=MemoryAction.SET,
                        at=event_at,
                        source=new_fact.source,
                        reason=new_fact.reason,
                        kind=FactKind.MUTABLE,
                        key=key,
                        new_present=True,
                        new_value=new_fact.value,
                        new_recorded_at=new_fact.recorded_at,
                        result=result_kind,
                    )
                    self._commit(snapshot, (event,))
                    return MemoryResult(
                        result_kind.value, fact=None, event=event,
                        events=(event,))

                event = self._event(
                    snapshot.revision + 1,
                    action=MemoryAction.SET,
                    at=event_at,
                    source=new_fact.source,
                    reason=new_fact.reason,
                    kind=FactKind.MUTABLE,
                    key=key,
                    new_present=True,
                    new_value=new_fact.value,
                    new_recorded_at=new_fact.recorded_at,
                    result=EventResult.APPLIED,
                )
                mutable = dict(snapshot.mutable)
                mutable[key] = new_fact
                self._commit(snapshot, (event,), mutable=mutable)
                return MemoryResult(
                    "applied", fact=new_fact, event=event, events=(event,))

            if new_fact.recorded_at < current.recorded_at:
                event = self._event(
                    snapshot.revision + 1,
                    action=MemoryAction.SET,
                    at=event_at,
                    source=new_fact.source,
                    reason=new_fact.reason,
                    kind=FactKind.MUTABLE,
                    key=key,
                    old_present=True,
                    old_value=current.value,
                    old_recorded_at=current.recorded_at,
                    new_present=True,
                    new_value=new_fact.value,
                    new_recorded_at=new_fact.recorded_at,
                    result=EventResult.REJECTED_STALE,
                )
                self._commit(snapshot, (event,))
                return MemoryResult(
                    "rejected_stale", fact=current, event=event,
                    events=(event,))

            if new_fact.recorded_at == current.recorded_at:
                if new_fact.value == current.value:
                    return MemoryResult("unchanged", fact=current)
                event = self._event(
                    snapshot.revision + 1,
                    action=MemoryAction.SET,
                    at=event_at,
                    source=new_fact.source,
                    reason=new_fact.reason,
                    kind=FactKind.MUTABLE,
                    key=key,
                    old_present=True,
                    old_value=current.value,
                    old_recorded_at=current.recorded_at,
                    new_present=True,
                    new_value=new_fact.value,
                    new_recorded_at=new_fact.recorded_at,
                    result=EventResult.REJECTED_EQUAL,
                )
                self._commit(snapshot, (event,))
                return MemoryResult(
                    "rejected_equal", fact=current, event=event,
                    events=(event,))

            # later timestamp
            updated = replace(
                current,
                value=new_fact.value,
                recorded_at=new_fact.recorded_at,
                source=new_fact.source,
                reason=new_fact.reason,
                version=self._fact_version(current),
            )
            same_value = new_fact.value == current.value
            event = self._event(
                snapshot.revision + 1,
                action=MemoryAction.SET,
                at=event_at,
                source=new_fact.source,
                reason=new_fact.reason,
                kind=FactKind.MUTABLE,
                key=key,
                old_present=True,
                old_value=current.value,
                old_recorded_at=current.recorded_at,
                new_present=True,
                new_value=new_fact.value,
                new_recorded_at=new_fact.recorded_at,
                result=(
                    EventResult.UNCHANGED if same_value
                    else EventResult.APPLIED
                ),
            )
            mutable = dict(snapshot.mutable)
            mutable[key] = updated
            self._commit(snapshot, (event,), mutable=mutable)
            return MemoryResult(
                "confirmed" if same_value else "applied",
                fact=updated,
                event=event,
                events=(event,),
            )

    def append_decision(
        self,
        value,
        *,
        recorded_at: datetime | None = None,
        source: str,
        reason: str = "",
    ) -> MemoryResult:
        source = _validate_source(source)
        reason = _validate_reason(reason)
        with self._store.locked():
            snapshot = self._store._load_unlocked()
            entry_id = f"dec-{snapshot.revision + 1:08d}"
            fact = Fact(
                kind=FactKind.CUMULATIVE,
                key=None,
                value=value,
                recorded_at=recorded_at if recorded_at is not None else self._now(),
                source=source,
                reason=reason,
                entry_id=entry_id,
            )
            event = self._event(
                snapshot.revision + 1,
                action=MemoryAction.APPEND,
                at=self._now(),
                source=source,
                reason=reason,
                kind=FactKind.CUMULATIVE,
                entry_id=entry_id,
                new_present=True,
                new_value=fact.value,
                new_recorded_at=fact.recorded_at,
                result=EventResult.APPLIED,
            )
            cumulative = snapshot.cumulative + (fact,)
            self._commit(snapshot, (event,), cumulative=cumulative)
            return MemoryResult(
                "applied", fact=fact, event=event, events=(event,))

    # ----- temporary -----

    def set_temporary(
        self,
        task_id: str,
        key: str,
        value,
        *,
        recorded_at: datetime | None = None,
        source: str,
        reason: str = "",
    ) -> MemoryResult:
        task_id = _validate_task_id(task_id)
        key = _validate_key(key)
        new_fact = Fact(
            kind=FactKind.TEMPORARY,
            key=key,
            task_id=task_id,
            value=value,
            recorded_at=recorded_at if recorded_at is not None else self._now(),
            source=source,
            reason=reason,
        )
        event_at = self._now()

        with self._store.locked():
            snapshot = self._store._load_unlocked()
            if task_id in snapshot.completed_tasks:
                event = self._event(
                    snapshot.revision + 1,
                    action=MemoryAction.SET,
                    at=event_at,
                    source=new_fact.source,
                    reason=new_fact.reason,
                    kind=FactKind.TEMPORARY,
                    key=key,
                    task_id=task_id,
                    new_present=True,
                    new_value=new_fact.value,
                    new_recorded_at=new_fact.recorded_at,
                    result=EventResult.REJECTED_TASK_COMPLETED,
                )
                self._commit(snapshot, (event,))
                return MemoryResult(
                    "rejected_task_completed", fact=None, event=event,
                    events=(event,))

            task_facts = dict(snapshot.temporary.get(task_id, {}))
            current = task_facts.get(key)
            tombstone = snapshot.temporary_tombstones.get(task_id, {}).get(key)

            if current is None:
                if tombstone is not None and new_fact.recorded_at <= tombstone.deleted_at:
                    result_kind = (
                        EventResult.REJECTED_EQUAL
                        if new_fact.recorded_at == tombstone.deleted_at
                        else EventResult.REJECTED_STALE
                    )
                    event = self._event(
                        snapshot.revision + 1,
                        action=MemoryAction.SET,
                        at=event_at,
                        source=new_fact.source,
                        reason=new_fact.reason,
                        kind=FactKind.TEMPORARY,
                        key=key,
                        task_id=task_id,
                        new_present=True,
                        new_value=new_fact.value,
                        new_recorded_at=new_fact.recorded_at,
                        result=result_kind,
                    )
                    self._commit(snapshot, (event,))
                    return MemoryResult(
                        result_kind.value, fact=None, event=event,
                        events=(event,))

                event = self._event(
                    snapshot.revision + 1,
                    action=MemoryAction.SET,
                    at=event_at,
                    source=new_fact.source,
                    reason=new_fact.reason,
                    kind=FactKind.TEMPORARY,
                    key=key,
                    task_id=task_id,
                    new_present=True,
                    new_value=new_fact.value,
                    new_recorded_at=new_fact.recorded_at,
                    result=EventResult.APPLIED,
                )
                task_facts[key] = new_fact
                temporary = dict(snapshot.temporary)
                temporary[task_id] = task_facts
                self._commit(snapshot, (event,), temporary=temporary)
                return MemoryResult(
                    "applied", fact=new_fact, event=event, events=(event,))

            if new_fact.recorded_at < current.recorded_at:
                event = self._event(
                    snapshot.revision + 1,
                    action=MemoryAction.SET,
                    at=event_at,
                    source=new_fact.source,
                    reason=new_fact.reason,
                    kind=FactKind.TEMPORARY,
                    key=key,
                    task_id=task_id,
                    old_present=True,
                    old_value=current.value,
                    old_recorded_at=current.recorded_at,
                    new_present=True,
                    new_value=new_fact.value,
                    new_recorded_at=new_fact.recorded_at,
                    result=EventResult.REJECTED_STALE,
                )
                self._commit(snapshot, (event,))
                return MemoryResult(
                    "rejected_stale", fact=current, event=event,
                    events=(event,))

            if new_fact.recorded_at == current.recorded_at:
                if new_fact.value == current.value:
                    return MemoryResult("unchanged", fact=current)
                event = self._event(
                    snapshot.revision + 1,
                    action=MemoryAction.SET,
                    at=event_at,
                    source=new_fact.source,
                    reason=new_fact.reason,
                    kind=FactKind.TEMPORARY,
                    key=key,
                    task_id=task_id,
                    old_present=True,
                    old_value=current.value,
                    old_recorded_at=current.recorded_at,
                    new_present=True,
                    new_value=new_fact.value,
                    new_recorded_at=new_fact.recorded_at,
                    result=EventResult.REJECTED_EQUAL,
                )
                self._commit(snapshot, (event,))
                return MemoryResult(
                    "rejected_equal", fact=current, event=event,
                    events=(event,))

            updated = replace(
                current,
                value=new_fact.value,
                recorded_at=new_fact.recorded_at,
                source=new_fact.source,
                reason=new_fact.reason,
                version=self._fact_version(current),
            )
            same_value = new_fact.value == current.value
            event = self._event(
                snapshot.revision + 1,
                action=MemoryAction.SET,
                at=event_at,
                source=new_fact.source,
                reason=new_fact.reason,
                kind=FactKind.TEMPORARY,
                key=key,
                task_id=task_id,
                old_present=True,
                old_value=current.value,
                old_recorded_at=current.recorded_at,
                new_present=True,
                new_value=new_fact.value,
                new_recorded_at=new_fact.recorded_at,
                result=(
                    EventResult.UNCHANGED if same_value
                    else EventResult.APPLIED
                ),
            )
            task_facts[key] = updated
            temporary = dict(snapshot.temporary)
            temporary[task_id] = task_facts
            self._commit(snapshot, (event,), temporary=temporary)
            return MemoryResult(
                "confirmed" if same_value else "applied",
                fact=updated, event=event, events=(event,),
            )

    # ----- delete -----

    def delete_long_term(
        self,
        key: str,
        *,
        recorded_at: datetime | None = None,
        source: str,
        reason: str = "",
    ) -> MemoryResult:
        key = _validate_key(key)
        source = _validate_source(source)
        reason = _validate_reason(reason)
        deleted_at = (
            recorded_at if recorded_at is not None else self._now()
        )
        deleted_at = _validate_datetime(deleted_at, "deleted_at")
        event_at = self._now()

        with self._store.locked():
            snapshot = self._store._load_unlocked()
            current = snapshot.mutable.get(key)
            if current is None:
                return MemoryResult("not_found", fact=None)

            if deleted_at < current.recorded_at:
                event = self._event(
                    snapshot.revision + 1,
                    action=MemoryAction.DELETE,
                    at=event_at,
                    source=source,
                    reason=reason,
                    kind=FactKind.MUTABLE,
                    key=key,
                    old_present=True,
                    old_value=current.value,
                    old_recorded_at=current.recorded_at,
                    new_present=False,
                    new_recorded_at=deleted_at,
                    result=EventResult.REJECTED_STALE,
                )
                self._commit(snapshot, (event,))
                return MemoryResult(
                    "rejected_stale", fact=current, event=event,
                    events=(event,))

            if deleted_at == current.recorded_at:
                event = self._event(
                    snapshot.revision + 1,
                    action=MemoryAction.DELETE,
                    at=event_at,
                    source=source,
                    reason=reason,
                    kind=FactKind.MUTABLE,
                    key=key,
                    old_present=True,
                    old_value=current.value,
                    old_recorded_at=current.recorded_at,
                    new_present=False,
                    new_recorded_at=deleted_at,
                    result=EventResult.REJECTED_EQUAL,
                )
                self._commit(snapshot, (event,))
                return MemoryResult(
                    "rejected_equal", fact=current, event=event,
                    events=(event,))

            mutable = dict(snapshot.mutable)
            del mutable[key]
            tombstones = dict(snapshot.mutable_tombstones)
            tombstones[key] = Tombstone(
                kind=FactKind.MUTABLE,
                key=key,
                deleted_at=deleted_at,
                source=source,
                reason=reason,
            )
            event = self._event(
                snapshot.revision + 1,
                action=MemoryAction.DELETE,
                at=event_at,
                source=source,
                reason=reason,
                kind=FactKind.MUTABLE,
                key=key,
                old_present=True,
                old_value=current.value,
                old_recorded_at=current.recorded_at,
                new_present=False,
                new_recorded_at=deleted_at,
                result=EventResult.APPLIED,
            )
            self._commit(
                snapshot, (event,),
                mutable=mutable, mutable_tombstones=tombstones)
            return MemoryResult(
                "applied", fact=None, event=event, events=(event,))

    def delete_temporary(
        self,
        task_id: str,
        key: str,
        *,
        recorded_at: datetime | None = None,
        source: str,
        reason: str = "",
    ) -> MemoryResult:
        task_id = _validate_task_id(task_id)
        key = _validate_key(key)
        source = _validate_source(source)
        reason = _validate_reason(reason)
        deleted_at = _validate_datetime(
            recorded_at if recorded_at is not None else self._now(),
            "deleted_at",
        )
        event_at = self._now()

        with self._store.locked():
            snapshot = self._store._load_unlocked()
            if task_id in snapshot.completed_tasks:
                return MemoryResult("not_found", fact=None)
            task_facts = dict(snapshot.temporary.get(task_id, {}))
            current = task_facts.get(key)
            if current is None:
                return MemoryResult("not_found", fact=None)

            if deleted_at < current.recorded_at:
                event = self._event(
                    snapshot.revision + 1,
                    action=MemoryAction.DELETE,
                    at=event_at,
                    source=source,
                    reason=reason,
                    kind=FactKind.TEMPORARY,
                    key=key,
                    task_id=task_id,
                    old_present=True,
                    old_value=current.value,
                    old_recorded_at=current.recorded_at,
                    new_present=False,
                    new_recorded_at=deleted_at,
                    result=EventResult.REJECTED_STALE,
                )
                self._commit(snapshot, (event,))
                return MemoryResult(
                    "rejected_stale", fact=current, event=event,
                    events=(event,))

            if deleted_at == current.recorded_at:
                event = self._event(
                    snapshot.revision + 1,
                    action=MemoryAction.DELETE,
                    at=event_at,
                    source=source,
                    reason=reason,
                    kind=FactKind.TEMPORARY,
                    key=key,
                    task_id=task_id,
                    old_present=True,
                    old_value=current.value,
                    old_recorded_at=current.recorded_at,
                    new_present=False,
                    new_recorded_at=deleted_at,
                    result=EventResult.REJECTED_EQUAL,
                )
                self._commit(snapshot, (event,))
                return MemoryResult(
                    "rejected_equal", fact=current, event=event,
                    events=(event,))

            del task_facts[key]
            temporary = dict(snapshot.temporary)
            if task_facts:
                temporary[task_id] = task_facts
            else:
                temporary.pop(task_id, None)
            tombstone_map = dict(snapshot.temporary_tombstones)
            task_tombstones = dict(tombstone_map.get(task_id, {}))
            task_tombstones[key] = Tombstone(
                kind=FactKind.TEMPORARY,
                key=key,
                task_id=task_id,
                deleted_at=deleted_at,
                source=source,
                reason=reason,
            )
            tombstone_map[task_id] = task_tombstones
            event = self._event(
                snapshot.revision + 1,
                action=MemoryAction.DELETE,
                at=event_at,
                source=source,
                reason=reason,
                kind=FactKind.TEMPORARY,
                key=key,
                task_id=task_id,
                old_present=True,
                old_value=current.value,
                old_recorded_at=current.recorded_at,
                new_present=False,
                new_recorded_at=deleted_at,
                result=EventResult.APPLIED,
            )
            self._commit(
                snapshot, (event,),
                temporary=temporary,
                temporary_tombstones=tombstone_map,
            )
            return MemoryResult(
                "applied", fact=None, event=event, events=(event,))

    # ----- task lifecycle -----

    def complete_task(
        self,
        task_id: str,
        *,
        completed_at: datetime | None = None,
        source: str = "system",
        reason: str = "task completed",
    ) -> MemoryResult:
        task_id = _validate_task_id(task_id)
        source = _validate_source(source)
        reason = _validate_reason(reason)
        completed_at = _validate_datetime(
            completed_at if completed_at is not None else self._now(),
            "completed_at",
        )

        with self._store.locked():
            snapshot = self._store._load_unlocked()
            existing = snapshot.completed_tasks.get(task_id)
            if existing is not None:
                return MemoryResult(
                    "already_completed", completion=existing,
                    cleared_count=0)

            temp_facts = dict(snapshot.temporary.get(task_id, {}))
            existing_tombstones = snapshot.temporary_tombstones.get(task_id, {})
            relevant = [fact.recorded_at for fact in temp_facts.values()]
            relevant.extend(tombstone.deleted_at
                            for tombstone in existing_tombstones.values())
            if any(moment >= completed_at for moment in relevant):
                raise MemoryValidationError(
                    "complete_task: completed_at должен быть позже всех "
                    "temporary-фактов и tombstones задачи")

            temporary = dict(snapshot.temporary)
            temporary.pop(task_id, None)
            tombstone_map = dict(snapshot.temporary_tombstones)
            task_tombstones = dict(existing_tombstones)
            events: list[MemoryEvent] = []
            revision = snapshot.revision

            for key in sorted(temp_facts):
                fact = temp_facts[key]
                revision += 1
                events.append(self._event(
                    revision,
                    action=MemoryAction.DELETE,
                    at=completed_at,
                    source=source,
                    reason=reason,
                    kind=FactKind.TEMPORARY,
                    key=key,
                    task_id=task_id,
                    old_present=True,
                    old_value=fact.value,
                    old_recorded_at=fact.recorded_at,
                    new_present=False,
                    new_recorded_at=completed_at,
                    result=EventResult.APPLIED,
                ))
                task_tombstones[key] = Tombstone(
                    kind=FactKind.TEMPORARY,
                    key=key,
                    task_id=task_id,
                    deleted_at=completed_at,
                    source=source,
                    reason=reason,
                )

            tombstone_map[task_id] = task_tombstones
            completion = TaskCompletion(
                task_id=task_id,
                completed_at=completed_at,
                source=source,
                reason=reason,
            )
            completed_tasks = dict(snapshot.completed_tasks)
            completed_tasks[task_id] = completion

            revision += 1
            complete_event = self._event(
                revision,
                action=MemoryAction.COMPLETE_TASK,
                at=completed_at,
                source=source,
                reason=reason,
                task_id=task_id,
                result=EventResult.APPLIED,
            )
            events.append(complete_event)

            self._commit(
                snapshot,
                tuple(events),
                temporary=temporary,
                temporary_tombstones=tombstone_map,
                completed_tasks=completed_tasks,
            )
            return MemoryResult(
                "applied",
                event=complete_event,
                events=tuple(events),
                completion=completion,
                cleared_count=len(temp_facts),
            )

    def reopen_task(
        self,
        task_id: str,
        *,
        reopened_at: datetime | None = None,
        source: str = "system",
        reason: str = "task reopened",
    ) -> MemoryResult:
        task_id = _validate_task_id(task_id)
        source = _validate_source(source)
        reason = _validate_reason(reason)
        reopened_at = _validate_datetime(
            reopened_at if reopened_at is not None else self._now(),
            "reopened_at",
        )

        with self._store.locked():
            snapshot = self._store._load_unlocked()
            completion = snapshot.completed_tasks.get(task_id)
            if completion is None:
                return MemoryResult("not_found", fact=None)

            completed_tasks = dict(snapshot.completed_tasks)
            del completed_tasks[task_id]
            event = self._event(
                snapshot.revision + 1,
                action=MemoryAction.REOPEN_TASK,
                at=reopened_at,
                source=source,
                reason=reason,
                task_id=task_id,
                result=EventResult.APPLIED,
            )
            self._commit(
                snapshot, (event,), completed_tasks=completed_tasks)
            return MemoryResult(
                "applied", event=event, events=(event,))

    # ----- reads -----

    def get_long_term(self, key: str) -> Fact | None:
        key = _validate_key(key)
        with self._store.locked():
            return self._store._load_unlocked().mutable.get(key)

    def get_temporary(self, task_id: str, key: str) -> Fact | None:
        task_id = _validate_task_id(task_id)
        key = _validate_key(key)
        with self._store.locked():
            snapshot = self._store._load_unlocked()
            return snapshot.temporary.get(task_id, {}).get(key)

    def active_facts(self, task_id: str | None = None) -> tuple[Fact, ...]:
        with self._store.locked():
            return self._store._load_unlocked().active_facts(task_id)

    def history(self) -> tuple[MemoryEvent, ...]:
        with self._store.locked():
            return self._store._load_unlocked().journal


__all__ = ["Clock", "MemoryResult", "MemoryService", "SystemClock"]
