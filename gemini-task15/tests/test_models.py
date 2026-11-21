"""Тесты моделей памяти Дня 15 (Этап 1)."""

from datetime import datetime, timedelta, timezone

import pytest

from memory.errors import MemoryError, MemoryValidationError
from memory.models import (
    EventResult,
    Fact,
    FactKind,
    MemoryAction,
    MemoryEvent,
    MemorySnapshot,
    TaskCompletion,
    Tombstone,
)

UTC = timezone.utc


def at(day=1, hour=0, minute=0, second=0):
    return datetime(2026, 12, day, hour, minute, second, tzinfo=UTC)


def make_fact(
    kind=FactKind.MUTABLE,
    *,
    key="project.deadline",
    value="2026-12-01",
    recorded_at=None,
    source="user",
    reason="",
    task_id=None,
    entry_id=None,
    version=1,
):
    return Fact(
        kind=kind,
        value=value,
        recorded_at=recorded_at or at(),
        source=source,
        reason=reason,
        key=key,
        task_id=task_id,
        entry_id=entry_id,
        version=version,
    )


def make_set_event(
    *,
    revision=1,
    kind=FactKind.MUTABLE,
    key="project.deadline",
    task_id=None,
    old_present=False,
    old_value=None,
    old_at=None,
    new_value="2026-12-01",
    new_at=None,
    result=EventResult.APPLIED,
    event_id=None,
):
    return MemoryEvent(
        event_id=event_id or f"evt-{revision:08d}",
        revision=revision,
        action=MemoryAction.SET,
        at=at(1, 1),
        source="user",
        kind=kind,
        key=key,
        task_id=task_id,
        old_value=old_value,
        new_value=new_value,
        old_present=old_present,
        new_present=True,
        old_recorded_at=old_at,
        new_recorded_at=new_at or at(1, 1),
        result=result,
    )


def make_delete_event(
    *,
    revision=2,
    kind=FactKind.MUTABLE,
    key="project.deadline",
    task_id=None,
    old_present=True,
    old_value="2026-12-01",
    old_at=None,
    result=EventResult.APPLIED,
):
    return MemoryEvent(
        event_id=f"evt-{revision:08d}",
        revision=revision,
        action=MemoryAction.DELETE,
        at=at(2, 1),
        source="user",
        kind=kind,
        key=key,
        task_id=task_id,
        old_value=old_value if old_present else None,
        old_present=old_present,
        old_recorded_at=(old_at or at(2)) if old_present else None,
        new_present=False,
        new_value=None,
        new_recorded_at=at(2, 1),
        result=result,
    )


def make_append_event(*, revision=1, entry_id="decision-001"):
    return MemoryEvent(
        event_id=f"evt-{revision:08d}",
        revision=revision,
        action=MemoryAction.APPEND,
        at=at(1, 1),
        source="assistant",
        kind=FactKind.CUMULATIVE,
        entry_id=entry_id,
        new_present=True,
        new_value="Use PostgreSQL",
        new_recorded_at=at(1, 1),
        result=EventResult.APPLIED,
    )


def make_reopen_event(*, revision=2, task_id="task-1"):
    return MemoryEvent(
        event_id=f"evt-{revision:08d}",
        revision=revision,
        action=MemoryAction.REOPEN_TASK,
        at=at(2, 1),
        source="user",
        task_id=task_id,
        result=EventResult.APPLIED,
    )


def test_validation_error_is_memory_and_value_error():
    with pytest.raises(MemoryValidationError) as exc:
        make_fact(key="BAD")
    assert isinstance(exc.value, MemoryError)
    assert isinstance(exc.value, ValueError)


def test_fact_normalizes_timestamp_to_utc():
    source_tz = timezone(timedelta(hours=3))
    fact = make_fact(recorded_at=datetime(2026, 12, 1, 12, tzinfo=source_tz))
    assert fact.recorded_at.utcoffset() == timedelta(0)
    assert fact.recorded_at.hour == 9


def test_fact_mutable_roundtrip():
    fact = make_fact(value="2026-12-01")
    restored = Fact.from_dict(fact.to_dict())
    assert restored == fact
    assert restored.to_dict() == fact.to_dict()


def test_fact_temporary_requires_task_id():
    with pytest.raises(MemoryValidationError):
        make_fact(kind=FactKind.TEMPORARY, task_id=None)


def test_fact_cumulative_rules():
    with pytest.raises(MemoryValidationError):
        make_fact(kind=FactKind.CUMULATIVE, key=None, entry_id=None)
    with pytest.raises(MemoryValidationError):
        make_fact(kind=FactKind.CUMULATIVE, key="decision", entry_id="decision-001")


def test_fact_rejects_naive_timestamp():
    with pytest.raises(MemoryValidationError):
        make_fact(recorded_at=datetime(2026, 12, 1, 12))


@pytest.mark.parametrize("key", [
    "",
    "Project.Deadline",
    "project deadline",
    "project/deadline",
    "a" * 129,
])
def test_fact_rejects_invalid_key(key):
    with pytest.raises(MemoryValidationError):
        make_fact(key=key)


@pytest.mark.parametrize("source", ["", " ", "user name", "a" * 65])
def test_fact_rejects_invalid_source(source):
    with pytest.raises(MemoryValidationError):
        make_fact(source=source)


@pytest.mark.parametrize("version", [0, -1, True, "1"])
def test_fact_rejects_invalid_version(version):
    with pytest.raises(MemoryValidationError):
        make_fact(version=version)


@pytest.mark.parametrize("value", [
    {1, 2},
    object(),
    float("nan"),
    float("inf"),
    b"bytes",
    {1: "not-string-key"},
])
def test_fact_rejects_non_json_value(value):
    with pytest.raises(MemoryValidationError):
        make_fact(value=value)


def test_fact_rejects_circular_value():
    value = []
    value.append(value)
    with pytest.raises(MemoryValidationError):
        make_fact(value=value)


def test_fact_copies_input_value():
    value = {"items": [1]}
    fact = make_fact(value=value)
    value["items"].append(2)
    assert fact.value == {"items": [1]}


def test_fact_from_dict_rejects_unknown_and_missing_fields():
    fact = make_fact()
    data = fact.to_dict()
    data["unknown"] = 1
    with pytest.raises(MemoryValidationError):
        Fact.from_dict(data)

    missing = fact.to_dict()
    del missing["source"]
    with pytest.raises(MemoryValidationError):
        Fact.from_dict(missing)


def test_tombstone_mutable_roundtrip():
    tombstone = Tombstone(
        kind=FactKind.MUTABLE,
        key="project.deadline",
        deleted_at=at(2),
        source="user",
        reason="deadline removed",
    )
    assert Tombstone.from_dict(tombstone.to_dict()) == tombstone


def test_tombstone_temporary_requires_task_id():
    with pytest.raises(MemoryValidationError):
        Tombstone(
            kind=FactKind.TEMPORARY,
            key="task.step",
            deleted_at=at(2),
            source="task_completion",
        )


def test_tombstone_rejects_cumulative_kind():
    with pytest.raises(MemoryValidationError):
        Tombstone(
            kind=FactKind.CUMULATIVE,
            key="decision",
            deleted_at=at(2),
            source="user",
        )


def test_tombstone_rejects_naive_datetime():
    with pytest.raises(MemoryValidationError):
        Tombstone(
            kind=FactKind.MUTABLE,
            key="project.deadline",
            deleted_at=datetime(2026, 12, 2),
            source="user",
        )


def test_task_completion_roundtrip():
    completion = TaskCompletion(
        task_id="task-1",
        completed_at=at(3),
        source="system",
        reason="task completed",
    )
    assert TaskCompletion.from_dict(completion.to_dict()) == completion


def test_task_completion_rejects_invalid_task_id():
    with pytest.raises(MemoryValidationError):
        TaskCompletion(
            task_id="Task 1",
            completed_at=at(3),
            source="system",
        )


def test_memory_event_set_roundtrip():
    event = make_set_event(
        old_present=True,
        old_value="2026-12-01",
        old_at=at(1),
        new_value="2026-12-15",
        new_at=at(2),
    )
    assert MemoryEvent.from_dict(event.to_dict()) == event


def test_memory_event_set_unchanged_requires_old_present():
    with pytest.raises(MemoryValidationError):
        make_set_event(
            old_present=False,
            result=EventResult.UNCHANGED,
        )


def test_memory_event_set_rejected_without_old_allowed():
    event = make_set_event(
        old_present=False,
        result=EventResult.REJECTED_STALE,
    )
    assert event.old_present is False
    assert event.result is EventResult.REJECTED_STALE


def test_memory_event_rejected_task_completed():
    event = make_set_event(
        kind=FactKind.TEMPORARY,
        task_id="task-1",
        old_present=False,
        result=EventResult.REJECTED_TASK_COMPLETED,
    )
    assert event.result is EventResult.REJECTED_TASK_COMPLETED
    assert event.task_id == "task-1"


def test_memory_event_complete_task_roundtrip():
    event = MemoryEvent(
        event_id="evt-00000003",
        revision=3,
        action=MemoryAction.COMPLETE_TASK,
        at=at(3),
        source="system",
        task_id="task-1",
        result=EventResult.APPLIED,
    )
    assert MemoryEvent.from_dict(event.to_dict()) == event


def test_memory_event_delete_roundtrip():
    event = make_delete_event()
    assert MemoryEvent.from_dict(event.to_dict()) == event


def test_memory_event_delete_not_found():
    event = make_delete_event(
        old_present=False,
        old_value=None,
        result=EventResult.NOT_FOUND,
    )
    assert MemoryEvent.from_dict(event.to_dict()) == event


def test_memory_event_append_roundtrip():
    event = make_append_event()
    assert MemoryEvent.from_dict(event.to_dict()) == event


def test_memory_event_reopen_roundtrip():
    event = make_reopen_event()
    assert MemoryEvent.from_dict(event.to_dict()) == event


@pytest.mark.parametrize("action,result", [
    ("unknown", EventResult.APPLIED),
    (MemoryAction.SET, "unknown"),
])
def test_memory_event_rejects_unknown_action_or_result(action, result):
    with pytest.raises(MemoryValidationError):
        MemoryEvent(
            event_id="evt-00000001",
            revision=1,
            action=action,
            at=at(1),
            source="user",
            result=result,
        )


def test_memory_event_scope_rules():
    with pytest.raises(MemoryValidationError):
        make_set_event(kind=FactKind.TEMPORARY, task_id=None)
    with pytest.raises(MemoryValidationError):
        make_set_event(kind=FactKind.MUTABLE, task_id="task-1")


def test_memory_event_from_dict_rejects_unknown_field():
    event = make_set_event().to_dict()
    event["unknown"] = 1
    with pytest.raises(MemoryValidationError):
        MemoryEvent.from_dict(event)


def test_snapshot_empty_roundtrip():
    snapshot = MemorySnapshot()
    restored = MemorySnapshot.from_dict(snapshot.to_dict())
    assert restored == snapshot
    assert restored.active_facts() == ()


def test_snapshot_active_facts_sorted_and_scoped():
    mutable_a = make_fact(key="project.status", value="in_progress", recorded_at=at(2))
    mutable_b = make_fact(key="project.deadline", value="2026-12-01", recorded_at=at(1))
    decision = make_fact(
        kind=FactKind.CUMULATIVE,
        key=None,
        entry_id="decision-001",
        value="Use PostgreSQL",
        recorded_at=at(1, 2),
    )
    temp_1 = make_fact(
        kind=FactKind.TEMPORARY,
        task_id="task-1",
        key="task.step",
        value="collect requirements",
        recorded_at=at(1, 3),
    )
    temp_2 = make_fact(
        kind=FactKind.TEMPORARY,
        task_id="task-2",
        key="task.step",
        value="another task",
        recorded_at=at(1, 3),
    )
    snapshot = MemorySnapshot(
        revision=0,
        mutable={
            mutable_a.key: mutable_a,
            mutable_b.key: mutable_b,
        },
        cumulative=(decision,),
        temporary={
            "task-1": {temp_1.key: temp_1},
            "task-2": {temp_2.key: temp_2},
        },
    )

    active = snapshot.active_facts("task-1")
    assert [item.key or item.entry_id for item in active] == [
        "project.deadline",
        "project.status",
        "decision-001",
        "task.step",
    ]
    assert all(item.value != "another task" for item in active)
    assert snapshot.active_facts() == (
        mutable_b, mutable_a, decision)


def test_snapshot_active_facts_excludes_journal_values():
    old_value = "OLD-DEADLINE"
    new_value = "NEW-DEADLINE"
    fact = make_fact(value=new_value)
    journal = make_set_event(
        old_present=True,
        old_value=old_value,
        old_at=at(1),
        new_value=new_value,
        new_at=at(2),
    )
    snapshot = MemorySnapshot(
        revision=1,
        mutable={fact.key: fact},
        journal=(journal,),
    )
    active_values = [item.value for item in snapshot.active_facts()]
    assert active_values == [new_value]
    assert old_value not in active_values


def test_snapshot_rejects_active_fact_not_newer_than_tombstone():
    fact = make_fact(recorded_at=at(1))
    tombstone = Tombstone(
        kind=FactKind.MUTABLE,
        key=fact.key,
        deleted_at=at(2),
        source="user",
    )
    with pytest.raises(MemoryValidationError):
        MemorySnapshot(
            mutable={fact.key: fact},
            mutable_tombstones={fact.key: tombstone},
        )


def test_snapshot_allows_restored_fact_newer_than_tombstone():
    fact = make_fact(recorded_at=at(3))
    tombstone = Tombstone(
        kind=FactKind.MUTABLE,
        key=fact.key,
        deleted_at=at(2),
        source="user",
    )
    snapshot = MemorySnapshot(
        mutable={fact.key: fact},
        mutable_tombstones={fact.key: tombstone},
    )
    assert snapshot.active_facts() == (fact,)


def test_snapshot_rejects_completed_task_with_temporary_facts():
    temp = make_fact(
        kind=FactKind.TEMPORARY,
        task_id="task-1",
        key="task.step",
    )
    completion = TaskCompletion(
        task_id="task-1",
        completed_at=at(2),
        source="system",
    )
    with pytest.raises(MemoryValidationError):
        MemorySnapshot(
            temporary={"task-1": {temp.key: temp}},
            completed_tasks={"task-1": completion},
        )


def test_snapshot_rejects_duplicate_cumulative_entry_ids():
    first = make_fact(
        kind=FactKind.CUMULATIVE,
        key=None,
        entry_id="decision-001",
        value="one",
    )
    second = make_fact(
        kind=FactKind.CUMULATIVE,
        key=None,
        entry_id="decision-001",
        value="two",
    )
    with pytest.raises(MemoryValidationError):
        MemorySnapshot(cumulative=(first, second))


def test_snapshot_rejects_duplicate_event_ids_and_revisions():
    first = make_set_event(revision=1)
    duplicate_id = make_set_event(revision=2, event_id=first.event_id)
    with pytest.raises(MemoryValidationError):
        MemorySnapshot(revision=2, journal=(first, duplicate_id))

    second = make_set_event(revision=1, event_id="evt-00000099")
    with pytest.raises(MemoryValidationError):
        MemorySnapshot(revision=1, journal=(first, second))


def test_snapshot_rejects_event_revision_beyond_snapshot():
    event = make_set_event(revision=2)
    with pytest.raises(MemoryValidationError):
        MemorySnapshot(revision=1, journal=(event,))


def test_snapshot_from_dict_rejects_unknown_field():
    data = MemorySnapshot().to_dict()
    data["unknown"] = 1
    with pytest.raises(MemoryValidationError):
        MemorySnapshot.from_dict(data)


def test_snapshot_roundtrip_with_sections():
    fact = make_fact(value="2026-12-01")
    decision = make_fact(
        kind=FactKind.CUMULATIVE,
        key=None,
        entry_id="decision-001",
        value="Use PostgreSQL",
    )
    temp = make_fact(
        kind=FactKind.TEMPORARY,
        task_id="task-1",
        key="task.step",
        value="collect requirements",
    )
    tombstone = Tombstone(
        kind=FactKind.MUTABLE,
        key="project.cancelled",
        deleted_at=at(2),
        source="user",
    )
    completion = TaskCompletion(
        task_id="task-2",
        completed_at=at(2),
        source="system",
    )
    event = make_set_event()
    snapshot = MemorySnapshot(
        revision=1,
        updated_at=at(2),
        mutable={fact.key: fact},
        cumulative=(decision,),
        temporary={"task-1": {temp.key: temp}},
        mutable_tombstones={tombstone.key: tombstone},
        completed_tasks={"task-2": completion},
        journal=(event,),
    )
    restored = MemorySnapshot.from_dict(snapshot.to_dict())
    assert restored == snapshot
    assert restored.to_dict() == snapshot.to_dict()
