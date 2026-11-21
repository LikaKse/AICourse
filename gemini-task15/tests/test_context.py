"""Тесты чистого ContextBuilder."""

import json

import pytest

from memory.context import ContextBuilder, MemoryContext
from memory.errors import MemoryValidationError
from memory.models import (
    Fact,
    FactKind,
    MemoryAction,
    MemoryEvent,
    MemorySnapshot,
    Tombstone,
)
from tests.fakes import at


def fact(
    kind,
    *,
    key=None,
    value="value",
    when=None,
    source="user",
    task_id=None,
    entry_id=None,
):
    return Fact(
        kind=kind,
        key=key,
        value=value,
        recorded_at=when or at(1),
        source=source,
        task_id=task_id,
        entry_id=entry_id,
    )


def set_event(*, old="OLD-SECRET", new="NEW"):
    return MemoryEvent(
        event_id="evt-00000001",
        revision=1,
        action=MemoryAction.SET,
        at=at(2),
        source="user",
        kind=FactKind.MUTABLE,
        key="project.deadline",
        old_present=True,
        old_value=old,
        old_recorded_at=at(1),
        new_present=True,
        new_value=new,
        new_recorded_at=at(2),
    )


def test_empty_context_is_stable():
    builder = ContextBuilder()
    snapshot = MemorySnapshot()
    assert builder.render(snapshot) == "[ACTIVE MEMORY]\n(empty)"
    assert builder.render(snapshot) == builder.render(snapshot)


def test_mutable_context_contains_only_current_value():
    current = fact(
        FactKind.MUTABLE, key="project.deadline", value="NEW",
        when=at(2))
    snapshot = MemorySnapshot(
        revision=1,
        updated_at=at(2),
        mutable={current.key: current},
        journal=(set_event(),),
    )

    prompt = ContextBuilder().render(snapshot)

    assert "project.deadline = \"NEW\"" in prompt
    assert "OLD-SECRET" not in prompt
    assert "journal" not in prompt.lower()
    assert "tombstone" not in prompt.lower()


def test_tombstones_are_not_rendered():
    current = fact(
        FactKind.MUTABLE, key="project.status", value="active", when=at(1))
    tombstone = Tombstone(
        kind=FactKind.MUTABLE,
        key="project.cancelled",
        deleted_at=at(2),
        source="user",
    )
    snapshot = MemorySnapshot(
        revision=1,
        updated_at=at(2),
        mutable={current.key: current},
        mutable_tombstones={tombstone.key: tombstone},
        journal=(set_event(),),
    )

    prompt = ContextBuilder().render(snapshot)

    assert "project.status = \"active\"" in prompt
    assert "project.cancelled" not in prompt
    assert "tombstone" not in prompt.lower()


def test_cumulative_decisions_are_rendered_as_list():
    first = fact(
        FactKind.CUMULATIVE, entry_id="decision-001",
        value="Use PostgreSQL", when=at(1))
    second = fact(
        FactKind.CUMULATIVE, entry_id="decision-002",
        value="Use Redis", when=at(2))
    snapshot = MemorySnapshot(cumulative=(second, first))

    prompt = ContextBuilder().render(snapshot)

    assert "decisions:" in prompt
    assert prompt.index("Use PostgreSQL") < prompt.index("Use Redis")
    assert "  - \"Use PostgreSQL\"" in prompt
    assert "  - \"Use Redis\"" in prompt


def test_temporary_facts_are_scoped_to_task():
    first = fact(
        FactKind.TEMPORARY, key="task.step", value="one",
        when=at(1), task_id="task-1")
    second = fact(
        FactKind.TEMPORARY, key="task.step", value="two",
        when=at(2), task_id="task-2")
    snapshot = MemorySnapshot(
        temporary={
            "task-1": {"task.step": first},
            "task-2": {"task.step": second},
        },
    )

    context_1 = ContextBuilder().build(snapshot, "task-1")
    context_2 = ContextBuilder().build(snapshot, "task-2")
    context_none = ContextBuilder().build(snapshot)

    assert [item.value for item in context_1.temporary] == ["one"]
    assert [item.value for item in context_2.temporary] == ["two"]
    assert context_none.temporary == ()
    assert "two" not in context_1.render()
    assert "one" not in context_2.render()


def test_context_is_deterministic_for_different_insertion_order():
    deadline = fact(
        FactKind.MUTABLE, key="project.deadline", value="d", when=at(1))
    status = fact(
        FactKind.MUTABLE, key="project.status", value="s", when=at(1))
    decision_a = fact(
        FactKind.CUMULATIVE, entry_id="decision-001",
        value="a", when=at(1))
    decision_b = fact(
        FactKind.CUMULATIVE, entry_id="decision-002",
        value="b", when=at(2))
    first = MemorySnapshot(
        mutable={deadline.key: deadline, status.key: status},
        cumulative=(decision_a, decision_b),
    )
    second = MemorySnapshot(
        mutable={status.key: status, deadline.key: deadline},
        cumulative=(decision_b, decision_a),
    )

    assert ContextBuilder().render(first) == ContextBuilder().render(second)


def test_nested_json_values_are_canonical():
    current = fact(
        FactKind.MUTABLE,
        key="project.data",
        value={"b": 2, "a": [1, 2]},
        when=at(1),
    )
    snapshot = MemorySnapshot(mutable={current.key: current})

    prompt = ContextBuilder().render(snapshot)

    assert "project.data = " + json.dumps(
        {"a": [1, 2], "b": 2},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) in prompt


def test_invalid_task_id_is_rejected():
    with pytest.raises(MemoryValidationError):
        ContextBuilder().build(MemorySnapshot(), "BAD ID")


def test_builder_rejects_non_snapshot():
    with pytest.raises(MemoryValidationError):
        ContextBuilder().build({}, "task-1")


def test_memory_context_rejects_wrong_section_kind():
    current = fact(
        FactKind.MUTABLE, key="project.deadline", value="d", when=at(1))
    with pytest.raises(MemoryValidationError):
        MemoryContext(long_term=(current,), temporary=(current,))
