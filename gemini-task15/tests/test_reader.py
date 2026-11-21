"""Тесты read-only MemoryReader поверх настоящего store/service."""

import pytest

from memory.errors import MemoryCorrupt
from memory.models import Fact, FactKind, MemorySnapshot
from memory.reader import MemoryReader
from memory.service import MemoryService
from memory.store import MemoryStore
from tests.fakes import FrozenClock, at


def setup(tmp_path):
    store = MemoryStore(tmp_path)
    service = MemoryService(store, clock=FrozenClock(at(1)))
    reader = MemoryReader(store)
    return store, service, reader


def test_reader_sees_only_current_after_replacement(tmp_path):
    store, service, reader = setup(tmp_path)
    service.set_long_term(
        "project.deadline", "OLD-DEADLINE",
        recorded_at=at(1), source="user")
    service.set_long_term(
        "project.deadline", "NEW-DEADLINE",
        recorded_at=at(2), source="user")

    prompt = reader.prompt_context()

    assert "NEW-DEADLINE" in prompt
    assert "OLD-DEADLINE" not in prompt
    assert [item.value for item in reader.context().long_term] == [
        "NEW-DEADLINE"]


def test_reader_hides_deleted_value_tombstone_and_journal(tmp_path):
    store, service, reader = setup(tmp_path)
    service.set_long_term(
        "project.requirement", "OLD-REQUIREMENT",
        recorded_at=at(1), source="user")
    service.delete_long_term(
        "project.requirement", recorded_at=at(2), source="user")

    prompt = reader.prompt_context()

    assert "OLD-REQUIREMENT" not in prompt
    assert "project.requirement" not in prompt
    assert "tombstone" not in prompt.lower()
    assert "journal" not in prompt.lower()
    assert reader.active_facts() == ()


def test_reader_keeps_cumulative_decisions_as_list(tmp_path):
    store, service, reader = setup(tmp_path)
    service.append_decision(
        "Use PostgreSQL", recorded_at=at(1), source="assistant")
    service.append_decision(
        "Use Redis", recorded_at=at(2), source="assistant")

    prompt = reader.prompt_context()

    assert "decisions:" in prompt
    assert prompt.index("Use PostgreSQL") < prompt.index("Use Redis")
    assert [item.value for item in reader.context().decisions] == [
        "Use PostgreSQL", "Use Redis"]


def test_reader_temporary_scope_and_complete(tmp_path):
    store, service, reader = setup(tmp_path)
    service.set_long_term(
        "project.deadline", "long", recorded_at=at(1), source="user")
    service.append_decision(
        "use postgres", recorded_at=at(1, 1), source="assistant")
    service.set_temporary(
        "task-1", "task.step", "one", recorded_at=at(2), source="user")
    service.set_temporary(
        "task-2", "task.step", "two", recorded_at=at(2), source="user")

    task_1 = reader.prompt_context("task-1")
    task_2 = reader.prompt_context("task-2")
    assert "one" in task_1 and "two" not in task_1
    assert "two" in task_2 and "one" not in task_2

    service.complete_task("task-1", completed_at=at(3), source="system")
    task_1_after = reader.prompt_context("task-1")
    task_2_after = reader.prompt_context("task-2")

    assert "one" not in task_1_after
    assert "long" in task_1_after
    assert "use postgres" in task_1_after
    assert "two" in task_2_after


def test_reader_has_no_write_api(tmp_path):
    store, service, reader = setup(tmp_path)

    for name in (
        "set_long_term", "append_decision", "set_temporary",
        "delete_long_term", "delete_temporary", "complete_task",
        "reopen_task", "history", "store",
    ):
        assert not hasattr(reader, name)


def test_reader_uses_store_load(tmp_path):
    class SpyStore:
        def __init__(self, snapshot):
            self.snapshot = snapshot
            self.calls = 0

        def load(self):
            self.calls += 1
            return self.snapshot

    item = Fact(
        kind=FactKind.MUTABLE,
        key="project.deadline",
        value="spy",
        recorded_at=at(1),
        source="user",
    )
    spy = SpyStore(MemorySnapshot(mutable={item.key: item}))
    reader = MemoryReader(spy)

    prompt = reader.prompt_context()

    assert spy.calls == 1
    assert "spy" in prompt


def test_reader_corrupt_store_fails_closed(tmp_path):
    store = MemoryStore(tmp_path)
    store.state_dir.mkdir(parents=True, exist_ok=True)
    store.memory_path.write_text("{broken", encoding="utf-8")
    reader = MemoryReader(store)

    with pytest.raises(MemoryCorrupt):
        reader.prompt_context()


def test_reader_deterministic_across_restart(tmp_path):
    store, service, reader = setup(tmp_path)
    service.set_long_term(
        "project.deadline", "value", recorded_at=at(1), source="user")
    first = reader.prompt_context()
    second_reader = MemoryReader(MemoryStore(tmp_path))
    assert second_reader.prompt_context() == first


def test_reader_after_reopen_does_not_restore_old_temp(tmp_path):
    store, service, reader = setup(tmp_path)
    service.set_temporary(
        "task-1", "task.step", "old", recorded_at=at(1), source="user")
    service.complete_task("task-1", completed_at=at(2), source="system")
    service.reopen_task("task-1", reopened_at=at(3), source="user")

    assert "old" not in reader.prompt_context("task-1")

    service.set_temporary(
        "task-1", "task.step", "new", recorded_at=at(4), source="user")
    prompt = reader.prompt_context("task-1")
    assert "new" in prompt
    assert "old" not in prompt
