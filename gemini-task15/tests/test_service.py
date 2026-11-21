"""Тесты сервиса памяти: конфликты, tombstones, задачи, транзакции."""

import pytest

from memory.errors import MemoryCorrupt, MemoryValidationError, MemoryWriteError
from memory.models import EventResult, FactKind, MemoryAction, MemorySnapshot
from memory.service import MemoryService
from memory.store import MemoryStore
from tests.fakes import FrozenClock, at


def service(tmp_path, moment=None):
    store = MemoryStore(tmp_path)
    return MemoryService(store, clock=FrozenClock(moment or at(1)))


def snapshot(tmp_path):
    return MemoryStore(tmp_path).load()


def test_set_and_get_long_term(tmp_path):
    svc = service(tmp_path, at(1, 1))
    result = svc.set_long_term(
        "project.deadline",
        "2026-12-01",
        recorded_at=at(2),
        source="user",
        reason="initial",
    )

    assert result.status == "applied"
    assert result.applied is True
    assert result.fact is not None
    assert result.fact.value == "2026-12-01"
    assert svc.get_long_term("project.deadline") == result.fact
    assert svc.active_facts() == (result.fact,)
    history = svc.history()
    assert len(history) == 1
    assert history[0].action is MemoryAction.SET
    assert history[0].result is EventResult.APPLIED
    assert history[0].new_value == "2026-12-01"
    assert snapshot(tmp_path).revision == 1


def test_replace_old_fact(tmp_path):
    svc = service(tmp_path)
    svc.set_long_term(
        "project.deadline", "old", recorded_at=at(1), source="user")
    result = svc.set_long_term(
        "project.deadline", "new", recorded_at=at(2), source="user")

    assert result.status == "applied"
    assert svc.get_long_term("project.deadline").value == "new"
    history = svc.history()
    assert len(history) == 2
    assert history[-1].old_value == "old"
    assert history[-1].new_value == "new"
    assert history[-1].result is EventResult.APPLIED
    assert "old" not in [item.value for item in svc.active_facts()]


def test_stale_timestamp_rejected(tmp_path):
    svc = service(tmp_path)
    svc.set_long_term("project.deadline", "new", recorded_at=at(2), source="user")
    result = svc.set_long_term(
        "project.deadline", "old", recorded_at=at(1), source="user")

    assert result.status == "rejected_stale"
    assert result.fact.value == "new"
    assert svc.get_long_term("project.deadline").value == "new"
    event = svc.history()[-1]
    assert event.action is MemoryAction.SET
    assert event.result is EventResult.REJECTED_STALE
    assert event.old_value == "new"
    assert event.new_value == "old"
    assert snapshot(tmp_path).revision == 2


def test_equal_timestamp_same_value_unchanged(tmp_path):
    svc = service(tmp_path)
    svc.set_long_term("project.deadline", "same", recorded_at=at(1), source="user")
    result = svc.set_long_term(
        "project.deadline", "same", recorded_at=at(1), source="user")

    assert result.status == "unchanged"
    assert result.fact.value == "same"
    assert len(svc.history()) == 1
    assert snapshot(tmp_path).revision == 1


def test_equal_timestamp_different_value_rejected_equal(tmp_path):
    svc = service(tmp_path)
    svc.set_long_term("project.deadline", "one", recorded_at=at(1), source="user")
    result = svc.set_long_term(
        "project.deadline", "two", recorded_at=at(1), source="user")

    assert result.status == "rejected_equal"
    assert result.fact.value == "one"
    assert svc.get_long_term("project.deadline").value == "one"
    event = svc.history()[-1]
    assert event.result is EventResult.REJECTED_EQUAL
    assert event.old_value == "one"
    assert event.new_value == "two"


def test_same_value_later_updates_recorded_at(tmp_path):
    svc = service(tmp_path)
    svc.set_long_term("project.deadline", "same", recorded_at=at(1), source="user")
    result = svc.set_long_term(
        "project.deadline", "same", recorded_at=at(2), source="user")

    assert result.status == "confirmed"
    assert result.applied is True
    assert result.fact.recorded_at == at(2)
    assert result.fact.value == "same"
    assert svc.get_long_term("project.deadline").recorded_at == at(2)
    event = svc.history()[-1]
    assert event.result is EventResult.UNCHANGED
    assert event.old_value == event.new_value == "same"
    assert event.old_recorded_at == at(1)
    assert event.new_recorded_at == at(2)


def test_delete_creates_tombstone_and_blocks_stale_restore(tmp_path):
    svc = service(tmp_path)
    svc.set_long_term("project.deadline", "one", recorded_at=at(2), source="user")
    deleted = svc.delete_long_term(
        "project.deadline", recorded_at=at(3), source="user", reason="cancelled")

    assert deleted.status == "applied"
    assert svc.get_long_term("project.deadline") is None
    snap = snapshot(tmp_path)
    assert snap.mutable_tombstones["project.deadline"].deleted_at == at(3)
    assert snap.mutable == {}

    stale = svc.set_long_term(
        "project.deadline", "stale", recorded_at=at(2), source="user")
    assert stale.status == "rejected_stale"
    assert svc.get_long_term("project.deadline") is None

    equal = svc.set_long_term(
        "project.deadline", "equal", recorded_at=at(3), source="user")
    assert equal.status == "rejected_equal"
    assert svc.get_long_term("project.deadline") is None

    restored = svc.set_long_term(
        "project.deadline", "restored", recorded_at=at(4), source="user")
    assert restored.status == "applied"
    assert svc.get_long_term("project.deadline").value == "restored"
    snap = snapshot(tmp_path)
    assert snap.mutable_tombstones["project.deadline"].deleted_at == at(3)


def test_delete_stale_and_equal_do_not_remove_fact(tmp_path):
    svc = service(tmp_path)
    svc.set_long_term("project.deadline", "active", recorded_at=at(2), source="user")

    stale = svc.delete_long_term(
        "project.deadline", recorded_at=at(1), source="user")
    assert stale.status == "rejected_stale"
    assert svc.get_long_term("project.deadline").value == "active"

    equal = svc.delete_long_term(
        "project.deadline", recorded_at=at(2), source="user")
    assert equal.status == "rejected_equal"
    assert svc.get_long_term("project.deadline").value == "active"
    assert svc.history()[-1].result is EventResult.REJECTED_EQUAL


def test_temporary_isolation(tmp_path):
    svc = service(tmp_path)
    svc.set_long_term("project.deadline", "long", recorded_at=at(1), source="user")
    svc.set_temporary(
        "task-1", "task.step", "one", recorded_at=at(1), source="user")
    svc.set_temporary(
        "task-2", "task.step", "two", recorded_at=at(2), source="user")

    assert svc.get_temporary("task-1", "task.step").value == "one"
    assert svc.get_temporary("task-2", "task.step").value == "two"
    assert svc.get_long_term("project.deadline").value == "long"
    assert "two" not in [item.value for item in svc.active_facts("task-1")]
    assert "one" not in [item.value for item in svc.active_facts("task-2")]
    assert all(item.kind is not FactKind.TEMPORARY for item in svc.active_facts())


def test_complete_task_clears_only_its_temporary_facts(tmp_path):
    svc = service(tmp_path)
    svc.set_long_term("project.deadline", "long", recorded_at=at(1), source="user")
    svc.append_decision("use postgres", recorded_at=at(1, 1), source="assistant")
    svc.set_temporary("task-1", "task.step", "one", recorded_at=at(2), source="user")
    svc.set_temporary("task-2", "task.step", "two", recorded_at=at(2), source="user")

    result = svc.complete_task(
        "task-1", completed_at=at(3), source="system", reason="done")

    assert result.status == "applied"
    assert result.cleared_count == 1
    assert result.completion.completed_at == at(3)
    assert svc.get_temporary("task-1", "task.step") is None
    assert svc.get_temporary("task-2", "task.step").value == "two"
    assert svc.get_long_term("project.deadline").value == "long"
    assert any(item.entry_id for item in svc.active_facts())
    snap = snapshot(tmp_path)
    assert "task-1" in snap.completed_tasks
    assert "task-1" not in snap.temporary
    assert snap.temporary_tombstones["task-1"]["task.step"].deleted_at == at(3)
    assert svc.history()[-1].action is MemoryAction.COMPLETE_TASK


def test_complete_task_is_idempotent(tmp_path):
    svc = service(tmp_path)
    first = svc.complete_task(
        "task-1", completed_at=at(2), source="system", reason="done")
    revision = snapshot(tmp_path).revision
    history_len = len(svc.history())

    second = svc.complete_task(
        "task-1", completed_at=at(3), source="system", reason="again")

    assert first.status == "applied"
    assert second.status == "already_completed"
    assert second.completion.completed_at == at(2)
    assert second.cleared_count == 0
    assert snapshot(tmp_path).revision == revision
    assert len(svc.history()) == history_len


def test_late_temporary_write_rejected_until_reopen(tmp_path):
    svc = service(tmp_path)
    svc.set_temporary("task-1", "task.step", "before", recorded_at=at(1), source="user")
    svc.complete_task("task-1", completed_at=at(2), source="system")

    late = svc.set_temporary(
        "task-1", "task.step", "late", recorded_at=at(3), source="user")
    assert late.status == "rejected_task_completed"
    assert late.fact is None
    assert svc.get_temporary("task-1", "task.step") is None
    assert svc.history()[-1].result is EventResult.REJECTED_TASK_COMPLETED

    reopened = svc.reopen_task(
        "task-1", reopened_at=at(4), source="user", reason="continue")
    assert reopened.status == "applied"
    assert "task-1" not in snapshot(tmp_path).completed_tasks

    after = svc.set_temporary(
        "task-1", "task.step", "after", recorded_at=at(5), source="user")
    assert after.status == "applied"
    assert svc.get_temporary("task-1", "task.step").value == "after"


def test_reopen_unknown_task_is_not_found(tmp_path):
    svc = service(tmp_path)
    result = svc.reopen_task("task-1", reopened_at=at(2), source="user")
    assert result.status == "not_found"
    assert snapshot(tmp_path).revision == 0
    assert svc.history() == ()


def test_append_decision_is_cumulative(tmp_path):
    svc = service(tmp_path)
    first = svc.append_decision(
        "Use PostgreSQL", recorded_at=at(1), source="assistant")
    second = svc.append_decision(
        "Use Redis", recorded_at=at(2), source="assistant")

    assert first.status == second.status == "applied"
    assert first.fact.kind is FactKind.CUMULATIVE
    assert first.fact.entry_id
    assert second.fact.entry_id != first.fact.entry_id
    assert [item.value for item in svc.active_facts()] == [
        "Use PostgreSQL", "Use Redis"]
    assert [event.action for event in svc.history()] == [
        MemoryAction.APPEND, MemoryAction.APPEND]


def test_journal_and_active_facts_are_separate(tmp_path):
    svc = service(tmp_path)
    svc.set_long_term("project.deadline", "old", recorded_at=at(1), source="user")
    svc.set_long_term("project.deadline", "new", recorded_at=at(2), source="user")

    active_values = [item.value for item in svc.active_facts()]
    assert active_values == ["new"]
    assert "old" not in active_values
    assert [event.new_value for event in svc.history()] == ["old", "new"]


def test_delete_missing_is_not_found(tmp_path):
    svc = service(tmp_path)
    result = svc.delete_long_term(
        "project.missing", recorded_at=at(1), source="user")
    assert result.status == "not_found"
    assert snapshot(tmp_path).revision == 0
    assert svc.history() == ()


def test_corrupt_store_fails_closed(tmp_path):
    store = MemoryStore(tmp_path)
    store.state_dir.mkdir(parents=True, exist_ok=True)
    store.memory_path.write_text("{broken", encoding="utf-8")
    before = store.memory_path.read_bytes()
    svc = MemoryService(store, clock=FrozenClock(at(1)))

    with pytest.raises(MemoryCorrupt):
        svc.set_long_term("project.deadline", "x", recorded_at=at(1), source="user")
    with pytest.raises(MemoryCorrupt):
        svc.get_long_term("project.deadline")
    assert store.memory_path.read_bytes() == before


def test_write_failure_is_not_success(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path)
    svc = MemoryService(store, clock=FrozenClock(at(1)))
    svc.set_long_term("project.deadline", "old", recorded_at=at(1), source="user")
    old = store.load()

    def broken_replace(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("memory.store.os.replace", broken_replace)
    with pytest.raises(MemoryWriteError):
        svc.set_long_term("project.deadline", "new", recorded_at=at(2), source="user")

    assert store.load() == old
    assert list(tmp_path.glob(".memory-*.tmp")) == []


def test_delete_temporary_tombstone_and_restore(tmp_path):
    svc = service(tmp_path)
    svc.set_temporary(
        "task-1", "task.step", "one", recorded_at=at(1), source="user")

    deleted = svc.delete_temporary(
        "task-1", "task.step", recorded_at=at(2), source="user")
    assert deleted.status == "applied"
    assert svc.get_temporary("task-1", "task.step") is None
    snap = snapshot(tmp_path)
    assert snap.temporary_tombstones["task-1"]["task.step"].deleted_at == at(2)

    stale = svc.set_temporary(
        "task-1", "task.step", "stale", recorded_at=at(1), source="user")
    assert stale.status == "rejected_stale"
    equal = svc.set_temporary(
        "task-1", "task.step", "equal", recorded_at=at(2), source="user")
    assert equal.status == "rejected_equal"
    assert svc.get_temporary("task-1", "task.step") is None

    restored = svc.set_temporary(
        "task-1", "task.step", "restored", recorded_at=at(3), source="user")
    assert restored.status == "applied"
    assert svc.get_temporary("task-1", "task.step").value == "restored"


def test_revision_and_journal_sequence(tmp_path):
    svc = service(tmp_path)
    svc.set_long_term(
        "project.deadline", "same", recorded_at=at(1), source="user")
    svc.set_long_term(
        "project.deadline", "same", recorded_at=at(2), source="user")
    svc.set_temporary(
        "task-1", "task.step", "one", recorded_at=at(3), source="user")
    svc.complete_task("task-1", completed_at=at(4), source="system")

    snap = snapshot(tmp_path)
    history = svc.history()
    assert snap.revision == 5
    assert len(history) == 5
    assert [event.revision for event in history] == [1, 2, 3, 4, 5]


def test_complete_task_rejects_time_not_after_temporary_fact(tmp_path):
    svc = service(tmp_path)
    svc.set_temporary("task-1", "task.step", "one", recorded_at=at(2), source="user")
    with pytest.raises(MemoryValidationError):
        svc.complete_task("task-1", completed_at=at(2), source="system")
    assert svc.get_temporary("task-1", "task.step").value == "one"
    assert snapshot(tmp_path).completed_tasks == {}
