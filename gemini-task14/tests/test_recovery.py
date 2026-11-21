"""Crash consistency: журнал транзакции и восстановление после сбоя."""

import json
import multiprocessing
import os

import pytest

from invariants.commands import InvariantCommandService
from invariants.errors import InvariantStoreCorrupt
from invariants.models import InvariantCategory, InvariantCheck, InvariantRule
from invariants.store import InvariantStore


def make_rule():
    return InvariantRule(
        id="arch-001",
        category=InvariantCategory.ARCHITECTURE,
        text="Архитектура: модульный монолит",
        checks=(InvariantCheck(kind="forbidden_import", value="flask"),),
    )


def add_rule(store):
    return InvariantCommandService(store).add(
        make_rule(), actor="operator", reason="initial")


def assert_consistent(store):
    snapshot = store.load()
    history = store.read_history()
    assert snapshot.revision == 1
    assert len(snapshot.rules) == 1
    assert snapshot.rules[0].id == "arch-001"
    assert len(history) == 1
    assert history[0].event_id
    assert history[0].action == "add"
    assert history[0].old is None
    assert history[0].new["id"] == "arch-001"
    assert history[0].store_revision == snapshot.revision
    assert not store.journal_path.exists()


def test_crash_after_journal_before_snapshot_recovers(tmp_path, monkeypatch):
    store = InvariantStore(tmp_path)

    def crash(_snapshot):
        raise SystemExit("simulated crash before snapshot")

    monkeypatch.setattr(store, "_write_snapshot", crash)
    with pytest.raises(SystemExit):
        add_rule(store)

    assert store.journal_path.exists()
    recovered = InvariantStore(tmp_path)
    assert_consistent(recovered)
    assert_consistent(recovered)  # повторный запуск восстановления


def test_crash_after_snapshot_before_history_recovers(tmp_path, monkeypatch):
    store = InvariantStore(tmp_path)

    def crash(_change):
        raise SystemExit("simulated crash before history")

    monkeypatch.setattr(store, "_append_history_unlocked", crash)
    with pytest.raises(SystemExit):
        add_rule(store)

    assert store.journal_path.exists()
    assert store.snapshot_path.exists()
    assert not store.history_path.exists()

    recovered = InvariantStore(tmp_path)
    assert_consistent(recovered)


def test_crash_after_history_before_journal_remove_recovers(tmp_path, monkeypatch):
    store = InvariantStore(tmp_path)

    def crash():
        raise SystemExit("simulated crash before journal removal")

    monkeypatch.setattr(store, "_remove_journal_unlocked", crash)
    with pytest.raises(SystemExit):
        add_rule(store)

    assert store.journal_path.exists()
    recovered = InvariantStore(tmp_path)
    assert_consistent(recovered)


def test_crash_during_history_append_repairs_partial_line(tmp_path, monkeypatch):
    store = InvariantStore(tmp_path)

    def crash(change):
        expected = store._history_line(change)
        store.history_path.write_bytes(expected[:max(1, len(expected) // 2)])
        raise SystemExit("simulated partial append")

    monkeypatch.setattr(store, "_append_history_unlocked", crash)
    with pytest.raises(SystemExit):
        add_rule(store)

    assert store.journal_path.exists()
    recovered = InvariantStore(tmp_path)
    assert_consistent(recovered)


def _leave_snapshot_without_history(tmp_path, monkeypatch):
    store = InvariantStore(tmp_path)

    def crash(_change):
        raise SystemExit("stop before history")

    monkeypatch.setattr(store, "_append_history_unlocked", crash)
    with pytest.raises(SystemExit):
        add_rule(store)
    assert store.journal_path.exists()
    return store


def test_unexpected_history_tail_is_not_discarded(tmp_path, monkeypatch):
    store = _leave_snapshot_without_history(tmp_path, monkeypatch)
    store.history_path.write_bytes(b"garbage tail")

    with pytest.raises(InvariantStoreCorrupt):
        InvariantStore(tmp_path).load()


@pytest.mark.parametrize("field,value", [
    ("transaction_id", "not-a-transaction"),
    ("status", "committed"),
    ("format_version", True),
])
def test_invalid_journal_identity_or_state_fails_closed(tmp_path, monkeypatch, field, value):
    store = _leave_snapshot_without_history(tmp_path, monkeypatch)
    journal = json.loads(store.journal_path.read_text(encoding="utf-8"))
    journal[field] = value
    store.journal_path.write_text(json.dumps(journal), encoding="utf-8")

    from invariants.errors import InvariantStoreCorrupt
    with pytest.raises(InvariantStoreCorrupt):
        InvariantStore(tmp_path).load()


def test_unknown_journal_field_fails_closed(tmp_path, monkeypatch):
    store = _leave_snapshot_without_history(tmp_path, monkeypatch)
    journal = json.loads(store.journal_path.read_text(encoding="utf-8"))
    journal["unexpected"] = "field"
    store.journal_path.write_text(json.dumps(journal), encoding="utf-8")

    from invariants.errors import InvariantStoreCorrupt
    with pytest.raises(InvariantStoreCorrupt):
        InvariantStore(tmp_path).load()


def _real_crash_worker(state_dir, stage):
    store = InvariantStore(state_dir)

    if stage == "before_snapshot":
        def crash_snapshot(_snapshot):
            os._exit(91)

        store._write_snapshot = crash_snapshot
    elif stage == "before_history":
        def crash_history(_change):
            os._exit(92)

        store._append_history_unlocked = crash_history
    else:
        os._exit(93)

    InvariantCommandService(store).add(
        make_rule(), actor="crashed-process", reason="real crash")


@pytest.mark.parametrize("stage,expected_code", [
    ("before_snapshot", 91),
    ("before_history", 92),
])
def test_real_process_crash_recovers(tmp_path, stage, expected_code):
    try:
        context = multiprocessing.get_context("fork")
    except ValueError:
        pytest.skip("fork context недоступен на этой платформе")

    process = context.Process(
        target=_real_crash_worker, args=(tmp_path, stage))
    process.start()
    process.join(timeout=15)

    assert process.exitcode == expected_code
    store = InvariantStore(tmp_path)
    assert store.journal_path.exists()
    assert_consistent(store)


def test_corrupt_journal_fails_closed(tmp_path):
    store = InvariantStore(tmp_path)
    store.state_dir.mkdir(parents=True, exist_ok=True)
    store.journal_path.write_text("{not json", encoding="utf-8")

    from invariants.errors import InvariantStoreCorrupt
    with pytest.raises(InvariantStoreCorrupt):
        store.load()


def test_journal_with_unexpected_revision_fails_closed(tmp_path):
    store = InvariantStore(tmp_path)
    store.state_dir.mkdir(parents=True, exist_ok=True)
    store.journal_path.write_text(json.dumps({
        "format_version": 1,
        "old_revision": 0,
        "new_snapshot": {"format_version": 1, "revision": 5,
                         "updated_at": "", "rules": []},
        "change": {},
        "history_before_size": 0,
    }), encoding="utf-8")

    from invariants.errors import InvariantStoreCorrupt
    with pytest.raises(InvariantStoreCorrupt):
        store.load()
