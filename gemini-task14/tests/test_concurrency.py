"""Межпроцессная блокировка и конфликт версий."""

import multiprocessing
import time

import pytest

from invariants.commands import InvariantCommandService
from invariants.models import InvariantCategory, InvariantCheck, InvariantRule
from invariants.store import InvariantStore


def make_rule(rule_id="arch-001"):
    return InvariantRule(
        id=rule_id,
        category=InvariantCategory.ARCHITECTURE,
        text=f"Архитектура: модульный монолит {rule_id}",
        checks=(InvariantCheck(kind="forbidden_import", value="flask"),),
    )


def _update_worker(state_dir, text, ready, results):
    try:
        service = InvariantCommandService(InvariantStore(state_dir))
        ready.wait(timeout=10)
        change = service.update(
            "arch-001",
            text=text,
            checks=(InvariantCheck(kind="forbidden_import", value="flask"),),
            actor="worker",
            reason="concurrent update",
            expected_version=1,
        )
        results.put(("ok", change.event_id))
    except Exception as exc:  # noqa: BLE001 - результат нужен тесту
        results.put(("error", type(exc).__name__))


def _slow_first_writer(state_dir, ready, results):
    store = InvariantStore(state_dir)
    original = store._append_history_unlocked

    def slow_append(change):
        time.sleep(0.15)
        return original(change)

    store._append_history_unlocked = slow_append
    ready.wait(timeout=10)
    try:
        InvariantCommandService(store).add(
            make_rule(), actor="writer", reason="first write")
        results.put(("writer", "ok"))
    except Exception as exc:  # noqa: BLE001
        results.put(("writer", type(exc).__name__))


def _first_reader(state_dir, ready, results):
    ready.wait(timeout=10)
    last_revision = None
    for _ in range(40):
        try:
            store = InvariantStore(state_dir)
            last_revision = store.load().revision
            store.read_history()
        except Exception as exc:  # noqa: BLE001
            results.put(("reader_error", type(exc).__name__))
            return
        time.sleep(0.01)
    results.put(("reader", last_revision))


def test_concurrent_first_read_never_sees_partial_store(tmp_path):
    try:
        context = multiprocessing.get_context("fork")
    except ValueError:
        pytest.skip("fork context недоступен на этой платформе")

    ready = context.Event()
    results = context.Queue()
    writer = context.Process(
        target=_slow_first_writer, args=(tmp_path, ready, results))
    reader = context.Process(
        target=_first_reader, args=(tmp_path, ready, results))

    writer.start()
    reader.start()
    ready.set()
    writer.join(timeout=15)
    reader.join(timeout=15)

    assert writer.exitcode == 0
    assert reader.exitcode == 0
    outcomes = [results.get(timeout=5) for _ in range(2)]
    assert not any(item[0] == "reader_error" for item in outcomes)
    assert ("writer", "ok") in outcomes

    store = InvariantStore(tmp_path)
    snapshot = store.load()
    history = store.read_history()
    assert snapshot.revision == 1
    assert len(snapshot.rules) == 1
    assert len(history) == 1
    assert not store.journal_path.exists()


def test_two_processes_update_same_rule_with_one_winner(tmp_path):
    service = InvariantCommandService(InvariantStore(tmp_path))
    service.add(make_rule(), actor="operator", reason="initial")

    try:
        context = multiprocessing.get_context("fork")
    except ValueError:
        pytest.skip("fork context недоступен на этой платформе")

    ready = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_update_worker,
            args=(tmp_path, f"variant-{index}", ready, results),
        )
        for index in (1, 2)
    ]
    for process in processes:
        process.start()
    ready.set()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    outcomes = [results.get(timeout=5) for _ in processes]
    assert sorted(item[0] for item in outcomes) == ["error", "ok"]
    assert any(item[1] == "InvariantVersionConflict"
               for item in outcomes if item[0] == "error")

    snapshot = InvariantStore(tmp_path).load()
    history = InvariantStore(tmp_path).read_history()
    assert snapshot.revision == 2
    assert snapshot.rule("arch-001").version == 2
    assert len(history) == 2
    assert history[-1].action == "update"
