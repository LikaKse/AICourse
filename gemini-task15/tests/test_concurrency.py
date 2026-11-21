"""Межпроцессные тесты блокировки и отсутствия потерянных обновлений."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta

from memory.service import MemoryService
from memory.store import MemoryStore
from tests.fakes import FrozenClock, at


def _set_unique(args):
    state_dir, index, moment = args
    service = MemoryService(
        MemoryStore(state_dir), clock=FrozenClock(moment))
    result = service.set_long_term(
        f"key.{index}",
        f"value-{index}",
        recorded_at=moment,
        source="user",
    )
    return result.status


def _set_same_key(args):
    state_dir, index, moment = args
    service = MemoryService(
        MemoryStore(state_dir), clock=FrozenClock(moment))
    result = service.set_long_term(
        "project.deadline",
        f"value-{index}",
        recorded_at=moment,
        source="user",
    )
    return result.status


def _run_parallel(worker, args):
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=4, mp_context=context) as executor:
        return list(executor.map(worker, args))


def test_parallel_unique_keys_are_not_lost(tmp_path):
    base = at(1)
    args = [
        (str(tmp_path), index, base + timedelta(seconds=index))
        for index in range(8)
    ]

    statuses = _run_parallel(_set_unique, args)

    assert statuses == ["applied"] * 8
    snapshot = MemoryStore(tmp_path).load()
    assert set(snapshot.mutable) == {f"key.{index}" for index in range(8)}
    assert snapshot.revision == 8
    assert len(snapshot.journal) == 8


def test_parallel_same_key_later_timestamp_wins(tmp_path):
    base = at(2)
    args = [
        (str(tmp_path), index, base + timedelta(seconds=index))
        for index in range(6)
    ]

    _run_parallel(_set_same_key, args)

    snapshot = MemoryStore(tmp_path).load()
    fact = snapshot.mutable["project.deadline"]
    assert fact.value == "value-5"
    assert fact.recorded_at == base + timedelta(seconds=5)
    assert snapshot.revision == 6
    assert len(snapshot.journal) == 6
