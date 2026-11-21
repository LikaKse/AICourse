"""Тесты атомарного хранилища памяти (Этап 2)."""

import json

import pytest

from memory.errors import MemoryCorrupt, MemoryWriteError
from memory.models import (
    Fact,
    FactKind,
    MemoryAction,
    MemoryEvent,
    MemorySnapshot,
)
from memory.store import MemoryStore
from tests.fakes import at


def make_snapshot(value="v1", revision=1, moment=None):
    moment = moment or at(1)
    fact = Fact(
        kind=FactKind.MUTABLE,
        key="project.deadline",
        value=value,
        recorded_at=moment,
        source="user",
    )
    event = MemoryEvent(
        event_id=f"evt-{revision:08d}",
        revision=revision,
        action=MemoryAction.SET,
        at=moment,
        source="user",
        kind=FactKind.MUTABLE,
        key="project.deadline",
        new_present=True,
        new_value=value,
        new_recorded_at=moment,
    )
    return MemorySnapshot(
        revision=revision,
        updated_at=moment,
        mutable={fact.key: fact},
        journal=(event,),
    )


def make_snapshot_v2(moment=None):
    moment = moment or at(2)
    old = make_snapshot("v1", revision=1, moment=at(1))
    fact = Fact(
        kind=FactKind.MUTABLE,
        key="project.deadline",
        value="v2",
        recorded_at=moment,
        source="user",
    )
    event = MemoryEvent(
        event_id="evt-00000002",
        revision=2,
        action=MemoryAction.SET,
        at=moment,
        source="user",
        kind=FactKind.MUTABLE,
        key="project.deadline",
        old_present=True,
        old_value="v1",
        old_recorded_at=at(1),
        new_present=True,
        new_value="v2",
        new_recorded_at=moment,
    )
    return MemorySnapshot(
        revision=2,
        updated_at=moment,
        mutable={fact.key: fact},
        journal=old.journal + (event,),
    )


def test_missing_store_is_empty(tmp_path):
    store = MemoryStore(tmp_path)
    snapshot = store.load()
    assert snapshot.revision == 0
    assert snapshot.mutable == {}
    assert not store.memory_path.exists()


def test_save_and_load_roundtrip(tmp_path):
    store = MemoryStore(tmp_path)
    original = make_snapshot()
    store.save(original)

    loaded = store.load()
    assert loaded == original
    assert store.memory_path.exists()


def test_save_rejects_revision_regression(tmp_path):
    store = MemoryStore(tmp_path)
    store.save(make_snapshot())

    with pytest.raises(MemoryWriteError):
        store.save(MemorySnapshot(revision=0))

    assert store.load() == make_snapshot()


def test_corrupt_json_raises_memory_corrupt(tmp_path):
    store = MemoryStore(tmp_path)
    store.state_dir.mkdir(parents=True, exist_ok=True)
    store.memory_path.write_text("{not-json", encoding="utf-8")
    before = store.memory_path.read_bytes()

    with pytest.raises(MemoryCorrupt):
        store.load()
    assert store.memory_path.read_bytes() == before


def test_corrupt_structure_raises_memory_corrupt(tmp_path):
    store = MemoryStore(tmp_path)
    store.state_dir.mkdir(parents=True, exist_ok=True)
    store.memory_path.write_text(
        json.dumps({"format_version": 99, "revision": 0}),
        encoding="utf-8",
    )

    with pytest.raises(MemoryCorrupt):
        store.load()


def test_revision_without_matching_journal_is_corrupt(tmp_path):
    store = MemoryStore(tmp_path)
    first = make_snapshot()
    invalid = MemorySnapshot(
        revision=2,
        updated_at=at(2),
        mutable=first.mutable,
        journal=first.journal,
    )
    store.state_dir.mkdir(parents=True, exist_ok=True)
    store.memory_path.write_text(
        json.dumps(invalid.to_dict()), encoding="utf-8")

    with pytest.raises(MemoryCorrupt):
        store.load()


def test_revision_without_updated_at_is_corrupt(tmp_path):
    store = MemoryStore(tmp_path)
    base = make_snapshot()
    invalid = MemorySnapshot(
        revision=1,
        updated_at=None,
        mutable=base.mutable,
        journal=base.journal,
    )
    store.state_dir.mkdir(parents=True, exist_ok=True)
    store.memory_path.write_text(
        json.dumps(invalid.to_dict()), encoding="utf-8")

    with pytest.raises(MemoryCorrupt):
        store.load()


def test_revision_zero_with_active_data_is_corrupt(tmp_path):
    store = MemoryStore(tmp_path)
    fact = make_snapshot().mutable["project.deadline"]
    invalid = MemorySnapshot(
        revision=0,
        mutable={fact.key: fact},
    )
    store.state_dir.mkdir(parents=True, exist_ok=True)
    store.memory_path.write_text(
        json.dumps(invalid.to_dict()), encoding="utf-8")

    with pytest.raises(MemoryCorrupt):
        store.load()


def test_save_rejects_inconsistent_snapshot(tmp_path):
    store = MemoryStore(tmp_path)
    fact = make_snapshot().mutable["project.deadline"]
    invalid = MemorySnapshot(revision=0, mutable={fact.key: fact})

    with pytest.raises(MemoryWriteError):
        store.save(invalid)
    assert not store.memory_path.exists()


def test_write_failure_keeps_old_snapshot(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path)
    old = make_snapshot()
    store.save(old)

    def broken_replace(*args, **kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr("memory.store.os.replace", broken_replace)
    with pytest.raises(MemoryWriteError):
        store.save(make_snapshot_v2())

    assert store.load() == old
    assert list(tmp_path.glob(".memory-*.tmp")) == []


def test_fsync_failure_keeps_old_snapshot(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path)
    old = make_snapshot()
    store.save(old)

    def broken_fsync(fd):
        raise OSError("fsync failed")

    monkeypatch.setattr("memory.store.os.fsync", broken_fsync)
    with pytest.raises(MemoryWriteError):
        store.save(make_snapshot_v2())

    assert store.load() == old
    assert list(tmp_path.glob(".memory-*.tmp")) == []
