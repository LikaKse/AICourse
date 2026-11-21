"""Хранилище: отдельные файлы, атомарность, fail-closed и revision."""

import json

import pytest

from invariants.errors import InvariantStoreCorrupt, InvariantStoreError
from invariants.models import (
    InvariantCategory,
    InvariantChange,
    InvariantCheck,
    InvariantRule,
    InvariantSnapshot,
)
from invariants.store import InvariantStore


def make_rule(rule_id="arch-001", *, version=1, active=True):
    return InvariantRule(
        id=rule_id,
        category=InvariantCategory.ARCHITECTURE,
        text="Архитектура: модульный монолит",
        checks=(InvariantCheck(kind="forbidden_import", value="flask"),),
        active=active,
        version=version,
        created_at="2026-11-19T10:00:00+00:00",
        updated_at="2026-11-19T10:00:00+00:00",
    )


def write_snapshot(store, snapshot):
    store.state_dir.mkdir(parents=True, exist_ok=True)
    store.snapshot_path.write_text(
        json.dumps(snapshot.to_dict(), ensure_ascii=False), encoding="utf-8")


def test_missing_store_is_empty_and_does_not_create_files(tmp_path):
    store = InvariantStore(tmp_path)

    assert store.load().revision == 0
    assert store.read_history() == ()
    assert not store.snapshot_path.exists()
    assert not store.history_path.exists()


def test_store_and_history_are_separate(tmp_path):
    store = InvariantStore(tmp_path)
    assert store.snapshot_path.name == "invariants.json"
    assert store.history_path.name == "invariant_history.jsonl"
    assert store.snapshot_path != store.history_path

    (tmp_path / "dialogue.jsonl").write_text("dialog only\n", encoding="utf-8")
    assert store.load().rules == ()


def test_snapshot_roundtrip_and_no_temporary_files(tmp_path):
    store = InvariantStore(tmp_path)
    snapshot = InvariantSnapshot(revision=1, rules=(make_rule(),))

    store._commit(
        snapshot,
        InvariantChange(
            event_id="evt-1",
            invariant_id="arch-001",
            action="add",
            old=None,
            new=make_rule().to_dict(),
            actor="operator",
            reason="initial",
            at="2026-11-19T10:00:00+00:00",
            store_revision=1,
        ),
    )

    assert store.load() == snapshot
    assert not list(tmp_path.glob(".*.tmp"))


def test_corrupt_snapshot_fails_closed(tmp_path):
    store = InvariantStore(tmp_path)
    store.state_dir.mkdir(parents=True, exist_ok=True)
    store.snapshot_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(InvariantStoreCorrupt):
        store.load()


def test_corrupt_history_fails_closed(tmp_path):
    store = InvariantStore(tmp_path)
    snapshot = InvariantSnapshot(revision=1, rules=(make_rule(),))
    write_snapshot(store, snapshot)
    store.history_path.write_text("{not json\n", encoding="utf-8")

    with pytest.raises(InvariantStoreCorrupt):
        store.load()


def test_history_without_snapshot_fails_closed(tmp_path):
    store = InvariantStore(tmp_path)
    store.state_dir.mkdir(parents=True, exist_ok=True)
    store.history_path.write_text(
        json.dumps({
            "event_id": "evt-1",
            "invariant_id": "arch-001",
            "action": "add",
            "old": None,
            "new": make_rule().to_dict(),
            "actor": "operator",
            "reason": "x",
            "at": "2026-11-19T10:00:00+00:00",
            "store_revision": 1,
        }) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(InvariantStoreCorrupt):
        store.load()


def test_snapshot_revision_without_history_fails_closed(tmp_path):
    store = InvariantStore(tmp_path)
    write_snapshot(store, InvariantSnapshot(revision=1, rules=(make_rule(),)))

    with pytest.raises(InvariantStoreCorrupt):
        store.load()


def test_revision_mismatch_between_snapshot_and_history_fails_closed(tmp_path):
    store = InvariantStore(tmp_path)
    snapshot = InvariantSnapshot(revision=1, rules=(make_rule(),))
    write_snapshot(store, snapshot)
    store.history_path.write_text(
        json.dumps({
            "event_id": "evt-1",
            "invariant_id": "arch-001",
            "action": "add",
            "old": None,
            "new": make_rule().to_dict(),
            "actor": "operator",
            "reason": "x",
            "at": "2026-11-19T10:00:00+00:00",
            "store_revision": 2,
        }) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(InvariantStoreCorrupt):
        store.load()


def test_store_has_no_delete_or_mutation_api(tmp_path):
    store = InvariantStore(tmp_path)

    for name in ("add", "update", "disable", "delete", "clear", "delete_history"):
        assert not hasattr(store, name)
