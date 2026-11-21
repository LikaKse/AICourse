"""Полная проверка последовательности и уникальности истории."""

import json

import pytest

from invariants.commands import InvariantCommandService
from invariants.errors import InvariantStoreCorrupt
from invariants.models import InvariantCategory, InvariantCheck, InvariantRule
from invariants.store import InvariantStore


def make_rule(rule_id):
    return InvariantRule(
        id=rule_id,
        category=InvariantCategory.ARCHITECTURE,
        text=f"Правило {rule_id}",
        checks=(InvariantCheck(kind="forbidden_import", value="flask"),),
    )


def seeded_store(tmp_path):
    store = InvariantStore(tmp_path)
    service = InvariantCommandService(store)
    service.add(make_rule("arch-001"), actor="operator", reason="first")
    service.add(make_rule("arch-002"), actor="operator", reason="second")
    return store


def lines(store):
    return store.history_path.read_text(encoding="utf-8").splitlines()


def write_lines(store, values):
    store.history_path.write_text("\n".join(values) + "\n", encoding="utf-8")


def test_history_revision_sequence_is_checked(tmp_path):
    store = seeded_store(tmp_path)
    # Пропуск первого события: остаётся только revision 2.
    write_lines(store, lines(store)[1:])

    with pytest.raises(InvariantStoreCorrupt, match="revision"):
        store.load()


def test_history_gap_is_rejected(tmp_path):
    store = seeded_store(tmp_path)
    values = lines(store)
    second = json.loads(values[1])
    second["store_revision"] = 3
    values[1] = json.dumps(second, ensure_ascii=False, separators=(",", ":"))
    write_lines(store, values)

    with pytest.raises(InvariantStoreCorrupt):
        store.load()


def test_history_duplicate_event_id_is_rejected(tmp_path):
    store = seeded_store(tmp_path)
    values = lines(store)
    first = json.loads(values[0])
    second = json.loads(values[1])
    second["event_id"] = first["event_id"]
    values[1] = json.dumps(second, ensure_ascii=False, separators=(",", ":"))
    write_lines(store, values)

    with pytest.raises(InvariantStoreCorrupt, match="event_id"):
        store.load()


def test_history_reordered_events_are_rejected(tmp_path):
    store = seeded_store(tmp_path)
    values = lines(store)
    write_lines(store, [values[1], values[0]])

    with pytest.raises(InvariantStoreCorrupt, match="revision"):
        store.load()


def test_history_extra_event_is_rejected(tmp_path):
    store = seeded_store(tmp_path)
    values = lines(store)
    extra = json.loads(values[0])
    extra["event_id"] = "evt-extra"
    extra["store_revision"] = 3
    values.append(json.dumps(extra, ensure_ascii=False, separators=(",", ":")))
    write_lines(store, values)

    with pytest.raises(InvariantStoreCorrupt):
        store.load()


def test_public_read_history_also_checks_sequence(tmp_path):
    store = seeded_store(tmp_path)
    write_lines(store, lines(store)[1:])

    with pytest.raises(InvariantStoreCorrupt):
        store.read_history()
