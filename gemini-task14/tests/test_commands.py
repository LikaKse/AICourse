"""Команды изменения инвариантов: аудит, версии, запрет обхода."""

import inspect

import pytest

import invariants
from invariants.commands import InvariantCommandService
from invariants.errors import (
    InvariantCommandError,
    InvariantDuplicateError,
    InvariantNotFoundError,
    InvariantRevisionConflict,
    InvariantValidationError,
    InvariantVersionConflict,
)
from invariants.models import (
    InvariantCategory,
    InvariantCheck,
    InvariantRule,
)
from invariants.store import InvariantStore


def make_rule(rule_id="arch-001"):
    return InvariantRule(
        id=rule_id,
        category=InvariantCategory.ARCHITECTURE,
        text="Архитектура: модульный монолит",
        checks=(InvariantCheck(kind="forbidden_import", value="flask"),),
    )


def service(tmp_path):
    return InvariantCommandService(InvariantStore(tmp_path))


def add_rule(svc, rule=None):
    return svc.add(
        rule or make_rule(),
        actor="local-operator",
        reason="Начальное правило",
    )


def test_add_and_read_rule(tmp_path):
    svc = service(tmp_path)
    change = add_rule(svc)
    snapshot = svc._store.load()

    assert snapshot.revision == 1
    assert len(snapshot.rules) == 1
    assert snapshot.rules[0].id == "arch-001"
    assert snapshot.rules[0].version == 1
    assert change.action == "add"
    assert change.old is None
    assert change.new["id"] == "arch-001"


def test_history_records_old_new_id_actor_time_reason(tmp_path):
    svc = service(tmp_path)
    add_rule(svc)
    svc.update(
        "arch-001",
        text="Архитектура: модульный монолит на Python",
        checks=(InvariantCheck(kind="forbidden_import", value="flask"),),
        actor="local-operator",
        reason="Уточнение решения",
        expected_version=1,
    )

    history = svc._store.read_history()
    assert len(history) == 2
    event = history[-1]
    assert event.invariant_id == "arch-001"
    assert event.action == "update"
    assert event.old["text"] == "Архитектура: модульный монолит"
    assert event.new["text"].endswith("на Python")
    assert event.actor == "local-operator"
    assert event.reason == "Уточнение решения"
    assert event.at
    assert event.store_revision == 2


def test_update_increments_version_and_preserves_id(tmp_path):
    svc = service(tmp_path)
    add_rule(svc)
    change = svc.update(
        "arch-001",
        text="Новый текст",
        checks=(InvariantCheck(kind="forbidden_term", value="Docker"),),
        actor="local-operator",
        reason="Новое решение",
        expected_version=1,
    )

    rule = svc._store.load().rule("arch-001")
    assert rule.id == "arch-001"
    assert rule.version == 2
    assert rule.text == "Новый текст"
    assert change.old["id"] == change.new["id"] == "arch-001"


def test_update_requires_expected_version(tmp_path):
    svc = service(tmp_path)
    add_rule(svc)

    with pytest.raises(TypeError):
        svc.update(
            "arch-001",
            text="x",
            checks=(InvariantCheck(kind="forbidden_term", value="x"),),
            actor="operator",
            reason="x",
        )


def test_version_conflict_does_not_write(tmp_path):
    svc = service(tmp_path)
    add_rule(svc)
    before = svc._store.load()

    with pytest.raises(InvariantVersionConflict):
        svc.update(
            "arch-001",
            text="x",
            checks=(InvariantCheck(kind="forbidden_term", value="x"),),
            actor="operator",
            reason="x",
            expected_version=99,
        )

    assert svc._store.load() == before
    assert len(svc._store.read_history()) == 1


def test_revision_conflict_does_not_write(tmp_path):
    svc = service(tmp_path)
    add_rule(svc)

    with pytest.raises(InvariantRevisionConflict):
        svc.add(
            make_rule("arch-002"),
            actor="operator",
            reason="x",
            expected_revision=99,
        )

    assert len(svc._store.load().rules) == 1
    assert len(svc._store.read_history()) == 1


def test_disable_keeps_history_and_rule(tmp_path):
    svc = service(tmp_path)
    add_rule(svc)
    change = svc.disable(
        "arch-001",
        actor="operator",
        reason="Правило больше не действует",
        expected_version=1,
    )

    rule = svc._store.load().rule("arch-001")
    assert rule is not None
    assert rule.active is False
    assert rule.version == 2
    assert change.action == "disable"
    assert change.old["active"] is True
    assert change.new["active"] is False
    assert len(svc._store.read_history()) == 2


@pytest.mark.parametrize("actor,reason", [
    ("", "reason"),
    ("operator", ""),
    (" ", "reason"),
    ("operator", "   "),
])
def test_add_requires_actor_and_reason(tmp_path, actor, reason):
    svc = service(tmp_path)

    with pytest.raises(InvariantCommandError):
        svc.add(make_rule(), actor=actor, reason=reason)

    assert not svc._store.snapshot_path.exists()
    assert not svc._store.history_path.exists()


@pytest.mark.parametrize("actor,reason", [
    ("", "reason"),
    ("operator", ""),
])
def test_update_and_disable_require_actor_and_reason(tmp_path, actor, reason):
    svc = service(tmp_path)
    add_rule(svc)

    with pytest.raises(InvariantCommandError):
        svc.update(
            "arch-001",
            text="x",
            checks=(InvariantCheck(kind="forbidden_term", value="x"),),
            actor=actor,
            reason=reason,
            expected_version=1,
        )
    with pytest.raises(InvariantCommandError):
        svc.disable("arch-001", actor=actor, reason=reason, expected_version=1)

    assert svc._store.load().revision == 1


def test_duplicate_and_unknown_rules(tmp_path):
    svc = service(tmp_path)
    add_rule(svc)

    with pytest.raises(InvariantDuplicateError):
        add_rule(svc)

    with pytest.raises(InvariantNotFoundError):
        svc.update(
            "missing",
            text="x",
            checks=(InvariantCheck(kind="forbidden_term", value="x"),),
            actor="operator",
            reason="x",
            expected_version=1,
        )

    with pytest.raises(InvariantNotFoundError):
        svc.disable("missing", actor="operator", reason="x", expected_version=1)


def test_disable_rejects_non_integer_expected_version(tmp_path):
    svc = service(tmp_path)
    add_rule(svc)

    with pytest.raises(InvariantCommandError):
        svc.disable("arch-001", actor="operator", reason="x", expected_version=True)

    assert svc._store.load().rule("arch-001").active is True


def test_disable_requires_expected_version(tmp_path):
    svc = service(tmp_path)
    add_rule(svc)

    with pytest.raises(TypeError):
        svc.disable("arch-001", actor="operator", reason="x")


def test_update_to_active_without_checks_is_rejected(tmp_path):
    svc = service(tmp_path)
    add_rule(svc)

    with pytest.raises(InvariantValidationError):
        svc.update(
            "arch-001",
            text="x",
            checks=(),
            actor="operator",
            reason="x",
            expected_version=1,
        )

    assert svc._store.load().revision == 1


def test_history_is_append_only_and_has_no_delete_api(tmp_path):
    svc = service(tmp_path)
    add_rule(svc)
    svc.update(
        "arch-001",
        text="x",
        checks=(InvariantCheck(kind="forbidden_term", value="x"),),
        actor="operator",
        reason="x",
        expected_version=1,
    )

    history = svc._store.read_history()
    assert len(history) == 2
    assert not hasattr(svc._store, "delete_history")
    assert not hasattr(svc._store, "clear_history")


def test_commands_are_not_exported_by_package():
    # Это API-разграничение, а не криптографическая защита: подмодуль
    # commands по-прежнему можно импортировать напрямую.
    assert "InvariantCommandService" not in vars(invariants)
    assert "commands" not in getattr(invariants, "__all__", ())


def test_update_signature_has_required_expected_version():
    signature = inspect.signature(InvariantCommandService.update)
    parameter = signature.parameters["expected_version"]
    assert parameter.default is inspect.Parameter.empty
