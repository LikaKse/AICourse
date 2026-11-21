"""Модели инвариантов: обязательные структурные проверки и сериализация."""

import pytest

from invariants.models import (
    InvariantCategory,
    InvariantCheck,
    InvariantRule,
    InvariantSnapshot,
)


def rule(rule_id="arch-001", *, active=True, checks=None):
    return InvariantRule(
        id=rule_id,
        category=InvariantCategory.ARCHITECTURE,
        text="Сервис остаётся модульным монолитом",
        checks=checks if checks is not None else (
            InvariantCheck(kind="forbidden_import", value="flask"),
        ),
        active=active,
    )


def test_active_rule_requires_at_least_one_structured_check():
    with pytest.raises(ValueError, match="хотя бы одной проверки"):
        rule(checks=())

    disabled = rule(active=False, checks=())
    assert disabled.checks == ()


@pytest.mark.parametrize("kind", ["always_true", "always_false", "noop", ""])
def test_unknown_or_fake_check_kinds_are_rejected(kind):
    with pytest.raises(ValueError, match="неизвестный вид"):
        InvariantCheck(kind=kind, value="x")


@pytest.mark.parametrize("kind,value", [
    ("forbidden_term", ""),
    ("forbidden_term", "   "),
    ("forbidden_regex", "["),
    ("required_regex", "("),
    ("forbidden_import", "not-a-module"),
    ("forbidden_import", ""),
])
def test_invalid_check_values_are_rejected(kind, value):
    with pytest.raises(ValueError):
        InvariantCheck(kind=kind, value=value)


def test_check_roundtrip():
    check = InvariantCheck(
        kind="forbidden_regex", value=r"\bDocker\b", case_sensitive=True)
    assert InvariantCheck.from_dict(check.to_dict()) == check


def test_snapshot_roundtrip_and_unique_ids():
    snapshot = InvariantSnapshot(
        revision=2,
        updated_at="2026-11-19T10:00:00+00:00",
        rules=(rule(), rule("arch-002", checks=(
            InvariantCheck(kind="forbidden_term", value="Docker"),))),
    )

    restored = InvariantSnapshot.from_dict(snapshot.to_dict())
    assert restored == snapshot
    assert restored.rule("arch-001") == snapshot.rules[0]


def test_snapshot_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="повторяющиеся"):
        InvariantSnapshot(rules=(rule(), rule()))


def test_snapshot_rejects_bool_format_version():
    with pytest.raises(ValueError, match="версия"):
        InvariantSnapshot(format_version=True)


def test_rule_rejects_unknown_category():
    with pytest.raises(ValueError, match="категория"):
        InvariantRule(
            id="arch-001",
            category="unknown",
            text="x",
            checks=(InvariantCheck(kind="forbidden_term", value="x"),),
        )
