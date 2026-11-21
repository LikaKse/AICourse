"""Модели инвариантов Дня 14.

Инварианты — это отдельные от диалога правила. Активное правило обязано
содержать хотя бы одну структурированную программную проверку. Текст правила
предназначен для промпта, но авторитетом является только ``checks``.
"""
from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class InvariantCategory(str, Enum):
    ARCHITECTURE = "architecture"
    DECISION = "decision"
    STACK = "stack"
    BUSINESS = "business"


#: Разрешённые виды программных проверок. Никаких "always_true"/"always_false":
#: такой вид не входит в набор и будет отвергнут.
CHECK_KINDS = frozenset({
    "forbidden_term",
    "forbidden_regex",
    "required_regex",
    "forbidden_import",
    "max_value",
})

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_IMPORT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _require_nonempty_str(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: ожидается непустая строка")
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{where}: неизвестные поля {sorted(unknown)}")


def _copy_optional_mapping(value: object, where: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{where}: ожидается объект или null")
    return copy.deepcopy(dict(value))


@dataclass(frozen=True)
class InvariantCheck:
    """Одна структурированная проверка.

    Виды:
    - forbidden_term: запрещённая подстрока;
    - forbidden_regex / required_regex: регулярное выражение;
    - forbidden_import: запрещённый импорт в Python-коде;
    - max_value: числовое ограничение по ключу.
    """

    kind: str
    value: str | int | float
    key: str | None = None
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        if self.kind not in CHECK_KINDS:
            raise ValueError(f"InvariantCheck: неизвестный вид проверки {self.kind!r}")
        if type(self.case_sensitive) is not bool:
            raise ValueError("InvariantCheck: case_sensitive должен быть bool")

        if self.kind in {"forbidden_term", "forbidden_regex", "required_regex"}:
            _require_nonempty_str(self.value, "InvariantCheck.value")
            if not isinstance(self.value, str):
                raise ValueError(f"InvariantCheck({self.kind}): value должен быть строкой")
            if self.kind.endswith("regex"):
                try:
                    re.compile(self.value)
                except re.error as exc:
                    raise ValueError(
                        f"InvariantCheck: некорректный regex {self.value!r}: {exc}") from exc
            if self.key is not None:
                raise ValueError(f"InvariantCheck({self.kind}): key недопустим")
        elif self.kind == "forbidden_import":
            if not isinstance(self.value, str) or not _IMPORT_RE.fullmatch(self.value):
                raise ValueError(
                    "InvariantCheck(forbidden_import): value должен быть именем модуля")
            if self.key is not None:
                raise ValueError("InvariantCheck(forbidden_import): key недопустим")
        elif self.kind == "max_value":
            if type(self.value) not in {int, float} or isinstance(self.value, bool):
                raise ValueError("InvariantCheck(max_value): value должен быть числом")
            _require_nonempty_str(self.key, "InvariantCheck.key")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "key": self.key,
            "case_sensitive": self.case_sensitive,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InvariantCheck":
        if not isinstance(data, Mapping):
            raise ValueError("InvariantCheck: ожидается объект")
        _reject_unknown(
            data, {"kind", "value", "key", "case_sensitive"}, "InvariantCheck")
        return cls(
            kind=data.get("kind"),  # type: ignore[arg-type]
            value=data.get("value"),
            key=data.get("key"),
            case_sensitive=data.get("case_sensitive", False),
        )


@dataclass(frozen=True)
class InvariantRule:
    id: str
    category: InvariantCategory
    text: str
    checks: tuple[InvariantCheck, ...]
    active: bool = True
    version: int = 1
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _ID_RE.fullmatch(self.id):
            raise ValueError(
                "InvariantRule: id должен соответствовать [a-z0-9][a-z0-9._-]{0,63}")
        category = self.category
        if not isinstance(category, InvariantCategory):
            if not isinstance(category, str):
                raise ValueError(
                    f"InvariantRule: неизвестная категория {self.category!r}")
            try:
                category = InvariantCategory(category)
            except ValueError as exc:
                raise ValueError(
                    f"InvariantRule: неизвестная категория {self.category!r}") from exc
            object.__setattr__(self, "category", category)

        _require_nonempty_str(self.text, "InvariantRule.text")
        if type(self.active) is not bool:
            raise ValueError("InvariantRule: active должен быть bool")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("InvariantRule: version должен быть целым >= 1")
        if not isinstance(self.checks, tuple):
            object.__setattr__(self, "checks", tuple(self.checks))
        if any(not isinstance(item, InvariantCheck) for item in self.checks):
            raise ValueError("InvariantRule: checks должен содержать InvariantCheck")
        if self.active and not self.checks:
            raise ValueError(
                "InvariantRule: активное правило требует хотя бы одной проверки")
        if not isinstance(self.created_at, str) or not isinstance(self.updated_at, str):
            raise ValueError("InvariantRule: created_at/updated_at должны быть строками")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category.value,
            "text": self.text,
            "checks": [item.to_dict() for item in self.checks],
            "active": self.active,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InvariantRule":
        if not isinstance(data, Mapping):
            raise ValueError("InvariantRule: ожидается объект")
        _reject_unknown(
            data,
            {"id", "category", "text", "checks", "active", "version",
             "created_at", "updated_at"},
            "InvariantRule",
        )
        raw_checks = data.get("checks", ())
        if not isinstance(raw_checks, (list, tuple)):
            raise ValueError("InvariantRule: checks должен быть списком")
        return cls(
            id=data.get("id"),  # type: ignore[arg-type]
            category=data.get("category"),  # type: ignore[arg-type]
            text=data.get("text"),  # type: ignore[arg-type]
            checks=tuple(
                item if isinstance(item, InvariantCheck)
                else InvariantCheck.from_dict(item)
                for item in raw_checks
            ),
            active=data.get("active", True),
            version=data.get("version", 1),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass(frozen=True)
class InvariantSnapshot:
    format_version: int = 1
    revision: int = 0
    updated_at: str = ""
    rules: tuple[InvariantRule, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if type(self.format_version) is not int or self.format_version != 1:
            raise ValueError(
                f"InvariantSnapshot: неподдерживаемая версия {self.format_version!r}")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("InvariantSnapshot: revision должен быть целым >= 0")
        if not isinstance(self.rules, tuple):
            object.__setattr__(self, "rules", tuple(self.rules))
        if any(not isinstance(item, InvariantRule) for item in self.rules):
            raise ValueError("InvariantSnapshot: rules должен содержать InvariantRule")
        ids = [item.id for item in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("InvariantSnapshot: повторяющиеся id правил")
        if not isinstance(self.updated_at, str):
            raise ValueError("InvariantSnapshot: updated_at должен быть строкой")

    def rule(self, invariant_id: str) -> InvariantRule | None:
        for item in self.rules:
            if item.id == invariant_id:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "revision": self.revision,
            "updated_at": self.updated_at,
            "rules": [item.to_dict() for item in self.rules],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InvariantSnapshot":
        if not isinstance(data, Mapping):
            raise ValueError("InvariantSnapshot: ожидается объект")
        _reject_unknown(
            data, {"format_version", "revision", "updated_at", "rules"},
            "InvariantSnapshot")
        raw_rules = data.get("rules", ())
        if not isinstance(raw_rules, (list, tuple)):
            raise ValueError("InvariantSnapshot: rules должен быть списком")
        return cls(
            format_version=data.get("format_version", 1),
            revision=data.get("revision", 0),
            updated_at=data.get("updated_at", ""),
            rules=tuple(
                item if isinstance(item, InvariantRule)
                else InvariantRule.from_dict(item)
                for item in raw_rules
            ),
        )


@dataclass(frozen=True)
class InvariantChange:
    """Событие изменения инварианта для append-only истории."""

    event_id: str
    invariant_id: str
    action: str
    old: dict[str, Any] | None
    new: dict[str, Any] | None
    actor: str
    reason: str
    at: str
    store_revision: int

    def __post_init__(self) -> None:
        _require_nonempty_str(self.event_id, "InvariantChange.event_id")
        _require_nonempty_str(self.invariant_id, "InvariantChange.invariant_id")
        if self.action not in {"add", "update", "disable"}:
            raise ValueError(f"InvariantChange: неизвестное действие {self.action!r}")
        _require_nonempty_str(self.actor, "InvariantChange.actor")
        _require_nonempty_str(self.reason, "InvariantChange.reason")
        _require_nonempty_str(self.at, "InvariantChange.at")
        if type(self.store_revision) is not int or self.store_revision < 1:
            raise ValueError("InvariantChange: store_revision должен быть >= 1")
        object.__setattr__(
            self, "old", _copy_optional_mapping(self.old, "InvariantChange.old"))
        object.__setattr__(
            self, "new", _copy_optional_mapping(self.new, "InvariantChange.new"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "invariant_id": self.invariant_id,
            "action": self.action,
            "old": copy.deepcopy(self.old),
            "new": copy.deepcopy(self.new),
            "actor": self.actor,
            "reason": self.reason,
            "at": self.at,
            "store_revision": self.store_revision,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InvariantChange":
        if not isinstance(data, Mapping):
            raise ValueError("InvariantChange: ожидается объект")
        _reject_unknown(
            data,
            {"event_id", "invariant_id", "action", "old", "new", "actor",
             "reason", "at", "store_revision"},
            "InvariantChange",
        )
        return cls(
            event_id=data.get("event_id"),  # type: ignore[arg-type]
            invariant_id=data.get("invariant_id"),  # type: ignore[arg-type]
            action=data.get("action"),  # type: ignore[arg-type]
            old=data.get("old"),
            new=data.get("new"),
            actor=data.get("actor"),  # type: ignore[arg-type]
            reason=data.get("reason"),  # type: ignore[arg-type]
            at=data.get("at"),  # type: ignore[arg-type]
            store_revision=data.get("store_revision"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class Violation:
    invariant_id: str
    check_kind: str
    message: str

    def __post_init__(self) -> None:
        _require_nonempty_str(self.invariant_id, "Violation.invariant_id")
        _require_nonempty_str(self.check_kind, "Violation.check_kind")
        _require_nonempty_str(self.message, "Violation.message")

    def to_dict(self) -> dict[str, str]:
        return {
            "invariant_id": self.invariant_id,
            "check_kind": self.check_kind,
            "message": self.message,
        }


__all__ = [
    "CHECK_KINDS",
    "InvariantCategory",
    "InvariantChange",
    "InvariantCheck",
    "InvariantRule",
    "InvariantSnapshot",
    "Violation",
]
