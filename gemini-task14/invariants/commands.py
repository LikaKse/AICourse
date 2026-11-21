"""Явные команды изменения инвариантов.

Этот модуль не должен импортироваться ассистентом/чатом. Штатный путь —
отдельный ``invariant_cli.py``. В одном Python-процессе это API-граница, а
не криптографическая авторизация.

Проверка expected_revision/expected_version и запись выполняются в одной
критической секции ``InvariantStore.locked()``.
"""
from __future__ import annotations

import copy
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Iterable

from .errors import (
    InvariantCommandError,
    InvariantDuplicateError,
    InvariantNotFoundError,
    InvariantRevisionConflict,
    InvariantValidationError,
    InvariantVersionConflict,
)
from .models import InvariantCategory, InvariantChange, InvariantCheck, InvariantRule
from .store import InvariantStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvariantCommandError(f"{field} обязателен и не должен быть пустым")
    return value.strip()


class InvariantCommandService:
    """add/update/disable с обязательным аудитом.

    ``expected_revision`` (если передан) защищает от конкурентного изменения
    всего store. ``expected_version`` обязателен и для update, и для disable.
    """

    def __init__(self, store: InvariantStore):
        self._store = store

    def add(
        self,
        rule: InvariantRule,
        *,
        actor: str,
        reason: str,
        expected_revision: int | None = None,
    ) -> InvariantChange:
        if not isinstance(rule, InvariantRule):
            raise InvariantCommandError("add ожидает InvariantRule")
        actor = _require_text(actor, "actor")
        reason = _require_text(reason, "reason")

        with self._store.locked():
            snapshot = self._store._load_unlocked()
            self._check_revision(snapshot.revision, expected_revision)
            if snapshot.rule(rule.id) is not None:
                raise InvariantDuplicateError(f"инвариант {rule.id!r} уже существует")

            now = _now()
            try:
                created = replace(
                    rule,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
            except ValueError as exc:
                raise InvariantValidationError(str(exc)) from exc
            new_snapshot = replace(
                snapshot,
                revision=snapshot.revision + 1,
                updated_at=now,
                rules=(*snapshot.rules, created),
            )
            change = InvariantChange(
                event_id=_event_id(),
                invariant_id=created.id,
                action="add",
                old=None,
                new=created.to_dict(),
                actor=actor,
                reason=reason,
                at=now,
                store_revision=new_snapshot.revision,
            )
            self._store._commit_unlocked(new_snapshot, change)
            return change

    def update(
        self,
        invariant_id: str,
        *,
        text: str,
        checks: Iterable[InvariantCheck],
        actor: str,
        reason: str,
        expected_version: int,
        category: InvariantCategory | str | None = None,
        expected_revision: int | None = None,
    ) -> InvariantChange:
        actor = _require_text(actor, "actor")
        reason = _require_text(reason, "reason")
        if type(expected_version) is not int:
            raise InvariantCommandError("expected_version обязателен и должен быть int")

        with self._store.locked():
            snapshot = self._store._load_unlocked()
            self._check_revision(snapshot.revision, expected_revision)
            current = snapshot.rule(invariant_id)
            if current is None:
                raise InvariantNotFoundError(f"инвариант {invariant_id!r} не найден")
            if current.version != expected_version:
                raise InvariantVersionConflict(
                    f"версия {current.version} не совпадает с expected_version "
                    f"{expected_version}")

            now = _now()
            try:
                updated = replace(
                    current,
                    category=category if category is not None else current.category,
                    text=text,
                    checks=tuple(checks),
                    version=current.version + 1,
                    updated_at=now,
                )
            except ValueError as exc:
                raise InvariantValidationError(str(exc)) from exc
            # replace не меняет id; это проверяется тестом.
            new_rules = tuple(
                updated if item.id == invariant_id else item
                for item in snapshot.rules
            )
            new_snapshot = replace(
                snapshot,
                revision=snapshot.revision + 1,
                updated_at=now,
                rules=new_rules,
            )
            change = InvariantChange(
                event_id=_event_id(),
                invariant_id=invariant_id,
                action="update",
                old=copy.deepcopy(current.to_dict()),
                new=copy.deepcopy(updated.to_dict()),
                actor=actor,
                reason=reason,
                at=now,
                store_revision=new_snapshot.revision,
            )
            self._store._commit_unlocked(new_snapshot, change)
            return change

    def disable(
        self,
        invariant_id: str,
        *,
        actor: str,
        reason: str,
        expected_version: int,
        expected_revision: int | None = None,
    ) -> InvariantChange:
        actor = _require_text(actor, "actor")
        reason = _require_text(reason, "reason")
        if type(expected_version) is not int:
            raise InvariantCommandError("expected_version обязателен и должен быть int")

        with self._store.locked():
            snapshot = self._store._load_unlocked()
            self._check_revision(snapshot.revision, expected_revision)
            current = snapshot.rule(invariant_id)
            if current is None:
                raise InvariantNotFoundError(f"инвариант {invariant_id!r} не найден")
            if current.version != expected_version:
                raise InvariantVersionConflict(
                    f"версия {current.version} не совпадает с expected_version "
                    f"{expected_version}")
            if not current.active:
                raise InvariantCommandError(f"инвариант {invariant_id!r} уже отключён")

            now = _now()
            disabled = replace(
                current,
                active=False,
                version=current.version + 1,
                updated_at=now,
            )
            new_rules = tuple(
                disabled if item.id == invariant_id else item
                for item in snapshot.rules
            )
            new_snapshot = replace(
                snapshot,
                revision=snapshot.revision + 1,
                updated_at=now,
                rules=new_rules,
            )
            change = InvariantChange(
                event_id=_event_id(),
                invariant_id=invariant_id,
                action="disable",
                old=copy.deepcopy(current.to_dict()),
                new=copy.deepcopy(disabled.to_dict()),
                actor=actor,
                reason=reason,
                at=now,
                store_revision=new_snapshot.revision,
            )
            self._store._commit_unlocked(new_snapshot, change)
            return change

    @staticmethod
    def _check_revision(current: int, expected: int | None) -> None:
        if expected is not None:
            if type(expected) is not int:
                raise InvariantCommandError("expected_revision должен быть int")
            if current != expected:
                raise InvariantRevisionConflict(
                    f"revision {current} не совпадает с expected_revision {expected}")


def _event_id() -> str:
    return f"evt-{uuid.uuid4().hex}"


__all__ = ["InvariantCommandService"]
