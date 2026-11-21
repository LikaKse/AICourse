"""Модели памяти Дня 15 и их базовая валидация.

Здесь нет store, service, разрешения конфликтов и сборки prompt. Модели
описывают только форму данных и структурные инварианты:

- Fact: активный факт (mutable / cumulative / temporary);
- Tombstone: последнее удаление ключа и его время;
- TaskCompletion: явное завершение задачи;
- MemoryEvent: запись журнала изменений;
- MemorySnapshot: активные факты + tombstones + завершения + журнал.

История и tombstones не являются частью активного контекста. Это будет
обеспечено ContextBuilder на следующем этапе.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, TypeAlias

from .errors import MemoryValidationError

FORMAT_VERSION = 1
MAX_KEY_LENGTH = 128
MAX_TASK_ID_LENGTH = 64
MAX_ENTRY_ID_LENGTH = 64
MAX_EVENT_ID_LENGTH = 64
MAX_SOURCE_LENGTH = 64
MAX_REASON_LENGTH = 500
MAX_JSON_DEPTH = 32
MAX_JSON_STRING = 100_000
MAX_JSON_KEY = 256

_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_TASK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_ENTRY_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_EVENT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SOURCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,63}$")

JsonValue: TypeAlias = (
    str | int | float | bool | None
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)


class FactKind(str, Enum):
    MUTABLE = "mutable"
    CUMULATIVE = "cumulative"
    TEMPORARY = "temporary"


class MemoryAction(str, Enum):
    SET = "set"
    APPEND = "append"
    DELETE = "delete"
    COMPLETE_TASK = "complete_task"
    REOPEN_TASK = "reopen_task"


class EventResult(str, Enum):
    APPLIED = "applied"
    UNCHANGED = "unchanged"
    REJECTED_STALE = "rejected_stale"
    REJECTED_EQUAL = "rejected_equal"
    REJECTED_TASK_COMPLETED = "rejected_task_completed"
    NOT_FOUND = "not_found"


def _validate_key(value: object, field_name: str = "key") -> str:
    if not isinstance(value, str) or not _KEY_RE.fullmatch(value):
        raise MemoryValidationError(
            f"{field_name}: ожидается ключ [a-z0-9][a-z0-9._-]{{0,127}}")
    return value


def _validate_task_id(value: object, field_name: str = "task_id") -> str:
    if not isinstance(value, str) or not _TASK_ID_RE.fullmatch(value):
        raise MemoryValidationError(
            f"{field_name}: ожидается [a-z0-9][a-z0-9._-]{{0,63}}")
    return value


def _validate_entry_id(value: object, field_name: str = "entry_id") -> str:
    if not isinstance(value, str) or not _ENTRY_ID_RE.fullmatch(value):
        raise MemoryValidationError(
            f"{field_name}: ожидается [a-z0-9][a-z0-9._-]{{0,63}}")
    return value


def _validate_event_id(value: object, field_name: str = "event_id") -> str:
    if not isinstance(value, str) or not _EVENT_ID_RE.fullmatch(value):
        raise MemoryValidationError(
            f"{field_name}: ожидается [a-z0-9][a-z0-9._-]{{0,63}}")
    return value


def _validate_source(value: object, field_name: str = "source") -> str:
    if not isinstance(value, str) or not _SOURCE_RE.fullmatch(value):
        raise MemoryValidationError(
            f"{field_name}: ожидается [A-Za-z0-9][A-Za-z0-9_.:@-]{{0,63}}")
    return value


def _validate_reason(value: object, field_name: str = "reason") -> str:
    if not isinstance(value, str):
        raise MemoryValidationError(f"{field_name}: ожидается строка")
    value = value.strip()
    if len(value) > MAX_REASON_LENGTH:
        raise MemoryValidationError(
            f"{field_name}: длина не должна превышать {MAX_REASON_LENGTH}")
    return value


def _validate_datetime(value: object, field_name: str) -> datetime:
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            value = datetime.fromisoformat(text)
        except ValueError as exc:
            raise MemoryValidationError(
                f"{field_name}: некорректный ISO timestamp {value!r}") from exc
    if not isinstance(value, datetime):
        raise MemoryValidationError(f"{field_name}: ожидается datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise MemoryValidationError(
            f"{field_name}: timestamp должен быть timezone-aware")
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _validate_json_value(
    value: Any,
    *,
    path: str,
    depth: int = 0,
    seen: set[int] | None = None,
) -> JsonValue:
    if depth > MAX_JSON_DEPTH:
        raise MemoryValidationError(
            f"{path}: превышена максимальная глубина {MAX_JSON_DEPTH}")
    if seen is None:
        seen = set()

    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise MemoryValidationError(f"{path}: NaN/Infinity недопустимы")
        return value
    if isinstance(value, str):
        if len(value) > MAX_JSON_STRING:
            raise MemoryValidationError(
                f"{path}: строка длиннее {MAX_JSON_STRING} символов")
        return value
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in seen:
            raise MemoryValidationError(f"{path}: циклическая структура")
        seen.add(identity)
        try:
            return [
                _validate_json_value(
                    item, path=f"{path}[{index}]", depth=depth + 1, seen=seen)
                for index, item in enumerate(value)
            ]
        finally:
            seen.remove(identity)
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            raise MemoryValidationError(f"{path}: циклическая структура")
        seen.add(identity)
        try:
            result: dict[str, JsonValue] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise MemoryValidationError(
                        f"{path}: ключи JSON-объекта должны быть строками")
                if len(key) > MAX_JSON_KEY:
                    raise MemoryValidationError(
                        f"{path}: ключ длиннее {MAX_JSON_KEY} символов")
                result[key] = _validate_json_value(
                    item, path=f"{path}.{key}", depth=depth + 1, seen=seen)
            return result
        finally:
            seen.remove(identity)
    raise MemoryValidationError(
        f"{path}: значение должно быть JSON-совместимым")


def _require_mapping(
    value: object, allowed: frozenset[str], where: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MemoryValidationError(f"{where}: ожидается JSON-объект")
    unknown = set(value) - allowed
    if unknown:
        raise MemoryValidationError(
            f"{where}: неизвестные поля {sorted(unknown)}")
    return value


def _require_present(data: Mapping[str, Any], field_name: str, where: str) -> Any:
    if field_name not in data:
        raise MemoryValidationError(f"{where}: отсутствует поле {field_name!r}")
    return data[field_name]


def _coerce_kind(value: object, field_name: str) -> FactKind:
    if isinstance(value, FactKind):
        return value
    if isinstance(value, str):
        try:
            return FactKind(value)
        except ValueError:
            pass
    raise MemoryValidationError(
        f"{field_name}: неизвестный тип факта {value!r}")


def _coerce_action(value: object) -> MemoryAction:
    if isinstance(value, MemoryAction):
        return value
    if isinstance(value, str):
        try:
            return MemoryAction(value)
        except ValueError:
            pass
    raise MemoryValidationError(f"MemoryEvent.action: неизвестное действие {value!r}")


def _coerce_result(value: object) -> EventResult:
    if isinstance(value, EventResult):
        return value
    if isinstance(value, str):
        try:
            return EventResult(value)
        except ValueError:
            pass
    raise MemoryValidationError(f"MemoryEvent.result: неизвестный результат {value!r}")


@dataclass(frozen=True)
class Fact:
    """Активный факт памяти."""

    kind: FactKind
    value: JsonValue
    recorded_at: datetime
    source: str
    reason: str = ""
    key: str | None = None
    task_id: str | None = None
    entry_id: str | None = None
    version: int = 1

    def __post_init__(self) -> None:
        kind = _coerce_kind(self.kind, "Fact.kind")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self, "value",
            _validate_json_value(self.value, path="Fact.value"))
        object.__setattr__(
            self, "recorded_at",
            _validate_datetime(self.recorded_at, "Fact.recorded_at"))
        object.__setattr__(
            self, "source", _validate_source(self.source, "Fact.source"))
        object.__setattr__(
            self, "reason", _validate_reason(self.reason, "Fact.reason"))

        if type(self.version) is not int or self.version < 1:
            raise MemoryValidationError("Fact.version: ожидается int >= 1")

        if self.key is not None:
            object.__setattr__(
                self, "key", _validate_key(self.key, "Fact.key"))
        if self.task_id is not None:
            object.__setattr__(
                self, "task_id", _validate_task_id(self.task_id, "Fact.task_id"))
        if self.entry_id is not None:
            object.__setattr__(
                self, "entry_id",
                _validate_entry_id(self.entry_id, "Fact.entry_id"))

        if kind is FactKind.MUTABLE:
            if self.key is None:
                raise MemoryValidationError("Fact(mutable): требуется key")
            if self.task_id is not None:
                raise MemoryValidationError("Fact(mutable): task_id недопустим")
            if self.entry_id is not None:
                raise MemoryValidationError("Fact(mutable): entry_id недопустим")
        elif kind is FactKind.TEMPORARY:
            if self.key is None:
                raise MemoryValidationError("Fact(temporary): требуется key")
            if self.task_id is None:
                raise MemoryValidationError(
                    "Fact(temporary): требуется task_id")
            if self.entry_id is not None:
                raise MemoryValidationError("Fact(temporary): entry_id недопустим")
        else:  # CUMULATIVE
            if self.key is not None:
                raise MemoryValidationError("Fact(cumulative): key недопустим")
            if self.task_id is not None:
                raise MemoryValidationError(
                    "Fact(cumulative): task_id недопустим")
            if self.entry_id is None:
                raise MemoryValidationError(
                    "Fact(cumulative): требуется entry_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "value": _validate_json_value(self.value, path="Fact.value"),
            "recorded_at": _iso(self.recorded_at),
            "source": self.source,
            "reason": self.reason,
            "key": self.key,
            "task_id": self.task_id,
            "entry_id": self.entry_id,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Fact":
        data = _require_mapping(
            data,
            frozenset({
                "kind", "value", "recorded_at", "source", "reason",
                "key", "task_id", "entry_id", "version",
            }),
            "Fact",
        )
        return cls(
            kind=_require_present(data, "kind", "Fact"),
            value=_require_present(data, "value", "Fact"),
            recorded_at=_require_present(data, "recorded_at", "Fact"),
            source=_require_present(data, "source", "Fact"),
            reason=data.get("reason", ""),
            key=data.get("key"),
            task_id=data.get("task_id"),
            entry_id=data.get("entry_id"),
            version=data.get("version", 1),
        )


@dataclass(frozen=True)
class Tombstone:
    """Последнее удаление ключа. Поздняя запись может восстановить факт."""

    kind: FactKind
    key: str
    deleted_at: datetime
    source: str
    reason: str = ""
    task_id: str | None = None

    def __post_init__(self) -> None:
        kind = _coerce_kind(self.kind, "Tombstone.kind")
        if kind not in {FactKind.MUTABLE, FactKind.TEMPORARY}:
            raise MemoryValidationError(
                "Tombstone: допустимы только mutable/temporary")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self, "key", _validate_key(self.key, "Tombstone.key"))
        object.__setattr__(
            self, "deleted_at",
            _validate_datetime(self.deleted_at, "Tombstone.deleted_at"))
        object.__setattr__(
            self, "source", _validate_source(self.source, "Tombstone.source"))
        object.__setattr__(
            self, "reason", _validate_reason(self.reason, "Tombstone.reason"))

        if kind is FactKind.MUTABLE and self.task_id is not None:
            raise MemoryValidationError(
                "Tombstone(mutable): task_id недопустим")
        if kind is FactKind.TEMPORARY:
            if self.task_id is None:
                raise MemoryValidationError(
                    "Tombstone(temporary): требуется task_id")
            object.__setattr__(
                self, "task_id",
                _validate_task_id(self.task_id, "Tombstone.task_id"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "key": self.key,
            "deleted_at": _iso(self.deleted_at),
            "source": self.source,
            "reason": self.reason,
            "task_id": self.task_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Tombstone":
        data = _require_mapping(
            data,
            frozenset({
                "kind", "key", "deleted_at", "source", "reason", "task_id",
            }),
            "Tombstone",
        )
        return cls(
            kind=_require_present(data, "kind", "Tombstone"),
            key=_require_present(data, "key", "Tombstone"),
            deleted_at=_require_present(data, "deleted_at", "Tombstone"),
            source=_require_present(data, "source", "Tombstone"),
            reason=data.get("reason", ""),
            task_id=data.get("task_id"),
        )


@dataclass(frozen=True)
class TaskCompletion:
    """Явное завершение задачи и его время."""

    task_id: str
    completed_at: datetime
    source: str
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "task_id",
            _validate_task_id(self.task_id, "TaskCompletion.task_id"))
        object.__setattr__(
            self, "completed_at",
            _validate_datetime(self.completed_at, "TaskCompletion.completed_at"))
        object.__setattr__(
            self, "source",
            _validate_source(self.source, "TaskCompletion.source"))
        object.__setattr__(
            self, "reason",
            _validate_reason(self.reason, "TaskCompletion.reason"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "completed_at": _iso(self.completed_at),
            "source": self.source,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskCompletion":
        data = _require_mapping(
            data,
            frozenset({"task_id", "completed_at", "source", "reason"}),
            "TaskCompletion",
        )
        return cls(
            task_id=_require_present(data, "task_id", "TaskCompletion"),
            completed_at=_require_present(data, "completed_at", "TaskCompletion"),
            source=_require_present(data, "source", "TaskCompletion"),
            reason=data.get("reason", ""),
        )


@dataclass(frozen=True)
class MemoryEvent:
    """Запись журнала. В prompt не попадает."""

    event_id: str
    revision: int
    action: MemoryAction
    at: datetime
    source: str
    kind: FactKind | None = None
    key: str | None = None
    task_id: str | None = None
    entry_id: str | None = None
    old_value: JsonValue | None = None
    new_value: JsonValue | None = None
    old_present: bool = False
    new_present: bool = False
    old_recorded_at: datetime | None = None
    new_recorded_at: datetime | None = None
    reason: str = ""
    result: EventResult = EventResult.APPLIED

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "event_id",
            _validate_event_id(self.event_id, "MemoryEvent.event_id"))
        if type(self.revision) is not int or self.revision < 1:
            raise MemoryValidationError("MemoryEvent.revision: ожидается int >= 1")
        action = _coerce_action(self.action)
        result = _coerce_result(self.result)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "result", result)
        object.__setattr__(
            self, "at", _validate_datetime(self.at, "MemoryEvent.at"))
        object.__setattr__(
            self, "source", _validate_source(self.source, "MemoryEvent.source"))
        object.__setattr__(
            self, "reason", _validate_reason(self.reason, "MemoryEvent.reason"))

        if self.kind is not None:
            object.__setattr__(
                self, "kind",
                _coerce_kind(self.kind, "MemoryEvent.kind"))
        if self.key is not None:
            object.__setattr__(
                self, "key", _validate_key(self.key, "MemoryEvent.key"))
        if self.task_id is not None:
            object.__setattr__(
                self, "task_id",
                _validate_task_id(self.task_id, "MemoryEvent.task_id"))
        if self.entry_id is not None:
            object.__setattr__(
                self, "entry_id",
                _validate_entry_id(self.entry_id, "MemoryEvent.entry_id"))

        if type(self.old_present) is not bool:
            raise MemoryValidationError("MemoryEvent.old_present: ожидается bool")
        if type(self.new_present) is not bool:
            raise MemoryValidationError("MemoryEvent.new_present: ожидается bool")
        if not self.old_present and self.old_value is not None:
            raise MemoryValidationError(
                "MemoryEvent: old_value задан при old_present=False")
        if not self.new_present and self.new_value is not None:
            raise MemoryValidationError(
                "MemoryEvent: new_value задан при new_present=False")
        if self.old_present:
            object.__setattr__(
                self, "old_value",
                _validate_json_value(self.old_value, path="MemoryEvent.old_value"))
            if self.old_recorded_at is None:
                raise MemoryValidationError(
                    "MemoryEvent: при old_present=True требуется old_recorded_at")
        elif self.old_recorded_at is not None:
            raise MemoryValidationError(
                "MemoryEvent: old_recorded_at задан при old_present=False")
        if self.new_present:
            object.__setattr__(
                self, "new_value",
                _validate_json_value(self.new_value, path="MemoryEvent.new_value"))

        if self.old_recorded_at is not None:
            object.__setattr__(
                self, "old_recorded_at",
                _validate_datetime(
                    self.old_recorded_at, "MemoryEvent.old_recorded_at"))
        if self.new_recorded_at is not None:
            object.__setattr__(
                self, "new_recorded_at",
                _validate_datetime(
                    self.new_recorded_at, "MemoryEvent.new_recorded_at"))

        if action is MemoryAction.SET:
            if self.kind not in {FactKind.MUTABLE, FactKind.TEMPORARY}:
                raise MemoryValidationError(
                    "MemoryEvent(set): kind должен быть mutable/temporary")
            if self.key is None:
                raise MemoryValidationError("MemoryEvent(set): требуется key")
            if self.entry_id is not None:
                raise MemoryValidationError(
                    "MemoryEvent(set): entry_id недопустим")
            if not self.new_present:
                raise MemoryValidationError(
                    "MemoryEvent(set): требуется new_present=True")
            if self.new_recorded_at is None:
                raise MemoryValidationError(
                    "MemoryEvent(set): требуется new_recorded_at")
            self._validate_scope()
            if result not in {
                EventResult.APPLIED,
                EventResult.UNCHANGED,
                EventResult.REJECTED_STALE,
                EventResult.REJECTED_EQUAL,
                EventResult.REJECTED_TASK_COMPLETED,
            }:
                raise MemoryValidationError(
                    f"MemoryEvent(set): недопустимый result {result.value!r}")
            # rejected_stale/rejected_equal могут не иметь old-факта:
            # попытку восстановления мог отклонить tombstone.
            if result is EventResult.UNCHANGED and not self.old_present:
                raise MemoryValidationError(
                    "MemoryEvent(set,unchanged): требуется old_present=True")
        elif action is MemoryAction.DELETE:
            if self.kind not in {FactKind.MUTABLE, FactKind.TEMPORARY}:
                raise MemoryValidationError(
                    "MemoryEvent(delete): kind должен быть mutable/temporary")
            if self.key is None:
                raise MemoryValidationError("MemoryEvent(delete): требуется key")
            if self.entry_id is not None:
                raise MemoryValidationError(
                    "MemoryEvent(delete): entry_id недопустим")
            if self.new_present:
                raise MemoryValidationError(
                    "MemoryEvent(delete): new_present должен быть False")
            if self.new_value is not None:
                raise MemoryValidationError(
                    "MemoryEvent(delete): new_value должен быть None")
            if self.new_recorded_at is None:
                raise MemoryValidationError(
                    "MemoryEvent(delete): требуется new_recorded_at (время удаления)")
            self._validate_scope()
            if result not in {
                EventResult.APPLIED,
                EventResult.REJECTED_STALE,
                EventResult.REJECTED_EQUAL,
                EventResult.NOT_FOUND,
            }:
                raise MemoryValidationError(
                    f"MemoryEvent(delete): недопустимый result {result.value!r}")
            if result is EventResult.NOT_FOUND and self.old_present:
                raise MemoryValidationError(
                    "MemoryEvent(delete,not_found): old_present должен быть False")
            if result is not EventResult.NOT_FOUND and not self.old_present:
                raise MemoryValidationError(
                    f"MemoryEvent(delete,{result.value}): требуется old_present=True")
        elif action is MemoryAction.APPEND:
            if self.kind is not FactKind.CUMULATIVE:
                raise MemoryValidationError(
                    "MemoryEvent(append): kind должен быть cumulative")
            if self.key is not None:
                raise MemoryValidationError("MemoryEvent(append): key недопустим")
            if self.task_id is not None:
                raise MemoryValidationError(
                    "MemoryEvent(append): task_id недопустим")
            if self.entry_id is None:
                raise MemoryValidationError(
                    "MemoryEvent(append): требуется entry_id")
            if self.old_present:
                raise MemoryValidationError(
                    "MemoryEvent(append): old_present должен быть False")
            if not self.new_present:
                raise MemoryValidationError(
                    "MemoryEvent(append): требуется new_present=True")
            if self.new_recorded_at is None:
                raise MemoryValidationError(
                    "MemoryEvent(append): требуется new_recorded_at")
            if result is not EventResult.APPLIED:
                raise MemoryValidationError(
                    "MemoryEvent(append): result должен быть applied")
        elif action in {MemoryAction.REOPEN_TASK, MemoryAction.COMPLETE_TASK}:
            name = action.value
            if self.kind is not None:
                raise MemoryValidationError(
                    f"MemoryEvent({name}): kind недопустим")
            if self.key is not None:
                raise MemoryValidationError(
                    f"MemoryEvent({name}): key недопустим")
            if self.entry_id is not None:
                raise MemoryValidationError(
                    f"MemoryEvent({name}): entry_id недопустим")
            if self.task_id is None:
                raise MemoryValidationError(
                    f"MemoryEvent({name}): требуется task_id")
            if self.old_present or self.new_present:
                raise MemoryValidationError(
                    f"MemoryEvent({name}): old/new недопустимы")
            if self.old_value is not None or self.new_value is not None:
                raise MemoryValidationError(
                    f"MemoryEvent({name}): old/new value недопустимы")
            if self.old_recorded_at is not None or self.new_recorded_at is not None:
                raise MemoryValidationError(
                    f"MemoryEvent({name}): old/new recorded_at недопустимы")
            if result is not EventResult.APPLIED:
                raise MemoryValidationError(
                    f"MemoryEvent({name}): result должен быть applied")

    def _validate_scope(self) -> None:
        if self.kind is FactKind.TEMPORARY:
            if self.task_id is None:
                raise MemoryValidationError(
                    "MemoryEvent(temporary): требуется task_id")
        elif self.task_id is not None:
            raise MemoryValidationError(
                "MemoryEvent(mutable): task_id недопустим")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "revision": self.revision,
            "action": self.action.value,
            "at": _iso(self.at),
            "source": self.source,
            "kind": self.kind.value if self.kind is not None else None,
            "key": self.key,
            "task_id": self.task_id,
            "entry_id": self.entry_id,
            "old_value": (
                _validate_json_value(self.old_value, path="MemoryEvent.old_value")
                if self.old_present else None
            ),
            "new_value": (
                _validate_json_value(self.new_value, path="MemoryEvent.new_value")
                if self.new_present else None
            ),
            "old_present": self.old_present,
            "new_present": self.new_present,
            "old_recorded_at": (
                _iso(self.old_recorded_at)
                if self.old_recorded_at is not None else None
            ),
            "new_recorded_at": (
                _iso(self.new_recorded_at)
                if self.new_recorded_at is not None else None
            ),
            "reason": self.reason,
            "result": self.result.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryEvent":
        data = _require_mapping(
            data,
            frozenset({
                "event_id", "revision", "action", "at", "source",
                "kind", "key", "task_id", "entry_id",
                "old_value", "new_value", "old_present", "new_present",
                "old_recorded_at", "new_recorded_at", "reason", "result",
            }),
            "MemoryEvent",
        )
        return cls(
            event_id=_require_present(data, "event_id", "MemoryEvent"),
            revision=_require_present(data, "revision", "MemoryEvent"),
            action=_require_present(data, "action", "MemoryEvent"),
            at=_require_present(data, "at", "MemoryEvent"),
            source=_require_present(data, "source", "MemoryEvent"),
            kind=data.get("kind"),
            key=data.get("key"),
            task_id=data.get("task_id"),
            entry_id=data.get("entry_id"),
            old_value=data.get("old_value"),
            new_value=data.get("new_value"),
            old_present=data.get("old_present", False),
            new_present=data.get("new_present", False),
            old_recorded_at=data.get("old_recorded_at"),
            new_recorded_at=data.get("new_recorded_at"),
            reason=data.get("reason", ""),
            result=data.get("result", EventResult.APPLIED.value),
        )


@dataclass(frozen=True)
class MemorySnapshot:
    """Активные данные памяти + отдельный журнал изменений."""

    format_version: int = FORMAT_VERSION
    revision: int = 0
    updated_at: datetime | None = None
    mutable: dict[str, Fact] = field(default_factory=dict)
    cumulative: tuple[Fact, ...] = ()
    temporary: dict[str, dict[str, Fact]] = field(default_factory=dict)
    mutable_tombstones: dict[str, Tombstone] = field(default_factory=dict)
    temporary_tombstones: dict[str, dict[str, Tombstone]] = field(
        default_factory=dict)
    completed_tasks: dict[str, TaskCompletion] = field(default_factory=dict)
    journal: tuple[MemoryEvent, ...] = ()

    def __post_init__(self) -> None:
        if type(self.format_version) is not int or self.format_version != FORMAT_VERSION:
            raise MemoryValidationError(
                f"MemorySnapshot.format_version: поддерживается {FORMAT_VERSION}")
        if type(self.revision) is not int or self.revision < 0:
            raise MemoryValidationError(
                "MemorySnapshot.revision: ожидается int >= 0")
        if self.updated_at is not None:
            object.__setattr__(
                self, "updated_at",
                _validate_datetime(self.updated_at, "MemorySnapshot.updated_at"))

        mutable = dict(self.mutable)
        for key, fact in mutable.items():
            _validate_key(key, "MemorySnapshot.mutable key")
            if not isinstance(fact, Fact):
                raise MemoryValidationError(
                    "MemorySnapshot.mutable: значения должны быть Fact")
            if fact.kind is not FactKind.MUTABLE:
                raise MemoryValidationError(
                    f"MemorySnapshot.mutable[{key!r}]: ожидается mutable")
            if fact.key != key or fact.task_id is not None or fact.entry_id is not None:
                raise MemoryValidationError(
                    f"MemorySnapshot.mutable[{key!r}]: несовпадение адреса")
        object.__setattr__(self, "mutable", mutable)

        cumulative = tuple(self.cumulative)
        seen_entries: set[str] = set()
        for fact in cumulative:
            if not isinstance(fact, Fact) or fact.kind is not FactKind.CUMULATIVE:
                raise MemoryValidationError(
                    "MemorySnapshot.cumulative: ожидаются cumulative Fact")
            if fact.entry_id is None or fact.entry_id in seen_entries:
                raise MemoryValidationError(
                    "MemorySnapshot.cumulative: повторяющийся/пустой entry_id")
            seen_entries.add(fact.entry_id)
        object.__setattr__(self, "cumulative", cumulative)

        temporary: dict[str, dict[str, Fact]] = {}
        for task_id, facts in dict(self.temporary).items():
            _validate_task_id(task_id, "MemorySnapshot.temporary task_id")
            if not isinstance(facts, Mapping):
                raise MemoryValidationError(
                    f"MemorySnapshot.temporary[{task_id!r}]: ожидается объект")
            task_facts: dict[str, Fact] = {}
            for key, fact in dict(facts).items():
                _validate_key(key, "MemorySnapshot.temporary key")
                if not isinstance(fact, Fact) or fact.kind is not FactKind.TEMPORARY:
                    raise MemoryValidationError(
                        f"MemorySnapshot.temporary[{task_id!r}][{key!r}]: "
                        "ожидается temporary Fact")
                if fact.task_id != task_id or fact.key != key:
                    raise MemoryValidationError(
                        f"MemorySnapshot.temporary[{task_id!r}][{key!r}]: "
                        "несовпадение адреса")
                task_facts[key] = fact
            temporary[task_id] = task_facts
        object.__setattr__(self, "temporary", temporary)

        mutable_tombstones = dict(self.mutable_tombstones)
        for key, tombstone in mutable_tombstones.items():
            _validate_key(key, "MemorySnapshot.mutable_tombstones key")
            if not isinstance(tombstone, Tombstone):
                raise MemoryValidationError(
                    "MemorySnapshot.mutable_tombstones: ожидается Tombstone")
            if (tombstone.kind is not FactKind.MUTABLE
                    or tombstone.key != key or tombstone.task_id is not None):
                raise MemoryValidationError(
                    f"MemorySnapshot.mutable_tombstones[{key!r}]: несовпадение")
        object.__setattr__(self, "mutable_tombstones", mutable_tombstones)

        temporary_tombstones: dict[str, dict[str, Tombstone]] = {}
        for task_id, tombstones in dict(self.temporary_tombstones).items():
            _validate_task_id(task_id, "MemorySnapshot.temporary_tombstones task_id")
            if not isinstance(tombstones, Mapping):
                raise MemoryValidationError(
                    f"MemorySnapshot.temporary_tombstones[{task_id!r}]: "
                    "ожидается объект")
            task_tombstones: dict[str, Tombstone] = {}
            for key, tombstone in dict(tombstones).items():
                _validate_key(key, "MemorySnapshot.temporary_tombstones key")
                if not isinstance(tombstone, Tombstone):
                    raise MemoryValidationError(
                        f"MemorySnapshot.temporary_tombstones[{task_id!r}]"
                        f"[{key!r}]: ожидается Tombstone")
                if (tombstone.kind is not FactKind.TEMPORARY
                        or tombstone.task_id != task_id
                        or tombstone.key != key):
                    raise MemoryValidationError(
                        f"MemorySnapshot.temporary_tombstones[{task_id!r}]"
                        f"[{key!r}]: несовпадение")
                task_tombstones[key] = tombstone
            temporary_tombstones[task_id] = task_tombstones
        object.__setattr__(self, "temporary_tombstones", temporary_tombstones)

        completed_tasks = dict(self.completed_tasks)
        for task_id, completion in completed_tasks.items():
            _validate_task_id(task_id, "MemorySnapshot.completed_tasks task_id")
            if not isinstance(completion, TaskCompletion):
                raise MemoryValidationError(
                    "MemorySnapshot.completed_tasks: ожидается TaskCompletion")
            if completion.task_id != task_id:
                raise MemoryValidationError(
                    f"MemorySnapshot.completed_tasks[{task_id!r}]: несовпадение")
        object.__setattr__(self, "completed_tasks", completed_tasks)

        journal = tuple(self.journal)
        seen_event_ids: set[str] = set()
        seen_revisions: set[int] = set()
        for event in journal:
            if not isinstance(event, MemoryEvent):
                raise MemoryValidationError(
                    "MemorySnapshot.journal: ожидаются MemoryEvent")
            if event.event_id in seen_event_ids:
                raise MemoryValidationError(
                    f"MemorySnapshot.journal: повторяющийся event_id "
                    f"{event.event_id!r}")
            if event.revision in seen_revisions:
                raise MemoryValidationError(
                    f"MemorySnapshot.journal: повторяющийся revision "
                    f"{event.revision}")
            if event.revision > self.revision:
                raise MemoryValidationError(
                    "MemorySnapshot.journal: revision события больше revision "
                    "snapshot")
            seen_event_ids.add(event.event_id)
            seen_revisions.add(event.revision)
        object.__setattr__(self, "journal", journal)

        # Tombstone не может быть «старше» активного факта: активный факт
        # должен быть записан строго позже удаления.
        for key, fact in self.mutable.items():
            tombstone = self.mutable_tombstones.get(key)
            if tombstone is not None and fact.recorded_at <= tombstone.deleted_at:
                raise MemoryValidationError(
                    f"MemorySnapshot.mutable[{key!r}]: активный факт не новее "
                    "tombstone")
        for task_id, facts in self.temporary.items():
            for key, fact in facts.items():
                tombstone = self.temporary_tombstones.get(task_id, {}).get(key)
                if tombstone is not None and fact.recorded_at <= tombstone.deleted_at:
                    raise MemoryValidationError(
                        f"MemorySnapshot.temporary[{task_id!r}][{key!r}]: "
                        "активный факт не новее tombstone")

        # Завершённая задача не может иметь активную temporary-память.
        for task_id in self.completed_tasks:
            if self.temporary.get(task_id):
                raise MemoryValidationError(
                    f"MemorySnapshot: завершённая задача {task_id!r} имеет "
                    "активные temporary-факты")

    def active_facts(self, task_id: str | None = None) -> tuple[Fact, ...]:
        """Только активные факты; журнал/tombstones не включаются."""
        facts: list[Fact] = [
            self.mutable[key] for key in sorted(self.mutable)
        ]
        facts.extend(sorted(
            self.cumulative,
            key=lambda item: (item.recorded_at, item.entry_id or ""),
        ))
        if task_id is not None:
            _validate_task_id(task_id, "MemorySnapshot.active_facts task_id")
            task_facts = self.temporary.get(task_id, {})
            facts.extend(task_facts[key] for key in sorted(task_facts))
        return tuple(facts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "revision": self.revision,
            "updated_at": _iso(self.updated_at) if self.updated_at else None,
            "mutable": {key: fact.to_dict() for key, fact in self.mutable.items()},
            "cumulative": [fact.to_dict() for fact in self.cumulative],
            "temporary": {
                task_id: {
                    key: fact.to_dict() for key, fact in facts.items()
                }
                for task_id, facts in self.temporary.items()
            },
            "mutable_tombstones": {
                key: tombstone.to_dict()
                for key, tombstone in self.mutable_tombstones.items()
            },
            "temporary_tombstones": {
                task_id: {
                    key: tombstone.to_dict()
                    for key, tombstone in tombstones.items()
                }
                for task_id, tombstones in self.temporary_tombstones.items()
            },
            "completed_tasks": {
                task_id: completion.to_dict()
                for task_id, completion in self.completed_tasks.items()
            },
            "journal": [event.to_dict() for event in self.journal],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemorySnapshot":
        data = _require_mapping(
            data,
            frozenset({
                "format_version", "revision", "updated_at",
                "mutable", "cumulative", "temporary",
                "mutable_tombstones", "temporary_tombstones",
                "completed_tasks", "journal",
            }),
            "MemorySnapshot",
        )
        raw_mutable = data.get("mutable", {})
        if not isinstance(raw_mutable, Mapping):
            raise MemoryValidationError(
                "MemorySnapshot.mutable: ожидается объект")
        raw_cumulative = data.get("cumulative", [])
        if not isinstance(raw_cumulative, (list, tuple)):
            raise MemoryValidationError(
                "MemorySnapshot.cumulative: ожидается список")
        raw_temporary = data.get("temporary", {})
        if not isinstance(raw_temporary, Mapping):
            raise MemoryValidationError(
                "MemorySnapshot.temporary: ожидается объект")
        raw_mutable_tombstones = data.get("mutable_tombstones", {})
        if not isinstance(raw_mutable_tombstones, Mapping):
            raise MemoryValidationError(
                "MemorySnapshot.mutable_tombstones: ожидается объект")
        raw_temporary_tombstones = data.get("temporary_tombstones", {})
        if not isinstance(raw_temporary_tombstones, Mapping):
            raise MemoryValidationError(
                "MemorySnapshot.temporary_tombstones: ожидается объект")
        raw_completed_tasks = data.get("completed_tasks", {})
        if not isinstance(raw_completed_tasks, Mapping):
            raise MemoryValidationError(
                "MemorySnapshot.completed_tasks: ожидается объект")
        raw_journal = data.get("journal", [])
        if not isinstance(raw_journal, (list, tuple)):
            raise MemoryValidationError(
                "MemorySnapshot.journal: ожидается список")

        mutable = {
            key: Fact.from_dict(value)
            for key, value in raw_mutable.items()
        }
        cumulative = tuple(
            Fact.from_dict(value) for value in raw_cumulative)
        temporary: dict[str, dict[str, Fact]] = {}
        for task_id, facts in raw_temporary.items():
            if not isinstance(facts, Mapping):
                raise MemoryValidationError(
                    f"MemorySnapshot.temporary[{task_id!r}]: ожидается объект")
            temporary[task_id] = {
                key: Fact.from_dict(value)
                for key, value in facts.items()
            }
        mutable_tombstones = {
            key: Tombstone.from_dict(value)
            for key, value in raw_mutable_tombstones.items()
        }
        temporary_tombstones: dict[str, dict[str, Tombstone]] = {}
        for task_id, tombstones in raw_temporary_tombstones.items():
            if not isinstance(tombstones, Mapping):
                raise MemoryValidationError(
                    f"MemorySnapshot.temporary_tombstones[{task_id!r}]: "
                    "ожидается объект")
            temporary_tombstones[task_id] = {
                key: Tombstone.from_dict(value)
                for key, value in tombstones.items()
            }
        completed_tasks = {
            task_id: TaskCompletion.from_dict(value)
            for task_id, value in raw_completed_tasks.items()
        }
        journal = tuple(
            MemoryEvent.from_dict(value) for value in raw_journal)

        return cls(
            format_version=data.get("format_version", FORMAT_VERSION),
            revision=data.get("revision", 0),
            updated_at=data.get("updated_at"),
            mutable=mutable,
            cumulative=cumulative,
            temporary=temporary,
            mutable_tombstones=mutable_tombstones,
            temporary_tombstones=temporary_tombstones,
            completed_tasks=completed_tasks,
            journal=journal,
        )


__all__ = [
    "EventResult",
    "Fact",
    "FactKind",
    "FORMAT_VERSION",
    "JsonValue",
    "MAX_JSON_DEPTH",
    "MemoryAction",
    "MemoryEvent",
    "MemorySnapshot",
    "TaskCompletion",
    "Tombstone",
]
