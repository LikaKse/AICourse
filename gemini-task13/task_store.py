"""Read-only adapter for the Day 12 ``working_memory.json`` format.

This module is the 2.1 sub-stage only. It:

- reads and validates the file header;
- distinguishes a missing file, a file with ``task=null``, a legacy task and a
  task with an embedded FSM envelope;
- extracts the reserved ``task.values["_task_fsm"]`` envelope;
- never writes, migrates, signs, creates keys, backups or checkpoints.

The embedded FSM is stored under a reserved key inside ``task.values``. Day 12's
``TaskState.to_dict()`` preserves unknown value keys, while its ``public_view()``
and ``render()`` ignore them. That keeps the old layer compatible without
modifying ``gemini-task12``.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1
LAYER = "working"
FSM_KEY = "_task_fsm"
FSM_SCHEMA = "task_fsm/v1"

#: Exact field set of the v1 signed envelope. Unknown fields are rejected,
#: not silently dropped, so serialization cannot lose data.
ENVELOPE_FIELDS = frozenset({
    "schema",
    "task_id",
    "revision",
    "key_id",
    "prev_digest",
    "digest",
    "hmac",
    "snapshot",
    "validation_receipts",
    "migrated_from_legacy",
})

#: Fields of the reserved ``task.values["_task_fsm"]`` wrapper.
WRAPPER_FIELDS = frozenset({"value", "kind", "evidence", "source_message"})


class TaskStoreError(Exception):
    """Base error for the read-only WorkingMemory adapter."""


class WorkingMemoryLoadError(TaskStoreError):
    """The working_memory.json file is missing required structure or invalid."""


class FsmEnvelopeError(TaskStoreError):
    """The reserved _task_fsm envelope is malformed or bound to another task."""


class DocumentState(str, Enum):
    """What the read-only adapter found in working_memory.json.

    ``FSM_TASK`` means only that an envelope was structurally parsed. It does
    NOT mean that HMAC/digest were verified or that the snapshot is trusted.
    """

    MISSING = "missing"
    NO_ACTIVE_TASK = "no_active_task"
    LEGACY_TASK = "legacy_task"
    FSM_TASK = "fsm_task"  # structurally parsed envelope only; NOT trusted


@dataclass(frozen=True)
class FsmEnvelope:
    """Structurally parsed ``task.values["_task_fsm"].value``.

    This is NOT a trust verification. ``digest`` and ``hmac`` are parsed but
    not checked in sub-stage 2.1. Use :attr:`trusted` /
    :attr:`verification_status` to see the current verification state.
    """

    schema: str
    task_id: int
    revision: int
    digest: str
    hmac: str
    _snapshot: dict[str, Any] = field(repr=False)
    key_id: str
    prev_digest: str | None = None
    _validation_receipts: tuple[dict[str, Any], ...] = field(default=(), repr=False)
    migrated_from_legacy: bool = False

    def __post_init__(self) -> None:
        """Defensively copy mutable constructor inputs.

        ``frozen=True`` prevents assignment but not mutation of nested dicts.
        Copies are made once, on construction, so an external object passed by
        a caller cannot change the envelope later.
        """
        object.__setattr__(self, "_snapshot", copy.deepcopy(self._snapshot))
        object.__setattr__(
            self,
            "_validation_receipts",
            tuple(copy.deepcopy(item) for item in self._validation_receipts),
        )

    @property
    def snapshot(self) -> dict[str, Any]:
        """Defensive copy: mutating it does not change the envelope."""
        return copy.deepcopy(self._snapshot)

    @property
    def validation_receipts(self) -> tuple[dict[str, Any], ...]:
        """Defensive copies: receipt objects cannot be mutated in place."""
        return tuple(copy.deepcopy(item) for item in self._validation_receipts)

    @property
    def trusted(self) -> bool:
        """Always False in sub-stage 2.1: HMAC is not verified here."""
        return False

    @property
    def verification_status(self) -> str:
        """Explicit unverified marker for callers and future stages."""
        return "unverified"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task_id": self.task_id,
            "revision": self.revision,
            "key_id": self.key_id,
            "prev_digest": self.prev_digest,
            "digest": self.digest,
            "hmac": self.hmac,
            "snapshot": self.snapshot,
            "validation_receipts": list(self.validation_receipts),
            "migrated_from_legacy": self.migrated_from_legacy,
        }


@dataclass(frozen=True)
class WorkingMemoryDocument:
    """Read-only view of a working_memory.json file."""

    path: Path
    status: DocumentState
    _payload: dict[str, Any] = field(repr=False)
    task_id: int | None = None
    task_status: str | None = None
    envelope: FsmEnvelope | None = None

    def __post_init__(self) -> None:
        """Deep-copy the payload so callers cannot mutate the document."""
        object.__setattr__(self, "_payload", copy.deepcopy(self._payload))

    @property
    def payload(self) -> dict[str, Any]:
        """Defensive copy: callers cannot mutate the parsed document in place."""
        return copy.deepcopy(self._payload)

    @property
    def exists(self) -> bool:
        return self.status is not DocumentState.MISSING

    @property
    def has_active_task(self) -> bool:
        return self.status in {DocumentState.LEGACY_TASK, DocumentState.FSM_TASK}

    @property
    def has_fsm(self) -> bool:
        """True when an envelope is present and structurally parsed.

        This does NOT mean the envelope is cryptographically verified or
        trusted. In sub-stage 2.1 HMAC/digest are not checked.
        """
        return self.envelope is not None

    @property
    def envelope_verified(self) -> bool:
        """Always False in sub-stage 2.1; verification is a later stage."""
        return False

    @property
    def is_legacy(self) -> bool:
        return self.status is DocumentState.LEGACY_TASK


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise WorkingMemoryLoadError(f"{path.name}: не удалось прочитать файл: {exc}") from exc

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WorkingMemoryLoadError(f"{path.name}: файл не в UTF-8") from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WorkingMemoryLoadError(
            f"{path.name}: некорректный JSON (строка {exc.lineno}): {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise WorkingMemoryLoadError("working: ожидался JSON-объект верхнего уровня")
    return payload


def _validate_header(payload: Mapping[str, Any]) -> None:
    version = payload.get("format_version")
    if type(version) is not int or version != FORMAT_VERSION:
        raise WorkingMemoryLoadError(
            f"working: неподдерживаемая версия формата {version!r}")
    if payload.get("layer") != LAYER:
        raise WorkingMemoryLoadError(
            f"working: файл принадлежит слою {payload.get('layer')!r}")


def _legacy_task_id(task: Mapping[str, Any]) -> int:
    value = task.get("task_id", 1)
    if type(value) is int and value >= 1:
        return value
    if isinstance(value, str) and value.isdigit() and int(value) >= 1:
        return int(value)
    raise WorkingMemoryLoadError("working: task.task_id должен быть положительным целым")


def _task_status(values: Mapping[str, Any]) -> str | None:
    entry = values.get("status")
    if not isinstance(entry, Mapping):
        return None
    value = entry.get("value")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _require_exact_str(mapping: Mapping[str, Any], key: str, *, where: str) -> str:
    """Require a non-empty string and preserve it byte-for-byte.

    Security-sensitive fields must not be silently normalized with ``strip()``.
    Leading/trailing whitespace is rejected explicitly.
    """
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise FsmEnvelopeError(f"{where}: поле {key!r} должно быть непустой строкой")
    if value != value.strip():
        raise FsmEnvelopeError(
            f"{where}: поле {key!r} не должно содержать внешних пробелов")
    return value


def _optional_exact_str(mapping: Mapping[str, Any], key: str, *,
                        where: str, default: str | None = None) -> str | None:
    value = mapping.get(key)
    if value is None:
        return default
    if not isinstance(value, str) or not value:
        raise FsmEnvelopeError(
            f"{where}: поле {key!r} должно быть непустой строкой или null")
    if value != value.strip():
        raise FsmEnvelopeError(
            f"{where}: поле {key!r} не должно содержать внешних пробелов")
    return value


def _require_int(mapping: Mapping[str, Any], key: str, *, where: str,
                 minimum: int = 0) -> int:
    value = mapping.get(key)
    if type(value) is not int or value < minimum:
        raise FsmEnvelopeError(
            f"{where}: поле {key!r} должно быть целым числом >= {minimum}")
    return value


def _parse_envelope(value: Mapping[str, Any], *, expected_task_id: int) -> FsmEnvelope:
    unknown = set(value) - ENVELOPE_FIELDS
    if unknown:
        raise FsmEnvelopeError(
            f"working: _task_fsm: неизвестные поля envelope {sorted(unknown)}")

    schema = _require_exact_str(value, "schema", where="working: _task_fsm")
    if schema != FSM_SCHEMA:
        raise FsmEnvelopeError(
            f"working: _task_fsm: неподдерживаемая схема {schema!r}")

    task_id = _require_int(value, "task_id", where="working: _task_fsm", minimum=1)
    if task_id != expected_task_id:
        raise FsmEnvelopeError(
            "working: _task_fsm привязан к task_id "
            f"{task_id}, а текущая задача имеет task_id {expected_task_id}")

    revision = _require_int(value, "revision", where="working: _task_fsm")
    digest = _require_exact_str(value, "digest", where="working: _task_fsm")
    hmac = _require_exact_str(value, "hmac", where="working: _task_fsm")

    raw_snapshot = value.get("snapshot")
    if not isinstance(raw_snapshot, Mapping):
        raise FsmEnvelopeError("working: _task_fsm: snapshot должен быть объектом")
    snapshot = copy.deepcopy(dict(raw_snapshot))

    key_id = _require_exact_str(value, "key_id", where="working: _task_fsm")
    prev_digest = _optional_exact_str(
        value, "prev_digest", where="working: _task_fsm", default=None)

    raw_receipts = value.get("validation_receipts", [])
    if not isinstance(raw_receipts, list):
        raise FsmEnvelopeError(
            "working: _task_fsm: validation_receipts должен быть списком")
    receipts: list[dict[str, Any]] = []
    for index, item in enumerate(raw_receipts, 1):
        if not isinstance(item, Mapping):
            raise FsmEnvelopeError(
                f"working: _task_fsm: validation_receipts[{index}] должен быть объектом")
        receipts.append(copy.deepcopy(dict(item)))

    migrated = value.get("migrated_from_legacy", False)
    if type(migrated) is not bool:
        raise FsmEnvelopeError(
            "working: _task_fsm: migrated_from_legacy должен быть bool")

    return FsmEnvelope(
        schema=schema,
        task_id=task_id,
        revision=revision,
        digest=digest,
        hmac=hmac,
        _snapshot=snapshot,
        key_id=key_id,
        prev_digest=prev_digest,
        _validation_receipts=tuple(receipts),
        migrated_from_legacy=migrated,
    )


def load_working_memory(path: str | Path) -> WorkingMemoryDocument:
    """Read working_memory.json without writing anything.

    Raises ``WorkingMemoryLoadError`` or ``FsmEnvelopeError`` for malformed data.
    A missing file is a valid state and does not create the file.
    """
    path = Path(path)

    if not path.exists():
        return WorkingMemoryDocument(
            path=path, status=DocumentState.MISSING, _payload={})

    if path.is_dir():
        raise WorkingMemoryLoadError(f"{path}: ожидался файл, а не каталог")

    payload = _read_json(path)
    _validate_header(payload)

    raw_task = payload.get("task")
    if raw_task is None:
        return WorkingMemoryDocument(
            path=path, status=DocumentState.NO_ACTIVE_TASK, _payload=payload)

    if not isinstance(raw_task, Mapping):
        raise WorkingMemoryLoadError("working: task должен быть объектом или null")

    task = dict(raw_task)
    task_id = _legacy_task_id(task)

    raw_values = task.get("values")
    if raw_values is None:
        raw_values = {}
    if not isinstance(raw_values, Mapping):
        raise WorkingMemoryLoadError("working: task.values должен быть объектом")
    values = dict(raw_values)

    envelope: FsmEnvelope | None = None
    if FSM_KEY in values:
        raw_fsm = values[FSM_KEY]
        if not isinstance(raw_fsm, Mapping):
            raise FsmEnvelopeError("working: task.values._task_fsm должен быть объектом")
        unknown = set(raw_fsm) - WRAPPER_FIELDS
        if unknown:
            raise FsmEnvelopeError(
                "working: task.values._task_fsm: неизвестные поля "
                f"{sorted(unknown)}")
        if "value" not in raw_fsm:
            raise FsmEnvelopeError(
                "working: task.values._task_fsm.value отсутствует")
        raw_value = raw_fsm["value"]
        if not isinstance(raw_value, Mapping):
            raise FsmEnvelopeError(
                "working: task.values._task_fsm.value должен быть объектом")
        envelope = _parse_envelope(raw_value, expected_task_id=task_id)

    status = DocumentState.FSM_TASK if envelope is not None else DocumentState.LEGACY_TASK
    return WorkingMemoryDocument(
        path=path,
        status=status,
        _payload=payload,
        task_id=task_id,
        task_status=_task_status(values),
        envelope=envelope,
    )


__all__ = [
    "DocumentState",
    "FSM_KEY",
    "FSM_SCHEMA",
    "FsmEnvelope",
    "FsmEnvelopeError",
    "TaskStoreError",
    "WorkingMemoryDocument",
    "WorkingMemoryLoadError",
    "load_working_memory",
]
