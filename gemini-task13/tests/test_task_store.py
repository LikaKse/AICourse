"""Read-only adapter for working_memory.json: no writes, no migration, no trust checks."""

import hashlib
import json

import pytest

from task_store import (
    DocumentState,
    FSM_KEY,
    FsmEnvelope,
    FsmEnvelopeError,
    WorkingMemoryDocument,
    WorkingMemoryLoadError,
    load_working_memory,
)


def header(**body):
    return {"format_version": 1, "layer": "working", "saved_at": "2026-11-20T12:00:00", **body}


def legacy_task(**overrides):
    task = {
        "task_id": 7,
        "created_at": "2026-11-20T10:00:00",
        "values": {
            "goal": {
                "value": "сделать отчёт",
                "kind": "user",
                "evidence": "Нужно сделать отчёт",
                "source_message": 1,
            },
            "status": {
                "value": "in_progress",
                "kind": "user",
                "evidence": "Начинаю",
                "source_message": 1,
            },
            "legacy_custom": {"value": "сохранить", "kind": "system", "evidence": "", "source_message": None},
        },
        "items": {"constraints": [], "results": []},
        "history": [{"field": "goal", "old": None, "new": "сделать отчёт", "message": 1}],
        "unknown_task_field": {"keep": True},
    }
    task.update(overrides)
    return task


def envelope(task_id=7, **overrides):
    value = {
        "schema": "task_fsm/v1",
        "task_id": task_id,
        "revision": 3,
        "key_id": "local-v1",
        "prev_digest": "sha256:previous",
        "digest": "sha256:current",
        "hmac": "hmac-sha256:signature",
        "migrated_from_legacy": False,
        "snapshot": {"state": "planning", "plan": [], "transitions": []},
        "validation_receipts": [
            {"receipt_id": "r-1", "task_id": task_id, "step_id": "step-1", "attempt": 1}
        ],
    }
    value.update(overrides)
    return value


def with_fsm(task=None, env=None, **task_overrides):
    task = legacy_task(**task_overrides) if task is None else task
    task["values"][FSM_KEY] = {
        "value": envelope() if env is None else env,
        "kind": "system",
        "evidence": "task_controller",
        "source_message": None,
    }
    return header(task=task, version=4, archived=[], journal=[])


def write_doc(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_missing_file_is_missing_and_does_not_create_file(tmp_path):
    path = tmp_path / "nested" / "working_memory.json"

    document = load_working_memory(path)

    assert document.status is DocumentState.MISSING
    assert document.exists is False
    assert document.has_active_task is False
    assert document.has_fsm is False
    assert document.is_legacy is False
    assert document.task_id is None
    assert document.envelope is None
    assert document.payload == {}
    assert not path.exists()
    assert not path.parent.exists()


def test_task_null_is_no_active_task(tmp_path):
    path = write_doc(tmp_path / "working_memory.json", header(task=None))
    before = file_hash(path)

    document = load_working_memory(path)

    assert document.status is DocumentState.NO_ACTIVE_TASK
    assert document.exists is True
    assert document.has_active_task is False
    assert document.has_fsm is False
    assert document.is_legacy is False
    assert document.task_id is None
    assert document.payload["task"] is None
    assert file_hash(path) == before


def test_absent_task_key_is_no_active_task(tmp_path):
    path = write_doc(tmp_path / "working_memory.json", header())

    document = load_working_memory(path)

    assert document.status is DocumentState.NO_ACTIVE_TASK
    assert document.has_active_task is False
    assert document.payload.get("task") is None


def test_legacy_task_without_fsm_is_detected(tmp_path):
    payload = header(task=legacy_task(), version=4, archived=[{"task_id": 1}],
                     journal=[{"message": 1}], top_level_unknown={"keep": True})
    path = write_doc(tmp_path / "working_memory.json", payload)
    before = file_hash(path)

    document = load_working_memory(path)

    assert document.status is DocumentState.LEGACY_TASK
    assert document.has_active_task is True
    assert document.has_fsm is False
    assert document.is_legacy is True
    assert document.task_id == 7
    assert document.task_status == "in_progress"
    assert document.envelope is None
    assert document.payload["task"]["unknown_task_field"] == {"keep": True}
    assert document.payload["task"]["values"]["legacy_custom"]["value"] == "сохранить"
    assert document.payload["archived"] == [{"task_id": 1}]
    assert document.payload["journal"] == [{"message": 1}]
    assert document.payload["top_level_unknown"] == {"keep": True}
    assert file_hash(path) == before


def test_fsm_task_extracts_envelope(tmp_path):
    path = write_doc(tmp_path / "working_memory.json", with_fsm())

    document = load_working_memory(path)

    assert document.status is DocumentState.FSM_TASK
    assert document.has_active_task is True
    assert document.has_fsm is True
    assert document.is_legacy is False
    assert document.task_id == 7
    assert document.task_status == "in_progress"
    assert document.envelope is not None
    assert document.envelope.schema == "task_fsm/v1"
    assert document.envelope.task_id == 7
    assert document.envelope.revision == 3
    assert document.envelope.key_id == "local-v1"
    assert document.envelope.prev_digest == "sha256:previous"
    assert document.envelope.digest == "sha256:current"
    assert document.envelope.hmac == "hmac-sha256:signature"
    assert document.envelope.snapshot == {"state": "planning", "plan": [], "transitions": []}
    assert document.envelope.validation_receipts[0]["receipt_id"] == "r-1"
    assert document.envelope.migrated_from_legacy is False


def test_fsm_envelope_task_id_must_match_current_task(tmp_path):
    path = write_doc(tmp_path / "working_memory.json", with_fsm(env=envelope(task_id=8)))

    with pytest.raises(FsmEnvelopeError) as exc:
        load_working_memory(path)

    assert "task_id" in str(exc.value)


def test_fsm_envelope_value_must_be_object(tmp_path):
    payload = header(task=legacy_task())
    payload["task"]["values"][FSM_KEY] = {"value": "not-an-object"}
    path = write_doc(tmp_path / "working_memory.json", payload)

    with pytest.raises(FsmEnvelopeError):
        load_working_memory(path)


def test_fsm_envelope_entry_must_be_object(tmp_path):
    payload = header(task=legacy_task())
    payload["task"]["values"][FSM_KEY] = "not-an-object"
    path = write_doc(tmp_path / "working_memory.json", payload)

    with pytest.raises(FsmEnvelopeError):
        load_working_memory(path)


@pytest.mark.parametrize("missing", [
    "schema", "task_id", "revision", "digest", "hmac", "snapshot",
])
def test_fsm_envelope_requires_core_fields(tmp_path, missing):
    env = envelope()
    env.pop(missing)
    path = write_doc(tmp_path / "working_memory.json", with_fsm(env=env))

    with pytest.raises(FsmEnvelopeError):
        load_working_memory(path)


def test_fsm_envelope_rejects_unknown_schema(tmp_path):
    path = write_doc(tmp_path / "working_memory.json", with_fsm(env=envelope(schema="task_fsm/v2")))

    with pytest.raises(FsmEnvelopeError) as exc:
        load_working_memory(path)

    assert "схема" in str(exc.value)


@pytest.mark.parametrize("revision", [-1, "3", True, 3.5, None])
def test_fsm_envelope_rejects_invalid_revision(tmp_path, revision):
    path = write_doc(tmp_path / "working_memory.json", with_fsm(env=envelope(revision=revision)))

    with pytest.raises(FsmEnvelopeError):
        load_working_memory(path)


@pytest.mark.parametrize("snapshot", [None, [], "state", 1])
def test_fsm_envelope_snapshot_must_be_object(tmp_path, snapshot):
    path = write_doc(tmp_path / "working_memory.json", with_fsm(env=envelope(snapshot=snapshot)))

    with pytest.raises(FsmEnvelopeError):
        load_working_memory(path)


@pytest.mark.parametrize("receipts", [None, {}, "r", [1], [None]])
def test_fsm_envelope_receipts_must_be_list_of_objects(tmp_path, receipts):
    path = write_doc(tmp_path / "working_memory.json", with_fsm(env=envelope(validation_receipts=receipts)))

    with pytest.raises(FsmEnvelopeError):
        load_working_memory(path)


def test_fsm_envelope_allows_null_prev_digest(tmp_path):
    env = envelope()
    env["prev_digest"] = None
    path = write_doc(tmp_path / "working_memory.json", with_fsm(env=env))

    document = load_working_memory(path)

    assert document.envelope is not None
    assert document.envelope.prev_digest is None
    assert document.envelope.key_id == "local-v1"


def test_fsm_envelope_requires_key_id(tmp_path):
    env = envelope()
    env.pop("key_id")
    path = write_doc(tmp_path / "working_memory.json", with_fsm(env=env))

    with pytest.raises(FsmEnvelopeError):
        load_working_memory(path)


def test_fsm_envelope_migrated_flag_must_be_bool(tmp_path):
    path = write_doc(tmp_path / "working_memory.json",
                     with_fsm(env=envelope(migrated_from_legacy="false")))

    with pytest.raises(FsmEnvelopeError):
        load_working_memory(path)


@pytest.mark.parametrize("payload", [
    [],
    {"format_version": 2, "layer": "working", "task": None},
    {"format_version": 1, "layer": "short_term", "task": None},
])
def test_header_validation(tmp_path, payload):
    path = write_doc(tmp_path / "working_memory.json", payload)

    with pytest.raises(WorkingMemoryLoadError):
        load_working_memory(path)


def test_task_must_be_object_or_null(tmp_path):
    path = write_doc(tmp_path / "working_memory.json", header(task=[]))

    with pytest.raises(WorkingMemoryLoadError):
        load_working_memory(path)


def test_values_must_be_object(tmp_path):
    payload = header(task={"task_id": 1, "values": []})
    path = write_doc(tmp_path / "working_memory.json", payload)

    with pytest.raises(WorkingMemoryLoadError):
        load_working_memory(path)


def test_task_id_must_be_positive_integer(tmp_path):
    for task_id in (0, -1, "x", None, True):
        payload = header(task={"task_id": task_id, "values": {}})
        path = write_doc(tmp_path / f"working_memory_{task_id!r}.json", payload)
        with pytest.raises(WorkingMemoryLoadError):
            load_working_memory(path)


def test_numeric_string_task_id_is_normalized(tmp_path):
    task = legacy_task(task_id="7")
    task["values"] = {}
    path = write_doc(tmp_path / "working_memory.json", with_fsm(task=task))

    document = load_working_memory(path)

    assert document.task_id == 7
    assert document.envelope is not None
    assert document.envelope.task_id == 7


def test_missing_task_id_defaults_to_one_like_day12(tmp_path):
    task = legacy_task(task_id=1)
    task.pop("task_id")
    path = write_doc(tmp_path / "working_memory.json", with_fsm(task=task, env=envelope(task_id=1)))

    document = load_working_memory(path)

    assert document.task_id == 1
    assert document.envelope is not None


def test_malformed_status_is_none(tmp_path):
    task = legacy_task()
    task["values"]["status"] = {"value": 123}
    path = write_doc(tmp_path / "working_memory.json", header(task=task))

    document = load_working_memory(path)

    assert document.task_status is None


def test_invalid_json_is_rejected(tmp_path):
    path = tmp_path / "working_memory.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(WorkingMemoryLoadError):
        load_working_memory(path)


def test_non_utf8_file_is_rejected(tmp_path):
    path = tmp_path / "working_memory.json"
    path.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(WorkingMemoryLoadError):
        load_working_memory(path)


def test_directory_path_is_rejected(tmp_path):
    with pytest.raises(WorkingMemoryLoadError):
        load_working_memory(tmp_path)


def test_payload_is_defensive_copy(tmp_path):
    path = write_doc(tmp_path / "working_memory.json", header(task=legacy_task()))

    document = load_working_memory(path)
    payload = document.payload
    payload["task"]["values"]["goal"]["value"] = "изменено"
    payload["task"]["unknown_task_field"]["keep"] = False

    assert document.payload["task"]["values"]["goal"]["value"] == "сделать отчёт"
    assert document.payload["task"]["unknown_task_field"] == {"keep": True}


def test_reading_does_not_change_file_for_legacy_and_fsm(tmp_path):
    for name, payload in (
        ("legacy.json", header(task=legacy_task())),
        ("fsm.json", with_fsm()),
    ):
        path = write_doc(tmp_path / name, payload)
        before = file_hash(path)
        document = load_working_memory(path)
        assert document.payload
        assert file_hash(path) == before


def test_no_active_task_does_not_use_archived_fsm(tmp_path):
    archived = legacy_task(task_id=1)
    archived["values"][FSM_KEY] = {
        "value": envelope(task_id=1),
        "kind": "system",
        "evidence": "task_controller",
        "source_message": None,
    }
    path = write_doc(tmp_path / "working_memory.json", header(task=None, archived=[archived]))

    document = load_working_memory(path)

    assert document.status is DocumentState.NO_ACTIVE_TASK
    assert document.envelope is None
    assert document.has_fsm is False


def test_archived_fsm_is_not_used_for_current_legacy_task(tmp_path):
    archived = legacy_task(task_id=1)
    archived["values"][FSM_KEY] = {
        "value": envelope(task_id=1),
        "kind": "system",
        "evidence": "task_controller",
        "source_message": None,
    }
    path = write_doc(tmp_path / "working_memory.json",
                     header(task=legacy_task(task_id=2), archived=[archived]))

    document = load_working_memory(path)

    assert document.status is DocumentState.LEGACY_TASK
    assert document.task_id == 2
    assert document.envelope is None


# ---------------------------------------------------------------------------
# Исправления после аудита 2.1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw_fsm", [None, "not-an-object", [], 1, False])
def test_present_task_fsm_with_invalid_type_is_rejected(tmp_path, raw_fsm):
    payload = header(task=legacy_task())
    payload["task"]["values"][FSM_KEY] = raw_fsm
    path = write_doc(tmp_path / "working_memory.json", payload)

    with pytest.raises(FsmEnvelopeError):
        load_working_memory(path)


def test_task_fsm_wrapper_requires_value(tmp_path):
    payload = header(task=legacy_task())
    payload["task"]["values"][FSM_KEY] = {"kind": "system"}
    path = write_doc(tmp_path / "working_memory.json", payload)

    with pytest.raises(FsmEnvelopeError):
        load_working_memory(path)


def test_task_fsm_wrapper_rejects_unknown_fields(tmp_path):
    payload = header(task=legacy_task())
    payload["task"]["values"][FSM_KEY] = {"value": envelope(), "extra": "x"}
    path = write_doc(tmp_path / "working_memory.json", payload)

    with pytest.raises(FsmEnvelopeError):
        load_working_memory(path)


def test_fsm_envelope_rejects_unknown_fields(tmp_path):
    env = envelope()
    env["extra"] = "x"
    path = write_doc(tmp_path / "working_memory.json", with_fsm(env=env))

    with pytest.raises(FsmEnvelopeError):
        load_working_memory(path)


def test_fsm_envelope_snapshot_is_defensive_copy(tmp_path):
    original = {
        "state": "planning",
        "plan": [{"step_id": "step-1"}],
        "nested": {"value": 1},
    }
    path = write_doc(tmp_path / "working_memory.json",
                     with_fsm(env=envelope(snapshot=original)))

    document = load_working_memory(path)
    snapshot = document.envelope.snapshot
    snapshot["state"] = "done"
    snapshot["plan"][0]["step_id"] = "changed"
    snapshot["nested"]["value"] = 2

    assert document.envelope.snapshot == original


def test_fsm_envelope_receipts_are_defensive_copy(tmp_path):
    original = {"receipt_id": "r-1", "nested": {"attempt": 1}}
    path = write_doc(tmp_path / "working_memory.json",
                     with_fsm(env=envelope(validation_receipts=[original])))

    document = load_working_memory(path)
    receipts = document.envelope.validation_receipts
    receipts[0]["receipt_id"] = "changed"
    receipts[0]["nested"]["attempt"] = 999

    assert document.envelope.validation_receipts[0] == original


def test_fsm_envelope_is_explicitly_unverified(tmp_path):
    path = write_doc(tmp_path / "working_memory.json", with_fsm())

    document = load_working_memory(path)

    assert document.envelope is not None
    assert document.envelope.trusted is False
    assert document.envelope.verification_status == "unverified"
    assert document.envelope_verified is False


@pytest.mark.parametrize("field", ["schema", "digest", "hmac", "key_id", "prev_digest"])
def test_security_fields_reject_outer_whitespace(tmp_path, field):
    env = envelope()
    env[field] = f" {env[field]} "
    path = write_doc(tmp_path / "working_memory.json", with_fsm(env=env))

    with pytest.raises(FsmEnvelopeError):
        load_working_memory(path)


@pytest.mark.parametrize("field", ["schema", "digest", "hmac", "key_id", "prev_digest"])
def test_security_fields_reject_empty_strings(tmp_path, field):
    env = envelope()
    env[field] = ""
    path = write_doc(tmp_path / "working_memory.json", with_fsm(env=env))

    with pytest.raises(FsmEnvelopeError):
        load_working_memory(path)


@pytest.mark.parametrize("field", ["schema", "digest", "hmac", "key_id", "prev_digest"])
@pytest.mark.parametrize("value", [1, True, [], {}])
def test_security_fields_reject_wrong_types(tmp_path, field, value):
    env = envelope()
    env[field] = value
    path = write_doc(tmp_path / "working_memory.json", with_fsm(env=env))

    with pytest.raises(FsmEnvelopeError):
        load_working_memory(path)


def test_key_id_null_is_rejected(tmp_path):
    env = envelope(key_id=None)
    path = write_doc(tmp_path / "working_memory.json", with_fsm(env=env))

    with pytest.raises(FsmEnvelopeError):
        load_working_memory(path)


@pytest.mark.parametrize("task_id", [0, -1, "7", 7.0, True, None])
def test_envelope_task_id_rejects_invalid_types_and_values(tmp_path, task_id):
    env = envelope(task_id=task_id)
    path = write_doc(tmp_path / "working_memory.json", with_fsm(env=env))

    with pytest.raises(FsmEnvelopeError):
        load_working_memory(path)


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "working_memory.json"
    path.write_text("", encoding="utf-8")

    with pytest.raises(WorkingMemoryLoadError):
        load_working_memory(path)


@pytest.mark.parametrize("payload", [None, "text", 1, True, []])
def test_root_json_must_be_object(tmp_path, payload):
    path = write_doc(tmp_path / "working_memory.json", payload)

    with pytest.raises(WorkingMemoryLoadError):
        load_working_memory(path)


@pytest.mark.parametrize("values", ["text", 1, True, []])
def test_values_must_be_object_non_dict_types(tmp_path, values):
    payload = header(task={"task_id": 1, "values": values})
    path = write_doc(tmp_path / "working_memory.json", payload)

    with pytest.raises(WorkingMemoryLoadError):
        load_working_memory(path)


@pytest.mark.parametrize("version", [True, 1.0, "1", None])
def test_format_version_must_be_strict_integer(tmp_path, version):
    payload = {"format_version": version, "layer": "working", "task": None}
    path = write_doc(tmp_path / "working_memory.json", payload)

    with pytest.raises(WorkingMemoryLoadError):
        load_working_memory(path)


def test_missing_values_is_legacy_task(tmp_path):
    path = write_doc(tmp_path / "working_memory.json", header(task={"task_id": 7}))

    document = load_working_memory(path)

    assert document.status is DocumentState.LEGACY_TASK
    assert document.task_id == 7
    assert document.task_status is None
    assert document.envelope is None


def test_reading_errors_do_not_change_source_file(tmp_path):
    payload_with_null_fsm = header(task=legacy_task())
    payload_with_null_fsm["task"]["values"][FSM_KEY] = None

    cases = [
        ("invalid-json", "{not json", WorkingMemoryLoadError),
        ("bad-version", json.dumps({"format_version": True, "layer": "working", "task": None}),
         WorkingMemoryLoadError),
        ("null-fsm", json.dumps(payload_with_null_fsm, ensure_ascii=False), FsmEnvelopeError),
        ("task-mismatch", json.dumps(with_fsm(env=envelope(task_id=8)), ensure_ascii=False),
         FsmEnvelopeError),
    ]

    for name, text, expected in cases:
        path = tmp_path / f"{name}.json"
        path.write_text(text, encoding="utf-8")
        before = file_hash(path)

        with pytest.raises(expected):
            load_working_memory(path)

        assert file_hash(path) == before


# ---------------------------------------------------------------------------
# Защита от алиасинга через публичные конструкторы
# ---------------------------------------------------------------------------


def direct_envelope(snapshot=None, receipts=()):
    return FsmEnvelope(
        schema="task_fsm/v1",
        task_id=7,
        revision=3,
        digest="sha256:current",
        hmac="hmac-sha256:signature",
        _snapshot={"state": "planning"} if snapshot is None else snapshot,
        key_id="local-v1",
        _validation_receipts=receipts,
    )


def test_fsm_envelope_constructor_copies_snapshot():
    source = {"state": "planning", "nested": {"value": 1}}
    original = {"state": "planning", "nested": {"value": 1}}
    env = direct_envelope(snapshot=source)

    source["state"] = "done"
    source["nested"]["value"] = 2

    assert env.snapshot == original


def test_fsm_envelope_constructor_copies_validation_receipts():
    receipt = {"receipt_id": "r-1", "nested": {"attempt": 1}}
    original = {"receipt_id": "r-1", "nested": {"attempt": 1}}
    env = direct_envelope(receipts=(receipt,))

    receipt["receipt_id"] = "changed"
    receipt["nested"]["attempt"] = 999

    assert env.validation_receipts[0] == original


def test_working_memory_document_constructor_copies_payload(tmp_path):
    source = {"task": {"task_id": 1}, "nested": {"value": 1}}
    original = {"task": {"task_id": 1}, "nested": {"value": 1}}
    document = WorkingMemoryDocument(
        path=tmp_path / "working_memory.json",
        status=DocumentState.NO_ACTIVE_TASK,
        _payload=source,
    )

    source["task"]["task_id"] = 99
    source["nested"]["value"] = 2

    assert document.payload == original


def test_to_dict_returns_independent_copy():
    snapshot = {"state": "planning", "nested": {"value": 1}}
    receipt = {"receipt_id": "r-1", "nested": {"attempt": 1}}
    env = direct_envelope(snapshot=snapshot, receipts=(receipt,))
    original_snapshot = {"state": "planning", "nested": {"value": 1}}
    original_receipt = {"receipt_id": "r-1", "nested": {"attempt": 1}}

    data = env.to_dict()
    data["snapshot"]["state"] = "done"
    data["snapshot"]["nested"]["value"] = 2
    data["validation_receipts"][0]["receipt_id"] = "changed"
    data["validation_receipts"][0]["nested"]["attempt"] = 999

    assert env.snapshot == original_snapshot
    assert env.validation_receipts[0] == original_receipt
