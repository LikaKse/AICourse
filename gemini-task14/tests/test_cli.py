"""Тесты CLI управления инвариантами (этап 4)."""

import json

import pytest

from invariant_cli import main
from invariants.models import InvariantCategory, InvariantCheck
from invariants.store import InvariantStore


def check_json(kind="forbidden_term", value="Docker", **extra):
    data = {"kind": kind, "value": value}
    data.update(extra)
    return json.dumps(data, ensure_ascii=False)


def run(tmp_path, *args):
    return main(["--state-dir", str(tmp_path), *args])


def add_rule(tmp_path, rule_id="stack-001", reason="initial"):
    return run(
        tmp_path,
        "add",
        "--id", rule_id,
        "--category", "stack",
        "--text", "No Docker",
        "--check", check_json(),
        "--reason", reason,
        "--actor", "tester",
    )


def test_list_empty(tmp_path, capsys):
    code = run(tmp_path, "list", "--json")
    out = capsys.readouterr()

    assert code == 0
    data = json.loads(out.out)
    assert data["revision"] == 0
    assert data["rules"] == []


def test_add_and_list(tmp_path, capsys):
    code = add_rule(tmp_path)
    out = capsys.readouterr()

    assert code == 0
    assert "stack-001" in out.out

    snapshot = InvariantStore(tmp_path).load()
    assert snapshot.revision == 1
    rule = snapshot.rule("stack-001")
    assert rule is not None
    assert rule.version == 1
    assert rule.active is True
    assert rule.category == InvariantCategory.STACK
    assert rule.text == "No Docker"
    assert rule.checks == (
        InvariantCheck(kind="forbidden_term", value="Docker"),
    )

    code = run(tmp_path, "list", "--json")
    out = capsys.readouterr()
    assert code == 0
    data = json.loads(out.out)
    assert data["revision"] == 1
    assert data["rules"][0]["id"] == "stack-001"
    assert data["rules"][0]["checks"][0]["kind"] == "forbidden_term"


def test_list_human_output(tmp_path, capsys):
    assert add_rule(tmp_path) == 0
    capsys.readouterr()

    code = run(tmp_path, "list")
    out = capsys.readouterr()

    assert code == 0
    assert "revision: 1" in out.out
    assert "stack-001" in out.out
    assert "active" in out.out
    assert "forbidden_term" in out.out


def test_add_max_value_check(tmp_path, capsys):
    code = run(
        tmp_path,
        "add",
        "--id", "budget-001",
        "--category", "business",
        "--text", "Budget",
        "--check", check_json("max_value", 100, key="budget_usd"),
        "--reason", "budget rule",
        "--actor", "tester",
    )
    out = capsys.readouterr()

    assert code == 0
    assert "budget-001" in out.out
    rule = InvariantStore(tmp_path).load().rule("budget-001")
    assert rule is not None
    assert rule.checks[0].kind == "max_value"
    assert rule.checks[0].value == 100
    assert rule.checks[0].key == "budget_usd"


def test_update_preserves_id_and_increments_version(tmp_path, capsys):
    assert add_rule(tmp_path, rule_id="arch-001") == 0
    capsys.readouterr()

    code = run(
        tmp_path,
        "update",
        "--id", "arch-001",
        "--version", "1",
        "--text", "New text",
        "--check", check_json(value="Podman"),
        "--reason", "tighten",
        "--actor", "tester",
    )
    out = capsys.readouterr()

    assert code == 0
    assert "arch-001" in out.out

    snapshot = InvariantStore(tmp_path).load()
    rule = snapshot.rule("arch-001")
    assert rule is not None
    assert rule.id == "arch-001"
    assert rule.version == 2
    assert rule.text == "New text"
    assert rule.checks == (
        InvariantCheck(kind="forbidden_term", value="Podman"),
    )
    assert snapshot.rule("new-id") is None
    assert len(snapshot.rules) == 1


def test_disable(tmp_path, capsys):
    assert add_rule(tmp_path) == 0
    capsys.readouterr()

    code = run(
        tmp_path,
        "disable",
        "--id", "stack-001",
        "--version", "1",
        "--reason", "retired",
        "--actor", "tester",
    )
    out = capsys.readouterr()

    assert code == 0
    assert "stack-001" in out.out

    snapshot = InvariantStore(tmp_path).load()
    rule = snapshot.rule("stack-001")
    assert rule is not None
    assert rule.active is False
    assert rule.version == 2
    history = InvariantStore(tmp_path).read_history()
    assert [item.action for item in history] == ["add", "disable"]


def test_history_json(tmp_path, capsys):
    assert add_rule(tmp_path) == 0
    capsys.readouterr()
    assert run(
        tmp_path,
        "update",
        "--id", "stack-001",
        "--version", "1",
        "--text", "Updated",
        "--check", check_json(value="Podman"),
        "--reason", "tighten",
        "--actor", "tester",
    ) == 0
    capsys.readouterr()
    assert run(
        tmp_path,
        "disable",
        "--id", "stack-001",
        "--version", "2",
        "--reason", "retired",
        "--actor", "tester",
    ) == 0
    capsys.readouterr()

    code = run(tmp_path, "history", "--json")
    out = capsys.readouterr()

    assert code == 0
    data = json.loads(out.out)
    assert [item["action"] for item in data] == ["add", "update", "disable"]
    assert [item["store_revision"] for item in data] == [1, 2, 3]
    assert [item["reason"] for item in data] == [
        "initial", "tighten", "retired"]
    assert all(item["actor"] == "tester" for item in data)


def test_history_human_output(tmp_path, capsys):
    assert add_rule(tmp_path) == 0
    capsys.readouterr()

    code = run(tmp_path, "history")
    out = capsys.readouterr()

    assert code == 0
    assert "add" in out.out
    assert "initial" in out.out
    assert "tester" in out.out


def test_add_requires_reason(tmp_path, capsys):
    code = run(
        tmp_path,
        "add",
        "--id", "stack-001",
        "--category", "stack",
        "--text", "No Docker",
        "--check", check_json(),
        "--actor", "tester",
    )
    out = capsys.readouterr()

    assert code == 1
    assert "причина" in out.err
    store = InvariantStore(tmp_path)
    assert not store.snapshot_path.exists()
    assert not store.history_path.exists()


@pytest.mark.parametrize("command, extra", [
    ("update", ["--version", "1", "--text", "x", "--check", check_json()]),
    ("disable", ["--version", "1"]),
])
def test_update_and_disable_require_reason(tmp_path, capsys, command, extra):
    assert add_rule(tmp_path) == 0
    capsys.readouterr()

    code = run(
        tmp_path, command,
        "--id", "stack-001",
        *extra,
        "--actor", "tester",
    )
    out = capsys.readouterr()

    assert code == 1
    assert "причина" in out.err
    snapshot = InvariantStore(tmp_path).load()
    assert snapshot.revision == 1
    assert len(InvariantStore(tmp_path).read_history()) == 1


@pytest.mark.parametrize("command, extra", [
    ("update", ["--text", "x", "--check", check_json()]),
    ("disable", []),
])
def test_wrong_version_does_not_write(tmp_path, capsys, command, extra):
    assert add_rule(tmp_path) == 0
    capsys.readouterr()

    code = run(
        tmp_path, command,
        "--id", "stack-001",
        "--version", "99",
        *extra,
        "--reason", "wrong version",
        "--actor", "tester",
    )
    out = capsys.readouterr()

    assert code == 1
    assert "версия" in out.err
    snapshot = InvariantStore(tmp_path).load()
    assert snapshot.revision == 1
    assert snapshot.rule("stack-001").version == 1
    assert len(InvariantStore(tmp_path).read_history()) == 1


@pytest.mark.parametrize("command, extra", [
    ("update", ["--text", "x", "--check", check_json()]),
    ("disable", []),
])
def test_unknown_id_does_not_write(tmp_path, capsys, command, extra):
    assert add_rule(tmp_path) == 0
    capsys.readouterr()

    code = run(
        tmp_path, command,
        "--id", "missing",
        "--version", "1",
        *extra,
        "--reason", "unknown",
        "--actor", "tester",
    )
    out = capsys.readouterr()

    assert code == 1
    assert "не найден" in out.err
    snapshot = InvariantStore(tmp_path).load()
    assert snapshot.revision == 1
    assert len(InvariantStore(tmp_path).read_history()) == 1


def test_update_rejects_new_id_option(tmp_path, capsys):
    assert add_rule(tmp_path, rule_id="arch-001") == 0
    capsys.readouterr()

    with pytest.raises(SystemExit) as exc:
        run(
            tmp_path,
            "update",
            "--id", "arch-001",
            "--new-id", "arch-002",
            "--version", "1",
            "--text", "x",
            "--check", check_json(),
            "--reason", "rename",
            "--actor", "tester",
        )

    assert exc.value.code == 2
    out = capsys.readouterr()
    assert "new-id" in out.err

    snapshot = InvariantStore(tmp_path).load()
    assert snapshot.revision == 1
    assert snapshot.rule("arch-001") is not None
    assert snapshot.rule("arch-002") is None


def test_duplicate_add_does_not_write(tmp_path, capsys):
    assert add_rule(tmp_path) == 0
    capsys.readouterr()

    code = add_rule(tmp_path, reason="duplicate")
    out = capsys.readouterr()

    assert code == 1
    assert "уже существует" in out.err
    snapshot = InvariantStore(tmp_path).load()
    assert snapshot.revision == 1
    assert len(snapshot.rules) == 1
    assert len(InvariantStore(tmp_path).read_history()) == 1


def test_add_requires_check(tmp_path, capsys):
    code = run(
        tmp_path,
        "add",
        "--id", "stack-001",
        "--category", "stack",
        "--text", "No Docker",
        "--reason", "x",
        "--actor", "tester",
    )
    out = capsys.readouterr()

    assert code == 1
    assert "проверка" in out.err
    assert not InvariantStore(tmp_path).snapshot_path.exists()


def test_invalid_check_json_fails_before_write(tmp_path, capsys):
    code = run(
        tmp_path,
        "add",
        "--id", "stack-001",
        "--category", "stack",
        "--text", "No Docker",
        "--check", "not-json",
        "--reason", "x",
        "--actor", "tester",
    )
    out = capsys.readouterr()

    assert code == 1
    assert "JSON" in out.err
    store = InvariantStore(tmp_path)
    assert not store.snapshot_path.exists()
    assert not store.history_path.exists()


def test_history_empty_store_succeeds(tmp_path, capsys):
    code = run(tmp_path, "history", "--json")
    out = capsys.readouterr()

    assert code == 0
    assert json.loads(out.out) == []


def test_history_rejects_corrupt_snapshot_without_history(tmp_path, capsys):
    store = InvariantStore(tmp_path)
    store.state_dir.mkdir(parents=True, exist_ok=True)
    store.snapshot_path.write_text("{broken", encoding="utf-8")
    before = store.snapshot_path.read_bytes()

    code = run(tmp_path, "history", "--json")
    out = capsys.readouterr()

    assert code == 1
    assert out.out == ""
    assert "Ошибка" in out.err
    assert store.snapshot_path.read_bytes() == before


@pytest.mark.parametrize("json_flag", [(), ("--json",)])
def test_history_rejects_corrupt_snapshot_with_history(
        tmp_path, capsys, json_flag):
    assert add_rule(tmp_path) == 0
    capsys.readouterr()
    store = InvariantStore(tmp_path)
    store.snapshot_path.write_text("{broken", encoding="utf-8")
    before = store.snapshot_path.read_bytes()

    code = run(tmp_path, "history", *json_flag)
    out = capsys.readouterr()

    assert code == 1
    assert out.out == ""
    assert "Ошибка" in out.err
    assert store.snapshot_path.read_bytes() == before


def test_history_rejects_corrupt_history(tmp_path, capsys):
    assert add_rule(tmp_path) == 0
    capsys.readouterr()
    store = InvariantStore(tmp_path)
    store.history_path.write_text("{broken\n", encoding="utf-8")

    code = run(tmp_path, "history", "--json")
    out = capsys.readouterr()

    assert code == 1
    assert out.out == ""
    assert "Ошибка" in out.err
