"""CLI, ключ API и .env."""

from pathlib import Path

import pytest
from fakes import FakeClient, text_response

import config
import main


def test_gitignore_excludes_env():
    lines = (config.ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in lines and "state/" in lines


def test_no_hardcoded_key_in_sources():
    for path in config.ROOT.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "AIza" not in text, path


def test_missing_key_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv(config.API_KEY_ENV, raising=False)
    with pytest.raises(main.ConfigError, match=".env.example"):
        main.load_api_key(tmp_path / "missing.env")


def test_key_loaded_from_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv(config.API_KEY_ENV, raising=False)
    env = tmp_path / ".env"
    env.write_text(f"{config.API_KEY_ENV}=test-key-from-dotenv\n", encoding="utf-8")
    assert main.load_api_key(env) == "test-key-from-dotenv"


def test_dry_run_without_api(capsys):
    assert main.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "50 сообщений" in out and "API не вызывался" in out


def test_experiment_requires_confirmation(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    monkeypatch.setattr(main, "make_client", lambda: pytest.fail("клиент не должен создаваться без подтверждения"))
    assert main.cmd_experiment("all", assume_yes=False, results_dir=Path("/nonexistent")) == 1
    assert "Отменено" in capsys.readouterr().out


def test_chat_persists_between_restarts(tmp_path, capsys):
    inputs = iter(["Меня зовут Лика", "stats", "exit"])
    client = FakeClient([text_response("Приятно познакомиться, Лика!")])
    assert main.cmd_chat(tmp_path, client=client, input_fn=lambda _: next(inputs)) == 0
    inputs = iter(["exit"])
    assert main.cmd_chat(tmp_path, client=FakeClient(), input_fn=lambda _: next(inputs)) == 0
    assert "Восстановлено: 2 сообщ." in capsys.readouterr().out


def test_chat_reports_corrupted_state(tmp_path, capsys):
    (tmp_path / config.FULL_HISTORY_FILE).write_text("not json", encoding="utf-8")
    assert main.cmd_chat(tmp_path, client=FakeClient(), input_fn=lambda _: "exit") == 1
    assert "Не удалось загрузить состояние" in capsys.readouterr().out


def test_audit_command_reads_only_and_guards_output(tmp_path, capsys):
    import experiments
    from test_compression_audit import _digest, _write_run

    results = tmp_path / "results"
    results.mkdir()
    _write_run(results, experiments.load_dialogue(), "- Лика живёт в Нячанге")
    before = _digest(results)
    assert main.main(["--audit", "--results-dir", str(results)]) == 0
    assert "# Офлайн-аудит сжатия контекста" in capsys.readouterr().out
    assert main.main(["--audit", "--results-dir", str(results), "--audit-output", str(results / "a.md")]) == 1
    assert not (results / "a.md").exists()
    out = tmp_path / "audit.md"
    assert main.main(["--audit", "--results-dir", str(results), "--audit-output", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("# Офлайн-аудит")
    assert _digest(results) == before


def test_audit_without_results(tmp_path, capsys):
    assert main.main(["--audit", "--results-dir", str(tmp_path)]) == 1
    assert "Аудит невозможен" in capsys.readouterr().out
