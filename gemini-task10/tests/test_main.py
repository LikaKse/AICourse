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


def test_dry_run_without_api(capsys, monkeypatch):
    monkeypatch.setattr(main, "make_client", lambda: pytest.fail("dry-run не должен создавать клиента"))
    assert main.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "50 сообщений" in out and "API не вызывался" in out
    assert "facts   : обработка диалога 25 + вопросы 15 = 40" in out
    assert "summary : обработка диалога  4 + вопросы 15 = 19" in out
    assert out.count("совпадает с прогнозом") == 3 and "НЕ совпадает" not in out


def test_dry_run_does_not_write_results(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    assert main.main(["--dry-run", "--strategy", "sliding"]) == 0
    assert not (tmp_path / "results").exists()


def test_experiment_requires_confirmation(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    monkeypatch.setattr(main, "make_client", lambda: pytest.fail("клиент не должен создаваться без подтверждения"))
    assert main.cmd_experiment("all", assume_yes=False, results_dir=Path("/nonexistent")) == 1
    out = capsys.readouterr().out
    assert "Отменено" in out


def test_experiment_with_fake_client_saves_results(tmp_path, monkeypatch, capsys):
    from fakes import echo_responder
    monkeypatch.setattr(main, "make_client", lambda: FakeClient(responder=echo_responder()))
    assert main.cmd_experiment("all", assume_yes=True, results_dir=tmp_path) == 0
    for name in ("experiment_sliding.json", "experiment_facts.json", "experiment_summary.json",
                 "comparison.json", "report.md"):
        assert (tmp_path / name).is_file(), name
    assert main.cmd_report(tmp_path) == 0
    assert main.cmd_audit("facts", tmp_path) == 0
    out = capsys.readouterr().out
    assert "| Стратегия | Деталей сохранено (из 36)" in out and "Офлайн-аудит" in out


def test_report_and_audit_without_results(tmp_path, capsys):
    assert main.cmd_report(tmp_path) == 1
    assert main.cmd_audit("sliding", tmp_path) == 1


def test_branch_demo_offline(capsys, monkeypatch):
    monkeypatch.setattr(main, "make_client", lambda: pytest.fail("branch-demo не вызывает API"))
    assert main.main(["--branch-demo", "--strategy", "facts"]) == 0
    out = capsys.readouterr().out
    assert "Ветка A" in out and "Ветка B" in out and "API не вызывался" in out


@pytest.mark.parametrize("strategy", config.STRATEGIES)
def test_chat_persists_between_restarts(tmp_path, capsys, strategy):
    from fakes import echo_responder
    inputs = iter(["Меня зовут Лика", "/stats", "exit"])
    assert main.cmd_chat(strategy, tmp_path, client=FakeClient(responder=echo_responder()),
                         input_fn=lambda _: next(inputs)) == 0
    inputs = iter(["exit"])
    assert main.cmd_chat(strategy, tmp_path, client=FakeClient(), input_fn=lambda _: next(inputs)) == 0
    out = capsys.readouterr().out
    assert "2 сообщ. в полной истории" in out
    assert (tmp_path / strategy / "branches" / "main" / config.FULL_HISTORY_FILE).is_file()


def test_chat_branch_commands(tmp_path, capsys):
    from fakes import echo_responder
    inputs = iter(["Меня зовут Лика", "/checkpoint cp1", "/branch A cp1", "Бюджет 22 000 000 VND",
                   "/branch B cp1", "Я живу в Дананге", "/switch A", "/memory", "/restore cp1", "/stats",
                   "/branches", "/switch Z", "/unknown", "exit"])
    assert main.cmd_chat("sliding", tmp_path, client=FakeClient(responder=echo_responder()),
                         input_fn=lambda _: next(inputs)) == 0
    out = capsys.readouterr().out
    memory_a = out.split("Текущая ветка: A.")[1].split("Ветка A возвращена")[0]
    assert "22 000 000" in memory_a and "Дананг" not in memory_a
    assert "Ветка A возвращена к checkpoint cp1" in out
    assert "Ошибка: Ветки 'Z' нет" in out and "Команды:" in out


def test_chat_reports_corrupted_state(tmp_path, capsys):
    d = tmp_path / "summary" / "branches" / "main"
    d.mkdir(parents=True)
    (tmp_path / "summary" / "session.json").write_text(
        '{"format_version": 1, "strategy": "summary", "current": "main", "branches": ["main"]}', encoding="utf-8")
    (d / config.FULL_HISTORY_FILE).write_text("not json", encoding="utf-8")
    assert main.cmd_chat("summary", tmp_path, client=FakeClient(), input_fn=lambda _: "exit") == 1
    assert "Не удалось загрузить состояние" in capsys.readouterr().out
