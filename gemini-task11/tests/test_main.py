"""CLI: без платных вызовов без подтверждения, офлайн-режимы, команды чата."""

import pytest

import main
from fakes import FakeClient, echo_responder


@pytest.fixture
def no_real_client(monkeypatch):
    def boom():
        raise AssertionError("реальный клиент Gemini создавать нельзя")
    monkeypatch.setattr(main, "make_client", boom)


def test_dry_run_does_not_create_real_client(no_real_client, capsys):
    assert main.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Платных запросов не было" in out and "Офлайн-матрица совпала с ожиданиями: да" in out


def test_experiment_without_confirmation_does_nothing(no_real_client, tmp_path, capsys):
    assert main.cmd_experiment(tmp_path, 6, False, False, input_fn=lambda _: "n") == 0
    assert "Отменено" in capsys.readouterr().out
    assert not any(tmp_path.iterdir())


def test_offline_experiment_and_overwrite_guard(no_real_client, tmp_path, capsys):
    assert main.main(["--offline-experiment", "--results-dir", str(tmp_path)]) == 0
    assert (tmp_path / "experiment.json").is_file() and (tmp_path / "report.md").is_file()
    assert (tmp_path / "state" / "base" / "long_term.json").is_file()
    assert main.main(["--offline-experiment", "--results-dir", str(tmp_path)]) == 2
    assert "уже есть результаты" in capsys.readouterr().err
    assert main.main(["--report", "offline", "--results-dir", str(tmp_path)]) == 0


def test_chat_commands(tmp_path, capsys):
    lines = iter(["Привет", "/memory", "/prompt", "/clear short", "/clear nope", "/task done", "/rules", "/x", "exit"])
    client = FakeClient(responder=echo_responder())
    assert main.cmd_chat(tmp_path, 6, client=client, input_fn=lambda _: next(lines)) == 0
    out = capsys.readouterr().out
    assert "Ответ по контексту" in out and "[память: WM +0, LT +0, отклонено 0]" in out
    assert "Очищено: Краткосрочная память" in out and "Неизвестный слой" in out
    assert "Нет активной задачи." in out and "[Запись]" in out
    assert (tmp_path / "short_term.json").is_file()


def test_negative_window_is_error():
    with pytest.raises(SystemExit):
        main.main(["--dry-run", "--window", "-1"])
