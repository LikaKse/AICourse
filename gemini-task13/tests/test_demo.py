"""Локальная демонстрация Дня 13: сценарий и отказы."""

from pathlib import Path

import demo


def test_demo_runs_full_scenario_and_is_deterministic():
    first = demo.run_demo()
    second = demo.run_demo()

    assert first == second
    assert len(first) >= 10
    assert not any("ОШИБКА ДЕМОНСТРАЦИИ" in line for line in first)


def test_demo_shows_required_stages_and_refusals():
    transcript = "\n".join(demo.run_demo())

    assert "[1] новая задача" in transcript
    assert "[2] агент предложил план" in transcript
    assert "[3] AgentReport" in transcript
    assert "authoritative=False" in transcript
    assert "[4a] агент пытается запустить шаг" in transcript
    assert "агенту запрещено" in transcript
    assert "[4b] контроллер пытается запустить шаг" in transcript
    assert "[5] контроллер утвердил план" in transcript
    assert "[7] пауза" in transcript
    assert "[8] продолжение" in transcript
    assert "[9] исполнитель выполнил шаг" in transcript
    assert "runner=build_weekly_report" in transcript
    assert "[12] валидатор подтвердил результат" in transcript
    assert "state=done" in transcript
    assert transcript.count("ОТКАЗ") >= 5


def test_demo_uses_real_local_runner_output():
    transcript = "\n".join(demo.run_demo())

    # Строка отчёта пришла из build_weekly_report, а не из AgentReport.
    assert "output_chars=" in transcript
    assert "executor_id=local" in transcript
    assert "actor_source=executor:local" in transcript


def test_demo_main_prints_transcript(capsys):
    demo.main()

    output = capsys.readouterr().out
    assert "[1] новая задача" in output
    assert "ОТКАЗ" in output


def test_demo_has_no_network_or_gemini_imports():
    root = Path(__file__).resolve().parent.parent
    text = (root / "demo.py").read_text(encoding="utf-8")

    for token in (
        "google.genai",
        "import requests",
        "requests.",
        "urllib.",
        "import socket",
        "socket.",
        "subprocess",
        "eval(",
        "exec(",
    ):
        assert token not in text, f"demo.py: найден запрещённый фрагмент {token!r}"
