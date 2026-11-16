"""Стратегии Sliding Window и Summary, общий интерфейс и переключение (fake-клиент, без API)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fakes import FakeClient, all_text, echo_responder, is_summary_request, text_response

import config
from agent import Agent
from context_manager import content_text, text_content
from strategies import SlidingWindowStrategy, SummaryStrategy, make_strategy, strategy_class


def fill(strategy, n, start=0):
    calls = []
    for i in range(start, start + n):
        calls += strategy.add_exchange(text_content("user", f"u{i}"), text_content("model", f"m{i}"))
    return calls


# ----- фабрика и переключение -----

@pytest.mark.parametrize("name", ["sliding", "facts", "summary"])
def test_factory_creates_each_strategy(name, tmp_path):
    s = make_strategy(name, tmp_path)
    assert s.name == name and strategy_class(name) is type(s)


def test_unknown_strategy():
    with pytest.raises(ValueError, match="Неизвестная стратегия"):
        make_strategy("magic")


def test_config_default_strategy_is_valid():
    assert config.STRATEGY in config.STRATEGIES
    assert config.WINDOW_SIZE == config.KEEP_LAST == 6


def test_agent_switches_strategy():
    client = FakeClient(responder=lambda c, cfg: text_response("ok"))
    agent = Agent(client, SlidingWindowStrategy(window_size=2))
    agent.ask("a")
    agent.use_strategy(make_strategy("summary"))
    assert agent.strategy.name == "summary" and agent.strategy.agent is agent
    agent.ask("b")
    assert len(client.requests) == 2


# ----- Sliding Window -----

def test_sliding_window_drops_old_messages_from_request():
    client = FakeClient(responder=lambda c, cfg: text_response("ok"))
    strategy = SlidingWindowStrategy(window_size=6)
    agent = Agent(client, strategy)
    fill(strategy, 10)                                 # 20 сообщений
    agent.ask("вопрос", remember=False)
    sent = [content_text(c) for c in client.requests[0]["contents"]]
    assert sent == ["u7", "m7", "u8", "m8", "u9", "m9", "вопрос"]
    text = all_text(client.requests[0]["contents"])
    for i in range(7):
        assert f"u{i}" not in text.split() and f"m{i}" not in text.split()
    assert len(strategy.full_history) == 20            # полная история не обрезается


def test_sliding_window_short_history_and_zero():
    s = SlidingWindowStrategy(window_size=6)
    fill(s, 2)
    assert len(s.build_contents()) == 4
    z = SlidingWindowStrategy(window_size=0)
    fill(z, 3)
    assert z.build_contents(text_content("user", "q"))[0].parts[0].text == "q"
    with pytest.raises(ValueError):
        SlidingWindowStrategy(window_size=-1)


def test_sliding_window_no_service_calls_and_persistence(tmp_path):
    client = FakeClient(responder=lambda c, cfg: text_response("ответ"))
    agent = Agent(client, SlidingWindowStrategy(tmp_path, window_size=4))
    for i in range(5):
        res = agent.ask(f"q{i}")
        assert res.service_calls == []
    assert len(client.requests) == 5                   # только ответы, служебных вызовов нет
    restored = SlidingWindowStrategy(tmp_path, window_size=4)
    assert len(restored.full_history) == 10
    assert [content_text(c) for c in restored.build_contents()] == ["q3", "ответ", "q4", "ответ"]


def test_sliding_memory_text_one_line_per_message():
    s = SlidingWindowStrategy(window_size=2)
    fill(s, 3)
    assert s.memory_text() == "Пользователь: u2\nАссистент: m2"


# ----- Summary (как в Дне 9) -----

def test_summary_strategy_threshold_and_calls_recorded():
    client = FakeClient(responder=echo_responder())
    strategy = SummaryStrategy()
    agent = Agent(client, strategy)
    calls = fill(strategy, 7)
    assert calls == []                                 # 14 < 16
    calls = fill(strategy, 1, 7)                       # 16 → сжатие 10 сообщений
    assert len(calls) == 1 and calls[0].kind == "summary" and calls[0].ok
    assert calls[0].usage.input_tokens > 0 and calls[0].request_tokens is not None
    assert strategy.context.summary.summarized_upto == 10
    contents = strategy.build_contents(text_content("user", "q"))
    assert len(contents) == 2 + 6 + 1
    assert content_text(contents[0]).startswith(config.SUMMARY_HEADER)
    assert agent.system_instruction == config.SYSTEM_INSTRUCTION


def test_summary_on_50_messages_makes_4_calls():
    client = FakeClient(responder=echo_responder())
    strategy = SummaryStrategy()
    Agent(client, strategy)
    calls = fill(strategy, 25)
    assert [c.meta["event"]["end"] for c in calls] == [10, 20, 30, 40]
    assert len(client.summary_requests()) == 4


def test_summary_strategy_persistence(tmp_path):
    client = FakeClient(responder=echo_responder())
    s = SummaryStrategy(tmp_path)
    Agent(client, s)
    fill(s, 8)
    r = SummaryStrategy(tmp_path)
    assert r.context.summary.summarized_upto == 10 and len(r.full_history) == 16
    assert (tmp_path / config.SUMMARY_FILE).is_file() and (tmp_path / config.FULL_HISTORY_FILE).is_file()


DAY9 = config.ROOT.parent / "gemini-task9"

DAY_RUNNER = r"""
import json, sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
sys.path.append(sys.argv[3])   # fake_client.py Дня 10 (в Дне 9 такого модуля нет)
from fakes import FakeClient, echo_responder
from agent import Agent
from context_manager import ContextManager, content_text, text_content
import experiments
client = FakeClient(responder=echo_responder())
agent = Agent(client, ContextManager())
dialogue = experiments.load_dialogue()
for i in range(0, len(dialogue), 2):
    agent.context.add_exchange(dialogue[i], dialogue[i + 1])
for q in ["Как меня зовут?", "Где я живу?"]:
    agent.ask(q, remember=False)
out = [{"system": r["config"].system_instruction,
        "thinking": str(r["config"].thinking_config.thinking_level) if r["config"].thinking_config else None,
        "max": r["config"].max_output_tokens,
        "contents": [[c.role, content_text(c)] for c in r["contents"]]} for r in client.requests]
print(json.dumps({"requests": out, "summary": agent.context.summary.text}, ensure_ascii=False))
"""


def run_project(root: Path) -> dict:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run([sys.executable, "-B", "-c", DAY_RUNNER, str(root), str(config.ROOT / "tests"), str(config.ROOT)],
                          cwd=root, capture_output=True, text=True, env=env, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


@pytest.mark.skipif(not DAY9.is_dir(), reason="рядом нет gemini-task9")
def test_summary_strategy_identical_to_day9():
    """Те же запросы (summary и ответы), та же системная инструкция и параметры — что и в коде Дня 9."""
    day9, day10 = run_project(DAY9), run_project(config.ROOT)
    assert len(day10["requests"]) == 4 + 2
    assert day10 == day9
