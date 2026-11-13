"""Агент: запросы со сжатым контекстом, создание summary через Gemini, обработка ошибок API."""

from fakes import FakeClient, api_error, empty_response, is_summary_request, text_response

import config
from agent import Agent, build_summary_request
from context_manager import ContextManager, content_text, text_content


def test_ask_ok_and_remembered():
    client = FakeClient([text_response("Привет!", prompt=40)])
    agent = Agent(client)
    res = agent.ask("Привет")
    assert res.ok and res.text == "Привет!"
    assert res.usage.input_tokens == 40 and res.request_tokens is not None
    assert [content_text(c) for c in agent.context.full_history] == ["Привет", "Привет!"]
    cfg = client.requests[0]["config"]
    assert cfg.system_instruction == config.SYSTEM_INSTRUCTION
    assert client.requests[0]["model"] == config.MODEL


def test_ask_without_remember_does_not_change_history():
    client = FakeClient([text_response("a"), text_response("b")])
    agent = Agent(client)
    agent.ask("вопрос 1", remember=False)
    agent.ask("вопрос 2", remember=False)
    assert agent.context.full_history == []
    assert [content_text(c) for c in client.requests[1]["contents"]] == ["вопрос 2"]


def test_api_error_is_reported_and_history_untouched(monkeypatch):
    monkeypatch.setenv(config.API_KEY_ENV, "SECRET-KEY-123456")
    client = FakeClient([api_error(503, "overloaded; key=SECRET-KEY-123456")])
    agent = Agent(client)
    res = agent.ask("Привет")
    assert res.status == "api_error"
    assert "временная ошибка сервера" in res.text and "503" in res.text
    assert "SECRET-KEY-123456" not in res.text
    assert agent.context.full_history == []


def test_network_error_and_auth_error_messages():
    agent = Agent(FakeClient([ConnectionError("no route"), api_error(401, "bad key", "UNAUTHENTICATED")]))
    r1 = agent.ask("a")
    r2 = agent.ask("b")
    assert r1.status == "api_error" and "Не удалось связаться" in r1.text
    assert r2.status == "api_error" and "неверный API-ключ" in r2.text


def test_empty_response():
    agent = Agent(FakeClient([empty_response("SAFETY")]))
    res = agent.ask("a")
    assert res.status == "empty_response" and "SAFETY" in res.text
    assert agent.context.full_history == []


def test_summarize_prompt_keeps_facts_and_previous_summary():
    client = FakeClient([text_response("- Имя: Лика", prompt=80, thoughts=3)])
    agent = Agent(client)
    out = agent.summarize("- Город: Нячанг", [text_content("user", "Меня зовут Лика"),
                                              text_content("model", "Приятно познакомиться")])
    assert out.text == "- Имя: Лика"
    req = client.requests[0]
    assert is_summary_request(req["config"])
    sent = content_text(req["contents"][0])
    assert "- Город: Нячанг" in sent and "Пользователь: Меня зовут Лика" in sent
    for word in ("имена", "даты", "бюджеты", "числа", "ограничения", "решения"):
        assert word in (config.SUMMARY_SYSTEM_INSTRUCTION + sent).lower()
    assert out.meta["usage"]["input_tokens"] == 80 and out.meta["usage"]["output_tokens"] == 3 + 3   # 3 слова ответа + 3 thinking
    assert out.meta["request_tokens"] is not None


def test_build_summary_request_without_previous():
    text = build_summary_request("", [text_content("user", "x")])
    assert "(пока нет)" in text and "[1] Пользователь: x" in text


def test_compression_through_agent_and_summary_api_error():
    """Ошибка API при создании summary не ломает ответ; при следующей попытке сжатие проходит."""
    state = {"fail": True}

    def responder(contents, cfg):
        if is_summary_request(cfg):
            if state["fail"]:
                state["fail"] = False
                return api_error(500, "internal", "INTERNAL")
            return text_response("SUMMARY")
        return text_response("ok")

    client = FakeClient(responder=responder)
    agent = Agent(client)
    for i in range(7):
        agent.context.add_exchange(text_content("user", f"u{i}"), text_content("model", f"m{i}"))
    res = agent.ask("q7")                         # 16 сообщений → попытка сжатия → 500
    assert res.ok
    assert len(res.compression) == 1 and not res.compression[0].ok
    assert "500" in res.compression[0].error
    assert agent.context.summary.summarized_upto == 0
    res = agent.ask("q8")                         # повторная попытка перед запросом — успешна
    assert res.compression[0].ok
    assert agent.context.summary.text == "SUMMARY"
    last = client.answer_requests()[-1]["contents"]
    assert content_text(last[0]).endswith("SUMMARY")
    assert len(last) == 2 + 6 + 1                 # summary-пара + 6 последних + вопрос
    assert len(client.summary_requests()) == 2


def test_agent_uses_given_context(tmp_path):
    cm = ContextManager.from_dir(tmp_path)
    agent = Agent(FakeClient([text_response("x")]), cm)
    assert cm.summarizer == agent.summarize
    agent.ask("a")
    assert len(ContextManager.from_dir(tmp_path).full_history) == 2
