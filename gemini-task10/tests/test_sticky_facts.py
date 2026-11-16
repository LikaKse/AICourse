"""Sticky Facts: обновление памяти после сообщений пользователя, история изменений, защита от выдумок,
различие «решение пользователя / принятое предложение / предложение ассистента» (fake-клиент)."""

import json

import pytest
from fakes import (
    FakeClient, all_text, api_error, facts_json, is_facts_request, request_user_message, text_response,
)

import config
from agent import Agent, build_facts_request
from context_manager import content_text, text_content
from facts_store import (
    FactsState, apply_operations, check_operation, normalize_key, numbers_supported, parse_operations, quote_in,
    render_facts,
)
from strategies.sticky_facts import StickyFactsStrategy


def scripted(facts_by_message: dict, answer="ок"):
    """facts_by_message: {номер сообщения пользователя: [операции]}; ответы — answer."""
    def respond(contents, cfg):
        if is_facts_request(cfg):
            n, _ = request_user_message(all_text(contents))
            return text_response(facts_json(*facts_by_message.get(n, [])), prompt=50, candidates=20, thoughts=5)
        return text_response(answer, prompt=30, candidates=2)
    return respond


def op(**kw):
    return {"op": "set", "kind": "user", "category": "other", "evidence": "", "proposal_evidence": "",
            "reason": "", "value": "", **kw}


# ----- обновление после каждого сообщения пользователя -----

def test_one_facts_call_per_user_message_before_answer(tmp_path):
    client = FakeClient(responder=scripted({1: [op(key="name", value="Лика", category="name",
                                                   evidence="Меня зовут Лика")]}))
    s = StickyFactsStrategy(tmp_path)
    agent = Agent(client, s)
    res = agent.ask("Привет! Меня зовут Лика.")
    assert [is_facts_request(r["config"]) for r in client.requests] == [True, False]   # facts ДО ответа
    assert len(res.service_calls) == 1 and res.service_calls[0].kind == "facts_update"
    assert res.service_calls[0].usage.input_tokens == 50 and res.service_calls[0].usage.output_tokens == 25
    assert s.facts.facts["name"].value == "Лика" and s.facts.facts["name"].source_message == 1
    # Факт уже виден в запросе-ответе на это же сообщение.
    answer_req = client.answer_requests()[0]["contents"]
    assert content_text(answer_req[0]).startswith(config.FACTS_HEADER) and "Лика" in content_text(answer_req[0])
    cfg = client.facts_requests()[0]["config"]
    assert cfg.response_mime_type == "application/json" and cfg.response_json_schema["required"] == ["operations"]
    assert str(cfg.thinking_config.thinking_level).endswith("LOW")
    assert client.answer_requests()[0]["config"].system_instruction == config.FACTS_ANSWER_SYSTEM_INSTRUCTION
    agent.ask("Как дела?")
    assert len(client.facts_requests()) == 2


def test_control_questions_do_not_update_facts():
    client = FakeClient(responder=scripted({}))
    s = StickyFactsStrategy()
    agent = Agent(client, s)
    agent.ask("Как меня зовут?", remember=False)
    assert client.facts_requests() == [] and s.facts.journal == [] and s.full_history == []


def test_facts_request_contains_only_facts_and_two_messages():
    text = build_facts_request({"name": {"value": "Лика"}}, "Совет: используйте venv.", "Ок, понятно", 7)
    assert "[6] Ассистент: Совет: используйте venv." in text and "[7] Пользователь: Ок, понятно" in text
    assert '"Лика"' in text and "НЕ решения пользователя" in text


def test_value_change_keeps_history_and_not_deleted():
    script = {
        1: [op(key="budget", value="15 000 000 VND", category="budget", evidence="бюджет — 15 000 000 VND")],
        3: [op(key="city", value="Нячанг", evidence="живу в Нячанге")],
        5: [op(key="budget", value="18 500 000 VND", category="budget", evidence="утвердили 18 500 000 VND")],
    }
    s = StickyFactsStrategy()
    agent = Agent(FakeClient(responder=scripted(script)), s)
    agent.ask("Предварительный бюджет — 15 000 000 VND.")
    agent.ask("Я живу в Нячанге.")
    agent.ask("Новости: утвердили 18 500 000 VND.")
    b = s.facts.facts["budget"]
    assert b.value == "18 500 000 VND" and b.source_message == 5
    assert b.history == [{"value": "15 000 000 VND", "kind": "user", "source_message": 1, "replaced_at_message": 5}]
    assert s.facts.facts["city"].value == "Нячанг"           # не тронутый факт не удалён
    text = render_facts(s.facts)
    assert "budget: 18 500 000 VND (ранее: 15 000 000 VND)" in text


def test_same_value_is_unchanged_not_new_version():
    state = FactsState()
    ops = [op(key="os", value="macOS", evidence="работаю на macOS")]
    apply_operations(state, ops, user_text="Я работаю на macOS", assistant_text=None, message_no=1)
    r = apply_operations(state, ops, user_text="Я работаю на macOS", assistant_text=None, message_no=3)
    assert r["unchanged"] and state.version == 1 and state.facts["os"].history == []


# ----- защита от выдумок -----

@pytest.mark.parametrize("operation, reason", [
    (op(key="name", value="Лика", evidence="Меня зовут Лика"), "цитаты нет"),                 # цитата выдумана
    (op(key="budget", value="20 000 000 VND", evidence="бюджет 15 000 000 VND"), "число"),    # число выдумано
    (op(key="", value="x", evidence="бюджет 15 000 000 VND"), "пустой ключ"),
    (op(key="x", value="", evidence="бюджет 15 000 000 VND"), "пустое значение"),
    (op(key="x", value="y", kind="boss", evidence="бюджет 15 000 000 VND"), "kind"),
    (op(key="tool", value="venv", kind="assistant_proposal", evidence="используйте poetry"), "ассистента"),
    (op(key="tool", value="venv", kind="accepted_proposal", evidence="бюджет 15 000 000 VND",
        proposal_evidence="используйте poetry"), "предложения ассистента"),
])
def test_invalid_operations_rejected(operation, reason):
    state = FactsState()
    r = apply_operations(state, [operation], user_text="Мой бюджет 15 000 000 VND.",
                         assistant_text="Советую: используйте venv.", message_no=3)
    assert not r["applied"] and reason in r["rejected"][0]["rejection"]
    assert state.facts == {} and state.version == 0


def test_delete_requires_reason_and_user_quote():
    state = FactsState()
    apply_operations(state, [op(key="docker", value="запрещён", category="constraint", evidence="Docker не использовать")],
                     user_text="Docker не использовать", assistant_text=None, message_no=1)
    r = apply_operations(state, [{"op": "delete", "key": "docker", "evidence": "", "reason": ""}],
                         user_text="Неважно", assistant_text=None, message_no=3)
    assert r["rejected"] and "docker" in state.facts
    r = apply_operations(state, [{"op": "delete", "key": "docker", "evidence": "ограничение про Docker снимаем",
                                  "reason": "пользователь отменил ограничение"}],
                         user_text="Ограничение про Docker снимаем.", assistant_text=None, message_no=5)
    assert r["applied"] and "docker" not in state.facts and state.deleted["docker"].history[-1]["deleted_at_message"] == 5
    assert "docker" not in render_facts(state)


def test_extraction_error_keeps_facts_and_is_counted():
    calls = {"n": 0}

    def respond(contents, cfg):
        if is_facts_request(cfg):
            calls["n"] += 1
            if calls["n"] == 2:
                return api_error(503)
            if calls["n"] == 3:
                return text_response("не json", prompt=40)
            return text_response(facts_json(op(key="name", value="Лика", evidence="зовут Лика")), prompt=40)
        return text_response("ок")

    s = StickyFactsStrategy()
    agent = Agent(FakeClient(responder=respond), s)
    agent.ask("Меня зовут Лика")
    r2 = agent.ask("Второе")
    r3 = agent.ask("Третье")
    assert r2.ok and not r2.service_calls[0].ok and "503" in r2.service_calls[0].error
    assert not r3.service_calls[0].ok and "не JSON" in r3.service_calls[0].error
    assert r3.service_calls[0].usage.input_tokens == 40           # неудачный, но оплаченный вызов учтён
    assert s.facts.facts["name"].value == "Лика"
    assert [e["ok"] for e in s.facts.journal] == [True, False, False]


# ----- предложение ассистента / принятое предложение / решение пользователя -----

def test_three_kinds_are_distinguished_and_rendered_separately():
    script = {
        3: [op(key="scheduler", value="launchd", kind="assistant_proposal", category="schedule",
               evidence="удобно сделать через launchd")],
        5: [op(key="http_library", value="httpx", kind="accepted_proposal", category="decision",
               evidence="Возьмём httpx", proposal_evidence="Подойдут requests или httpx")],
        7: [op(key="storage", value="JSON", category="decision", evidence="результаты храним в JSON")],
    }
    s = StickyFactsStrategy()
    agent = Agent(FakeClient(responder=scripted(script, answer="Хорошо")), s)
    s.add_exchange(text_content("user", "Запуск в 08:00"), text_content("model", "На macOS это удобно сделать через launchd."))
    agent.ask("Понятно.")                                     # #3: предложение без подтверждения
    s.full_history[-1] = text_content("model", "Подойдут requests или httpx.")
    agent.ask("Возьмём httpx.")                               # #5: явное согласие
    agent.ask("Давай так: результаты храним в JSON.")         # #7: собственное решение
    f = s.facts.facts
    assert (f["scheduler"].kind, f["http_library"].kind, f["storage"].kind) == (
        "assistant_proposal", "accepted_proposal", "user")
    assert f["scheduler"].source_message == 2                 # цитата — из сообщения ассистента
    text = render_facts(s.facts)
    blocks = text.split("\n")
    user_block = blocks[blocks.index("Данные и решения пользователя (сообщил или решил сам):") + 1]
    assert "storage: JSON" in user_block
    launchd_line = next(l for l in blocks if "launchd" in l)
    assert "предложение ассистента, не подтверждено" in launchd_line
    assert text.index("ЯВНО принятые") < text.index("http_library") < text.index("НЕ решения")


def test_proposal_cannot_overwrite_user_decision_but_acceptance_upgrades_proposal():
    state = FactsState()
    apply_operations(state, [op(key="storage", value="JSON", evidence="храним в JSON")],
                     user_text="Решено: храним в JSON", assistant_text=None, message_no=1)
    r = apply_operations(state, [op(key="storage", value="SQLite", kind="assistant_proposal",
                                    evidence="SQLite удобен")],
                         user_text="Ок", assistant_text="SQLite удобен для запросов", message_no=3)
    assert "не может заменить решение" in r["rejected"][0]["rejection"] and state.facts["storage"].value == "JSON"

    apply_operations(state, [op(key="formatter", value="ruff", kind="assistant_proposal", evidence="вариант — ruff")],
                     user_text="Какой форматтер?", assistant_text="Хороший вариант — ruff", message_no=5)
    r = apply_operations(state, [op(key="formatter", value="ruff", kind="accepted_proposal", evidence="берём ruff",
                                    proposal_evidence="вариант — ruff")],
                         user_text="Да, берём ruff", assistant_text="Хороший вариант — ruff", message_no=7)
    assert r["applied"] and state.facts["formatter"].kind == "accepted_proposal"
    assert state.facts["formatter"].history[0]["kind"] == "assistant_proposal"


def test_question_without_new_facts_gives_empty_operations():
    s = StickyFactsStrategy()
    Agent(FakeClient(responder=scripted({})), s).ask("Как оформить README?")
    assert s.facts.facts == {} and s.facts.journal[0]["proposed"] == 0


# ----- окно и сохранение -----

def test_window_and_facts_in_request_old_messages_dropped():
    script = {1: [op(key="name", value="Лика", evidence="зовут Лика")]}
    client = FakeClient(responder=scripted(script))
    s = StickyFactsStrategy(window_size=2)
    agent = Agent(client, s)
    agent.ask("Меня зовут Лика")
    for i in range(3):
        agent.ask(f"вопрос {i}")
    agent.ask("финал", remember=False)
    sent = [content_text(c) for c in client.answer_requests()[-1]["contents"]]
    assert len(sent) == 2 + 2 + 1 and sent[-1] == "финал"
    assert sent[2:4] == ["вопрос 2", "ок"]
    assert "Меня зовут Лика" not in sent                         # исходное сообщение вне окна…
    assert "Лика" in sent[0]                                     # …но факт сохранён в памяти


def test_persistence_between_restarts(tmp_path):
    s = StickyFactsStrategy(tmp_path)
    Agent(FakeClient(responder=scripted({1: [op(key="os", value="macOS", evidence="на macOS")]})), s).ask("Работаю на macOS")
    data = json.loads((tmp_path / config.STICKY_FACTS_FILE).read_text(encoding="utf-8"))
    assert data["facts"]["os"]["value"] == "macOS" and data["journal"][0]["applied"]
    r = StickyFactsStrategy(tmp_path)
    assert r.facts.facts["os"].value == "macOS" and len(r.full_history) == 2


def test_helpers():
    assert normalize_key(" Tests Count! ") == "tests_count"
    assert quote_in("живу в Нячанге", "Я живу в Нячанге, поэтому…") and not quote_in("ab", "ab")
    assert quote_in("Переносим на 09:30", "Переносим на 09:30.")
    assert numbers_supported("18 500 000 VND", "утвердили 18 500 000 VND") and not numbers_supported("19 млн", "18 млн")
    assert parse_operations('```json\n{"operations": []}\n```') == []
    with pytest.raises(Exception):
        parse_operations('{"ops": []}')
    assert check_operation(op(key="a", value="b", evidence="xyz"), FactsState(), "xyz!", None) == ("a", None)
