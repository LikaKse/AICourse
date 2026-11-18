"""Agent: один вызов менеджера памяти на сообщение, ошибки, токены по слоям, маскировка ключа."""

import json

from fakes import FakeClient, api_error, echo_responder, empty_response, is_memory_request, ops_json, text_response

from agent import Agent, build_memory_request
from memory import MemoryManager
from memory.storage import content_text


def name_op():
    return {"layer": "long_term", "key": "name", "value": "Лика", "category": "name", "evidence": "Меня зовут Лика"}


def test_one_memory_call_before_each_answer(tmp_path):
    client = FakeClient(responder=echo_responder(lambda text: ops_json(name_op()) if "Лика" in text else ops_json()))
    agent = Agent(client, MemoryManager(tmp_path))
    res = agent.ask("Меня зовут Лика.")
    assert res.ok and [is_memory_request(r["config"]) for r in client.requests] == [True, False]
    assert agent.memory.long_term.state.facts["name"].value == "Лика"
    assert "Лика" in content_text(client.requests[1]["contents"][0])     # уже в этом же ответе
    agent.ask("Как дела?")
    assert len(client.memory_requests()) == 2 and agent.memory.short_term.stats()["stored_messages"] == 4


def test_control_question_changes_nothing(tmp_path):
    client = FakeClient(responder=echo_responder())
    agent = Agent(client, MemoryManager(tmp_path))
    res = agent.ask("Как меня зовут?", remember=False)
    assert res.ok and client.memory_requests() == [] and not any(tmp_path.iterdir())


def test_memory_request_contains_only_memory_and_two_messages(tmp_path):
    client = FakeClient(responder=echo_responder())
    agent = Agent(client, MemoryManager(tmp_path))
    agent.play_exchange("первое сообщение", "первый ответ")
    agent.play_exchange("второе сообщение", "второй ответ")
    agent.ask("третье сообщение")
    text = content_text(client.memory_requests()[-1]["contents"][0])
    assert "[4] Ассистент: второй ответ" in text and "[5] Пользователь: третье сообщение" in text
    assert "первое сообщение" not in text and "первый ответ" not in text
    assert client.memory_requests()[-1]["config"].response_mime_type == "application/json"


def test_memory_errors_leave_layers_untouched(tmp_path):
    for bad in (api_error(503), text_response("не json"), empty_response()):
        client = FakeClient(script=[bad, text_response("ответ")])
        agent = Agent(client, MemoryManager(tmp_path / str(id(bad))))
        res = agent.ask("Меня зовут Лика.")
        call = res.service_calls[0]
        assert call.ok is False and call.error
        assert res.ok                                                  # ответ пользователю всё равно есть
        assert agent.memory.long_term.state.facts == {} and agent.memory.working.task is None
        assert agent.memory.short_term.stats()["stored_messages"] == 2


def test_failed_answer_not_written_to_short_term(tmp_path):
    client = FakeClient(script=[text_response(ops_json()), api_error(500, "boom", "INTERNAL")])
    agent = Agent(client, MemoryManager(tmp_path))
    res = agent.ask("Привет")
    assert res.status == "api_error" and agent.memory.short_term.is_empty()


def test_token_breakdown_per_layer_sums_to_total(tmp_path):
    client = FakeClient(responder=echo_responder(lambda text: ops_json(name_op()) if "Лика" in text else ops_json()))
    agent = Agent(client, MemoryManager(tmp_path))
    agent.play_exchange("Меня зовут Лика.", "Привет!")
    res = agent.ask("Как меня зовут?", remember=False)
    t = res.tokens
    assert t["layers"]["working"] == 0 and t["layers"]["long_term"] > 0 and t["layers"]["short_term"] > 0
    assert t["sum_of_parts"] == t["total"] == res.request_tokens      # фейковый счёт аддитивен
    assert res.prompt_layers == ["long_term", "short_term"]


def test_api_key_never_written_to_files(tmp_path, monkeypatch):
    secret = "AIzaSy-test-secret-0123456789"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    op = {"layer": "long_term", "key": "note", "value": f"ключ {secret}", "category": "knowledge",
          "evidence": f"мой ключ {secret}"}
    client = FakeClient(responder=echo_responder(lambda text: ops_json(op)))
    Agent(client, MemoryManager(tmp_path)).ask(f"Запомни: мой ключ {secret}")
    for f in tmp_path.iterdir():
        assert secret not in f.read_text(encoding="utf-8")


def test_build_memory_request_without_previous_answer():
    text = build_memory_request({"long_term": {}, "working": {}}, None, "привет", 1)
    assert "(нет)" in text and "[1] Пользователь: привет" in text
    assert json.dumps({}) in text


def test_memory_instruction_asks_to_record_unconfirmed_proposals():
    """Q7 реального прогона v1: launchd (предложение ассистента + уклончивый ответ) не был записан."""
    import config
    text = config.MEMORY_SYSTEM_INSTRUCTION
    assert "ОБЯЗАТЕЛЬНО проверь предыдущее сообщение ассистента" in text
    assert "«подумаю»" in text and "kind=\"assistant_proposal\"" in text
    client = FakeClient(responder=echo_responder())
    Agent(client).ask("Подумаю.")
    assert client.memory_requests()[0]["config"].system_instruction == text


def test_evasive_reply_records_proposal_and_new_task_from_one_message(tmp_path):
    """Одно сообщение пользователя → предложение ассистента в LT и новая задача в WM одновременно."""
    def extract(text):
        if "Подумаю над этим позже" not in text:
            return ops_json()
        return ops_json(
            {"key": "scheduler", "value": "launchd", "kind": "assistant_proposal", "category": "decision",
             "evidence": "можно использовать launchd"},
            {"layer": "working", "key": "goal", "value": "еженедельный отчёт", "evidence": "сделать еженедельный отчёт"})
    agent = Agent(FakeClient(responder=echo_responder(extract)), MemoryManager(tmp_path))
    agent.play_exchange("Чем запускать по расписанию?", "На macOS можно использовать launchd.")
    call = agent.play_exchange("Подумаю над этим позже. Теперь нужно сделать еженедельный отчёт.", "Хорошо.")
    assert call.ok and call.meta["decisions"]["long_term"]["applied"] and call.meta["decisions"]["working"]["applied"]
    fact = agent.memory.long_term.state.facts["scheduler"]
    assert fact.kind == "assistant_proposal" and fact.source_message == 2       # из сообщения ассистента
    assert "launchd — предложение ассистента, не подтверждено" in agent.memory.long_term.render()
    assert agent.memory.working.task.value("goal") == "еженедельный отчёт"
