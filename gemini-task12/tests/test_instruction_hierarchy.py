"""Иерархия инструкций: текущее сообщение → профиль → общие настройки; изоляция профиля и LT."""

from fake_client import FakeClient, echo_responder

import config
from agent import Agent
from memory import LongTermMemory, MemoryManager
from memory.long_term_store import Fact, FactsState
from memory.storage import content_text


def test_system_instruction_has_no_default_brevity_and_declares_priority():
    text = config.SYSTEM_INSTRUCTION
    low = text.lower()

    assert "кратко и по делу" not in low
    assert "кратк" not in low and "подробн" not in low       # нет жёсткой установки объёма
    assert "приоритет инструкций" in low
    assert low.index("текущем сообщении") < low.index("актуальный профиль") < low.index("общие настройки")
    assert "если профиль и общие настройки расходятся" in low


def test_memory_instruction_does_not_offer_preference_category_or_profile_access():
    text = config.MEMORY_SYSTEM_INSTRUCTION
    low = text.lower()

    assert "preference" not in low
    assert "профил" in low
    assert "не передаётся" in low
    assert "ОБЯЗАТЕЛЬНО проверь предыдущее сообщение ассистента" in text


def test_current_message_is_last_and_profile_is_separate(tmp_path):
    client = FakeClient(responder=echo_responder())
    agent = Agent(client, MemoryManager(tmp_path))
    agent.profile.update("length", "подробно", source="test", confidence=1, evidence="test")

    agent.ask("Сегодня ответь кратко.", remember=False)

    request = client.answer_requests()[-1]
    texts = [content_text(content) for content in request["contents"]]
    assert texts[-1] == "Сегодня ответь кратко."
    assert any("Длина ответа: подробно" in text for text in texts[:-1])
    assert "ПРИОРИТЕТ ИНСТРУКЦИЙ" in request["config"].system_instruction


def test_legacy_lt_preference_is_hidden_and_profile_wins(tmp_path):
    legacy = FactsState(facts={
        "answer_style": Fact(key="answer_style", value="кратко", kind="user", category="preference",
                             evidence="отвечай кратко", source_message=1),
        "no_docker": Fact(key="no_docker", value="не использовать Docker", kind="user", category="constraint",
                          evidence="никакого Docker", source_message=2),
        "no_emoji": Fact(key="no_emoji", value="без эмодзи", kind="user", category="constraint",
                         evidence="не используй эмодзи", source_message=2),
    })
    memory = MemoryManager(tmp_path)
    memory.long_term.state = legacy

    client = FakeClient(responder=echo_responder())
    agent = Agent(client, memory)
    agent.profile.update("length", "подробно", source="test", confidence=1, evidence="test")
    agent.ask("Объясни рекурсию.", remember=False)

    prompt = "\n".join(content_text(content) for content in client.answer_requests()[-1]["contents"])
    assert "Длина ответа: подробно" in prompt
    assert "answer_style" not in prompt
    assert "отвечай кратко" not in prompt
    assert "no_emoji" not in prompt and "эмодзи" not in prompt
    assert "no_docker" in prompt and "Docker" in prompt
    # Данные не удалены: они остаются в состоянии LT, но не попадают в промпт.
    assert {"answer_style", "no_emoji", "no_docker"} <= set(memory.long_term.state.facts)


def test_public_view_and_extraction_view_hide_legacy_but_keep_storage(tmp_path):
    path = tmp_path / "long_term.json"
    facts = {
        "answer_style": Fact(key="answer_style", value="кратко", kind="user", category="preference",
                             evidence="отвечай кратко", source_message=1),
        "favorite_language": Fact(key="favorite_language", value="Python", kind="user", category="preference",
                                  evidence="предпочитаю Python", source_message=2),
        "no_emoji": Fact(key="no_emoji", value="без эмодзи", kind="user", category="constraint",
                         evidence="не используй эмодзи", source_message=3),
        "response_style": Fact(key="response_style", value="формально", kind="user", category="knowledge",
                               evidence="отвечай формально", source_message=4),
        "no_docker": Fact(key="no_docker", value="не использовать Docker", kind="user", category="constraint",
                          evidence="никакого Docker", source_message=5),
    }
    lt = LongTermMemory(path)
    lt.state = FactsState(facts=facts)
    lt.save()
    before = path.read_bytes()

    assert set(lt.public_view()) == {"no_docker"}
    assert set(lt.dump()["facts"]) == set(facts)

    manager = MemoryManager(tmp_path)
    assert set(manager.extraction_view()["long_term"]) == {"no_docker"}
    assert set(manager.long_term.state.facts) == set(facts)
    assert path.read_bytes() == before


def test_new_profile_preferences_and_style_constraints_are_rejected():
    lt = LongTermMemory()
    user = "Отвечай кратко. Не используй эмодзи. Никакого Docker."
    ops = [
        {"layer": "long_term", "op": "set", "key": "answer_style", "value": "кратко", "kind": "user",
         "category": "preference", "evidence": "Отвечай кратко.", "proposal_evidence": "", "reason": ""},
        {"layer": "long_term", "op": "set", "key": "no_emoji", "value": "без эмодзи", "kind": "user",
         "category": "constraint", "evidence": "Не используй эмодзи.", "proposal_evidence": "", "reason": ""},
        {"layer": "long_term", "op": "set", "key": "response_style", "value": "кратко", "kind": "user",
         "category": "knowledge", "evidence": "Отвечай кратко.", "proposal_evidence": "", "reason": ""},
        {"layer": "long_term", "op": "set", "key": "no_docker", "value": "без Docker", "kind": "user",
         "category": "constraint", "evidence": "Никакого Docker.", "proposal_evidence": "", "reason": ""},
    ]

    out = lt.apply_operations(ops, user_text=user, assistant_text=None, message_no=1)

    assert [item["key"] for item in out["applied"]] == ["no_docker"]
    assert {item["key"] for item in out["rejected"]} == {"answer_style", "no_emoji", "response_style"}
    assert all("профил" in item["rejection"] for item in out["rejected"])
    assert set(lt.state.facts) == {"no_docker"}


def test_manager_cannot_delete_legacy_profile_fact():
    lt = LongTermMemory()
    lt.state = FactsState(facts={
        "answer_style": Fact(key="answer_style", value="кратко", kind="user", category="preference",
                             evidence="отвечай кратко", source_message=1),
    })
    operation = {"layer": "long_term", "op": "delete", "key": "answer_style", "value": "", "kind": "user",
                 "category": "preference", "evidence": "забудь про краткость", "proposal_evidence": "",
                 "reason": "пользователь изменил настройку"}

    out = lt.apply_operations([operation], user_text="Забудь про краткость.", assistant_text=None, message_no=2)

    assert out["rejected"] and "профил" in out["rejected"][0]["rejection"]
    assert "answer_style" in lt.state.facts
