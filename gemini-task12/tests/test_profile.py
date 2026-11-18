import json

import pytest

from agent import Agent
from fake_client import FakeClient, is_memory_request, ops_json, text_response
from memory import MemoryManager
from profile import ProfileStore
from memory.storage import content_text


def test_profile_preferences_are_separate_and_survive_restart(tmp_path):
    path = tmp_path / "user_profile.json"
    store = ProfileStore(path)
    store.observe("Предпочитаю краткие ответы и неформальный стиль.", 1)
    reloaded = ProfileStore(path)
    assert reloaded.profile.preferences["length"].value == "кратко"
    assert reloaded.profile.preferences["style"].source == "user_message:1"
    assert not (tmp_path / "long_term.json").exists()


def test_one_off_request_is_not_persisted(tmp_path):
    store = ProfileStore(tmp_path / "profile.json")
    assert store.observe("Для этого ответа отвечай кратко.", 1) == []
    assert not store.profile.preferences


def test_single_preference_command_is_candidate_until_repeated(tmp_path):
    path = tmp_path / "profile.json"
    store = ProfileStore(path)
    store.observe("Отвечай кратко.", 1)
    assert "length" not in store.profile.preferences
    store = ProfileStore(path)  # кандидаты тоже переживают перезапуск
    store.observe("Отвечай кратко.", 2)
    pref = store.profile.preferences["length"]
    assert pref.value == "кратко" and pref.confidence >= .8


def test_profile_added_to_each_prompt_and_counted(tmp_path):
    client = FakeClient(responder=lambda c, cfg: text_response("Ок"))
    memory = MemoryManager(tmp_path)
    agent = Agent(client, memory)
    agent.profile.update("length", "кратко", source="test", confidence=1, evidence="test")
    result = agent.ask("Вопрос", remember=False)
    prompt = "\n".join(content_text(c) for c in client.answer_requests()[0]["contents"])
    assert "Профиль пользователя" in prompt
    assert result.tokens["profile"] > 0
    assert result.tokens["sum_of_parts"] == result.tokens["total"]


def test_profile_is_not_sent_to_memory_manager(tmp_path):
    client = FakeClient(responder=lambda c, cfg: text_response(ops_json() if is_memory_request(cfg) else "Хорошо."))
    agent = Agent(client, MemoryManager(tmp_path))
    agent.profile.update("style", "формальный", source="test", confidence=1, evidence="test")
    result = agent.ask("Вопрос")
    memory_request = client.memory_requests()[0]
    prompt = "\n".join(content_text(c) for c in memory_request["contents"])
    assert "Профиль пользователя" not in prompt
    assert "Стиль общения" not in prompt
    assert "profile_tokens" not in result.service_calls[0].meta


def test_memory_manager_does_not_change_profile(tmp_path):
    client = FakeClient(responder=lambda c, cfg: text_response(ops_json()))
    agent = Agent(client, MemoryManager(tmp_path))
    agent.profile.update("code", "не показывать", source="test", confidence=1, evidence="test")
    before = (tmp_path / "user_profile.json").read_text(encoding="utf-8")

    record = agent.update_memory("Вопрос пользователя")

    prompt = "\n".join(content_text(c) for c in client.memory_requests()[0]["contents"])
    assert "Код: не показывать" not in prompt
    assert "profile_tokens" not in record.meta
    assert (tmp_path / "user_profile.json").read_text(encoding="utf-8") == before


def test_profile_reload_observes_external_update_and_deletion(tmp_path):
    client = FakeClient(responder=lambda c, cfg: text_response("Хорошо."))
    agent = Agent(client, MemoryManager(tmp_path))
    agent.profile.update("length", "кратко", source="test", confidence=1, evidence="test")
    external = ProfileStore(tmp_path / "user_profile.json")
    external.update("style", "формальный", source="external", confidence=1, evidence="external")
    agent.ask("Первый вопрос", remember=False)
    prompt = "\n".join(content_text(c) for c in client.answer_requests()[-1]["contents"])
    assert "Стиль общения: формальный" in prompt
    (tmp_path / "user_profile.json").unlink()
    agent.ask("Второй вопрос", remember=False)
    prompt = "\n".join(content_text(c) for c in client.answer_requests()[-1]["contents"])
    assert "Профиль пользователя" not in prompt


def test_cancel_without_an_extra_imperative_is_detected(tmp_path):
    store = ProfileStore(tmp_path / "profile.json")
    store.observe("Всегда предпочитаю неформальный стиль.", 1)
    assert "style" in store.profile.preferences
    actions = store.observe("Отменяю неформальный стиль.", 2)
    assert {"category": "style", "action": "cancel"} in actions
    assert "style" not in store.profile.preferences


def test_constraints_accumulate_and_targeted_cancel(tmp_path):
    store = ProfileStore(tmp_path / "profile.json")
    store.observe("Всегда не используй эмодзи.", 1)
    store.observe("Всегда не используй жаргон.", 2)
    assert len(store.profile.preferences["constraints"].value) == 2
    store.observe("Отменяю ограничение на эмодзи.", 3)
    assert store.profile.preferences["constraints"].value == ["Всегда не используй жаргон."]


def test_profile_file_redacts_api_key_from_preference_evidence(tmp_path, monkeypatch):
    secret = "FAKE_API_SECRET_123456789"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    path = tmp_path / "profile.json"
    store = ProfileStore(path)
    store.update("constraints", f"Всегда не используй жаргон; ключ {secret}",
                 source="test", confidence=1, evidence=f"ключ {secret}")
    assert secret not in path.read_text(encoding="utf-8")


def test_report_only_does_not_rewrite_archived_experiment_json(tmp_path, monkeypatch):
    from personalization_experiment import report_only

    folder = tmp_path / "results" / "personalization"
    folder.mkdir(parents=True)
    payload = {
        "model": "archived-model", "settings": {"thinking": "MINIMAL", "max_output_tokens": 100},
        "profiles": {"A": {"satisfied": 1, "total": 3, "percent": 33.33, "preferences": {},
                            "questions": [{"question": "Q", "answer": "A", "assessment": {
                                "satisfied": 1, "total": 3, "checks": {"length": True, "code": False, "style": False}}}]}},
        "stability": {"first_violation_question": 1, "turns": [{"question_no": 1, "answer": "A",
                     "assessment": {"satisfied": 1, "checks": {"length": True, "code": False, "style": False}},
                     "violation_reason": "code, style"}]}
    }
    source = folder / "personalization_experiment.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    before = source.read_bytes()
    monkeypatch.chdir(tmp_path)
    report_only()
    assert source.read_bytes() == before
    assert (folder / "report.md").is_file()


def test_explicit_update_and_cancel(tmp_path):
    store = ProfileStore(tmp_path / "profile.json")
    store.observe("Всегда показывай код.", 1)
    assert store.profile.preferences["code"].value == "показывать"
    store.observe("Отменяю: больше не показывай код.", 2)
    assert "code" not in store.profile.preferences


# ---------------------------------------------------------------------
# Вопросы не должны менять профиль
# ---------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Почему ты всегда отвечаешь кратко?",
    "Ты всегда приводишь код?",
    "Мне нравится, когда ответы краткие?",
    "Можно ли отвечать кратко?",
    "Какой стиль ответа ты предпочитаешь?",
])
def test_questions_do_not_change_profile(tmp_path, text):
    store = ProfileStore(tmp_path / "profile.json")
    assert store.observe(text, 1) == []
    assert store.profile.preferences == {}
    assert store.pending == {}


def test_question_does_not_remove_existing_preference(tmp_path):
    store = ProfileStore(tmp_path / "profile.json")
    store.observe("Всегда отвечай кратко.", 1)
    before = store.profile.to_dict()
    assert store.observe("Почему ты всегда отвечаешь кратко?", 2) == []
    assert store.profile.to_dict() == before


def test_declarative_question_word_is_not_a_question(tmp_path):
    store = ProfileStore(tmp_path / "profile.json")
    store.observe("Что касается длины, предпочитаю кратко.", 1)
    assert store.profile.preferences["length"].value == "кратко"


def test_declarative_preference_still_applies(tmp_path):
    store = ProfileStore(tmp_path / "profile.json")
    store.observe("Мне нравится, когда ответы краткие.", 1)
    assert store.profile.preferences["length"].value == "кратко"


@pytest.mark.parametrize("text", [
    "Это короткий ответ.",
    "Формальный стиль уместен в документации.",
    "В Python есть короткие списки.",
    "Краткость не нужна.",
    "Формат файла — JSON.",
])
def test_descriptive_statement_does_not_set_preference(tmp_path, text):
    store = ProfileStore(tmp_path / "profile.json")
    assert store.observe(text, 1) == []
    assert store.profile.preferences == {}
    assert store.pending == {}


# ---------------------------------------------------------------------
# Отрицания: не сохранять противоположное и корректно понимать просьбы
# ---------------------------------------------------------------------

@pytest.mark.parametrize("text,key", [
    ("Я не предпочитаю краткие ответы.", "length"),
    ("Я не предпочитаю формальный стиль.", "style"),
    ("Мне не нравится, когда ответы краткие.", "length"),
    ("Мне не нравится формальный стиль.", "style"),
])
def test_negated_statement_does_not_set_preference(tmp_path, text, key):
    store = ProfileStore(tmp_path / "profile.json")
    assert store.observe(text, 1) == []
    assert key not in store.profile.preferences


def test_negated_statement_removes_only_matching_preference(tmp_path):
    store = ProfileStore(tmp_path / "profile.json")
    store.update("length", "кратко", source="test", confidence=1, evidence="test")
    actions = store.observe("Я не предпочитаю краткие ответы.", 2)
    assert "length" not in store.profile.preferences
    assert {"category": "length", "action": "cancel"} in actions

    store.update("style", "неформальный", source="test", confidence=1, evidence="test")
    store.observe("Я не предпочитаю формальный стиль.", 3)
    assert store.profile.preferences["style"].value == "неформальный"


def test_negative_formal_command_maps_to_informal(tmp_path):
    store = ProfileStore(tmp_path / "profile.json")
    actions = store.observe("Не пиши формально.", 1)
    assert actions == [{"category": "style", "action": "candidate", "confidence": 0.55}]
    assert "style" not in store.profile.preferences
    assert store.pending["style"]["value"] == "неформальный"

    store.observe("Не пиши формально.", 2)
    assert store.profile.preferences["style"].value == "неформальный"


def test_negative_command_cancels_opposite_preference_immediately(tmp_path):
    store = ProfileStore(tmp_path / "profile.json")
    store.update("style", "формальный", source="test", confidence=1, evidence="test")
    actions = store.observe("Не пиши формально.", 2)
    assert "style" not in store.profile.preferences
    assert {"category": "style", "action": "cancel"} in actions
    assert store.pending["style"]["value"] == "неформальный"


@pytest.mark.parametrize("text", [
    "Не отвечай формально.",
    "Пожалуйста, не будь формальным.",
    "Не используй формальный стиль.",
])
def test_negative_formal_command_variants(tmp_path, text):
    store = ProfileStore(tmp_path / "profile.json")
    store.observe(text, 1)
    assert store.pending["style"]["value"] == "неформальный"
    store.observe(text, 2)
    assert store.profile.preferences["style"].value == "неформальный"


def test_always_negative_brief_sets_detailed_not_brief(tmp_path):
    store = ProfileStore(tmp_path / "profile.json")
    store.observe("Всегда не надо кратко.", 1)
    assert store.profile.preferences["length"].value == "подробно"


@pytest.mark.parametrize("text", [
    "Не отвечай кратко.",
    "Не надо кратко.",
])
def test_negative_brief_command_maps_to_detailed(tmp_path, text):
    store = ProfileStore(tmp_path / "profile.json")
    store.observe(text, 1)
    assert store.pending["length"]["value"] == "подробно"
    store.observe(text, 2)
    assert store.profile.preferences["length"].value == "подробно"


def test_one_off_negative_request_is_ignored(tmp_path):
    store = ProfileStore(tmp_path / "profile.json")
    assert store.observe("Сейчас не пиши формально.", 1) == []
    assert "style" not in store.profile.preferences
    assert store.pending == {}


def test_code_preference_still_understands_privodi(tmp_path):
    store = ProfileStore(tmp_path / "profile.json")
    store.observe("Всегда приводи примеры кода.", 1)
    assert store.profile.preferences["code"].value == "показывать"


def test_negative_code_command_stays_negative(tmp_path):
    store = ProfileStore(tmp_path / "profile.json")
    store.observe("Не показывай код.", 1)
    assert store.pending["code"]["value"] == "не показывать"
    store.observe("Не показывай код.", 2)
    assert store.profile.preferences["code"].value == "не показывать"


@pytest.mark.parametrize("text", [
    "Без подробностей.",
    "Без всяких подробностей.",
    "Без шуток, отвечай кратко.",
])
def test_bez_negates_only_the_next_preference_word(tmp_path, text):
    store = ProfileStore(tmp_path / "profile.json")
    store.observe(text, 1)
    assert store.pending["length"]["value"] == "кратко"


# ---------------------------------------------------------------------
# Явные команды формата и ограничений
# ---------------------------------------------------------------------

@pytest.mark.parametrize("text,value", [
    ("Отвечай списком.", "списком"),
    ("Отвечай в формате таблицы.", "таблицей"),
    ("В формате JSON.", "json"),
])
def test_explicit_format_command_is_extracted(tmp_path, text, value):
    store = ProfileStore(tmp_path / "profile.json")
    actions = store.observe(text, 1)
    assert actions == [{"category": "format", "action": "candidate", "confidence": 0.55}]
    assert store.pending["format"]["value"] == value

    store.observe(text, 2)
    assert store.profile.preferences["format"].value == value
    assert "constraints" not in store.profile.preferences


def test_explicit_no_emoji_constraint_is_extracted(tmp_path):
    store = ProfileStore(tmp_path / "profile.json")
    actions = store.observe("Без эмодзи.", 1)
    assert actions == [{"category": "constraints", "action": "candidate", "confidence": 0.55}]
    assert store.pending["constraints"]["value"] == "Без эмодзи."

    store.observe("Без эмодзи.", 2)
    assert store.profile.preferences["constraints"].value == ["Без эмодзи."]


def test_constraint_candidate_confirms_without_trailing_period(tmp_path):
    store = ProfileStore(tmp_path / "profile.json")
    store.observe("Без эмодзи", 1)
    store.observe("Без эмодзи.", 2)
    assert store.profile.preferences["constraints"].value == ["Без эмодзи."]


def test_format_preference_is_not_duplicated_in_constraints(tmp_path):
    store = ProfileStore(tmp_path / "profile.json")
    store.observe("Предпочитаю ответы списком.", 1)
    assert store.profile.preferences["format"].value == "списком"
    assert "constraints" not in store.profile.preferences


@pytest.mark.parametrize("text,value", [
    ("Всегда используй списки.", "списком"),
    ("Отвечай в формате таблицы.", "таблицей"),
    ("Предпочитаю ответы таблицей.", "таблицей"),
])
def test_format_commands_do_not_create_constraint_duplicates(tmp_path, text, value):
    store = ProfileStore(tmp_path / "profile.json")
    store.observe(text, 1)
    store.observe(text, 2)
    assert store.profile.preferences["format"].value == value
    assert "constraints" not in store.profile.preferences


def test_negative_format_does_not_become_positive(tmp_path):
    store = ProfileStore(tmp_path / "profile.json")
    store.observe("Не используй списки.", 1)
    assert "format" not in store.profile.preferences
    store.observe("Не используй списки.", 2)
    assert "format" not in store.profile.preferences
    assert any("списк" in item.lower() for item in store.profile.preferences["constraints"].value)


def test_negative_format_removes_existing_positive(tmp_path):
    store = ProfileStore(tmp_path / "profile.json")
    store.update("format", "списком", source="test", confidence=1, evidence="test")
    actions = store.observe("Не используй списки.", 2)
    assert "format" not in store.profile.preferences
    assert {"category": "format", "action": "cancel"} in actions


def test_bez_list_is_constraint_not_format(tmp_path):
    store = ProfileStore(tmp_path / "profile.json")
    store.observe("Без списков.", 1)
    store.observe("Без списков.", 2)
    assert "format" not in store.profile.preferences
    assert any("списк" in item.lower() for item in store.profile.preferences["constraints"].value)
