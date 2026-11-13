"""Эксперименты: данные, проверка фактов, сравнение со сжатием и без него (на фейковой модели)."""

import json
import re

import pytest
from fakes import FakeClient, all_text, echo_responder, lossless_summary

import config
import experiments
from context_manager import content_text


@pytest.fixture(scope="module")
def data():
    return experiments.load_dialogue(), experiments.load_questions(), experiments.load_facts()


# ----- данные -----

def test_dialogue_questions_and_facts_are_valid(data):
    dialogue, questions, facts = data
    assert len(dialogue) == 50
    assert [c.role for c in dialogue] == ["user", "model"] * 25
    assert len(questions) == len(facts) == 15
    assert [f["category"] for f in facts] == ["core"] * 10 + ["detail"] * 5
    experiments.validate_data(dialogue, questions, facts)


def test_new_questions_text(data):
    questions = {q["id"]: q["question"] for q in data[1]}
    assert questions[2] == "Где я живу?"
    assert questions[11] == "Почему ежедневный запуск перенесли с 08:00?"
    assert questions[12] == "Почему выбрали gemini-3.5-flash-lite вместо gemini-3.5-flash?"
    assert questions[13] == "Какие форматы хранения рассматривали помимо JSON?"
    assert questions[14] == "Какой бюджет был до окончательного утверждения?"
    assert questions[15] == "Кто предложил использовать launchd — пользователь или ассистент?"


def test_every_reference_answer_backed_by_dialogue_quote(data):
    dialogue, _, facts = data
    for f in facts:
        for ev in f["evidence"]:
            msg = dialogue[ev["message"] - 1]
            assert msg.role == ev["role"] and ev["quote"] in content_text(msg), f["id"]
    launchd = {f["key"]: f for f in facts}["launchd_author"]
    assert launchd["expected"] == "ассистент"
    users = [content_text(c).lower() for c in dialogue if c.role == "user"]
    assert not any("launchd" in u for u in users)       # пользователь launchd ни разу не упоминал


@pytest.mark.parametrize("mutate,error", [
    (lambda f: f[0]["evidence"][0].update(quote="Меня зовут Марина"), "цитата"),
    (lambda f: f[0]["evidence"][0].update(role="model"), "цитата"),
    (lambda f: f[0]["evidence"][0].update(message=99), "нет в диалоге"),
    (lambda f: f[0].update(evidence=[]), "evidence"),
    (lambda f: f[14]["absent_in"][0].update(role="model"), "встречается"),
    (lambda f: f[1].update(expected="город прогноза погоды — Нячанг"), "не проходит"),
])
def test_validate_data_rejects_unbacked_reference(data, mutate, error):
    import copy
    dialogue, questions, facts = data
    facts = copy.deepcopy(facts)
    mutate(facts)
    with pytest.raises(experiments.DataError, match=error):
        experiments.validate_data(dialogue, questions, facts)


def test_reference_facts_stored_separately_from_dialogue():
    dialogue_raw = json.loads(config.DIALOGUE_FILE.read_text(encoding="utf-8"))
    questions_raw = json.loads(config.QUESTIONS_FILE.read_text(encoding="utf-8"))
    assert "facts" not in dialogue_raw and "expected" not in json.dumps(questions_raw)
    expected = [f["expected"] for f in experiments.load_facts()]
    assert expected == ["Лика", "Нячанг", "18 500 000 VND", "17 декабря 2026 года", "24",
                        "gemini-3.5-flash-lite", "не использовать Docker", "macOS",
                        "хранить результаты в JSON", "09:30",
                        "в 08:00 ноутбук ещё выключен", "важнее скорость и цена", "CSV и SQLite",
                        "15 000 000 VND (предварительный)", "ассистент"]


def test_facts_in_part_that_will_be_compressed(data):
    """Все факты подтверждены сообщениями 1–40 (в эксперименте B они уйдут в summary),
    а в последних 10 сообщениях ответов на вопросы нет."""
    dialogue, _, facts = data
    recent = experiments.normalize_answer(all_text(dialogue[40:]))
    for f in facts:
        assert max(ev["message"] for ev in f["evidence"]) <= 40, f["id"]
        assert not any(re.search(p, recent) for p in f["accept_patterns"]), f["id"]


# ----- проверка ответов -----

@pytest.mark.parametrize("fact_id,answer,ok", [
    (1, "Вас зовут Лика.", True),
    # вопрос 2 «Где я живу?»: ответ про город прогноза погоды не засчитывается
    (2, "Вы живёте в Нячанге.", True),
    (2, "В Нячанге.", True),
    (2, "Судя по информации из проекта, для прогноза погоды используется город Нячанг.", False),
    (2, "Не знаю, где вы живёте; прогноз погоды собирается для Нячанга.", False),
    (2, "Город для прогноза погоды — Нячанг.", False),
    (11, "Потому что в 08:00 ноутбук ещё выключен.", True),
    (11, "Так было удобнее.", False),
    (12, "Для нас важнее скорость и цена.", True),
    (12, "Она быстрее и дешевле.", True),
    (12, "Она точнее.", False),
    (13, "Рассматривали CSV и SQLite.", True),
    (13, "Рассматривали CSV.", False),
    (14, "Предварительный бюджет был 15 000 000 VND.", True),
    (14, "18 500 000 VND", False),
    (15, "launchd предложил ассистент.", True),
    (15, "Это предложил я, ассистент.", True),
    (15, "Вы предложили launchd.", False),
    (15, "Пользователь предложил использовать launchd.", False),
    (15, "Предложил пользователь, а не ассистент.", False),
    (1, "Вас зовут **Лика**", True),
    (1, "Вас зовут Марина.", False),
    (1, "Это великий проект", False),
    (1, "Анжелика", False),
    (1, "Не знаю, как вас зовут.", False),
    (3, "Утверждён бюджет 18 500 000 VND.", True),
    (3, "Бюджет — 18,5 млн донгов", True),
    (3, "Бюджет 15 000 000 VND.", False),
    (4, "До 17 декабря 2026 года.", True),
    (5, "Мы планируем 24 теста.", True),
    (5, "20 тестов", False),
    (5, "245 тестов", False),
    (6, "Модель **gemini-3.5-flash-lite**", True),
    (6, "gemini-3.5-flash", False),
    (7, "Не использовать Docker.", True),
    (7, "Docker использовать нельзя", True),
    (7, "Использовать venv", False),
    (10, "Каждый день в 09:30.", True),
    (10, "В 9:30", True),
    (10, "В 19:30", False),
    (10, "В 08:00", False),
])
def test_check_answer(data, fact_id, answer, ok):
    fact = {f["id"]: f for f in data[2]}[fact_id]
    assert experiments.check_answer(answer, fact) is ok


# ----- сквозной прогон на фейковой модели -----

def _lossy_summary(text):
    """Суммаризатор с потерями: «забывает» всё про бюджет и Docker."""
    return "\n".join(l for l in lossless_summary(text).splitlines() if "VND" not in l and "Docker" not in l)


def run_both(tmp_path, data, summary_fn=None):
    dialogue, questions, facts = data
    client_a = FakeClient(responder=echo_responder(summary_fn))
    client_b = FakeClient(responder=echo_responder(summary_fn))
    a = experiments.run_experiment(client_a, compression=False, state_dir=tmp_path / "A",
                                   dialogue=dialogue, questions=questions, facts=facts)
    b = experiments.run_experiment(client_b, compression=True, state_dir=tmp_path / "B",
                                   dialogue=dialogue, questions=questions, facts=facts)
    return a, b, client_a, client_b


def test_experiment_without_compression(tmp_path, data):
    a, _, client_a, _ = run_both(tmp_path, data)
    assert client_a.summary_requests() == []
    assert a["context"]["recent_messages"] == 50 and a["context"]["summary_tokens"] == 0
    # Фейковая «модель» отвечает текстом контекста. На вопрос 15 («кто предложил launchd») такой эхо-ответ
    # не содержит указания автора, поэтому он не засчитывается — остальные 14 фактов есть в контексте.
    assert a["facts_correct"] == 14
    assert [x["id"] for x in a["answers"] if not x["correct"]] == [15]
    assert a["facts_by_category"] == {"core": {"correct": 10, "total": 10, "percent": 100.0},
                                      "detail": {"correct": 4, "total": 5, "percent": 80.0}}
    assert a["compression_audit"] is None and a["summary_versions"] == []
    assert len(a["dialogue_fingerprint"]) == 64
    # каждый вопрос ушёл с полной историей и только с одним вопросом
    for req in client_a.answer_requests():
        assert len(req["contents"]) == 51


def test_experiment_with_compression_and_comparison(tmp_path, data):
    a, b, client_a, client_b = run_both(tmp_path, data)
    # 4 вызова суммаризации (на 16, 26, 36, 46 сообщениях), последние 10 активны
    assert len(client_b.summary_requests()) == 4
    assert b["context"]["summarized_messages"] == 40 and b["context"]["recent_messages"] == 10
    assert b["context"]["full_history_tokens"] == a["context"]["full_history_tokens"]
    # вопросы независимы: в каждом запросе один вопрос и ни одного ответа на прошлые вопросы
    questions = [q["question"] for q in data[1]]
    for req, q in zip(client_b.answer_requests(), questions):
        texts = [content_text(c) for c in req["contents"]]
        assert texts[-1] == q
        assert sum(t in questions for t in texts) == 1
        assert not any(t.startswith("Ответ по контексту") for t in texts)
        assert len(req["contents"]) == 2 + 10 + 1
    # токены суммаризации учтены отдельно и в итоге
    t = b["tokens"]
    assert t["summarization"]["calls"] == 4 and t["summarization"]["input_tokens"] > 0
    assert t["total"]["total_tokens"] == t["control_questions"]["total_tokens"] + t["summarization"]["total_tokens"]

    cmp = experiments.compare(a, b)
    assert cmp["facts_retained"]["A"]["correct"] == 14
    assert cmp["facts_retained"]["by_category"]["B"]["core"]["correct"] == 10
    # все версии summary сохранены целиком, аудит выполнен офлайн
    assert [v["version"] for v in b["summary_versions"]] == [1, 2, 3, 4]
    assert all(v["text"] for v in b["summary_versions"])
    assert [v["end"] for v in b["summary_versions"]] == [10, 20, 30, 40]
    audit = b["compression_audit"]
    assert audit["versions_available"] == [1, 2, 3, 4]
    assert audit["totals"]["checked"] > 0
    assert cmp["control_requests"]["input_tokens"]["A"] > cmp["control_requests"]["input_tokens"]["B"]
    assert cmp["overall_with_summarization"]["B_summarization_tokens"] == t["summarization"]["total_tokens"]
    summary = experiments.build_summary(a, b, cmp)
    assert summary["savings_control_requests_input_percent"] > summary["savings_overall_with_summarization_percent"]


def test_lossy_compression_loses_facts(tmp_path, data):
    a, b, _, _ = run_both(tmp_path, data, summary_fn=_lossy_summary)
    cmp = experiments.compare(a, b)
    assert a["facts_correct"] == 14
    assert set(cmp["facts_retained"]["lost_in_B"]) == {3, 7, 14}
    assert b["facts_correct"] == 11
    # аудит summary тоже видит потерю бюджета и Docker
    statuses = {r["id"]: r["status"] for r in b["compression_audit"]["details"]}
    assert statuses["D26"] == "lost" and statuses["D08"] == "lost" and statuses["D12"] == "lost"
    assert statuses["D29"] == "preserved"


def test_reference_facts_never_sent_to_model(tmp_path, data):
    _, _, client_a, client_b = run_both(tmp_path, data)
    facts_markers = ["accept_patterns", "Эталон", "Срок проекта", "Решение по хранению", "evidence",
                     "Причина выбора модели", "line_lacks", "Каталог деталей"]
    for client in (client_a, client_b):
        for req in client.requests:
            sent = all_text(req["contents"]) + str(req["config"].system_instruction)
            assert not any(m in sent for m in facts_markers)


def test_api_errors_during_experiment_are_recorded(tmp_path, data):
    from fakes import api_error

    def responder(contents, cfg):
        return api_error(429, "quota", "RESOURCE_EXHAUSTED")

    b = experiments.run_experiment(FakeClient(responder=responder), compression=True, state_dir=tmp_path / "B",
                                   dialogue=data[0], questions=data[1], facts=data[2])
    assert all(x["status"] == "api_error" and not x["correct"] for x in b["answers"])
    assert b["tokens"]["summarization"]["failed_attempts"] > 0
    assert b["context"]["summarized_messages"] == 0            # ничего не потеряно
    assert b["context"]["full_history_messages"] == 50


def test_results_saved_and_compared_from_files(tmp_path, data):
    a, b, _, _ = run_both(tmp_path, data)
    experiments.save_results(tmp_path, "A", a)
    experiments.save_results(tmp_path, "B", b)
    comparison, summary = experiments.compare_saved(tmp_path)
    for name in experiments.RESULT_FILES.values():
        assert (tmp_path / name).is_file()
    report = experiments.markdown_report(a, b, comparison)
    assert "Проверка 15 контрольных вопросов" in report and "Итого с учётом суммаризации" in report
    assert "дополнительные детали" in report


def test_compare_rejects_different_scenarios(tmp_path, data):
    a, b, _, _ = run_both(tmp_path, data)
    b["answers"] = b["answers"][:-1]
    with pytest.raises(experiments.DataError):
        experiments.compare(a, b)
