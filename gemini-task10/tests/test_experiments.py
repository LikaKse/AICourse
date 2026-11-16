"""Эксперимент Дня 10: данные Дня 9, проверка ответов, прогон трёх стратегий и сравнение (fake-клиент)."""

import json
import re

import pytest
from fakes import FakeClient, all_text, api_error, echo_responder, is_facts_request, lossless_facts, lossless_summary, text_response

import config
import complexity
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

# ----- сквозной прогон трёх стратегий на фейковой модели -----

def run(tmp_path, data, name, **responder_kw):
    client = FakeClient(responder=echo_responder(**responder_kw))
    dialogue, questions, facts = data
    res = experiments.run_strategy_experiment(client, name, state_dir=tmp_path / f"state_{name}",
                                              dialogue=dialogue, questions=questions, facts=facts)
    return res, client


def run_all(tmp_path, data, **kw):
    return {n: run(tmp_path, data, n, **kw)[0] for n in config.STRATEGIES}


def test_sliding_window_counts_and_loses_old_details(tmp_path, data):
    res, client = run(tmp_path, data, "sliding")
    assert res["tokens"]["processing"]["calls"] == 0               # служебных вызовов нет
    assert res["tokens"]["control_questions"]["calls"] == 15
    assert res["tokens"]["total"]["generate_content_calls"] == len(client.requests) == 15
    assert all(a["request_messages"] == 6 + 1 for a in res["answers"])
    assert res["context"]["recent_messages"] == 6 and res["context"]["memory_block_tokens"] == 0
    # echo-«модель» отвечает текстом контекста: в последних 6 сообщениях нет ни одного эталонного факта
    assert res["facts_correct"] == 0
    assert res["detail_audit"]["totals"]["checked"] == 36
    # Окно — сообщения 45–50; единственная деталь в нём — httpx (D34) в сообщении 50.
    preserved = [d["id"] for d in res["detail_audit"]["details"] if d["status"] == "preserved"]
    assert preserved == ["D34"]


def test_summary_strategy_matches_day9_scheme(tmp_path, data):
    res, client = run(tmp_path, data, "summary")
    proc = res["tokens"]["processing"]
    assert proc["calls"] == 4 and list(proc["by_kind"]) == ["summary"]
    assert len(client.summary_requests()) == 4 and len(client.answer_requests()) == 15
    assert res["summarized_messages"] == 40 and res["context"]["recent_messages"] == 10
    assert res["summary_only_audit"]["totals"]["checked"] == 36
    # фейковое summary хранит только реплики пользователя → вопрос 15 (launchd предложил ассистент) неверен
    assert res["facts_correct"] == 14
    assert [a["id"] for a in res["answers"] if not a["correct"]] == [15]


def test_facts_strategy_one_call_per_user_message(tmp_path, data):
    res, client = run(tmp_path, data, "facts")
    proc = res["tokens"]["processing"]
    assert proc["calls"] == 25 == len(client.facts_requests())     # 25 сообщений пользователя
    assert proc["by_kind"]["facts_update"]["input_tokens"] > 0
    assert proc["input_tokens"] == sum(r["usage"]["input_tokens"] for r in res["service_calls"])
    assert res["tokens"]["total"]["generate_content_calls"] == 25 + 15
    assert res["facts_by_kind"]["user"] == 25 and res["operations"]["applied"] == 25
    assert res["facts_correct"] >= 10                              # факты пользователя все в памяти


def test_processing_and_control_costs_are_separated_and_summed(tmp_path, data):
    for name, res in run_all(tmp_path, data).items():
        t = res["tokens"]
        assert t["total"]["input_tokens"] == t["processing"]["input_tokens"] + t["control_questions"]["input_tokens"]
        assert t["total"]["total_tokens"] == t["processing"]["total_tokens"] + t["control_questions"]["total_tokens"]
        expected = experiments.estimate_cost(t["total"]["input_tokens"], t["total"]["output_tokens"])
        assert t["total"]["estimated_cost_usd"] == expected and t["total"]["complete"], name
        assert t["count_tokens_calls"] > 0


def test_failed_service_calls_are_still_counted(tmp_path, data):
    state = {"n": 0}

    def respond(contents, cfg):
        if is_facts_request(cfg):
            state["n"] += 1
            if state["n"] % 5 == 0:
                return api_error(503)
            return text_response(lossless_facts(all_text(contents)), prompt=100, candidates=10)
        return text_response("ответ", prompt=50, candidates=1)

    dialogue, questions, facts = data
    res = experiments.run_strategy_experiment(FakeClient(responder=respond), "facts", state_dir=tmp_path / "s",
                                              dialogue=dialogue, questions=questions, facts=facts)
    proc = res["tokens"]["processing"]
    assert proc["calls"] == 25 and proc["failed_calls"] == 5
    assert proc["input_tokens"] == 20 * 100                         # у ошибок usage нет — не выдумываем
    assert proc["complete"] is False                                # и это помечено


def test_reference_answers_never_sent_to_model(tmp_path, data):
    for name in config.STRATEGIES:
        _, client = run(tmp_path, data, name)
        sent = "\n".join(all_text(r["contents"]) + str(r["config"].system_instruction) for r in client.requests)
        assert "accept_patterns" not in sent and "summary_details" not in sent
        assert "Совет: открытые API погоды по координатам города" not in sent   # текст каталога деталей


def test_comparison_table_and_report(tmp_path, data):
    results = run_all(tmp_path, data)
    cmp = experiments.compare_strategies(results, complexity.measure_all())
    rows = {r["strategy"]: r for r in cmp["table"]}
    assert list(rows) == ["sliding", "facts", "summary"]
    assert rows["sliding"]["processing"]["calls"] == 0
    assert rows["facts"]["processing"]["calls"] == 25 and rows["summary"]["processing"]["calls"] == 4
    assert rows["facts"]["total"]["generate_content_calls"] == 40
    assert rows["sliding"]["details_preserved"] == 1 and len(rows["sliding"]["lost_ids"]) == 35
    assert rows["sliding"]["complexity"]["loc"] < rows["summary"]["complexity"]["loc"]
    assert len(cmp["per_question"]) == 15 and len(cmp["per_detail"]) == 36
    md = experiments.markdown_report(cmp)
    assert "| Стратегия | Деталей сохранено (из 36)" in md and "обновление facts" in md and "суммаризация" in md


def test_save_load_and_compare_from_files(tmp_path, data):
    results = run_all(tmp_path, data)
    for n, r in results.items():
        experiments.save_results(tmp_path / "results", n, r)
    cmp = experiments.compare_saved(tmp_path / "results")
    assert (tmp_path / "results" / "comparison.json").is_file() and cmp["strategies"] == list(config.STRATEGIES)
    assert experiments.compare_saved(tmp_path / "empty") is None


def test_compare_rejects_different_scenarios(tmp_path, data):
    results = run_all(tmp_path, data)
    results["facts"]["dialogue_fingerprint"] = "other"
    with pytest.raises(experiments.DataError, match="другом сценарии"):
        experiments.compare_strategies(results)
    results["facts"]["dialogue_fingerprint"] = results["sliding"]["dialogue_fingerprint"]
    results["facts"]["model"] = "other-model"
    with pytest.raises(experiments.DataError, match="другая модель"):
        experiments.compare_strategies(results)


def test_complexity_is_measured_from_code():
    m = complexity.measure_all()
    assert m["sliding"]["extra_model_call_types"] == 0 and m["sliding"]["extra_state_files"] == 0
    assert m["facts"]["extra_state_files"] == 1 and m["summary"]["extra_state_files"] == 1
    assert all(m[s]["loc"] > 0 for s in m)
    assert m["facts"]["model_output_validation"] is True
