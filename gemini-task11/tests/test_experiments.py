"""Сквозной офлайн-эксперимент: слои, промпт, токены и очистка каждого слоя (FakeClient)."""

import json

import pytest

import experiments
from fakes import FakeClient, echo_responder, scripted_extractor


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    client = FakeClient(responder=echo_responder(scripted_extractor(experiments.load_offline_extraction())))
    data = experiments.run_experiment(client, tmp_path_factory.mktemp("exp"))
    data["_client"] = client
    return data


def ids(layer):
    expected = experiments.load_expected()
    return sorted(i for i, e in expected.items() if e["layer"] == layer)


def test_data_is_valid():
    experiments.validate_data(experiments.load_scenario(), experiments.load_questions(), experiments.load_expected())


def test_expected_layers_cover_all_three():
    assert len(ids("long_term")) == 7 and len(ids("working")) == 5 and len(ids("short_term")) == 3


def test_answer_checker():
    exp = experiments.load_expected()[7]
    assert experiments.check_answer("Launchd предложил ассистент, вы его не подтверждали.", exp)
    assert not experiments.check_answer("Вы выбрали launchd, ассистент согласился.", exp)
    assert not experiments.check_answer("Не знаю.", exp)


def test_layers_after_scenario(result):
    la = result["layers_after_scenario"]
    assert la["short_term"]["stats"]["stored_messages"] == 30
    assert la["short_term"]["stats"]["in_prompt_messages"] == 6
    wm = la["working"]["content"]["task"]
    assert wm["goal"].startswith("еженедельный отчёт") and wm["status"] == "in_progress"
    assert wm["deadline"] == "20 ноября" and wm["next_step"] == "написать тесты для build_report"
    lt = la["long_term"]["content"]["facts"]
    assert {k: v["kind"] for k, v in lt.items() if v["kind"] != "user"} == {
        "test_framework": "accepted_proposal", "scheduler": "assistant_proposal"}
    assert "deadline" not in lt and "current_task" not in lt               # данные задачи не попали в LT


def test_write_rules_rejected_demo_operations(result):
    wl = result["write_log"]
    reasons = {r["rejection"] for layer in ("working", "long_term") for r in wl[layer]["rejected"]}
    reasons |= {r["rejection"] for r in wl["unrouted"]}
    assert "вопрос пользователя — не факт" in reasons
    assert "данные текущей задачи — место в рабочей памяти, не в долговременной" in reasons
    assert "неподтверждённое предложение ассистента — не данные задачи" in reasons
    assert "краткосрочную память пишет только код" in reasons
    assert any("не входит в рабочую память" in r for r in reasons)
    assert wl["failed_calls"] == []
    demo = sum(1 for ops in experiments.load_offline_extraction().values() for o in ops if "_demo" in o)
    rejected = sum(len(wl[layer]["rejected"]) for layer in ("working", "long_term")) + len(wl["unrouted"])
    assert rejected == demo                                                 # отклонены ровно демонстрационные


def test_calls_count(result):
    client = result["_client"]
    assert len(client.memory_requests()) == 15                             # 1 на реплику сценария
    assert len(client.answer_requests()) == 15 * len(experiments.VARIANTS)  # вопросы не обновляют память


def test_restart_check(result):
    assert result["restart_check"]["ok"]
    assert set(result["restart_check"]["files"]) == {"short_term", "working", "long_term"}


@pytest.mark.parametrize("variant, cleared", [("full", None), ("no_short_term", "short_term"),
                                              ("no_working", "working"), ("no_long_term", "long_term")])
def test_each_variant_loses_exactly_its_layer(result, variant, cleared):
    v = result["variants"][variant]
    s = v["summary"]
    assert s["matches_expectation"]
    assert s["lost_vs_full"] == (ids(cleared) if cleared else [])
    assert v["changed_files"] == ([cleared] if cleared else [])
    assert v["memory_unchanged_by_questions"]
    assert (cleared not in v["prompt_layers"]) if cleared else v["prompt_layers"] == ["long_term", "working", "short_term"]
    assert s["tokens"]["layers"][cleared] == 0 if cleared else all(s["tokens"]["layers"].values())


def test_tokens_per_layer_add_up(result):
    full = result["variants"]["full"]
    for a in full["answers"]:
        assert a["tokens"]["sum_of_parts"] == a["tokens"]["total"]
    layers = full["summary"]["tokens"]["layers"]
    for name, cleared in (("no_short_term", "short_term"), ("no_working", "working"), ("no_long_term", "long_term")):
        diff = full["summary"]["tokens"]["total_avg"] - result["variants"][name]["summary"]["tokens"]["total_avg"]
        assert diff == layers[cleared]


def test_save_refuses_overwrite_and_report(result, tmp_path):
    data = {k: v for k, v in result.items() if k != "_client"}
    paths = experiments.save_results(tmp_path, data)
    assert json.loads(paths[0].read_text(encoding="utf-8"))["meta"]["mode"] == "offline"
    report = paths[1].read_text(encoding="utf-8")
    assert "## 4. Варианты очистки" in report and "Очищена только рабочая память" in report
    with pytest.raises(experiments.ResultsExistError):
        experiments.save_results(tmp_path, data)
    with pytest.raises(experiments.ResultsExistError):
        experiments.ensure_can_write(tmp_path)
    experiments.save_results(tmp_path, data, overwrite=True)


def test_window_zero_moves_short_term_answers_out(tmp_path):
    client = FakeClient(responder=echo_responder(scripted_extractor(experiments.load_offline_extraction())))
    data = experiments.run_experiment(client, tmp_path, window=0)
    assert data["variants"]["full"]["summary"]["wrong_ids"] == ids("short_term")


def test_meta_records_instruction_versions(result):
    import config
    import hashlib
    meta = result["meta"]
    assert meta["memory_instruction_sha256"] == hashlib.sha256(config.MEMORY_SYSTEM_INSTRUCTION.encode()).hexdigest()[:12]
    assert len(meta["answer_instruction_sha256"]) == 12


def test_checker_accepts_first_person_assistant_answer():
    """v2, no_working, Q6: «Его предложил я, а ты подтвердила» — верный ответ, раньше шаблон его не принимал."""
    exp = experiments.load_expected()[6]
    assert experiments.check_answer("Мы выбрали **pytest**. Его предложил я, а ты подтвердила.", exp)
    assert not experiments.check_answer("Мы выбрали pytest, его предложила ты.", exp)
