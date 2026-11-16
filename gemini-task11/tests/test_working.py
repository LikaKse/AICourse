"""Рабочая память: поля задачи, обновление по ходу работы, статус, завершение и хранение."""

from memory import WorkingMemory
from memory.storage import content_text

USER = "Нужно сделать отчёт за неделю. Срок — 20 ноября. Отчёт на одну страницу. Начинаю."


def op(key, value, evidence, **kw):
    return {"layer": "working", "op": "set", "key": key, "value": value, "kind": "user", "category": "other",
            "evidence": evidence, "proposal_evidence": "", "reason": "", **kw}


def apply(wm, ops, user=USER, assistant=None, n=1):
    return wm.apply_operations(ops, user_text=user, assistant_text=assistant, message_no=n)


def test_fields_update_with_history():
    wm = WorkingMemory()
    apply(wm, [op("goal", "отчёт за неделю", "Нужно сделать отчёт за неделю"),
               op("deadline", "20 ноября", "Срок — 20 ноября")])
    out = apply(wm, [op("deadline", "25 ноября", "перенесли на 25 ноября")], user="Срок перенесли на 25 ноября.", n=3)
    assert len(out["applied"]) == 1
    assert wm.task.value("deadline") == "25 ноября"
    assert wm.task.history == [{"field": "deadline", "old": "20 ноября", "new": "25 ноября", "message": 3}]
    assert wm.state.version == 2


def test_same_value_unchanged_and_list_dedup():
    wm = WorkingMemory()
    apply(wm, [op("goal", "отчёт за неделю", "Нужно сделать отчёт за неделю"),
               op("constraints", "отчёт на одну страницу", "Отчёт на одну страницу")])
    out = apply(wm, [op("goal", "отчёт за неделю", "Нужно сделать отчёт за неделю"),
                     op("constraint", "Отчёт на одну страницу!", "Отчёт на одну страницу")])   # алиас поля
    assert len(out["unchanged"]) == 2 and out["applied"] == []
    assert len(wm.task.items["constraints"]) == 1


def test_status_aliases_and_invalid_status():
    wm = WorkingMemory()
    assert apply(wm, [op("status", "в работе", "Начинаю")])["applied"]
    assert wm.task.status == "in_progress"
    out = apply(wm, [op("status", "почти готово", "Начинаю")])
    assert "недопустимый статус" in out["rejected"][0]["rejection"]


def test_unknown_field_and_assistant_proposal_rejected():
    wm = WorkingMemory()
    out = apply(wm, [op("priority", "высокий", "Начинаю"),
                     op("results", "таблица готова", "таблица готова", kind="assistant_proposal")],
                assistant="Думаю, таблица готова.")
    reasons = [r["rejection"] for r in out["rejected"]]
    assert "не входит в рабочую память" in reasons[0]
    assert reasons[1] == "неподтверждённое предложение ассистента — не данные задачи"
    assert wm.task is None


def test_quote_and_numbers_checked():
    wm = WorkingMemory()
    out = apply(wm, [op("deadline", "21 ноября", "Срок — 20 ноября"),
                     op("goal", "отчёт", "этого нет в сообщении")])
    assert [r["rejection"] for r in out["rejected"]] == [
        "число в значении не встречается в сообщении пользователя", "цитаты нет в сообщении пользователя"]


def test_done_task_not_in_prompt_and_new_goal_archives():
    wm = WorkingMemory()
    apply(wm, [op("goal", "отчёт за неделю", "Нужно сделать отчёт за неделю")])
    assert wm.prompt_contents() and "отчёт за неделю" in content_text(wm.prompt_contents()[0])
    assert wm.finish_task() is True and wm.finish_task() is False
    assert wm.prompt_contents() == [] and wm.public_view() == {}
    out = apply(wm, [op("deadline", "20 ноября", "Срок — 20 ноября")])
    assert out["rejected"][0]["rejection"] == "текущая задача завершена: новая задача начинается с goal"
    apply(wm, [op("goal", "новая задача", "новая задача")], user="Теперь новая задача.")
    assert wm.task.task_id == 2 and wm.task.status == "planning"
    assert len(wm.state.archived) == 1 and wm.state.archived[0]["task_id"] == 1


def test_delete_rules():
    wm = WorkingMemory()
    apply(wm, [op("goal", "отчёт за неделю", "Нужно сделать отчёт за неделю"),
               op("constraints", "отчёт на одну страницу", "Отчёт на одну страницу")])
    user = "Ограничение про одну страницу снимаем."
    bad = apply(wm, [op("constraints", "отчёт на одну страницу", "снимаем", op="delete")], user=user)
    assert bad["rejected"][0]["rejection"] == "удаление без причины"
    good = apply(wm, [op("constraints", "отчёт на одну страницу", "снимаем", op="delete", reason="отменено")], user=user)
    assert good["applied"] and wm.task.items["constraints"] == []
    missing = apply(wm, [op("budget", "", "снимаем", op="delete", reason="x")], user=user)
    assert missing["rejected"][0]["rejection"] == "удаление пустого поля"


def test_no_ops_does_not_touch_file(tmp_path):
    path = tmp_path / "working_memory.json"
    wm = WorkingMemory(path)
    apply(wm, [])
    assert not path.exists()


def test_persistence(tmp_path):
    path = tmp_path / "working_memory.json"
    wm = WorkingMemory(path)
    apply(wm, [op("goal", "отчёт за неделю", "Нужно сделать отчёт за неделю"), op("status", "in_progress", "Начинаю")])
    again = WorkingMemory(path)
    assert again.task.public_view() == wm.task.public_view()
    assert again.render() == wm.render()
    again.clear()
    assert WorkingMemory(path).task is None
