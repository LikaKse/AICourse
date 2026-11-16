"""Долговременная память: Sticky Facts Дня 10 + правила слоя (данные задачи, вопрос — не факт)."""

import pytest

from memory import LongTermMemory
from memory.storage import content_text


def op(key, value, evidence, kind="user", category="other", **kw):
    return {"layer": "long_term", "op": "set", "key": key, "value": value, "kind": kind, "category": category,
            "evidence": evidence, "proposal_evidence": "", "reason": "", **kw}


def apply(lt, ops, user, assistant=None, n=1):
    return lt.apply_operations(ops, user_text=user, assistant_text=assistant, message_no=n)


def test_three_kinds_are_distinguished_and_rendered_separately():
    lt = LongTermMemory()
    apply(lt, [op("name", "Лика", "Меня зовут Лика", category="name")], "Меня зовут Лика.")
    apply(lt, [op("test_framework", "pytest", "Давай pytest", kind="accepted_proposal", category="decision",
                  proposal_evidence="удобнее всего pytest")],
          "Давай pytest.", "Удобнее всего pytest.", n=3)
    apply(lt, [op("scheduler", "launchd", "можно использовать launchd", kind="assistant_proposal")],
          "Подумаю.", "На macOS можно использовать launchd.", n=5)
    assert {k: f.kind for k, f in lt.state.facts.items()} == {
        "name": "user", "test_framework": "accepted_proposal", "scheduler": "assistant_proposal"}
    assert lt.state.facts["scheduler"].source_message == 4          # предложение — из сообщения ассистента
    text = lt.render()
    assert text.index("Данные и решения пользователя") < text.index("ЯВНО принятые") < text.index("НЕ решения")
    assert "launchd — предложение ассистента, не подтверждено" in text
    assert "[предложил ассистент: «удобнее всего pytest»]" in text


def test_unconfirmed_proposals_can_be_hidden_from_prompt():
    lt = LongTermMemory(include_unconfirmed=False, render_evidence=False)
    apply(lt, [op("scheduler", "launchd", "можно использовать launchd", kind="assistant_proposal")],
          "Подумаю.", "Можно использовать launchd.")
    assert lt.prompt_contents() == [] and lt.state.facts       # хранится, но в промпт не попадает


def test_proposal_cannot_overwrite_user_decision():
    lt = LongTermMemory()
    apply(lt, [op("scheduler", "cron", "Буду запускать через cron", category="decision")], "Буду запускать через cron.")
    out = apply(lt, [op("scheduler", "launchd", "используйте launchd", kind="assistant_proposal")],
                "Ок.", "Лучше используйте launchd.", n=3)
    assert out["rejected"][0]["rejection"] == "предложение ассистента не может заменить решение пользователя"
    assert lt.state.facts["scheduler"].value == "cron"


@pytest.mark.parametrize("operation, reason", [
    (op("budget", "5 USD", "бюджет 5 USD", category="budget"), "данные текущей задачи"),
    (op("deadline", "20 ноября", "до 20 ноября"), "данные текущей задачи"),          # по ключу
    (op("current_task", "отчёт", "сделать отчёт", category="goal"), "данные текущей задачи"),
    (op("x", "y", "Какой фреймворк посоветуешь"), "вопрос пользователя — не факт"),
    (op("city", "Нячанг", "живу в Дананге", category="location"), "цитаты нет"),
    (op("limit", "2000 запросов", "лимит 1000 запросов", category="knowledge"), "число в значении"),
    (op("x", "y", "Какой", category="weird"), "не разрешена"),
    (op("", "y", "z"), "пустой ключ"),
    ({**op("x", "y", "z"), "op": "merge"}, "неизвестная операция"),
])
def test_invalid_operations_rejected(operation, reason):
    lt = LongTermMemory()
    user = "Какой фреймворк посоветуешь? Я пишу: бюджет 5 USD, до 20 ноября, сделать отчёт, лимит 1000 запросов."
    out = apply(lt, [operation], user)
    assert out["applied"] == [] and reason in out["rejected"][0]["rejection"]
    assert lt.state.facts == {}


def test_value_change_keeps_history_and_same_value_unchanged():
    lt = LongTermMemory()
    apply(lt, [op("city", "Нячанг", "живу в Нячанге", category="location")], "Я живу в Нячанге.")
    out = apply(lt, [op("city", "Нячанг", "живу в Нячанге", category="location")], "Я живу в Нячанге.", n=3)
    assert out["unchanged"] and lt.state.version == 1
    apply(lt, [op("city", "Дананг", "переехала в Дананг", category="location")], "Я переехала в Дананг.", n=5)
    fact = lt.state.facts["city"]
    assert fact.value == "Дананг" and fact.history[0]["value"] == "Нячанг"
    assert "(ранее: Нячанг)" in lt.render()


def test_delete_requires_reason_and_user_quote():
    lt = LongTermMemory()
    apply(lt, [op("no_docker", "без Docker", "никакого Docker", category="constraint")], "Никакого Docker.")
    user = "Docker теперь можно."
    bad = apply(lt, [op("no_docker", "", "Docker теперь можно", op="delete")], user, n=3)
    assert bad["rejected"][0]["rejection"] == "удаление без причины"
    good = apply(lt, [op("no_docker", "", "Docker теперь можно", op="delete", reason="разрешила")], user, n=3)
    assert good["applied"] and "no_docker" not in lt.state.facts and "no_docker" in lt.state.deleted


def test_persistence_and_clear(tmp_path):
    path = tmp_path / "long_term.json"
    lt = LongTermMemory(path)
    apply(lt, [op("name", "Лика", "Меня зовут Лика", category="name")], "Меня зовут Лика.")
    again = LongTermMemory(path)
    assert again.public_view() == {"name": {"value": "Лика", "kind": "user", "category": "name"}}
    assert content_text(again.prompt_contents()[0]).startswith("Долговременная память")
    assert again.state.journal[0]["message"] == 1
    again.clear()
    assert LongTermMemory(path).state.facts == {}


ASSISTANT_14 = "На macOS можно использовать launchd: это системный планировщик, Docker не нужен. Вариант попроще — cron."
USER_15 = ("Подумаю над этим позже. Теперь к текущей задаче: нужно сделать еженедельный отчёт по курсам валют "
           "в формате Markdown.")


def test_regression_v2_proposal_quote_in_proposal_evidence_is_accepted():
    """Операция ровно из реального прогона v2 (results/gemini_v2): раньше отклонялась."""
    v2_op = op("scheduler_proposal", "использовать launchd или cron для планирования задач", "Подумаю над этим позже.",
               kind="assistant_proposal", category="alternative", proposal_evidence=ASSISTANT_14,
               reason="пользователь отложил предложение ассистента")
    lt = LongTermMemory()
    out = apply(lt, [v2_op], USER_15, ASSISTANT_14, n=15)
    assert out["rejected"] == [] and out["applied"][0]["normalized"] == "evidence ← proposal_evidence"
    fact = lt.state.facts["scheduler_proposal"]
    assert fact.kind == "assistant_proposal" and fact.evidence == ASSISTANT_14 and fact.source_message == 14
    assert "предложение ассистента, не подтверждено" in lt.render()


def test_proposal_without_assistant_quote_still_rejected():
    lt = LongTermMemory()
    bad = op("scheduler", "launchd", "Подумаю над этим позже.", kind="assistant_proposal",
             proposal_evidence="используйте systemd")                        # такого нет у ассистента
    out = apply(lt, [bad], USER_15, ASSISTANT_14, n=15)
    assert out["rejected"][0]["rejection"] == "цитаты нет в сообщении ассистента"
