import json
from pathlib import Path

import pytest

import personalization_experiment as pe
from fake_client import FakeClient, all_text, api_error, text_response
from memory import MemoryManager
from memory.storage import text_content


def test_neutral_formal_answer_is_unverified_not_a_style_violation():
    response = (
        "Функция в Python — это именованный блок кода, предназначенный для выполнения определенной задачи. "
        "Она позволяет избегать дублирования кода и структурировать программу. Функции могут принимать входные данные "
        "(аргументы) и возвращать результаты.\n\nПример:\n```python\ndef calculate_sum(a, b):\n    return a + b\n```"
    )
    result = pe.assess(response, {"length": "кратко", "code": "не показывать", "style": "формальный"})
    assert result["checks"] == {"length": True, "code": False, "style": None}
    assert result["unverified"] == ["style"]


def test_code_detection_ignores_inline_mentions_and_word_count_ignores_fenced_code():
    response = "Для примера можно написать `def f(): return 1`."
    result = pe.assess(response, {"length": "кратко", "code": "не показывать", "style": "формальный"})
    assert result["checks"]["code"] is True
    assert result["has_code"] is False
    assert result["word_count"] == 8


def test_style_evaluator_uses_explicit_markers_and_leaves_neutral_unscored():
    formal = pe.assess("Вы можете использовать этот способ. Следует проверить результат.",
                       {"length": "подробно", "code": "не показывать", "style": "формальный"})
    informal = pe.assess("Давай, попробуй этот способ.",
                         {"length": "кратко", "code": "показывать", "style": "неформальный"})
    assert formal["checks"]["style"] is True
    assert informal["checks"]["style"] is True
    neutral = pe.assess("Это определение функции.",
                        {"length": "кратко", "code": "показывать", "style": "формальный"})
    assert neutral["checks"]["style"] is None


def ok_client(answer: str = "Ответ.") -> FakeClient:
    return FakeClient(responder=lambda contents, config: text_response(answer))


def test_run_uses_fresh_state_and_ignores_old_state_dirs(tmp_path):
    stale = tmp_path / "state_A"
    memory = MemoryManager(stale)
    memory.record_exchange(text_content("user", "СТАРЫЙ КОНТЕКСТ"),
                           text_content("model", "СТАРЫЙ ОТВЕТ"))
    client = ok_client()

    data = pe.run(tmp_path, client=client, overwrite=True)

    assert data["state"]["isolation"].startswith("fresh temporary")
    assert data["state"]["reused_previous"] is False
    assert all("СТАРЫЙ КОНТЕКСТ" not in all_text(request["contents"]) for request in client.requests)
    assert (tmp_path / pe.RESULT_JSON).is_file()
    assert (tmp_path / pe.RESULT_REPORT).is_file()


def test_run_metadata_and_current_assessments(tmp_path):
    answer = "Давай, используй код.\n```python\nprint(1)\n```"
    client = ok_client(answer)

    data = pe.run(tmp_path, client=client, overwrite=True)

    assert data["generated_at"]
    assert data["model"] == pe.config.MODEL
    assert data["requests"] == {"planned": 25, "attempted": 25, "successful": 25,
                                "failed": 0, "generate_content": 25}
    assert data["assessment"]["version"] == pe.ASSESSMENT_VERSION
    assert data["errors"] == []
    for name, item in data["profiles"].items():
        assert item["answered"] == 5 and item["failed"] == 0 and item["complete"] is True
        for row in item["questions"]:
            assert row["status"] == "ok"
            assert row["assessment"] == pe.assess(row["answer"], item["preferences"])
    assert data["stability"]["answered"] == 10
    assert data["stability"]["failed"] == 0


def test_api_error_is_failed_not_assessed_and_not_recorded(tmp_path):
    def responder(contents, config):
        text = all_text(contents)
        if "Что такое рекурсия" in text and "Стиль общения: неформальный" in text:
            return api_error(503, "boom")
        return text_response("Ответ.")

    client = FakeClient(responder=responder)
    data = pe.run(tmp_path, client=client, overwrite=True)

    profile_a = data["profiles"]["A"]
    failed = [row for row in profile_a["questions"] if row["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["answer"] is None
    assert failed[0]["assessment"] is None
    assert failed[0]["error_status"] == "api_error"
    assert "boom" in failed[0]["error"]
    assert profile_a["answered"] == 4 and profile_a["failed"] == 1
    assert profile_a["complete"] is False
    assert data["requests"]["failed"] == 1
    assert data["requests"]["successful"] == 24
    assert data["errors"][0]["profile"] == "A"

    # Ошибка не должна попасть в short-term и, значит, в промпт следующего вопроса.
    later = [request for request in client.requests
             if "Как отсортировать список чисел" in all_text(request["contents"])
             and "Стиль общения: неформальный" in all_text(request["contents"])]
    assert later
    assert all("Ошибка Gemini API" not in all_text(request["contents"]) for request in later)
    assert all("boom" not in all_text(request["contents"]) for request in later)

    report = (tmp_path / pe.RESULT_REPORT).read_text(encoding="utf-8")
    assert "**Ошибка:**" in report
    assert "4/5" in report


def test_stability_failure_is_marked_and_makes_conclusion_incomplete(tmp_path):
    def responder(contents, config):
        if "Вопрос 7" in all_text(contents):
            return api_error(500, "stability failure")
        return text_response("Ответ.")

    data = pe.run(tmp_path, client=FakeClient(responder=responder), overwrite=True)

    stability = data["stability"]
    assert stability["answered"] == 9 and stability["failed"] == 1
    assert stability["complete"] is False
    assert stability["zero_violations"] is False
    failed = [row for row in stability["turns"] if row["status"] == "failed"]
    assert len(failed) == 1 and failed[0]["assessment"] is None
    assert failed[0]["violation_reason"] is None
    assert any(error["profile"] == "stability" for error in data["errors"])


def test_report_table_contains_all_profiles_before_sections(tmp_path):
    pe.run(tmp_path, client=ok_client(), overwrite=True)
    lines = (tmp_path / pe.RESULT_REPORT).read_text(encoding="utf-8").splitlines()

    header = next(index for index, line in enumerate(lines) if line.startswith("| Профиль"))
    table = []
    for line in lines[header:]:
        if line.startswith("|"):
            table.append(line)
        elif table:
            break

    assert len(table) == 5  # заголовок + разделитель + A + B + C
    assert any(line.startswith("| A |") for line in table)
    assert any(line.startswith("| B |") for line in table)
    assert any(line.startswith("| C |") for line in table)

    first_section = next(index for index, line in enumerate(lines) if line.startswith("## Профиль"))
    assert first_section > header + len(table) - 1


def test_run_refuses_overwrite_without_flag(tmp_path):
    (tmp_path / pe.RESULT_JSON).write_text("{}", encoding="utf-8")
    client = ok_client()

    with pytest.raises(pe.ResultsExistError):
        pe.run(tmp_path, client=client)

    assert client.requests == []


def test_run_allows_overwrite_with_flag(tmp_path):
    (tmp_path / pe.RESULT_JSON).write_text("{}", encoding="utf-8")
    data = pe.run(tmp_path, client=ok_client(), overwrite=True)
    assert data["requests"]["attempted"] == 25
    assert (tmp_path / pe.RESULT_JSON).read_text(encoding="utf-8") != "{}"


def test_paid_run_requires_confirmation_and_does_not_create_client(tmp_path, monkeypatch):
    def boom():
        raise AssertionError("реальный клиент Gemini создавать нельзя")

    monkeypatch.setattr(pe, "make_client", boom)

    assert pe.run(tmp_path, assume_yes=False, input_fn=lambda _: "n") is None
    assert not (tmp_path / pe.RESULT_JSON).exists()
    assert not (tmp_path / pe.RESULT_REPORT).exists()


def test_report_only_recomputes_stale_snapshot_and_keeps_json(tmp_path):
    payload = {
        "model": "archived-model",
        "settings": {"thinking": "MINIMAL", "max_output_tokens": 100},
        "profiles": {"A": {
            "preferences": {"length": "кратко", "code": "показывать", "style": "неформальный"},
            "satisfied": 1, "total": 3, "percent": 33.33,
            "questions": [{
                "question": "Q",
                "answer": "Используй конструкцию try-except.",
                "assessment": {"checks": {"length": True, "code": False, "style": False},
                               "satisfied": 1, "total": 3},
            }],
        }},
        "stability": {
            "profile": "C", "first_violation_question": 1,
            "turns": [{
                "question_no": 1, "answer": "Ответ.",
                "assessment": {"checks": {"length": True, "code": True, "style": False},
                               "satisfied": 2, "total": 3},
                "violation_reason": "style",
            }],
        },
    }
    source = tmp_path / pe.RESULT_JSON
    source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    before = source.read_bytes()

    target = pe.report_only(tmp_path)

    assert source.read_bytes() == before
    report = target.read_text(encoding="utf-8")
    assert "пересчитан" in report
    assert "стиль ✓" in report
