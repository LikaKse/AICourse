"""Сквозные тесты demo-сценария устаревания памяти."""

import contextlib

import pytest

import demo


def steps_by_name(report):
    return {step.name: step for step in report.steps}


def test_demo_answers_match_expected_values():
    report = demo.run_demo()
    steps = steps_by_name(report)

    assert steps["deadline-2"].actual == "DEADLINE-2"
    assert steps["deadline-2"].expected == "DEADLINE-2"
    assert steps["deadline-2"].passed

    assert steps["requirement-deleted"].actual == "UNKNOWN"
    assert steps["requirement-deleted"].expected == "UNKNOWN"
    assert steps["requirement-deleted"].passed

    assert steps["status-2"].actual == "STATUS-2"
    assert steps["status-2"].expected == "STATUS-2"
    assert steps["status-2"].passed

    assert steps["approach-2"].actual == "MICROSERVICES"
    assert steps["approach-2"].expected == "MICROSERVICES"
    assert steps["approach-2"].passed

    assert all(step.passed for step in report.steps)


def test_demo_prompts_contain_only_current_values():
    report = demo.run_demo()
    steps = steps_by_name(report)

    assert "DEADLINE-2" in steps["deadline-2"].prompt
    assert "DEADLINE-1" not in steps["deadline-2"].prompt

    assert "project.requirement" not in steps[
        "requirement-deleted"].prompt.split("[USER]", 1)[0]
    assert "REQUIREMENT-OLD" not in steps["requirement-deleted"].prompt

    assert "STATUS-2" in steps["status-2"].prompt
    assert "STATUS-1" not in steps["status-2"].prompt

    assert "MICROSERVICES" in steps["approach-2"].prompt
    assert "MONOLITH" not in steps["approach-2"].prompt

    for step in report.steps:
        assert "journal" not in step.prompt.lower()
        assert "tombstone" not in step.prompt.lower()
        assert "old_value" not in step.prompt
        assert "new_value" not in step.prompt


def test_demo_stale_counters_are_zero():
    report = demo.run_demo()
    steps = steps_by_name(report)

    assert set(report.stale_values) == {
        "DEADLINE-1",
        "REQUIREMENT-OLD",
        "STATUS-1",
        "MONOLITH",
        "TEMP-1",
        "TEMP-NOTE-1",
    }
    assert report.stale_prompt == 0
    assert report.stale_response == 0

    # Проверяем шаги, выполненные уже после того, как значение устарело.
    checks = {
        "deadline-2": {"DEADLINE-1"},
        "requirement-deleted": {"REQUIREMENT-OLD"},
        "status-2": {"STATUS-1"},
        "approach-2": {"MONOLITH"},
        "temporary-task-1-cleared": {"TEMP-1", "TEMP-NOTE-1"},
        "temporary-task-2-after-clear": {"TEMP-1", "TEMP-NOTE-1"},
        "final-deadline": {
            "DEADLINE-1", "REQUIREMENT-OLD", "STATUS-1",
            "MONOLITH", "TEMP-1", "TEMP-NOTE-1",
        },
        "final-status": {
            "DEADLINE-1", "REQUIREMENT-OLD", "STATUS-1",
            "MONOLITH", "TEMP-1", "TEMP-NOTE-1",
        },
        "final-approach": {
            "DEADLINE-1", "REQUIREMENT-OLD", "STATUS-1",
            "MONOLITH", "TEMP-1", "TEMP-NOTE-1",
        },
        "final-requirement": {
            "DEADLINE-1", "REQUIREMENT-OLD", "STATUS-1",
            "MONOLITH", "TEMP-1", "TEMP-NOTE-1",
        },
    }
    for name, stale in checks.items():
        step = steps[name]
        for value in stale:
            assert value not in step.prompt
            assert value not in step.actual


def test_demo_cumulative_decisions_are_kept_as_list():
    report = demo.run_demo()

    assert report.decisions == ("Use PostgreSQL", "Use Redis")
    assert "decisions:" in report.cumulative_prompt
    assert "  - \"Use PostgreSQL\"" in report.cumulative_prompt
    assert "  - \"Use Redis\"" in report.cumulative_prompt
    assert report.cumulative_prompt.index(
        "Use PostgreSQL") < report.cumulative_prompt.index("Use Redis")


def test_demo_temporary_memory_is_scoped_by_task():
    report = demo.run_demo()

    task_1 = report.temporary_prompt_task_1
    task_2 = report.temporary_prompt_task_2

    assert "TEMP-1" in task_1
    assert "TEMP-NOTE-1" in task_1
    assert "TEMP-2" not in task_1

    assert "TEMP-2" in task_2
    assert "TEMP-1" not in task_2
    assert "TEMP-NOTE-1" not in task_2


def test_demo_complete_task_clears_only_task_1():
    report = demo.run_demo()
    after = report.temporary_prompt_after_complete

    assert "TEMP-1" not in after
    assert "TEMP-NOTE-1" not in after
    assert "TEMP-2" not in after

    assert "DEADLINE-2" in after
    assert "STATUS-2" in after
    assert "MICROSERVICES" in after
    assert "Use PostgreSQL" in after
    assert "Use Redis" in after


def test_demo_final_state():
    report = demo.run_demo()

    assert report.long_term == {
        "project.deadline": "DEADLINE-2",
        "project.status": "STATUS-2",
        "stack.approach": "MICROSERVICES",
    }
    assert report.decisions == ("Use PostgreSQL", "Use Redis")
    assert report.remaining_temporary == {
        "task-1": {},
        "task-2": {"task.step": "TEMP-2"},
    }
    assert report.stale_prompt == 0
    assert report.stale_response == 0
    assert report.passed is True


def test_demo_does_not_write_project_state(tmp_path, monkeypatch):
    monkeypatch.setattr(
        demo.tempfile, "TemporaryDirectory",
        lambda **kwargs: contextlib.nullcontext(str(tmp_path)))
    report = demo.run_demo()

    assert report.passed is True
    assert (tmp_path / "memory.json").exists()


def test_demo_main_returns_zero_and_prints_pass(capsys):
    assert demo.main() == 0
    output = capsys.readouterr().out
    final = output.rsplit("=== FINAL MEMORY ===", 1)[-1]

    assert "RESULT: PASS" in final
    assert "Stale in prompt: 0" in final
    assert "Stale in responses: 0" in final
    assert "project.deadline = \"DEADLINE-2\"" in final
    assert "project.requirement" not in final
    assert "Use PostgreSQL" in final
    assert "Use Redis" in final
    assert "task-2.task.step = \"TEMP-2\"" in final
    assert "TEMP-1" not in final


@pytest.mark.parametrize(
    "name, expected",
    [
        ("deadline-2", "DEADLINE-2"),
        ("requirement-deleted", "UNKNOWN"),
        ("status-2", "STATUS-2"),
        ("approach-2", "MICROSERVICES"),
    ],
)
def test_required_answers_are_explicit(name, expected):
    report = demo.run_demo()
    step = steps_by_name(report)[name]

    assert step.actual == expected
    assert step.passed is True
