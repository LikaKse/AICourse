"""Ядро Task State Machine: переходы, запреты, история, pause/resume, валидация."""

import copy

import pytest

from task_state import (
    STATES,
    Actor,
    ActorContext,
    PlanStep,
    TaskFSM,
    TaskState,
    TransitionError,
    ValidationResult,
)

USER = ActorContext.controller(Actor.USER)
CODE = ActorContext.controller(Actor.CODE)
VALIDATOR = ActorContext.controller(Actor.VALIDATOR)
SYSTEM = ActorContext.controller(Actor.SYSTEM)
TRUSTED_ASSISTANT = ActorContext.controller(Actor.ASSISTANT)

#: Actor, который якобы пришёл из ответа LLM. Строка роли не делает его
#: доверенным: окончательное назначение actor — задача TaskController.
LLM_ASSISTANT = ActorContext(actor=Actor.ASSISTANT, source="llm", trusted=False)
LLM_USER = ActorContext(actor=Actor.USER, source="llm", trusted=False)

CHECKS = (
    {"name": "syntax", "passed": True, "evidence": "python -m py_compile: ok"},
    {"name": "tests", "passed": True, "evidence": "pytest: 3 passed"},
)


def make_planning(plan=("step one", "step two")):
    machine = TaskFSM()
    machine.propose_plan(plan, actor=LLM_ASSISTANT, at="2026-11-19T10:00:00+07:00")
    return machine


def make_execution(plan=("step one", "step two")):
    machine = make_planning(plan)
    machine.approve_plan(actor=USER, at="2026-11-19T10:01:00+07:00")
    return machine


def make_validation(plan=("step one",)):
    machine = make_execution(plan)
    machine.submit_step("результат шага", actor=CODE, at="2026-11-19T10:02:00+07:00")
    return machine


def make_done(plan=("step one",)):
    machine = make_validation(plan)
    machine.apply_validation(passing_result(machine), actor=VALIDATOR,
                             at="2026-11-19T10:03:00+07:00")
    return machine


def passing_result(machine):
    request = machine.validation_request()
    return request.create_result(
        CHECKS,
        evidence="все обязательные проверки пройдены",
    )


def failing_result(machine):
    request = machine.validation_request()
    return request.create_result(
        ({"name": "tests", "passed": False, "evidence": "2 tests failed"},),
        evidence="обязательная проверка не пройдена",
    )


def manual_result(step_id="step-1", *, passed=True, subject="0" * 64):
    return ValidationResult(
        step_id=step_id,
        checks=({"name": "manual", "passed": passed, "evidence": "manual check"},),
        evidence="manual evidence",
        validator="test",
        subject=subject,
    )


def assert_unchanged(machine, before):
    assert machine.snapshot() == before


def snapshot_after(machine, event, *, actor, **kwargs):
    clone = TaskFSM(machine.state)
    clone.apply(event, actor=actor, **kwargs)
    return clone.snapshot()


def test_initial_state_is_planning():
    machine = TaskFSM()
    state = machine.state

    assert state.state == "planning"
    assert state.expected_action == "create_plan"
    assert state.paused is False
    assert state.plan == []
    assert state.transitions == []
    assert set(STATES) == {"planning", "execution", "validation", "done"}


def test_can_transition_is_pure():
    machine = TaskFSM()
    before = machine.snapshot()

    allowed, reason = machine.can_transition(
        "propose_plan", actor=LLM_ASSISTANT, plan=["one"])

    assert allowed is True and reason == ""
    assert machine.snapshot() == before

    allowed, reason = machine.can_transition("submit_step", actor=CODE)
    assert allowed is False
    assert "не разрешён" in reason
    assert machine.snapshot() == before


def test_propose_and_approve_plan():
    machine = TaskFSM()
    machine.propose_plan(["first", "second"], actor=LLM_ASSISTANT,
                         at="2026-11-19T10:00:00+07:00")

    state = machine.state
    assert state.state == "planning"
    assert state.expected_action == "approve_plan"
    assert [step.status for step in state.plan] == ["pending", "pending"]
    assert state.plan_approved is False
    assert state.transitions[-1].event == "propose_plan"
    assert state.transitions[-1].actor == "assistant"
    assert state.transitions[-1].actor_source == "llm"
    assert state.transitions[-1].actor_trusted is False

    machine.approve_plan(actor=USER, reason="план согласован",
                         at="2026-11-19T10:01:00+07:00")

    state = machine.state
    assert state.state == "execution"
    assert state.plan_approved is True
    assert state.plan_approved_by == "user"
    assert state.current_step == "step-1"
    assert state.plan[0].status == "active"
    assert state.expected_action == "execute_step:step-1"
    assert state.transitions[-1].event == "approve_plan"
    assert state.transitions[-1].reason == "план согласован"
    assert state.transitions[-1].actor_source == "controller"
    assert state.transitions[-1].actor_trusted is True


def test_approve_plan_requires_user():
    machine = make_planning(("one",))
    before = machine.snapshot()

    with pytest.raises(TransitionError) as exc:
        machine.approve_plan(actor=TRUSTED_ASSISTANT)

    assert "только пользователь" in exc.value.reason
    assert_unchanged(machine, before)


def test_cannot_execute_without_approved_plan():
    machine = TaskFSM()
    before = machine.snapshot()

    with pytest.raises(TransitionError):
        machine.submit_step(actor=CODE)

    assert_unchanged(machine, before)

    machine.propose_plan(["one"], actor=LLM_ASSISTANT)
    before = machine.snapshot()
    with pytest.raises(TransitionError):
        machine.apply("submit_step", actor=CODE)
    assert_unchanged(machine, before)


def test_full_happy_path_single_step():
    machine = make_validation()
    assert machine.state.state == "validation"
    assert machine.state.expected_action == "validate_step:step-1"

    transition = machine.apply_validation(
        passing_result(machine), actor=VALIDATOR,
        at="2026-11-19T10:03:00+07:00",
    )

    state = machine.state
    assert transition.from_state == "validation"
    assert transition.to_state == "done"
    assert state.state == "done"
    assert state.current_step is None
    assert state.expected_action == "none"
    assert state.plan[0].status == "done"
    assert state.validation["passed"] is True
    assert state.validation["checks"]
    assert state.validation["evidence"]
    assert state.transitions[-1].event == "validation_passed"


def test_multistep_validation_advances_step():
    machine = make_execution(("one", "two"))
    machine.submit_step("первый", actor=CODE, at="2026-11-19T10:02:00+07:00")
    machine.apply_validation(passing_result(machine), actor=VALIDATOR,
                             at="2026-11-19T10:03:00+07:00")

    state = machine.state
    assert state.state == "execution"
    assert state.current_step == "step-2"
    assert state.plan[0].status == "done"
    assert state.plan[1].status == "active"
    assert state.expected_action == "execute_step:step-2"

    machine.submit_step("второй", actor=CODE, at="2026-11-19T10:04:00+07:00")
    machine.apply_validation(passing_result(machine), actor=VALIDATOR,
                             at="2026-11-19T10:05:00+07:00")

    state = machine.state
    assert state.state == "done"
    assert state.current_step is None
    assert [step.status for step in state.plan] == ["done", "done"]


def test_validation_failure_returns_to_execution_and_keeps_step():
    machine = make_validation()
    before = machine.snapshot()

    machine.apply_validation(failing_result(machine), actor=VALIDATOR)

    state = machine.state
    assert state.state == "execution"
    assert state.current_step == "step-1"
    assert state.plan[0].status == "active"
    assert state.expected_action == "execute_step:step-1"
    assert state.validation["passed"] is False
    assert state.state != before["state"]


def test_done_requires_successful_validation():
    machine = make_validation()
    before = machine.snapshot()
    failed = failing_result(machine)
    passed = passing_result(machine)

    with pytest.raises(TransitionError):
        machine.apply("validation_passed", actor=VALIDATOR, validation=failed)
    assert_unchanged(machine, before)

    with pytest.raises(TransitionError):
        machine.apply("validation_passed", actor=VALIDATOR)
    assert_unchanged(machine, before)

    with pytest.raises(TransitionError):
        machine.apply("validation_passed", actor=TRUSTED_ASSISTANT, validation=passed)
    assert_unchanged(machine, before)

    with pytest.raises(TransitionError):
        machine.apply("validation_failed", actor=VALIDATOR, validation=passed)
    assert_unchanged(machine, before)


def test_cannot_skip_stages():
    machine = TaskFSM()
    machine.propose_plan(["one"], actor=LLM_ASSISTANT)
    before = machine.snapshot()

    with pytest.raises(TransitionError):
        machine.apply("validation_passed", actor=VALIDATOR, validation=manual_result())
    assert_unchanged(machine, before)

    machine.approve_plan(actor=USER)
    before = machine.snapshot()
    with pytest.raises(TransitionError):
        machine.apply("validation_passed", actor=VALIDATOR, validation=manual_result())
    assert_unchanged(machine, before)


@pytest.mark.parametrize("setup", ["planning", "execution", "validation"])
def test_pause_resume_preserves_state_step_and_expected_action(setup):
    if setup == "planning":
        machine = make_planning(("one",))
    elif setup == "execution":
        machine = make_execution(("one",))
    else:
        machine = make_validation()

    before = machine.state
    machine.pause(actor=USER, reason="перерыв", at="2026-11-19T11:00:00+07:00")

    paused = machine.state
    assert paused.paused is True
    assert paused.state == before.state
    assert paused.current_step == before.current_step
    assert paused.expected_action == "resume"
    assert paused.paused_state == before.state
    assert paused.paused_step == before.current_step
    assert paused.paused_expected_action == before.expected_action
    assert paused.transitions[-1].event == "pause"
    assert paused.transitions[-1].actor == "user"
    assert paused.transitions[-1].reason == "перерыв"

    machine.resume(actor=USER, at="2026-11-19T11:05:00+07:00")

    resumed = machine.state
    assert resumed.paused is False
    assert resumed.state == before.state
    assert resumed.current_step == before.current_step
    assert resumed.expected_action == before.expected_action
    assert resumed.paused_at is None
    assert resumed.transitions[-1].event == "resume"


def test_pause_is_not_allowed_from_done():
    machine = make_done()
    before = machine.snapshot()

    with pytest.raises(TransitionError):
        machine.pause(actor=USER)

    assert_unchanged(machine, before)


def test_any_transition_is_blocked_while_paused():
    machine = make_execution()
    machine.pause(actor=USER)
    before = machine.snapshot()

    for event, kwargs in (
        ("approve_plan", {}),
        ("submit_step", {"step_result": "x"}),
        ("validation_passed", {"validation": manual_result()}),
        ("pause", {}),
    ):
        with pytest.raises(TransitionError):
            machine.apply(event, actor=USER, **kwargs)
        assert_unchanged(machine, before)


def test_resume_requires_pause():
    machine = TaskFSM()
    before = machine.snapshot()

    with pytest.raises(TransitionError):
        machine.resume(actor=USER)

    assert_unchanged(machine, before)


def test_done_is_terminal():
    machine = make_done()
    before = machine.snapshot()

    for event, kwargs in (
        ("propose_plan", {"plan": ["new"]}),
        ("approve_plan", {}),
        ("submit_step", {"step_result": "x"}),
        ("validation_passed", {"validation": manual_result()}),
        ("validation_failed", {"validation": manual_result(passed=False)}),
        ("pause", {}),
        ("resume", {}),
    ):
        with pytest.raises(TransitionError) as exc:
            machine.apply(event, actor=USER, **kwargs)
        assert "done" in exc.value.reason
        assert_unchanged(machine, before)


def test_assistant_cannot_submit_or_validate():
    machine = make_execution(("one",))
    before = machine.snapshot()

    with pytest.raises(TransitionError):
        machine.submit_step(actor=TRUSTED_ASSISTANT, result="готово")
    assert_unchanged(machine, before)

    machine.submit_step(actor=CODE, result="готово")
    before = machine.snapshot()
    with pytest.raises(TransitionError):
        machine.apply("validation_passed", actor=TRUSTED_ASSISTANT,
                      validation=passing_result(machine))
    assert_unchanged(machine, before)


def test_invalid_plan_does_not_change_state():
    for plan in ([], [""], [{"step_id": "x"}, {"step_id": "x"}]):
        machine = TaskFSM()
        before = machine.snapshot()
        with pytest.raises((TransitionError, ValueError)):
            machine.propose_plan(plan, actor=LLM_ASSISTANT)
        assert_unchanged(machine, before)


def test_state_roundtrip_and_history():
    machine = make_execution(("one",))
    machine.submit_step("result", actor=CODE, at="2026-11-19T12:00:00+07:00")
    before = machine.snapshot()

    restored = TaskFSM(TaskState.from_dict(before))

    assert restored.snapshot() == before
    assert [item["event"] for item in before["transitions"]] == [
        "propose_plan", "approve_plan", "submit_step"]
    for transition in before["transitions"]:
        assert transition["at"]
        assert transition["actor"]
        assert transition["actor_source"]
        assert "reason" in transition
        assert "actor_trusted" in transition


# ---------------------------------------------------------------------------
# Негативные сценарии из аудита
# ---------------------------------------------------------------------------


def test_validation_result_rejects_missing_checks_or_evidence():
    with pytest.raises(TypeError):
        ValidationResult(passed=True)

    with pytest.raises(TypeError):
        ValidationResult(step_id="step-1", checks=CHECKS, evidence="ок", passed=True)

    with pytest.raises(ValueError):
        ValidationResult(step_id="", checks=CHECKS, evidence="ок")

    with pytest.raises(ValueError):
        ValidationResult(step_id="step-1", checks=(), evidence="ок")

    with pytest.raises(ValueError):
        ValidationResult(step_id="step-1", checks=CHECKS, evidence="")

    with pytest.raises(ValueError):
        ValidationResult(step_id="step-1", checks=CHECKS, evidence="ок", validator="")


@pytest.mark.parametrize("checks", [
    ({"name": "tests", "passed": True},),
    ({"name": "tests", "evidence": "ok"},),
    ({"name": "tests", "passed": True, "evidence": ""},),
    ({"name": "", "passed": True, "evidence": "ok"},),
    ({"name": "tests", "passed": 1, "evidence": "ok"},),
    ({"name": "tests", "passed": True, "evidence": "ok", "extra": "x"},),
    ({"name": "tests", "passed": True, "evidence": "ok"},
     {"name": "tests", "passed": True, "evidence": "ok"}),
])
def test_validation_result_rejects_malformed_checks(checks):
    with pytest.raises(ValueError):
        ValidationResult(step_id="step-1", checks=checks, evidence="ок")


def test_validation_passed_is_derived_from_checks():
    good = manual_result()
    bad = manual_result(passed=False)

    assert good.passed is True
    assert bad.passed is False
    assert good.as_dict()["passed"] is True
    assert bad.as_dict()["passed"] is False

    with pytest.raises(ValueError):
        ValidationResult.from_dict({
            "step_id": "step-1",
            "passed": True,
            "checks": [{"name": "tests", "passed": False, "evidence": "failed"}],
            "evidence": "не прошло",
        })


def test_forged_validation_cannot_finish_task():
    machine = make_validation()
    before = machine.snapshot()
    request = machine.validation_request()

    forged_subject = ValidationResult(
        step_id=request.step_id,
        checks=({"name": "fake", "passed": True, "evidence": "сам подтвердил"},),
        evidence="поддельный результат",
        validator="llm",
        subject="0" * 64,
    )
    with pytest.raises(TransitionError) as exc:
        machine.apply_validation(forged_subject, actor=VALIDATOR)
    assert "привязан" in exc.value.reason
    assert_unchanged(machine, before)

    forged_step = ValidationResult(
        step_id="step-2",
        checks=({"name": "fake", "passed": True, "evidence": "сам подтвердил"},),
        evidence="поддельный результат",
        validator="llm",
        subject=request.subject,
    )
    with pytest.raises(TransitionError) as exc:
        machine.apply_validation(forged_step, actor=VALIDATOR)
    assert "относится к шагу" in exc.value.reason
    assert_unchanged(machine, before)

    # Даже структурно корректный результат не должен приниматься без
    # доверенного actor (роль validator из LLM — только claim).
    llm_validator = ActorContext(actor=Actor.VALIDATOR, source="llm", trusted=False)
    with pytest.raises(TransitionError) as exc:
        machine.apply_validation(passing_result(machine), actor=llm_validator)
    assert "доверенного ActorContext" in exc.value.reason
    assert_unchanged(machine, before)


def test_actor_substitution_is_not_identity_proof():
    machine = make_planning(("one",))
    before = machine.snapshot()

    with pytest.raises(TransitionError) as exc:
        machine.approve_plan(actor=LLM_USER)
    assert "доверенного ActorContext" in exc.value.reason
    assert_unchanged(machine, before)

    with pytest.raises(TransitionError) as exc:
        machine.apply("approve_plan", actor="user")
    assert "доверенного ActorContext" in exc.value.reason
    assert_unchanged(machine, before)

    with pytest.raises(TransitionError) as exc:
        machine.approve_plan(actor=TRUSTED_ASSISTANT)
    assert "только пользователь" in exc.value.reason
    assert_unchanged(machine, before)

    machine.approve_plan(actor=USER)
    transition = machine.state.transitions[-1]
    assert transition.actor == "user"
    assert transition.actor_source == "controller"
    assert transition.actor_trusted is True


@pytest.mark.parametrize("status", ["active", "done", "failed"])
@pytest.mark.parametrize("as_object", [False, True])
def test_plan_with_preset_statuses_rejected(status, as_object):
    machine = TaskFSM()
    before = machine.snapshot()
    if as_object:
        plan = [PlanStep(step_id="step-1", title="one", status=status)]
    else:
        plan = [{"step_id": "step-1", "title": "one", "status": status}]

    with pytest.raises(TransitionError) as exc:
        machine.propose_plan(plan, actor=LLM_ASSISTANT)

    assert "pending" in exc.value.reason
    assert_unchanged(machine, before)


def test_approve_plan_always_activates_first_step():
    machine = make_planning(("one", "two", "three"))
    assert [step.status for step in machine.state.plan] == ["pending", "pending", "pending"]

    machine.approve_plan(actor=USER)

    state = machine.state
    assert state.current_step == state.plan[0].step_id
    assert [step.status for step in state.plan] == ["active", "pending", "pending"]
    assert state.expected_action == f"execute_step:{state.plan[0].step_id}"


_ACTOR_CONSISTENCY_CASES = [
    ("propose valid", lambda: TaskFSM(), "propose_plan", LLM_ASSISTANT,
     lambda m: {"plan": ["one"]}, True),
    ("propose preset done", lambda: TaskFSM(), "propose_plan", LLM_ASSISTANT,
     lambda m: {"plan": [{"step_id": "s", "title": "one", "status": "done"}]}, False),
    ("approve trusted user", lambda: make_planning(("one",)), "approve_plan", USER,
     lambda m: {}, True),
    ("approve llm user", lambda: make_planning(("one",)), "approve_plan", LLM_USER,
     lambda m: {}, False),
    ("submit from execution", lambda: make_execution(("one",)), "submit_step", CODE,
     lambda m: {"step_result": "x"}, True),
    ("submit from planning", lambda: TaskFSM(), "submit_step", CODE,
     lambda m: {"step_result": "x"}, False),
    ("validation passed", lambda: make_validation(), "validation_passed", VALIDATOR,
     lambda m: {"validation": passing_result(m)}, True),
    ("validation forged subject", lambda: make_validation(), "validation_passed", VALIDATOR,
     lambda m: {"validation": manual_result(subject="0" * 64)}, False),
    ("validation failed", lambda: make_validation(), "validation_failed", VALIDATOR,
     lambda m: {"validation": failing_result(m)}, True),
    ("pause execution", lambda: make_execution(("one",)), "pause", USER,
     lambda m: {}, True),
    ("resume without pause", lambda: make_execution(("one",)), "resume", USER,
     lambda m: {}, False),
    ("pause done", lambda: make_done(), "pause", USER,
     lambda m: {}, False),
]


@pytest.mark.parametrize(
    "name,setup,event,actor,make_kwargs,expected",
    _ACTOR_CONSISTENCY_CASES,
    ids=[case[0] for case in _ACTOR_CONSISTENCY_CASES],
)
def test_can_transition_and_apply_agree(name, setup, event, actor, make_kwargs, expected):
    machine = setup()
    before = machine.snapshot()
    kwargs = make_kwargs(machine)

    allowed, reason = machine.can_transition(event, actor=actor, **kwargs)

    assert allowed is expected, reason
    clone = TaskFSM(machine.state)
    if expected:
        clone.apply(event, actor=actor, **kwargs)
    else:
        with pytest.raises(TransitionError):
            clone.apply(event, actor=actor, **kwargs)
    assert_unchanged(machine, before)


def test_validation_request_is_only_available_during_validation():
    with pytest.raises(TransitionError):
        TaskFSM().validation_request()

    machine = make_execution(("one",))
    with pytest.raises(TransitionError):
        machine.validation_request()


def test_from_dict_rejects_forged_done_without_checks():
    snapshot = make_done().snapshot()
    snapshot["validation"]["checks"] = []

    with pytest.raises(ValueError):
        TaskState.from_dict(snapshot)


def test_from_dict_rejects_forged_done_with_inconsistent_passed_flag():
    snapshot = make_done().snapshot()
    snapshot["validation"]["passed"] = False

    with pytest.raises(ValueError):
        TaskState.from_dict(snapshot)


def test_from_dict_rejects_done_without_validation():
    snapshot = make_execution(("one",)).snapshot()
    snapshot["state"] = "done"
    snapshot["current_step"] = None
    snapshot["expected_action"] = "none"
    snapshot["validation"] = None

    with pytest.raises(ValueError):
        TaskState.from_dict(snapshot)


def test_from_dict_rejects_done_without_history():
    snapshot = make_done().snapshot()
    snapshot["transitions"] = []
    snapshot["version"] = 0

    with pytest.raises(ValueError):
        TaskState.from_dict(snapshot)


def test_from_dict_rejects_done_whose_history_does_not_end_with_validation():
    snapshot = make_done().snapshot()
    snapshot["transitions"][-1]["to"] = "execution"

    with pytest.raises(ValueError):
        TaskState.from_dict(snapshot)


def test_from_dict_rejects_done_with_history_validation_mismatch():
    snapshot = make_done().snapshot()
    snapshot["transitions"][-1]["validation"]["evidence"] = "другая проверка"

    with pytest.raises(ValueError) as exc:
        TaskState.from_dict(snapshot)
    assert "validation не совпадает" in str(exc.value)


def test_from_dict_rejects_execution_without_approved_plan():
    snapshot = make_execution(("one",)).snapshot()
    snapshot["plan_approved"] = False

    with pytest.raises(ValueError) as exc:
        TaskState.from_dict(snapshot)
    assert "утверждённого плана" in str(exc.value)


def test_from_dict_rejects_current_step_missing_from_plan():
    snapshot = make_execution(("one",)).snapshot()
    snapshot["current_step"] = "ghost"

    with pytest.raises(ValueError) as exc:
        TaskState.from_dict(snapshot)
    assert "отсутствует в плане" in str(exc.value)


def test_from_dict_rejects_missing_current_step_in_execution():
    snapshot = make_execution(("one",)).snapshot()
    snapshot["current_step"] = None

    with pytest.raises(ValueError):
        TaskState.from_dict(snapshot)


def test_from_dict_rejects_paused_state_done():
    snapshot = make_execution(("one",)).snapshot()
    snapshot["paused"] = True
    snapshot["paused_at"] = "2026-11-19T11:00:00+07:00"
    snapshot["paused_state"] = "done"
    snapshot["paused_step"] = snapshot["current_step"]
    snapshot["paused_expected_action"] = snapshot["expected_action"]
    snapshot["expected_action"] = "resume"

    with pytest.raises(ValueError) as exc:
        TaskState.from_dict(snapshot)
    assert "paused_state=done" in str(exc.value)


def test_from_dict_rejects_paused_state_done_even_when_not_paused():
    snapshot = make_execution(("one",)).snapshot()
    snapshot["paused_state"] = "done"

    with pytest.raises(ValueError):
        TaskState.from_dict(snapshot)


@pytest.mark.parametrize("mutation", [
    lambda snap: snap["transitions"][-1].__setitem__("from", "validation"),
    lambda snap: snap["transitions"][-1].__setitem__("to", "planning"),
    lambda snap: snap.__setitem__("version", 999),
    lambda snap: snap.__setitem__("updated_at", "2026-11-19T23:59:59+07:00"),
    lambda snap: snap.__setitem__("expected_action", "none"),
    lambda snap: snap["plan"][0].__setitem__("status", "pending"),
    lambda snap: snap["transitions"][-1].__setitem__("actor_trusted", False),
    lambda snap: snap["transitions"][-1].__setitem__("actor", "assistant"),
])
def test_from_dict_rejects_inconsistent_execution(mutation):
    snapshot = make_execution(("one",)).snapshot()
    mutation(snapshot)

    with pytest.raises(ValueError):
        TaskState.from_dict(snapshot)


def test_from_dict_rejects_inconsistent_step_statuses_in_done():
    snapshot = make_done().snapshot()
    snapshot["plan"][0]["status"] = "pending"

    with pytest.raises(ValueError):
        TaskState.from_dict(snapshot)


@pytest.mark.parametrize("factory", [
    lambda: TaskFSM(),
    lambda: make_planning(("one",)),
    lambda: make_execution(("one", "two")),
    lambda: make_validation(),
    lambda: make_done(),
    lambda: (lambda m: (m.pause(actor=SYSTEM), m)[1])(make_execution(("one",))),
])
def test_from_dict_accepts_valid_states(factory):
    machine = factory()
    snapshot = machine.snapshot()

    restored = TaskState.from_dict(copy.deepcopy(snapshot))

    assert restored.to_dict() == snapshot
