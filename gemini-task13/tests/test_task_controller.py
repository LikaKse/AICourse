"""TaskController, ограниченный AgentAdapter, restore и receipt-проверки."""

import copy
from dataclasses import replace

import pytest

from execution import (
    AgentReport,
    ExecutionOutcome,
    ExecutionReceipt,
    LocalCodeExecutor,
)
import task_controller as controller_module

from task_controller import (
    AgentPermissionError,
    ControllerError,
    ControllerStatus,
    ExecutionAppliedError,
    RestoreTrust,
    TaskController,
    UntrustedStateError,
)
from task_state import (
    Actor,
    ActorContext,
    TaskFSM,
    TaskState,
    TransitionError,
    ValidationResult,
)


def runner_output():
    return "# отчёт\n| USD | 92.50 |"


def executor_factory(port):
    return LocalCodeExecutor(
        port, {"step-1": runner_output}, executor_id="local")


def make_controller(*, task_id=1, diagnostics=None, runner=None):
    def factory(port):
        return LocalCodeExecutor(
            port, {"step-1": runner or runner_output}, executor_id="local")

    return TaskController.new_task(
        task_id=task_id, executor_factory=factory, diagnostics=diagnostics)


def make_planning():
    controller = make_controller()
    controller.agent_view().propose_plan(["Собрать отчёт"])
    return controller


def make_execution():
    controller = make_planning()
    controller.approve_plan()
    return controller


def make_validation():
    controller = make_execution()
    controller.run_current_step()
    return controller


def make_done():
    controller = make_validation()
    request = controller.validation_request()
    result = request.create_result(
        ({"name": "report", "passed": True, "evidence": "отчёт проверен"},),
        evidence="локальная проверка",
    )
    controller.confirm_validation(result)
    return controller


def failing_result(controller):
    request = controller.validation_request()
    return request.create_result(
        ({"name": "report", "passed": False, "evidence": "отчёт пуст"},),
        evidence="локальная проверка не пройдена",
    )


def trusted_code(source="executor:local"):
    return ActorContext.controller(Actor.CODE, source=source)


class TamperingExecutor:
    """Тестовый исполнитель: подменяет receipt или actor перед отправкой."""

    def __init__(self, port, *, mutate=None, actor=None):
        self.port = port
        self.mutate = mutate
        self.actor = actor or trusted_code()

    def execute(self, request):
        base = ExecutionReceipt(
            controller_id=request.controller_id,
            run_seq=request.run_seq,
            task_id=request.task_id,
            step_id=request.step_id,
            attempt=request.attempt,
            output="подмена",
            executor_id="local",
            evidence="runner=fake; output_chars=8",
        )
        receipt = self.mutate(base) if self.mutate else base
        transition = self.port.submit_execution(receipt, actor=self.actor)
        return ExecutionOutcome(receipt=receipt, transition=transition)


class RecordingSink:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.reports = []
        self.outcomes = []

    def on_agent_report(self, report):
        self.reports.append(report)
        if self.fail:
            raise RuntimeError("diagnostics failed")

    def on_execution(self, outcome):
        self.outcomes.append(outcome)
        if self.fail:
            raise RuntimeError("diagnostics failed")


# ---------------------------------------------------------------------------
# Базовый жизненный цикл
# ---------------------------------------------------------------------------


def test_new_task_starts_in_planning():
    controller = make_controller()
    status = controller.status()

    assert status.state == "planning"
    assert status.paused is False
    assert status.current_step is None
    assert status.expected_action == "create_plan"
    assert status.plan_approved is False
    assert status.trust is RestoreTrust.FRESH
    assert status.history_trusted is True
    assert controller.task_id == 1


def test_agent_proposes_plan_and_controller_approves():
    controller = make_controller()
    controller.agent_view().propose_plan(["Первый шаг"])

    assert controller.status().expected_action == "approve_plan"
    assert controller.status().plan_approved is False

    controller.approve_plan()

    status = controller.status()
    assert status.state == "execution"
    assert status.plan_approved is True
    assert status.current_step == "step-1"
    assert status.expected_action == "execute_step:step-1"


def test_agent_report_never_changes_fsm():
    controller = make_planning()
    before = controller.snapshot()
    report = controller.agent_view().report_step_result("Я всё сделал")

    assert isinstance(report, AgentReport)
    assert report.authoritative is False
    assert controller.snapshot() == before
    assert controller.status().state == "planning"

    controller.approve_plan()
    before = controller.snapshot()
    report = controller.agent_view().report_step_result("Шаг выполнен")
    assert report.authoritative is False
    assert controller.snapshot() == before
    assert controller.status().state == "execution"


def test_agent_has_no_privileged_methods():
    agent = make_controller().agent_view()

    for name in (
        "approve_plan",
        "confirm_validation",
        "validation_request",
        "pause",
        "resume",
        "run_current_step",
        "submit_execution",
        "submit_result",
        "fsm",
        "_fsm",
        "actor",
        "_actor",
    ):
        with pytest.raises(AgentPermissionError):
            getattr(agent, name)

    assert agent.__slots__ == ("_status", "_propose", "_report")
    with pytest.raises(AgentPermissionError):
        getattr(agent, "_controller")
    with pytest.raises(AgentPermissionError):
        getattr(agent, "controller")


def test_run_before_approval_is_rejected_without_state_change():
    controller = make_controller()
    before = controller.snapshot()

    with pytest.raises(ControllerError):
        controller.run_current_step()

    assert controller.snapshot() == before
    assert controller.status().state == "planning"


def test_run_current_step_executes_and_moves_to_validation():
    calls = []

    def runner():
        calls.append("called")
        return "результат"

    controller = make_controller(runner=runner)
    controller.agent_view().propose_plan(["Шаг"])
    controller.approve_plan()

    outcome = controller.run_current_step()

    assert calls == ["called"]
    assert outcome.receipt.output == "результат"
    assert outcome.receipt.task_id == 1
    assert outcome.receipt.step_id == "step-1"
    assert outcome.receipt.attempt == 1
    assert outcome.receipt.executor_id == "local"
    assert outcome.transition.actor == "code"
    assert outcome.transition.actor_trusted is True
    assert outcome.transition.actor_source == "executor:local"

    status = controller.status()
    assert status.state == "validation"
    assert status.expected_action == "validate_step:step-1"


def test_pause_and_resume_preserve_step_and_action():
    controller = make_execution()
    before = controller.status()

    controller.pause()
    paused = controller.status()
    assert paused.paused is True
    assert paused.state == before.state
    assert paused.current_step == before.current_step
    assert paused.expected_action == "resume"

    with pytest.raises(ControllerError):
        controller.run_current_step()

    controller.resume()
    resumed = controller.status()
    assert resumed.paused is False
    assert resumed.state == before.state
    assert resumed.current_step == before.current_step
    assert resumed.expected_action == before.expected_action


def test_validation_success_and_done_terminal():
    controller = make_validation()
    request = controller.validation_request()
    result = request.create_result(
        ({"name": "report", "passed": True, "evidence": "ok"},),
        evidence="проверено",
    )

    controller.confirm_validation(result)

    status = controller.status()
    assert status.state == "done"
    assert status.current_step is None
    assert status.expected_action == "none"

    with pytest.raises((ControllerError, TransitionError)):
        controller.run_current_step()
    with pytest.raises(TransitionError):
        controller.pause()
    with pytest.raises(TransitionError):
        controller.approve_plan()


def test_validation_failure_returns_to_execution_and_increments_attempt():
    controller = make_validation()

    controller.confirm_validation(failing_result(controller))
    status = controller.status()
    assert status.state == "execution"
    assert status.current_step == "step-1"
    assert status.expected_action == "execute_step:step-1"

    outcome = controller.run_current_step()
    assert outcome.receipt.attempt == 2


def test_confirm_validation_requires_validation_result():
    controller = make_validation()

    with pytest.raises(ControllerError):
        controller.confirm_validation("not-a-result")


def test_validation_requires_issued_request_and_matching_capability():
    controller = make_validation()
    request = controller.validation_request()

    forged = ValidationResult(
        step_id=request.step_id,
        checks=({"name": "forged", "passed": True, "evidence": "forged"},),
        evidence="forged",
        validator="attacker",
        subject=request.subject,
        attempt=request.attempt,
    )
    with pytest.raises(ControllerError):
        controller.confirm_validation(forged)

    assert controller.status().state == "validation"


def test_validation_request_capability_is_one_shot_and_renews():
    controller = make_validation()
    first = controller.validation_request()
    first_result = first.create_result(
        ({"name": "x", "passed": True, "evidence": "ok"},), evidence="ok")

    second = controller.validation_request()
    with pytest.raises(ControllerError):
        controller.confirm_validation(first_result)

    second_result = second.create_result(
        ({"name": "x", "passed": True, "evidence": "ok"},), evidence="ok")
    controller.confirm_validation(second_result)

    with pytest.raises(ControllerError):
        controller.confirm_validation(second_result)


def test_capability_is_invalidated_by_pause_and_resume():
    controller = make_validation()
    request = controller.validation_request()
    old_result = request.create_result(
        ({"name": "x", "passed": True, "evidence": "old"},), evidence="old")

    controller.pause()
    controller.resume()

    with pytest.raises(ControllerError):
        controller.confirm_validation(old_result)

    new_result = controller.validation_request().create_result(
        ({"name": "x", "passed": True, "evidence": "new"},), evidence="new")
    controller.confirm_validation(new_result)
    assert controller.status().state == "done"


def test_validation_result_from_another_controller_is_rejected():
    first = make_validation()
    second = make_validation()
    first_request = first.validation_request()
    second.validation_request()
    result = first_request.create_result(
        ({"name": "x", "passed": True, "evidence": "ok"},), evidence="ok")

    with pytest.raises(ControllerError):
        second.confirm_validation(result)

    assert second.status().state == "validation"


def test_old_passing_result_cannot_finish_second_attempt():
    controller = make_validation()
    first_request = controller.validation_request()
    old_passing = first_request.create_result(
        ({"name": "x", "passed": True, "evidence": "old"},), evidence="old")
    failing = first_request.create_result(
        ({"name": "x", "passed": False, "evidence": "fail"},), evidence="fail")
    controller.confirm_validation(failing)

    controller.run_current_step()
    controller.validation_request()

    with pytest.raises(ControllerError):
        controller.confirm_validation(old_passing)

    assert controller.status().state == "validation"
    assert controller.status().state != "done"


def test_validation_request_is_only_available_in_validation():
    controller = make_execution()
    with pytest.raises(TransitionError):
        controller.validation_request()


# ---------------------------------------------------------------------------
# D2: capability запроса валидации
# ---------------------------------------------------------------------------


def test_validation_result_without_request_is_rejected():
    controller = make_validation()
    forged = ValidationResult(
        step_id="step-1",
        checks=({"name": "x", "passed": True, "evidence": "forged"},),
        evidence="forged",
        validator="attacker",
        subject="0" * 64,
        attempt=1,
    )

    with pytest.raises(ControllerError):
        controller.confirm_validation(forged)
    assert controller.status().state == "validation"


# ---------------------------------------------------------------------------
# D4: создание контроллера и защитная копия FSM
# ---------------------------------------------------------------------------


def test_controller_has_no_public_factory():
    assert not hasattr(TaskController, "_create")
    assert not hasattr(TaskController, "_build_controller")
    assert not hasattr(controller_module, "_build_controller")
    assert not hasattr(controller_module, "_FACTORY_TOKEN")
    assert not hasattr(controller_module, "_create")


def test_internal_factory_copies_state():
    source = make_execution()
    state = TaskState.from_dict(source.snapshot())
    before = state.to_dict()
    built = TaskController._TaskController__from_state(
        state,
        task_id=1,
        trust=RestoreTrust.FRESH,
        history_trusted=True,
        read_only=False,
        requires_reexecution=False,
        executor_factory=None,
        diagnostics=None,
    )

    state.plan[0].title = "mutated"
    assert state.to_dict() != before
    assert built.snapshot()["plan"][0]["title"] == before["plan"][0]["title"]


def test_restored_validation_requires_new_execution(monkeypatch):
    controller = make_validation()
    snapshot = controller.snapshot()
    restored = TaskController.restore(
        snapshot, allow_unverified=True, executor_factory=executor_factory)

    assert restored.requires_reexecution is True
    assert restored.history_trusted is False
    with pytest.raises(UntrustedStateError):
        restored.validation_request()
    with pytest.raises(UntrustedStateError):
        restored.confirm_validation(None)
    assert restored.status().state == "validation"

    transition = restored.restart_current_step()
    assert transition.event == "validation_failed"
    assert restored.status().state == "execution"
    assert restored.requires_reexecution is False

    outcome = restored.run_current_step()
    assert outcome.receipt.output == runner_output()
    request = restored.validation_request()
    result = request.create_result(
        ({"name": "x", "passed": True, "evidence": "new run"},),
        evidence="новая проверка",
    )
    restored.confirm_validation(result)
    assert restored.status().state == "done"
    assert restored.history_trusted is False


def test_restored_execution_still_runs_normally():
    controller = make_execution()
    restored = TaskController.restore(
        controller.snapshot(), allow_unverified=True,
        executor_factory=executor_factory)

    assert restored.status().state == "execution"
    assert restored.requires_reexecution is False
    outcome = restored.run_current_step()
    assert outcome.receipt.attempt == 1
    assert restored.status().state == "validation"


# ---------------------------------------------------------------------------
# D5/D6: исключения исполнителя
# ---------------------------------------------------------------------------


class RaisingExecutor:
    def execute(self, request):
        raise RuntimeError("runner boom")


class AfterSubmitExecutor:
    def __init__(self, port):
        self.port = port

    def execute(self, request):
        receipt = ExecutionReceipt(
            controller_id=request.controller_id,
            run_seq=request.run_seq,
            task_id=request.task_id,
            step_id=request.step_id,
            attempt=request.attempt,
            output="applied",
            executor_id="local",
            evidence="runner=after-submit",
        )
        self.port.submit_execution(receipt, actor=trusted_code())
        raise RuntimeError("boom after submit")


class InvalidOutcomeAfterSubmitExecutor:
    def __init__(self, port):
        self.port = port

    def execute(self, request):
        receipt = ExecutionReceipt(
            controller_id=request.controller_id,
            run_seq=request.run_seq,
            task_id=request.task_id,
            step_id=request.step_id,
            attempt=request.attempt,
            output="applied",
            executor_id="local",
            evidence="runner=invalid-outcome",
        )
        self.port.submit_execution(receipt, actor=trusted_code())
        return object()


def test_runner_exception_before_submit_keeps_execution_state():
    controller = make_execution()
    controller._executor = RaisingExecutor()

    with pytest.raises(RuntimeError):
        controller.run_current_step()

    assert controller.status().state == "execution"
    assert controller._active_request is None


def test_exception_after_submit_is_reported_as_applied():
    controller = make_execution()
    controller._executor = AfterSubmitExecutor(controller)

    with pytest.raises(ExecutionAppliedError) as exc:
        controller.run_current_step()

    assert exc.value.receipt_applied is True
    assert exc.value.state == "validation"
    assert controller.status().state == "validation"
    assert controller._active_request is None


def test_invalid_outcome_after_submit_is_reported_as_applied():
    controller = make_execution()
    controller._executor = InvalidOutcomeAfterSubmitExecutor(controller)

    with pytest.raises(ExecutionAppliedError) as exc:
        controller.run_current_step()

    assert exc.value.receipt_applied is True
    assert exc.value.state == "validation"
    assert controller.status().state == "validation"


# ---------------------------------------------------------------------------
# Receipt: привязка и повторное использование
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field, value", [
    ("controller_id", "other"),
    ("run_seq", 999),
    ("task_id", 2),
    ("step_id", "step-2"),
    ("attempt", 99),
])
def test_receipt_must_match_active_request(field, value):
    def mutate(receipt):
        return replace(receipt, **{field: value})

    controller = make_execution()
    controller._executor = TamperingExecutor(controller, mutate=mutate)
    before = controller.snapshot()

    with pytest.raises(ControllerError):
        controller.run_current_step()

    assert controller.snapshot() == before
    assert controller.status().state == "execution"


@pytest.mark.parametrize("actor", [
    ActorContext.controller(Actor.USER, source="executor:local"),
    ActorContext(actor=Actor.CODE, source="executor:local", trusted=False),
    ActorContext.controller(Actor.CODE, source="executor:other"),
])
def test_receipt_requires_code_actor_with_matching_source(actor):
    controller = make_execution()
    controller._executor = TamperingExecutor(controller, actor=actor)
    before = controller.snapshot()

    with pytest.raises(ControllerError):
        controller.run_current_step()

    assert controller.snapshot() == before


def test_receipt_cannot_be_reused_without_active_run():
    controller = make_validation()
    # После make_validation активного запуска нет: повторная отправка отклоняется.
    fake = ExecutionReceipt(
        controller_id="x", run_seq=1, task_id=1, step_id="step-1",
        attempt=1, output="x", executor_id="local", evidence="e")

    with pytest.raises(ControllerError):
        controller.submit_execution(fake, actor=trusted_code())


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------


def test_restore_clean_planning_without_confirmation():
    controller = make_planning()
    snapshot = controller.snapshot()

    restored = TaskController.restore(snapshot)

    assert restored.status().state == "planning"
    assert restored.status().trust is RestoreTrust.RESTORED_UNVERIFIED
    assert restored.history_trusted is False


@pytest.mark.parametrize("factory", [
    make_execution,
    make_validation,
    lambda: (lambda c: (c.pause(), c)[1])(make_execution()),
])
def test_restore_privileged_state_requires_confirmation(factory):
    snapshot = factory().snapshot()

    with pytest.raises(UntrustedStateError):
        TaskController.restore(snapshot)


@pytest.mark.parametrize("factory", [
    make_execution,
    make_validation,
    lambda: (lambda c: (c.pause(), c)[1])(make_execution()),
])
def test_restore_privileged_state_with_confirmation_stays_untrusted(factory):
    snapshot = factory().snapshot()

    restored = TaskController.restore(snapshot, allow_unverified=True)

    assert restored.history_trusted is False
    assert restored.status().trust is RestoreTrust.RESTORED_UNVERIFIED
    assert restored.snapshot() == snapshot


def test_restore_done_is_read_only():
    snapshot = make_done().snapshot()

    with pytest.raises(UntrustedStateError):
        TaskController.restore(snapshot)

    restored = TaskController.restore(snapshot, allow_unverified=True)
    assert restored.status().state == "done"
    assert restored.history_trusted is False

    with pytest.raises(ControllerError):
        restored.run_current_step()
    with pytest.raises(ControllerError):
        restored.pause()
    with pytest.raises(ControllerError):
        restored.approve_plan()
    with pytest.raises(ControllerError):
        restored.confirm_validation(None)


def test_restore_rejects_inconsistent_snapshot():
    snapshot = make_execution().snapshot()
    snapshot["plan_approved"] = False

    with pytest.raises(ValueError):
        TaskController.restore(snapshot, allow_unverified=True)


def test_restore_does_not_share_mutable_validation_with_input():
    snapshot = make_done().snapshot()

    restored = TaskController.restore(snapshot, allow_unverified=True)
    original = copy.deepcopy(restored.snapshot())

    snapshot["validation"]["evidence"] = "подмена"
    snapshot["transitions"][-1]["validation"]["evidence"] = "подмена"

    assert restored.snapshot() == original


def test_restored_history_is_consistent_but_not_trusted():
    snapshot = make_done().snapshot()

    restored = TaskController.restore(snapshot, allow_unverified=True)

    assert restored.status().state == "done"
    assert restored.history_trusted is False


# ---------------------------------------------------------------------------
# Диагностика
# ---------------------------------------------------------------------------


def test_diagnostics_are_separate_from_status():
    sink = RecordingSink()
    controller = make_controller(diagnostics=sink)
    controller.agent_view().propose_plan(["Шаг"])
    controller.approve_plan()

    controller.agent_view().report_step_result("готово")
    outcome = controller.run_current_step()

    assert isinstance(sink.reports[0], AgentReport)
    assert sink.outcomes == [outcome]
    status = controller.status()
    assert not hasattr(status, "last_agent_report")
    assert not hasattr(status, "last_execution")
    assert set(status.__dataclass_fields__) == set(
        ControllerStatus.__dataclass_fields__)


def test_diagnostics_failure_cannot_change_execution_result():
    sink = RecordingSink(fail=True)
    controller = make_controller(diagnostics=sink)
    controller.agent_view().propose_plan(["Шаг"])
    controller.approve_plan()

    report = controller.agent_view().report_step_result("готово")
    assert report.authoritative is False
    assert controller.status().state == "execution"

    outcome = controller.run_current_step()

    assert outcome.receipt.output == runner_output()
    assert controller.status().state == "validation"


# ---------------------------------------------------------------------------
# Границы: публичный ActorContext остаётся обходом
# ---------------------------------------------------------------------------


def test_public_actor_context_can_bypass_adapter_boundary():
    """Документирует ограничение: это не защита от произвольного кода."""
    machine = TaskFSM()
    machine.propose_plan(
        ["Шаг"], actor=ActorContext(actor=Actor.ASSISTANT, source="llm"))

    machine.approve_plan(actor=ActorContext.controller(Actor.USER))

    assert machine.state.state == "execution"
