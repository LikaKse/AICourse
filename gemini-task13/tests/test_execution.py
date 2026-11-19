"""Граница исполнения: AgentReport, ExecutionRequest/Receipt, LocalCodeExecutor."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from execution import (
    AgentReport,
    ExecutionError,
    ExecutionReceipt,
    ExecutionRequest,
    LocalCodeExecutor,
)
from task_state import Actor, ActorContext


class RecordingPort:
    def __init__(self):
        self.calls = []

    def submit_execution(self, receipt, *, actor):
        self.calls.append((receipt, actor))
        return "transition-sentinel"


def request(**overrides):
    base = {
        "controller_id": "controller-1",
        "run_seq": 1,
        "task_id": 1,
        "step_id": "step-1",
        "attempt": 1,
    }
    base.update(overrides)
    return ExecutionRequest(**base)


def receipt(**overrides):
    base = {
        "controller_id": "controller-1",
        "run_seq": 1,
        "task_id": 1,
        "step_id": "step-1",
        "attempt": 1,
        "output": "результат",
        "executor_id": "local",
        "evidence": "runner=build_report; output_chars=9",
    }
    base.update(overrides)
    return ExecutionReceipt(**base)


# ---------------------------------------------------------------------------
# AgentReport
# ---------------------------------------------------------------------------


def test_agent_report_is_never_authoritative():
    report = AgentReport(step_id="step-1", text="готово")

    assert report.authoritative is False
    assert report.step_id == "step-1"

    with pytest.raises(ValueError):
        AgentReport(step_id="step-1", text="готово", authoritative=True)


@pytest.mark.parametrize("kwargs", [
    {"step_id": "step-1", "text": ""},
    {"step_id": "step-1", "text": "   "},
    {"step_id": 1, "text": "ok"},
    {"step_id": "", "text": "ok"},
])
def test_agent_report_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        AgentReport(**kwargs)


# ---------------------------------------------------------------------------
# ExecutionRequest / ExecutionReceipt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("overrides", [
    {"controller_id": ""},
    {"controller_id": " "},
    {"controller_id": " x "},
    {"run_seq": 0},
    {"run_seq": True},
    {"run_seq": "1"},
    {"task_id": 0},
    {"task_id": True},
    {"step_id": ""},
    {"attempt": 0},
    {"attempt": False},
])
def test_execution_request_validation(overrides):
    with pytest.raises(ValueError):
        request(**overrides)


@pytest.mark.parametrize("overrides", [
    {"controller_id": ""},
    {"run_seq": 0},
    {"task_id": 0},
    {"step_id": ""},
    {"attempt": 0},
    {"output": 1},
    {"executor_id": ""},
    {"evidence": ""},
    {"evidence": " "},
])
def test_execution_receipt_validation(overrides):
    with pytest.raises(ValueError):
        receipt(**overrides)


def test_execution_request_and_receipt_are_frozen():
    req = request()
    rec = receipt()

    with pytest.raises(FrozenInstanceError):
        req.step_id = "other"
    with pytest.raises(FrozenInstanceError):
        rec.evidence = "other"

    assert replace(req, step_id="other").step_id == "other"
    assert replace(rec, output="").output == ""


def test_execution_receipt_allows_empty_output_but_not_empty_evidence():
    assert receipt(output="").output == ""
    with pytest.raises(ValueError):
        receipt(output="", evidence="")


# ---------------------------------------------------------------------------
# LocalCodeExecutor
# ---------------------------------------------------------------------------


def test_local_executor_really_calls_runner_and_submits():
    calls = []

    def runner():
        calls.append("called")
        return "# отчёт"

    port = RecordingPort()
    executor = LocalCodeExecutor(port, {"step-1": runner}, executor_id="local")
    outcome = executor.execute(request())

    assert calls == ["called"]
    assert outcome.receipt.output == "# отчёт"
    assert outcome.receipt.executor_id == "local"
    assert outcome.receipt.evidence.startswith("runner=runner;")
    assert len(port.calls) == 1
    submitted, actor = port.calls[0]
    assert submitted == outcome.receipt
    assert actor.actor is Actor.CODE
    assert actor.trusted is True
    assert actor.source == "executor:local"


def test_local_executor_uses_private_copy_of_runners():
    runners = {"step-1": lambda: "ok"}
    port = RecordingPort()
    executor = LocalCodeExecutor(port, runners)

    runners.pop("step-1")

    outcome = executor.execute(request())
    assert outcome.receipt.output == "ok"


def test_local_executor_is_deterministic():
    port = RecordingPort()
    executor = LocalCodeExecutor(port, {"step-1": lambda: "same"})

    first = executor.execute(request())
    second = executor.execute(request(run_seq=2))

    assert first.receipt.output == second.receipt.output == "same"
    assert first.receipt.evidence == second.receipt.evidence


def test_local_executor_rejects_missing_runner():
    executor = LocalCodeExecutor(RecordingPort(), {"step-1": lambda: "ok"})

    with pytest.raises(ExecutionError):
        executor.execute(request(step_id="step-2"))


def test_local_executor_rejects_non_string_runner_output():
    executor = LocalCodeExecutor(RecordingPort(), {"step-1": lambda: 123})

    with pytest.raises(ExecutionError):
        executor.execute(request())


def test_local_executor_rejects_invalid_runners():
    with pytest.raises(ValueError):
        LocalCodeExecutor(RecordingPort(), {})
    with pytest.raises(ValueError):
        LocalCodeExecutor(RecordingPort(), {"step-1": "not-callable"})
    with pytest.raises(ValueError):
        LocalCodeExecutor(RecordingPort(), {"step-1": lambda: "ok"}, executor_id=" ")


def test_local_executor_code_actor_is_created_at_execution_boundary():
    executor = LocalCodeExecutor(RecordingPort(), {"step-1": lambda: "ok"})

    actor = executor._actor  # документируем, где именно создаётся контекст
    assert isinstance(actor, ActorContext)
    assert actor.actor is Actor.CODE
    assert actor.trusted is True
    assert actor.source == "executor:local"


def test_execution_and_demo_have_no_network_or_arbitrary_execution():
    root = Path(__file__).resolve().parent.parent
    forbidden = (
        "import subprocess",
        "subprocess.",
        "eval(",
        "exec(",
        "import socket",
        "socket.",
        "import requests",
        "requests.",
        "urllib.",
        "google.genai",
    )
    for name in ("execution.py", "demo.py"):
        text = (root / name).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{name}: найден запрещённый фрагмент {token!r}"
