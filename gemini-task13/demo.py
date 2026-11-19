"""Локальная демонстрация Дня 13 без Gemini API и сетевых вызовов.

Сценарий показывает:
- предложение плана агентом;
- отказ выполнять шаг до утверждения плана;
- утверждение плана контроллером;
- пауза/продолжение;
- реальное локальное исполнение шага и переход к валидации;
- отказ агенту в подтверждении валидации;
- подтверждение валидатором и переход в done;
- запрет переходов после done.

AgentReport здесь намеренно не меняет FSM: он только печатается.
"""
from __future__ import annotations

from collections.abc import Callable

from execution import LocalCodeExecutor
from task_controller import (
    AgentPermissionError,
    ControllerError,
    TaskController,
)
from task_state import TransitionError, ValidationRequest, ValidationResult


def build_weekly_report() -> str:
    """Детерминированное локальное действие без сети и случайности."""
    rows = (
        ("USD", "92.50"),
        ("EUR", "100.10"),
        ("CNY", "12.80"),
    )
    lines = ["# Отчёт по курсам", "", "| Валюта | Курс |", "|---|---|"]
    lines.extend(f"| {code} | {rate} |" for code, rate in rows)
    return "\n".join(lines)


RUNNERS: dict[str, Callable[[], str]] = {"step-1": build_weekly_report}


class LocalValidator:
    """Внешний локальный валидатор: проверяет фактический вывод исполнителя."""

    def validate(self, request: ValidationRequest, output: str) -> ValidationResult:
        checks = (
            {
                "name": "report_not_empty",
                "passed": bool(output.strip()),
                "evidence": f"output_chars={len(output)}",
            },
            {
                "name": "report_header",
                "passed": output.startswith("# Отчёт по курсам"),
                "evidence": "первая строка отчёта",
            },
            {
                "name": "report_rows",
                "passed": all(
                    f"| {code} |" in output for code in ("USD", "EUR", "CNY")),
                "evidence": "три строки таблицы",
            },
        )
        return request.create_result(
            checks, evidence="локальный отчёт проверен")


def _executor_factory(port):
    return LocalCodeExecutor(port, RUNNERS, executor_id="local")


def _refusal(
    transcript: list[str],
    label: str,
    action: Callable[[], object],
    expected: tuple[type[BaseException], ...],
) -> None:
    try:
        action()
    except expected as exc:
        transcript.append(f"{label}: ОТКАЗ: {exc}")
    else:
        transcript.append(f"{label}: ОШИБКА ДЕМОНСТРАЦИИ: действие не отклонено")


def run_demo() -> list[str]:
    transcript: list[str] = []
    controller = TaskController.new_task(
        task_id=1, executor_factory=_executor_factory)
    agent = controller.agent_view()

    status = controller.status()
    transcript.append(
        f"[1] новая задача: state={status.state}, "
        f"expected_action={status.expected_action}, "
        f"history_trusted={status.history_trusted}")

    agent.propose_plan(["Собрать локальный отчёт"],)
    status = controller.status()
    transcript.append(
        f"[2] агент предложил план: state={status.state}, "
        f"expected_action={status.expected_action}")

    report = agent.report_step_result("Я уже всё сделал, можно в done")
    status = controller.status()
    transcript.append(
        f"[3] AgentReport: authoritative={report.authoritative}, "
        f"state={status.state}; FSM не изменён")

    before_refusals = controller.snapshot()
    _refusal(
        transcript,
        "[4a] агент пытается запустить шаг до утверждения плана",
        lambda: agent.run_current_step(),
        (AgentPermissionError,),
    )
    _refusal(
        transcript,
        "[4b] контроллер пытается запустить шаг до утверждения плана",
        controller.run_current_step,
        (ControllerError, TransitionError),
    )
    if controller.snapshot() != before_refusals:
        transcript.append("[4c] ОШИБКА ДЕМОНСТРАЦИИ: отказы изменили FSM")

    controller.approve_plan(reason="план согласован пользователем")
    status = controller.status()
    transcript.append(
        f"[5] контроллер утвердил план: state={status.state}, "
        f"step={status.current_step}, "
        f"expected_action={status.expected_action}")

    report = agent.report_step_result("Отчёт готов, проверьте")
    transcript.append(
        f"[6] AgentReport во время execution: authoritative={report.authoritative}; "
        f"state={controller.status().state}")

    controller.pause(reason="перерыв")
    status = controller.status()
    transcript.append(
        f"[7] пауза: paused={status.paused}, "
        f"expected_action={status.expected_action}, step={status.current_step}")

    controller.resume()
    status = controller.status()
    transcript.append(
        f"[8] продолжение: paused={status.paused}, state={status.state}, "
        f"step={status.current_step}, "
        f"expected_action={status.expected_action}")

    outcome = controller.run_current_step()
    status = controller.status()
    transcript.append(
        f"[9] исполнитель выполнил шаг: runner=build_weekly_report, "
        f"evidence={outcome.receipt.evidence}, state={status.state}, "
        f"expected_action={status.expected_action}")
    transcript.append(
        f"    executor_id={outcome.receipt.executor_id}, "
        f"attempt={outcome.receipt.attempt}, "
        f"actor_source={outcome.transition.actor_source}")

    _refusal(
        transcript,
        "[10] агент пытается подтвердить валидацию",
        lambda: agent.confirm_validation(None),
        (AgentPermissionError,),
    )
    _refusal(
        transcript,
        "[11] агент пытается отправить receipt",
        lambda: agent.submit_execution(None),
        (AgentPermissionError,),
    )

    request = controller.validation_request()
    result = LocalValidator().validate(request, outcome.receipt.output)
    controller.confirm_validation(
        result, reason="локальная проверка пройдена")
    status = controller.status()
    transcript.append(
        f"[12] валидатор подтвердил результат: state={status.state}, "
        f"current_step={status.current_step}, "
        f"expected_action={status.expected_action}")

    _refusal(
        transcript,
        "[13] запуск после done",
        controller.run_current_step,
        (ControllerError, TransitionError),
    )
    _refusal(
        transcript,
        "[14] пауза после done",
        lambda: controller.pause(),
        (ControllerError, TransitionError),
    )

    transcript.append(
        "[15] итог: порядок этапов соблюдён, AgentReport не стал свидетельством")
    return transcript


def main() -> None:
    for line in run_demo():
        print(line)


if __name__ == "__main__":
    main()
