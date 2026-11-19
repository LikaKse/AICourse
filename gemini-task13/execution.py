"""Граница исполнения шага: AgentReport, ExecutionRequest/Receipt, LocalCodeExecutor.

Модуль разделяет два принципиально разных объекта:

- ``AgentReport`` — сообщение агента. Оно никогда не меняет FSM и не является
  свидетельством выполнения.
- ``ExecutionReceipt`` — данные, созданные исполнителем после реального
  локального действия. Только receipt может попасть в ``submit_step``.

``LocalCodeExecutor`` вызывает функцию из фиксированного доверенного реестра.
Он не выполняет ``eval``/``exec``/``subprocess`` и не обращается к сети.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from task_state import Actor, ActorContext, Transition


class ExecutionError(Exception):
    """Локальное исполнение шага невозможно или вернуло неверный результат."""


def _require_nonempty_str(value: object, field: str, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: поле {field!r} должно быть непустой строкой")
    if value != value.strip():
        raise ValueError(f"{where}: поле {field!r} не должно иметь внешних пробелов")
    return value


def _require_positive_int(value: object, field: str, *, where: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{where}: поле {field!r} должно быть целым >= 1")
    return value


def _require_task_id(value: object, *, where: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise ValueError(f"{where}: task_id должен быть целым >= 1 или None")
    return value


@dataclass(frozen=True)
class AgentReport:
    """Советующее сообщение агента. Не является доказательством выполнения.

    ``authoritative`` существует только для явной маркировки и обязан быть
    ``False``. FSM не принимает AgentReport ни в одном событии.
    """

    step_id: str | None
    text: str
    authoritative: bool = False

    def __post_init__(self) -> None:
        if self.authoritative is not False:
            raise ValueError("AgentReport не может быть authoritative")
        if self.step_id is not None and (
                not isinstance(self.step_id, str) or not self.step_id.strip()):
            raise ValueError("AgentReport: step_id должен быть непустой строкой или None")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("AgentReport: text должен быть непустой строкой")


@dataclass(frozen=True)
class ExecutionRequest:
    """Запрос доверенного исполнителю на выполнение текущего шага.

    Создаётся контроллером на один запуск. Служит для привязки receipt к
    конкретному контроллеру, задаче, шагу и попытке.
    """

    controller_id: str
    run_seq: int
    task_id: int | None
    step_id: str
    attempt: int

    def __post_init__(self) -> None:
        _require_nonempty_str(self.controller_id, "controller_id",
                              where="ExecutionRequest")
        _require_positive_int(self.run_seq, "run_seq", where="ExecutionRequest")
        _require_task_id(self.task_id, where="ExecutionRequest")
        _require_nonempty_str(self.step_id, "step_id", where="ExecutionRequest")
        _require_positive_int(self.attempt, "attempt", where="ExecutionRequest")


@dataclass(frozen=True)
class ExecutionReceipt:
    """Результат реального локального исполнения.

    Все поля — неизменяемые примитивы, поэтому внешние ссылки не могут
    изменить receipt после создания. Проверка ``executor_id``/``evidence``
    является политикой и аудитом, а не криптографической аутентификацией.
    """

    controller_id: str
    run_seq: int
    task_id: int | None
    step_id: str
    attempt: int
    output: str
    executor_id: str
    evidence: str

    def __post_init__(self) -> None:
        where = "ExecutionReceipt"
        _require_nonempty_str(self.controller_id, "controller_id", where=where)
        _require_positive_int(self.run_seq, "run_seq", where=where)
        _require_task_id(self.task_id, where=where)
        _require_nonempty_str(self.step_id, "step_id", where=where)
        _require_positive_int(self.attempt, "attempt", where=where)
        if not isinstance(self.output, str):
            raise ValueError(f"{where}: output должен быть строкой")
        _require_nonempty_str(self.executor_id, "executor_id", where=where)
        _require_nonempty_str(self.evidence, "evidence", where=where)


@dataclass(frozen=True)
class ExecutionOutcome:
    """Receipt исполнителя вместе с уже применённым переходом FSM."""

    receipt: ExecutionReceipt
    transition: Transition

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, ExecutionReceipt):
            raise ValueError("ExecutionOutcome: receipt должен быть ExecutionReceipt")


class StepSubmissionPort(Protocol):
    """Узкий порт, через который исполнитель отправляет receipt контроллеру."""

    def submit_execution(
        self, receipt: ExecutionReceipt, *, actor: ActorContext
    ) -> Transition: ...


class CodeExecutor(Protocol):
    """Доверенная граница исполнения шага.

    Контракт: ``execute`` либо возвращает ``ExecutionOutcome``, либо
    выбрасывает исключение **до** отправки receipt. Если executor выбросил
    исключение после успешного ``submit_execution``, контроллер сообщает
    ``ExecutionAppliedError``: состояние уже изменено, обычный «неуспех»
    здесь недопустим.
    """

    def execute(self, request: ExecutionRequest) -> ExecutionOutcome: ...


class DiagnosticsSink(Protocol):
    """Необязательный приёмник диагностики.

    Не входит в состояние FSM и не влияет на переходы. Ошибки приёмника
    изолируются контроллером и не могут изменить исход операции.
    """

    def on_agent_report(self, report: AgentReport) -> None: ...

    def on_execution(self, outcome: ExecutionOutcome) -> None: ...


class LocalCodeExecutor:
    """Выполняет шаг через фиксированный реестр локальных функций.

    CODE-контекст создаётся здесь, на границе исполнителя, а не в
    контроллере. ``actor.source`` имеет вид ``executor:<executor_id>``.
    """

    def __init__(
        self,
        port: StepSubmissionPort,
        runners: Mapping[str, Callable[[], str]],
        *,
        executor_id: str = "local",
    ) -> None:
        _require_nonempty_str(executor_id, "executor_id", where="LocalCodeExecutor")
        if not isinstance(runners, Mapping) or not runners:
            raise ValueError("LocalCodeExecutor: runners должен быть непустым отображением")
        copied: dict[str, Callable[[], str]] = {}
        for step_id, runner in runners.items():
            _require_nonempty_str(step_id, "step_id", where="LocalCodeExecutor.runners")
            if not callable(runner):
                raise ValueError(
                    f"LocalCodeExecutor: runner для {step_id!r} должен быть вызываемым")
            copied[step_id] = runner

        self._port = port
        self._runners = copied
        self._executor_id = executor_id
        self._actor = ActorContext.controller(
            Actor.CODE, source=f"executor:{executor_id}")

    def execute(self, request: ExecutionRequest) -> ExecutionOutcome:
        if not isinstance(request, ExecutionRequest):
            raise ExecutionError("ожидался ExecutionRequest")
        runner = self._runners.get(request.step_id)
        if runner is None:
            raise ExecutionError(
                f"нет локального runner для шага {request.step_id!r}")

        output = runner()
        if not isinstance(output, str):
            raise ExecutionError("локальный runner должен вернуть строку")

        name = getattr(runner, "__name__", type(runner).__name__)
        receipt = ExecutionReceipt(
            controller_id=request.controller_id,
            run_seq=request.run_seq,
            task_id=request.task_id,
            step_id=request.step_id,
            attempt=request.attempt,
            output=output,
            executor_id=self._executor_id,
            evidence=f"runner={name}; output_chars={len(output)}",
        )
        transition = self._port.submit_execution(receipt, actor=self._actor)
        return ExecutionOutcome(receipt=receipt, transition=transition)


__all__ = [
    "AgentReport",
    "CodeExecutor",
    "DiagnosticsSink",
    "ExecutionError",
    "ExecutionOutcome",
    "ExecutionReceipt",
    "ExecutionRequest",
    "LocalCodeExecutor",
    "StepSubmissionPort",
]
