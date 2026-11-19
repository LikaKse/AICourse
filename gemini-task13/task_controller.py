"""TaskController и ограниченный интерфейс агента.

Разделение ролей:

- ``TaskController`` — доверенная сторона: утверждает план, управляет паузой,
  применяет результат валидатора и запускает исполнителя.
- ``AgentAdapter`` — ограниченный фасад: агент может предложить план и
  сообщить результат, но не может утверждать, валидировать, ставить паузу
  или отправлять авторитетный результат шага.
- ``LocalCodeExecutor`` (execution.py) — единственное место, где создаётся
  доверенный ``Actor.CODE``. Контроллер его не создаёт.

Граница безопасности: это архитектурное разделение, а не sandbox. Публичный
``ActorContext.controller()`` и интроспекция Python позволяют обойти фасад
из того же процесса. Никакой криптографической аутентификации здесь нет.
"""
from __future__ import annotations

import copy
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from execution import (
    AgentReport,
    CodeExecutor,
    DiagnosticsSink,
    ExecutionOutcome,
    ExecutionReceipt,
    ExecutionRequest,
    StepSubmissionPort,
)
from task_state import (
    PRIVILEGED_EVENTS,
    Actor,
    ActorContext,
    TaskFSM,
    TaskState,
    Transition,
    TransitionError,
    ValidationRequest,
    ValidationResult,
)


class ControllerError(Exception):
    """Операция контроллера невозможна."""


class AgentPermissionError(ControllerError):
    """Агентскому адаптеру запрещено это действие."""


class UntrustedStateError(ControllerError):
    """Восстановленное состояние требует явного подтверждения пользователя."""


class ExecutionAppliedError(ControllerError):
    """Исполнитель выбросил исключение уже после применения receipt.

    ``receipt_applied`` всегда True: FSM уже перешла в новое состояние, и
    вызывающий код не должен трактовать ошибку как обычный неуспех.
    """

    receipt_applied = True

    def __init__(self, message: str, *, state: str) -> None:
        self.state = state
        super().__init__(message)


class RestoreTrust(str, Enum):
    FRESH = "fresh"
    RESTORED_UNVERIFIED = "restored"


@dataclass(frozen=True)
class ControllerStatus:
    """Публичный снимок состояния без диагностических данных."""

    state: str
    paused: bool
    current_step: str | None
    expected_action: str
    plan_approved: bool
    trust: RestoreTrust
    history_trusted: bool


_UNTRUSTED_ASSISTANT = ActorContext(
    actor=Actor.ASSISTANT, source="llm", trusted=False)
_CONTROLLER_USER = ActorContext.controller(Actor.USER, source="controller")
_CONTROLLER_VALIDATOR = ActorContext.controller(
    Actor.VALIDATOR, source="controller")

ExecutorFactory = Callable[[StepSubmissionPort], CodeExecutor]


def _validate_task_id(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise ValueError("task_id должен быть целым >= 1 или None")
    return value


class AgentAdapter:
    """Ограниченный фасад агента.

    Адаптер не хранит контроллер: только три узкие функции. Closure-ссылки
    всё равно доступны через Python-интроспекцию; это API-разграничение, а
    не sandbox и не защита от произвольного кода в процессе.
    """

    __slots__ = ("_status", "_propose", "_report")

    def __init__(
        self,
        *,
        status: Callable[[], ControllerStatus],
        propose: Callable[[Iterable[Any]], Transition],
        report: Callable[[str], AgentReport],
    ) -> None:
        self._status = status
        self._propose = propose
        self._report = report

    def status(self) -> ControllerStatus:
        return self._status()

    def propose_plan(self, plan: Iterable[Any]) -> Transition:
        return self._propose(plan)

    def report_step_result(self, text: str) -> AgentReport:
        return self._report(text)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        raise AgentPermissionError(f"агенту запрещено обращаться к {name!r}")


class TaskController:
    """Доверенный владелец TaskFSM и границы исполнения."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(
            "TaskController создаётся через new_task() или restore()")

    # ----- создание -----

    @classmethod
    def __from_state(
        cls,
        state: TaskState,
        *,
        task_id: int | None,
        trust: RestoreTrust,
        history_trusted: bool,
        read_only: bool,
        requires_reexecution: bool,
        executor_factory: ExecutorFactory | None,
        diagnostics: DiagnosticsSink | None,
    ) -> "TaskController":
        """Внутренняя сборка из уже проверенного состояния."""
        if not isinstance(state, TaskState):
            raise TypeError("ожидался TaskState")
        controller = object.__new__(cls)
        controller._fsm = TaskFSM(copy.deepcopy(state))
        controller._task_id = _validate_task_id(task_id)
        controller._trust = trust
        controller._history_trusted = history_trusted
        controller._read_only = read_only
        controller._requires_reexecution = requires_reexecution
        controller._diagnostics = diagnostics
        controller._controller_id = uuid.uuid4().hex
        controller._run_seq = 0
        controller._active_request: ExecutionRequest | None = None
        controller._active_validation_request: ValidationRequest | None = None
        controller._active_validation_capability: object | None = None
        controller._executor = (
            executor_factory(controller) if executor_factory is not None else None)
        return controller

    @classmethod
    def new_task(
        cls,
        *,
        task_id: int | None = None,
        executor_factory: ExecutorFactory | None = None,
        diagnostics: DiagnosticsSink | None = None,
    ) -> "TaskController":
        return cls.__from_state(
            TaskState(),
            task_id=task_id,
            trust=RestoreTrust.FRESH,
            history_trusted=True,
            read_only=False,
            requires_reexecution=False,
            executor_factory=executor_factory,
            diagnostics=diagnostics,
        )

    @classmethod
    def restore(
        cls,
        data: Mapping[str, Any],
        *,
        allow_unverified: bool = False,
        task_id: int | None = None,
        executor_factory: ExecutorFactory | None = None,
        diagnostics: DiagnosticsSink | None = None,
    ) -> "TaskController":
        """Восстановить состояние, не считая историю достоверной.

        Без подтверждения разрешён только чистый planning без привилегированных
        переходов. Остальные состояния требуют ``allow_unverified=True``.
        Восстановленный ``done`` всегда доступен только для чтения.
        """
        if not isinstance(data, Mapping):
            raise ValueError("состояние должно быть объектом")
        state = TaskState.from_dict(dict(data))
        safe = cls._is_safe_restore(state)
        if not safe and not allow_unverified:
            raise UntrustedStateError(
                "восстановление состояния "
                f"{state.state!r} требует подтверждения пользователя "
                "(allow_unverified=True): история не аутентифицирована")
        read_only = state.state == "done"
        return cls.__from_state(
            state,
            task_id=task_id,
            trust=RestoreTrust.RESTORED_UNVERIFIED,
            history_trusted=False,
            read_only=read_only,
            requires_reexecution=state.state == "validation",
            executor_factory=executor_factory,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _is_safe_restore(state: TaskState) -> bool:
        return (
            state.state == "planning"
            and not state.paused
            and not state.plan_approved
            and all(item.event not in PRIVILEGED_EVENTS for item in state.transitions)
        )

    # ----- публичное состояние -----

    @property
    def task_id(self) -> int | None:
        return self._task_id

    @property
    def history_trusted(self) -> bool:
        return self._history_trusted

    @property
    def requires_reexecution(self) -> bool:
        return self._requires_reexecution

    def status(self) -> ControllerStatus:
        state = self._fsm.state
        return ControllerStatus(
            state=state.state,
            paused=state.paused,
            current_step=state.current_step,
            expected_action=state.expected_action,
            plan_approved=state.plan_approved,
            trust=self._trust,
            history_trusted=self._history_trusted,
        )

    def snapshot(self) -> dict:
        return self._fsm.snapshot()

    def agent_view(self) -> AgentAdapter:
        def status() -> ControllerStatus:
            return self.status()

        def propose(plan: Iterable[Any]) -> Transition:
            return self._propose_plan(plan)

        def report(text: str) -> AgentReport:
            return self._report_step_result(text)

        return AgentAdapter(status=status, propose=propose, report=report)

    # ----- операции контроллера -----

    def approve_plan(self, *, reason: str = "", at: str | None = None) -> Transition:
        self._require_writable()
        return self._fsm.approve_plan(
            actor=_CONTROLLER_USER, reason=reason, at=at)

    def pause(self, *, reason: str = "", at: str | None = None) -> Transition:
        self._require_writable()
        self._clear_validation_request()
        return self._fsm.pause(actor=_CONTROLLER_USER, reason=reason, at=at)

    def resume(self, *, reason: str = "", at: str | None = None) -> Transition:
        self._require_writable()
        self._clear_validation_request()
        return self._fsm.resume(actor=_CONTROLLER_USER, reason=reason, at=at)

    def validation_request(self) -> ValidationRequest:
        """Выдать запрос с одноразовым capability для валидатора."""
        self._require_writable()
        if self._requires_reexecution:
            raise UntrustedStateError(
                "восстановленное validation требует повторного выполнения шага")
        request = self._fsm.validation_request()
        self._active_validation_request = request
        self._active_validation_capability = request._capability
        return request

    def confirm_validation(
        self,
        result: ValidationResult,
        *,
        reason: str = "",
        at: str | None = None,
    ) -> Transition:
        """Применить результат только действующего запроса валидатора.

        Capability — защита от случайной подмены результата и повторного
        использования запроса. В одном Python-процессе она не является
        криптографической аутентификацией: интроспекция и публичный
        ActorContext остаются обходом.
        """
        self._require_writable()
        if self._requires_reexecution:
            raise UntrustedStateError(
                "восстановленное validation требует повторного выполнения шага")
        if not isinstance(result, ValidationResult):
            raise ControllerError("ожидался ValidationResult от внешнего валидатора")
        request = self._active_validation_request
        capability = self._active_validation_capability
        if request is None or capability is None:
            raise ControllerError("нет действующего запроса валидации")
        if result._capability is not capability:
            raise ControllerError(
                "ValidationResult не принадлежит действующему запросу")
        expected = (request.step_id, request.subject, request.attempt)
        actual = (result.step_id, result.subject, result.attempt)
        if actual != expected:
            raise ControllerError(
                "ValidationResult не соответствует выданному запросу")

        # Одноразовое использование: даже если FSM отклонит результат,
        # повторно применить этот запрос нельзя.
        self._active_validation_request = None
        self._active_validation_capability = None
        return self._fsm.apply_validation(
            result, actor=_CONTROLLER_VALIDATOR, reason=reason, at=at)

    def restart_current_step(self, *, reason: str = "", at: str | None = None) -> Transition:
        """Перевести восстановленное validation в execution для нового запуска.

        Используется только для состояния, восстановленного с
        ``history_trusted=False``: старый результат считается недоверенным и
        не может быть подтверждён. Мы фиксируем failed-валидацию «требуется
        повторное выполнение» и возвращаемся к execution.
        """
        self._require_writable()
        if not self._requires_reexecution:
            raise ControllerError("повторный запуск доступен только для восстановленного validation")
        request = self._fsm.validation_request()
        result = request.create_result(
            ({
                "name": "restored_validation_requires_reexecution",
                "passed": False,
                "evidence": "история восстановлена без доверия",
            },),
            evidence="требуется повторное выполнение шага",
            validator="controller",
        )
        transition = self._fsm.apply_validation(
            result, actor=_CONTROLLER_VALIDATOR, reason=reason or "restart restored step",
            at=at)
        self._requires_reexecution = False
        self._active_validation_request = None
        self._active_validation_capability = None
        return transition

    def run_current_step(self, *, at: str | None = None) -> ExecutionOutcome:
        """Выполнить текущий шаг через подключённого исполнителя."""
        self._require_writable()
        if self._executor is None:
            raise ControllerError("исполнитель не подключён")

        state = self._fsm.state
        if state.paused:
            raise ControllerError("задача на паузе; сначала resume")
        if state.state != "execution":
            raise ControllerError(
                f"выполнение шага доступно только в execution, сейчас {state.state!r}")
        step_id = state.current_step
        if step_id is None:
            raise ControllerError("нет текущего шага")
        if self._active_request is not None:
            raise ControllerError("предыдущий запуск ещё не завершён")

        attempt = 1 + sum(
            1 for item in state.transitions
            if item.event == "submit_step" and item.step_id == step_id)
        self._run_seq += 1
        request = ExecutionRequest(
            controller_id=self._controller_id,
            run_seq=self._run_seq,
            task_id=self._task_id,
            step_id=step_id,
            attempt=attempt,
        )
        self._active_request = request
        state_before = self._fsm.state
        try:
            outcome = self._executor.execute(request)
        except Exception as exc:
            consumed = self._active_request is None
            self._active_request = None
            state_after = self._fsm.state
            if consumed and state_after != state_before:
                raise ExecutionAppliedError(
                    "receipt уже применён, FSM перешла в "
                    f"{state_after.state!r}; executor выбросил "
                    f"{type(exc).__name__}",
                    state=state_after.state,
                ) from exc
            raise

        if self._active_request is not None:
            self._active_request = None
            raise ControllerError("исполнитель не отправил receipt")

        if not isinstance(outcome, ExecutionOutcome):
            state_after = self._fsm.state
            if state_after != state_before:
                raise ExecutionAppliedError(
                    "receipt уже применён, но executor вернул некорректный "
                    f"outcome; FSM в {state_after.state!r}",
                    state=state_after.state,
                )
            raise ControllerError("исполнитель вернул не ExecutionOutcome")
        try:
            self._check_receipt_matches(request, outcome.receipt)
        except ControllerError as exc:
            state_after = self._fsm.state
            if state_after != state_before:
                raise ExecutionAppliedError(
                    "receipt уже применён, но outcome не совпадает с запросом; "
                    f"FSM в {state_after.state!r}",
                    state=state_after.state,
                ) from exc
            raise
        self._emit("on_execution", outcome)
        return outcome

    # ----- реализация StepSubmissionPort -----

    def submit_execution(
        self, receipt: ExecutionReceipt, *, actor: ActorContext
    ) -> Transition:
        """Принять receipt только от активного запуска и доверенного CODE.

        Проверки actor.source/executor_id — дополнительная политика и аудит,
        а не криптографическая аутентификация.
        """
        self._require_writable()
        if not isinstance(receipt, ExecutionReceipt):
            raise ControllerError("ожидался ExecutionReceipt")
        request = self._active_request
        if request is None:
            raise ControllerError("нет активного запуска для receipt")
        self._check_receipt_matches(request, receipt)

        if not isinstance(actor, ActorContext):
            raise ControllerError("ожидался ActorContext исполнителя")
        if actor.actor is not Actor.CODE:
            raise ControllerError("receipt должен отправлять Actor.CODE")
        if actor.trusted is not True:
            raise ControllerError("receipt требует доверенного CODE-контекста")
        if actor.source != f"executor:{receipt.executor_id}":
            raise ControllerError(
                "actor.source не соответствует executor_id (политика, не аутентификация)")

        self._active_request = None
        return self._fsm.submit_step(
            receipt.output,
            actor=actor,
            reason=f"executor evidence: {receipt.evidence}",
        )

    # ----- агентский фасад -----

    def _propose_plan(self, plan: Iterable[Any]) -> Transition:
        self._require_writable()
        return self._fsm.propose_plan(plan, actor=_UNTRUSTED_ASSISTANT)

    def _report_step_result(self, text: str) -> AgentReport:
        status = self.status()
        report = AgentReport(step_id=status.current_step, text=text)
        self._emit("on_agent_report", report)
        return report

    # ----- внутренние -----

    def _require_writable(self) -> None:
        if self._read_only:
            raise ControllerError(
                "восстановленный done доступен только для чтения")

    def _clear_validation_request(self) -> None:
        self._active_validation_request = None
        self._active_validation_capability = None

    def _check_receipt_matches(
        self, request: ExecutionRequest, receipt: ExecutionReceipt
    ) -> None:
        mismatches = [
            name
            for name in ("controller_id", "run_seq", "task_id", "step_id", "attempt")
            if getattr(request, name) != getattr(receipt, name)
        ]
        if mismatches:
            raise ControllerError(
                f"receipt не принадлежит активному запуску: {mismatches}")

    def _emit(self, method: str, payload: Any) -> None:
        """Изолировать диагностику от управления состоянием.

        Ошибка приёмника подавляется: она не может ни откатить уже
        применённый переход, ни превратить неуспех в успех.
        """
        if self._diagnostics is None:
            return
        try:
            getattr(self._diagnostics, method)(payload)
        except Exception:
            pass


__all__ = [
    "AgentAdapter",
    "AgentPermissionError",
    "ControllerError",
    "ControllerStatus",
    "ExecutionAppliedError",
    "RestoreTrust",
    "TaskController",
    "UntrustedStateError",
]
