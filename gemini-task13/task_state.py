"""Чистое ядро Task State Machine для Дня 13.

Модуль не зависит от Gemini, памяти, профиля и CLI. Он только:
- хранит состояние задачи;
- проверяет допустимость перехода (`can_transition`);
- применяет переход после проверки (`apply`);
- ведёт историю переходов;
- поддерживает pause/resume;
- проверяет инварианты при восстановлении состояния.

Интеграция с `WorkingMemory` и агентом будет отдельным этапом.

Границы доверия
---------------
`actor`, `ValidationResult` и все поля состояния — это данные, а не
криптографические доказательства. Публичный Python API не умеет
аутентифицировать вызывающего и не защищает от подделки в памяти процесса.
Окончательная защита от подделки actor и validation будет обеспечиваться
доверенным `TaskController` на следующем этапе: контроллер сам назначает
`ActorContext` и только он создаёт доверенные контексты. Ответ LLM не должен
попадать в поле actor напрямую.
"""
from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

STATES = ("planning", "execution", "validation", "done")
STEP_STATUSES = ("pending", "active", "done", "failed")

#: Ключи, которые обязана содержать каждая проверка в ValidationResult.
CHECK_REQUIRED_KEYS = frozenset({"name", "passed", "evidence"})

#: Переходы, которые может инициировать только доверенный TaskController.
#: Строка actor сама по себе не даёт права на эти действия.
PRIVILEGED_EVENTS = frozenset({
    "approve_plan",
    "submit_step",
    "validation_passed",
    "validation_failed",
    "pause",
    "resume",
})

#: Явная таблица разрешённых переходов. Для validation_passed целевое
#: состояние зависит от того, остались ли ещё шаги плана.
TRANSITIONS: dict[str, dict[str, Any]] = {
    "propose_plan": {
        "from": {"planning"},
        "to": {"planning"},
        "actors": {"assistant", "code", "user"},
    },
    "approve_plan": {
        "from": {"planning"},
        "to": {"execution"},
        "actors": {"user"},
    },
    "submit_step": {
        "from": {"execution"},
        "to": {"validation"},
        "actors": {"code", "user"},
    },
    "validation_passed": {
        "from": {"validation"},
        "to": {"execution", "done"},
        "actors": {"validator", "code"},
    },
    "validation_failed": {
        "from": {"validation"},
        "to": {"execution"},
        "actors": {"validator", "code"},
    },
    "pause": {
        "from": {"planning", "execution", "validation"},
        "to": {"planning", "execution", "validation"},
        "actors": {"user", "code", "system"},
    },
    "resume": {
        "from": {"planning", "execution", "validation"},
        "to": {"planning", "execution", "validation"},
        "actors": {"user", "code", "system"},
    },
}


class TransitionError(Exception):
    """Недопустимый переход. Состояние FSM при этом не меняется."""

    def __init__(self, reason: str, *, state: str | None = None, event: str | None = None):
        self.reason = reason
        self.state = state
        self.event = event
        super().__init__(f"{event or 'переход'} из {state or '?'}: {reason}")


class Actor(str, Enum):
    """Роль инициатора перехода.

    Это только политическая метка, а не доказательство личности.
    """

    USER = "user"
    ASSISTANT = "assistant"
    CODE = "code"
    VALIDATOR = "validator"
    SYSTEM = "system"


@dataclass(frozen=True)
class ActorContext:
    """Контекст инициатора, который назначает доверенный TaskController.

    `trusted=True` — это архитектурный шов для следующего этапа, а не
    криптографическая гарантия. Публичный код всё ещё может создать такой
    объект; окончательная проверка полномочий остаётся за TaskController.
    """

    actor: Actor
    source: str = "caller_claim"
    trusted: bool = False

    def __post_init__(self) -> None:
        actor = self.actor
        if not isinstance(actor, Actor):
            if not isinstance(actor, str):
                raise ValueError("actor должен быть Actor или строкой")
            try:
                actor = Actor(actor)
            except ValueError as exc:
                raise ValueError(f"неизвестная роль actor {self.actor!r}") from exc
            object.__setattr__(self, "actor", actor)
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source контекста actor должен быть непустой строкой")
        if type(self.trusted) is not bool:
            raise ValueError("trusted должен быть bool")

    @classmethod
    def controller(cls, actor: Actor | str, *, source: str = "controller") -> "ActorContext":
        """Создать контекст, назначенный контроллером.

        Вызывать этот метод должен только доверенный TaskController следующего
        этапа (и unit-тесты ролей). Это не способ аутентификации.
        """
        return cls(actor=actor, source=source, trusted=True)


#: Контекст по умолчанию для предложения плана: LLM может предложить план,
#: но не может утвердить его или подтвердить валидацию.
_UNTRUSTED_ASSISTANT = ActorContext(actor=Actor.ASSISTANT, source="llm", trusted=False)


def _normalize_actor(value: ActorContext | Actor | str) -> ActorContext:
    if isinstance(value, ActorContext):
        return value
    if isinstance(value, Actor):
        return ActorContext(actor=value, source="caller_claim", trusted=False)
    if isinstance(value, str):
        return ActorContext(actor=value, source="caller_claim", trusted=False)
    raise ValueError("actor должен быть ActorContext, Actor или строкой")


def _result_fingerprint(value: str) -> str:
    """Отпечаток результата шага для привязки validation к конкретному submit.

    Это только проверка согласованности, а не подпись и не доказательство
    авторства: любой вызывающий может вычислить такой же отпечаток.
    """
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ValidationCheck:
    """Одна обязательная проверка с результатом и evidence."""

    name: str
    passed: bool
    evidence: str

    def as_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "evidence": self.evidence}


def _validate_checks(checks: Iterable[Mapping[str, Any] | ValidationCheck]
                     ) -> tuple[ValidationCheck, ...]:
    if isinstance(checks, (str, bytes)) or not isinstance(checks, Iterable):
        raise ValueError("checks должен быть непустым списком проверок")
    checks = tuple(checks)
    if not checks:
        raise ValueError("ValidationResult требует хотя бы одну проверку")

    normalized: list[ValidationCheck] = []
    seen: set[str] = set()
    for index, check in enumerate(checks, 1):
        if isinstance(check, ValidationCheck):
            name, passed, evidence = check.name, check.passed, check.evidence
        elif isinstance(check, Mapping):
            keys = set(check)
            missing = CHECK_REQUIRED_KEYS - keys
            extra = keys - CHECK_REQUIRED_KEYS
            if missing:
                raise ValueError(
                    f"проверка #{index}: отсутствуют обязательные поля {sorted(missing)}")
            if extra:
                raise ValueError(f"проверка #{index}: неизвестные поля {sorted(extra)}")
            name, passed, evidence = check["name"], check["passed"], check["evidence"]
        else:
            raise ValueError(f"проверка #{index}: ожидается объект")

        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"проверка #{index}: пустое имя")
        if type(passed) is not bool:
            raise ValueError(f"проверка #{index}: результат passed должен быть bool")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError(f"проверка #{index}: пустое evidence")
        if name.strip() in seen:
            raise ValueError(f"проверка #{index}: повторяющееся имя {name.strip()!r}")

        seen.add(name.strip())
        normalized.append(ValidationCheck(
            name=name.strip(), passed=passed, evidence=evidence.strip()))
    return tuple(normalized)


@dataclass(frozen=True)
class ValidationResult:
    """Результат программной проверки конкретного шага.

    `passed` не хранится отдельно: он вычисляется как конъюнкция результатов
    всех проверок. Поэтому нельзя создать успешный результат без checks.
    Создание результата отделено от его применения: результат создаёт
    валидатор (например, через `ValidationRequest.create_result`), а FSM
    только применяет готовый результат в `apply_validation`.
    """

    step_id: str
    checks: tuple[ValidationCheck, ...]
    evidence: str
    validator: str = "code"
    subject: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.step_id, str) or not self.step_id.strip():
            raise ValueError("ValidationResult требует непустой step_id")
        if not isinstance(self.evidence, str) or not self.evidence.strip():
            raise ValueError("ValidationResult требует непустой evidence")
        if not isinstance(self.validator, str) or not self.validator.strip():
            raise ValueError("ValidationResult требует непустой validator")
        if not isinstance(self.subject, str):
            raise ValueError("subject ValidationResult должен быть строкой")

        object.__setattr__(self, "step_id", self.step_id.strip())
        object.__setattr__(self, "evidence", self.evidence.strip())
        object.__setattr__(self, "validator", self.validator.strip())
        object.__setattr__(self, "checks", _validate_checks(self.checks))

    @property
    def passed(self) -> bool:
        """Успех возможен только если все обязательные проверки прошли."""
        return all(check.passed for check in self.checks)

    def as_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "passed": self.passed,
            "checks": [check.as_dict() for check in self.checks],
            "evidence": self.evidence,
            "validator": self.validator,
            "subject": self.subject,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationResult":
        if not isinstance(data, Mapping):
            raise ValueError("ValidationResult должен быть объектом")
        result = cls(
            step_id=data.get("step_id", ""),
            checks=data.get("checks", ()),
            evidence=data.get("evidence", ""),
            validator=data.get("validator", "code"),
            subject=data.get("subject", ""),
        )
        if "passed" in data and data["passed"] is not result.passed:
            raise ValueError("поле passed не согласовано с результатами проверок")
        return result


@dataclass(frozen=True)
class ValidationRequest:
    """Запрос на проверку, который FSM выдаёт валидатору.

    Содержит привязку к шагу и отпечаток отправленного результата. Запрос
    создаётся FSM, результат создаёт валидатор, применяет снова FSM.
    """

    step_id: str
    subject: str
    attempt: int

    def create_result(self, checks: Iterable[Mapping[str, Any]], *, evidence: str,
                      validator: str = "code") -> ValidationResult:
        return ValidationResult(
            step_id=self.step_id,
            checks=tuple(checks),
            evidence=evidence,
            validator=validator,
            subject=self.subject,
        )


@dataclass
class PlanStep:
    step_id: str
    title: str = ""
    status: str = "pending"          # pending | active | done | failed
    result: str = ""

    def to_dict(self) -> dict:
        return {"step_id": self.step_id, "title": self.title, "status": self.status, "result": self.result}

    @classmethod
    def from_dict(cls, data: dict) -> "PlanStep":
        if not isinstance(data, Mapping):
            raise ValueError("шаг плана должен быть объектом")
        return cls(step_id=str(data.get("step_id", "")), title=str(data.get("title", "")),
                   status=str(data.get("status", "pending")), result=str(data.get("result", "")))


@dataclass(frozen=True)
class Transition:
    at: str
    event: str
    actor: str
    reason: str
    from_state: str
    to_state: str
    step_id: str | None = None
    validation: dict | None = None
    actor_source: str = "caller_claim"
    actor_trusted: bool = False

    def to_dict(self) -> dict:
        return {"at": self.at, "event": self.event, "actor": self.actor, "reason": self.reason,
                "from": self.from_state, "to": self.to_state, "step_id": self.step_id,
                "validation": self.validation, "actor_source": self.actor_source,
                "actor_trusted": self.actor_trusted}

    @classmethod
    def from_dict(cls, data: dict) -> "Transition":
        if not isinstance(data, Mapping):
            raise ValueError("переход должен быть объектом")
        step_id = data.get("step_id")
        if step_id is not None and not isinstance(step_id, str):
            raise ValueError("step_id перехода должен быть строкой или None")
        actor_trusted = data.get("actor_trusted", False)
        if type(actor_trusted) is not bool:
            raise ValueError("actor_trusted перехода должен быть bool")
        return cls(at=str(data.get("at", "")), event=str(data.get("event", "")),
                   actor=str(data.get("actor", "")), reason=str(data.get("reason", "")),
                   from_state=str(data.get("from", data.get("from_state", ""))),
                   to_state=str(data.get("to", data.get("to_state", ""))),
                   step_id=step_id, validation=data.get("validation"),
                   actor_source=str(data.get("actor_source", "unknown")),
                   actor_trusted=actor_trusted)


@dataclass
class TaskState:
    state: str = "planning"
    paused: bool = False
    paused_at: str | None = None
    paused_state: str | None = None
    paused_step: str | None = None
    paused_expected_action: str | None = None

    plan: list[PlanStep] = field(default_factory=list)
    plan_approved: bool = False
    plan_approved_by: str | None = None
    plan_approved_at: str | None = None
    current_step: str | None = None
    expected_action: str = "create_plan"
    validation: dict | None = None
    last_result: str = ""

    transitions: list[Transition] = field(default_factory=list)
    version: int = 0
    updated_at: str = field(default_factory=lambda: _now())

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "paused": self.paused,
            "paused_at": self.paused_at,
            "paused_state": self.paused_state,
            "paused_step": self.paused_step,
            "paused_expected_action": self.paused_expected_action,
            "plan": [step.to_dict() for step in self.plan],
            "plan_approved": self.plan_approved,
            "plan_approved_by": self.plan_approved_by,
            "plan_approved_at": self.plan_approved_at,
            "current_step": self.current_step,
            "expected_action": self.expected_action,
            "validation": self.validation,
            "last_result": self.last_result,
            "transitions": [transition.to_dict() for transition in self.transitions],
            "version": self.version,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskState":
        if not isinstance(data, dict):
            raise ValueError("состояние FSM должно быть объектом")

        paused = data.get("paused", False)
        if type(paused) is not bool:
            raise ValueError("paused должен быть bool")
        plan_approved = data.get("plan_approved", False)
        if type(plan_approved) is not bool:
            raise ValueError("plan_approved должен быть bool")
        version = data.get("version", 0)
        if type(version) is not int or version < 0:
            raise ValueError("version должен быть неотрицательным int")

        raw_plan = data.get("plan") or []
        raw_transitions = data.get("transitions") or []
        if not isinstance(raw_plan, list):
            raise ValueError("plan должен быть списком")
        if not isinstance(raw_transitions, list):
            raise ValueError("transitions должен быть списком")

        state = cls(
            state=str(data.get("state", "planning")),
            paused=paused,
            paused_at=data.get("paused_at"),
            paused_state=data.get("paused_state"),
            paused_step=data.get("paused_step"),
            paused_expected_action=data.get("paused_expected_action"),
            plan=[PlanStep.from_dict(step) for step in raw_plan],
            plan_approved=plan_approved,
            plan_approved_by=data.get("plan_approved_by"),
            plan_approved_at=data.get("plan_approved_at"),
            current_step=data.get("current_step"),
            expected_action=str(data.get("expected_action", "create_plan")),
            validation=data.get("validation"),
            last_result=str(data.get("last_result", "")),
            transitions=[Transition.from_dict(transition) for transition in raw_transitions],
            version=version,
            updated_at=str(data.get("updated_at", _now())),
        )
        state.validate()
        return state

    # ----- инварианты восстановления -----

    def validate(self) -> None:
        """Проверить инварианты состояния. При нарушении — ValueError."""
        if self.state not in STATES:
            raise ValueError(f"неизвестное состояние FSM {self.state!r}")
        if type(self.paused) is not bool:
            raise ValueError("paused должен быть bool")
        if type(self.plan_approved) is not bool:
            raise ValueError("plan_approved должен быть bool")
        if type(self.version) is not int or self.version < 0:
            raise ValueError("version должен быть неотрицательным int")
        if not isinstance(self.expected_action, str) or not self.expected_action:
            raise ValueError("expected_action должен быть непустой строкой")
        if not isinstance(self.last_result, str):
            raise ValueError("last_result должен быть строкой")
        if not isinstance(self.updated_at, str) or not self.updated_at:
            raise ValueError("updated_at должен быть непустой строкой")

        self._validate_plan_steps()
        self._validate_approval()
        self._validate_current_step()
        self._validate_pause()
        self._validate_validation()
        self._validate_step_statuses()
        self._validate_expected_action()
        self._validate_history()

    def _validate_plan_steps(self) -> None:
        seen: set[str] = set()
        for index, step in enumerate(self.plan, 1):
            if not isinstance(step.step_id, str) or not step.step_id:
                raise ValueError(f"шаг плана #{index}: пустой step_id")
            if step.step_id in seen:
                raise ValueError(f"шаг плана #{index}: повторяющийся step_id {step.step_id!r}")
            if not isinstance(step.title, str) or not step.title.strip():
                raise ValueError(f"шаг плана #{index}: пустой title")
            if step.status not in STEP_STATUSES:
                raise ValueError(f"шаг плана #{index}: неизвестный статус {step.status!r}")
            if step.status == "failed":
                raise ValueError(
                    f"шаг плана #{index}: статус failed несовместим с историей FSM")
            if not isinstance(step.result, str):
                raise ValueError(f"шаг плана #{index}: result должен быть строкой")
            seen.add(step.step_id)

    def _validate_approval(self) -> None:
        if not self.plan_approved:
            if self.state != "planning":
                raise ValueError(
                    f"состояние {self.state!r} требует утверждённого плана")
            if self.plan_approved_by is not None or self.plan_approved_at is not None:
                raise ValueError("неутверждённый план не должен иметь plan_approved_by/at")
            return

        if not self.plan:
            raise ValueError("утверждённый план не может быть пустым")
        if not isinstance(self.plan_approved_by, str) or not self.plan_approved_by:
            raise ValueError("утверждённый план требует plan_approved_by")
        if not isinstance(self.plan_approved_at, str) or not self.plan_approved_at:
            raise ValueError("утверждённый план требует plan_approved_at")
        approvals = [item for item in self.transitions if item.event == "approve_plan"]
        if not approvals:
            raise ValueError("plan_approved=True без перехода approve_plan в истории")
        if approvals[0].at != self.plan_approved_at:
            raise ValueError("plan_approved_at не совпадает с историей")
        if approvals[0].actor != self.plan_approved_by:
            raise ValueError("plan_approved_by не совпадает с историей")

    def _validate_current_step(self) -> None:
        plan_ids = [step.step_id for step in self.plan]
        if self.current_step is not None:
            if self.current_step not in plan_ids:
                raise ValueError(
                    f"current_step {self.current_step!r} отсутствует в плане")
            current = next(step for step in self.plan if step.step_id == self.current_step)
            if current.status != "active":
                raise ValueError(
                    f"current_step {self.current_step!r} должен иметь статус active")

        if self.state in ("execution", "validation"):
            if self.current_step is None:
                raise ValueError(f"состояние {self.state!r} требует current_step")
        elif self.current_step is not None:
            raise ValueError(
                f"состояние {self.state!r} не должно иметь current_step")

    def _validate_pause(self) -> None:
        if self.paused:
            if self.paused_state is None or self.paused_state not in STATES:
                raise ValueError("paused=True требует корректного paused_state")
            if self.paused_state == "done":
                raise ValueError("paused_state=done недопустим")
            if self.paused_state != self.state:
                raise ValueError("paused_state должен совпадать с текущим state")
            if self.paused_step != self.current_step:
                raise ValueError("paused_step должен совпадать с current_step")
            if not isinstance(self.paused_at, str) or not self.paused_at:
                raise ValueError("paused=True требует paused_at")
            if not isinstance(self.paused_expected_action, str) or not self.paused_expected_action:
                raise ValueError("paused=True требует paused_expected_action")
            if self.paused_state == "planning":
                expected = "approve_plan" if self.plan else "create_plan"
            elif self.paused_state == "execution":
                expected = f"execute_step:{self.current_step}"
            elif self.paused_state == "validation":
                expected = f"validate_step:{self.current_step}"
            else:
                expected = "none"
            if self.paused_expected_action != expected:
                raise ValueError("paused_expected_action не согласован с paused_state")
        else:
            if self.paused_state is not None:
                raise ValueError("paused_state допустим только при paused=True")
            if self.paused_at is not None:
                raise ValueError("paused_at допустим только при paused=True")
            if self.paused_step is not None:
                raise ValueError("paused_step допустим только при paused=True")
            if self.paused_expected_action is not None:
                raise ValueError("paused_expected_action допустим только при paused=True")

    def _validate_validation(self) -> None:
        if self.state == "planning":
            if self.validation is not None:
                raise ValueError("в planning не должно быть validation")
            return

        if self.state == "validation":
            if self.validation is not None:
                raise ValueError("в validation результат ещё не применён")
            return

        if self.validation is None:
            if self.state == "done":
                raise ValueError("done требует применённой успешной валидации")
            return

        result = ValidationResult.from_dict(self.validation)
        if result.subject and result.subject != _result_fingerprint(self.last_result):
            raise ValueError("validation.subject не соответствует last_result")

        if self.state == "done":
            if not result.passed:
                raise ValueError("done требует validation.passed=True")
            if not self.plan or result.step_id != self.plan[-1].step_id:
                raise ValueError("done требует валидации последнего шага")

    def _validate_step_statuses(self) -> None:
        statuses = [step.status for step in self.plan]
        if self.state == "planning":
            if any(status != "pending" for status in statuses):
                raise ValueError("в planning все шаги должны быть pending")
            return

        if self.state == "done":
            if any(status != "done" for status in statuses):
                raise ValueError("в done все шаги должны быть done")
            return

        # execution / validation: ровно один active, до него done, после — pending.
        active_index = next(
            (index for index, step in enumerate(self.plan) if step.status == "active"), None)
        if active_index is None:
            raise ValueError(f"в состоянии {self.state!r} должен быть ровно один active шаг")
        if sum(1 for status in statuses if status == "active") != 1:
            raise ValueError(f"в состоянии {self.state!r} должен быть ровно один active шаг")
        if any(status != "done" for status in statuses[:active_index]):
            raise ValueError("шаги до active должны быть done")
        if any(status != "pending" for status in statuses[active_index + 1:]):
            raise ValueError("шаги после active должны быть pending")
        if self.current_step != self.plan[active_index].step_id:
            raise ValueError("current_step должен указывать на active шаг")

    def _validate_expected_action(self) -> None:
        if self.paused:
            expected = "resume"
        elif self.state == "planning":
            expected = "approve_plan" if self.plan else "create_plan"
        elif self.state == "execution":
            expected = f"execute_step:{self.current_step}"
        elif self.state == "validation":
            expected = f"validate_step:{self.current_step}"
        else:
            expected = "none"
        if self.expected_action != expected:
            raise ValueError(
                f"expected_action {self.expected_action!r} не согласован с state; "
                f"ожидалось {expected!r}")

    def _validate_history(self) -> None:
        if self.version != len(self.transitions):
            raise ValueError("version не совпадает с количеством переходов")
        if not self.transitions:
            if self.state != "planning" or self.plan or self.plan_approved:
                raise ValueError("непустое состояние без истории переходов недопустимо")
            return

        if self.transitions[0].from_state != "planning":
            raise ValueError("история должна начинаться из planning")
        if self.updated_at != self.transitions[-1].at:
            raise ValueError("updated_at не совпадает с последним переходом")

        plan_ids = [step.step_id for step in self.plan]
        expected_from = "planning"
        current: str | None = None
        paused = False
        passed_steps: list[str] = []

        for index, transition in enumerate(self.transitions, 1):
            prefix = f"переход #{index} ({transition.event!r})"
            if transition.from_state != expected_from:
                raise ValueError(
                    f"{prefix}: from={transition.from_state!r}, ожидалось {expected_from!r}")
            if not transition.at:
                raise ValueError(f"{prefix}: пустое время")
            if not transition.actor:
                raise ValueError(f"{prefix}: пустой actor")
            if not transition.actor_source:
                raise ValueError(f"{prefix}: пустой actor_source")
            if transition.event in PRIVILEGED_EVENTS and not transition.actor_trusted:
                raise ValueError(
                    f"{prefix}: привилегированный переход без доверенного ActorContext")
            rule = TRANSITIONS.get(transition.event)
            if rule is None:
                raise ValueError(f"{prefix}: неизвестное событие")
            if transition.from_state not in rule["from"]:
                raise ValueError(f"{prefix}: событие не разрешено из {transition.from_state!r}")
            if transition.to_state not in rule["to"]:
                raise ValueError(f"{prefix}: недопустимое целевое состояние {transition.to_state!r}")
            if transition.actor not in rule["actors"]:
                raise ValueError(
                    f"{prefix}: роль {transition.actor!r} не разрешена для события")

            if transition.event == "propose_plan":
                if transition.to_state != "planning":
                    raise ValueError(f"{prefix}: propose_plan должен оставаться в planning")
                if transition.validation is not None:
                    raise ValueError(f"{prefix}: propose_plan не несёт validation")
                current = None

            elif transition.event == "approve_plan":
                if not plan_ids:
                    raise ValueError(f"{prefix}: approve_plan без плана")
                current = plan_ids[0]
                if transition.step_id != current:
                    raise ValueError(f"{prefix}: approve_plan должен активировать первый шаг")
                if transition.validation is not None:
                    raise ValueError(f"{prefix}: approve_plan не несёт validation")

            elif transition.event == "submit_step":
                if current is None or transition.step_id != current:
                    raise ValueError(f"{prefix}: submit_step не для current_step")
                if transition.validation is not None:
                    raise ValueError(f"{prefix}: submit_step не несёт validation")

            elif transition.event == "validation_passed":
                if current is None or transition.step_id != current:
                    raise ValueError(f"{prefix}: validation_passed не для current_step")
                result = ValidationResult.from_dict(transition.validation or {})
                if not result.passed:
                    raise ValueError(f"{prefix}: validation_passed требует passed=True")
                if result.step_id != current:
                    raise ValueError(f"{prefix}: validation.step_id не совпадает с шагом")
                passed_steps.append(current)
                if transition.to_state == "execution":
                    position = plan_ids.index(current)
                    if position + 1 >= len(plan_ids):
                        raise ValueError(f"{prefix}: нет следующего шага для execution")
                    current = plan_ids[position + 1]
                elif transition.to_state == "done":
                    if current != plan_ids[-1]:
                        raise ValueError(f"{prefix}: done возможен только после последнего шага")
                    current = None
                else:
                    raise ValueError(f"{prefix}: недопустимый переход validation_passed")

            elif transition.event == "validation_failed":
                if current is None or transition.step_id != current:
                    raise ValueError(f"{prefix}: validation_failed не для current_step")
                result = ValidationResult.from_dict(transition.validation or {})
                if result.passed:
                    raise ValueError(f"{prefix}: validation_failed требует passed=False")
                if result.step_id != current:
                    raise ValueError(f"{prefix}: validation.step_id не совпадает с шагом")

            elif transition.event in ("pause", "resume"):
                if transition.from_state != transition.to_state:
                    raise ValueError(f"{prefix}: pause/resume не меняет состояние")
                if transition.step_id != current:
                    raise ValueError(f"{prefix}: step_id не совпадает с current_step")
                if transition.validation is not None:
                    raise ValueError(f"{prefix}: pause/resume не несёт validation")
                if transition.event == "pause":
                    if paused:
                        raise ValueError(f"{prefix}: повторная пауза")
                    paused = True
                else:
                    if not paused:
                        raise ValueError(f"{prefix}: resume без паузы")
                    paused = False

            expected_from = transition.to_state

        if paused != self.paused:
            raise ValueError("история паузы не согласована с paused")
        if self.transitions[-1].to_state != self.state:
            raise ValueError("последний переход не ведёт в текущее состояние")
        if current != self.current_step:
            raise ValueError("история не согласована с current_step")

        if self.state == "done":
            if self.transitions[-1].event != "validation_passed":
                raise ValueError("done должен завершаться validation_passed")
            if self.transitions[-1].to_state != "done":
                raise ValueError("последний переход должен вести в done")
            if set(passed_steps) != set(plan_ids) or len(passed_steps) != len(plan_ids):
                raise ValueError("в истории done должны быть пройдены все шаги плана")
            if self.validation != self.transitions[-1].validation:
                raise ValueError("validation не совпадает с последним переходом")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _normalize_plan(plan: Iterable[Any] | None) -> list[PlanStep]:
    """Принять список строк, dict или PlanStep; проверить план.

    Любой план до утверждения обязан состоять только из pending шагов.
    """
    if not plan:
        raise ValueError("план пуст")
    steps: list[PlanStep] = []
    seen: set[str] = set()
    for index, item in enumerate(plan, 1):
        if isinstance(item, PlanStep):
            step = copy.deepcopy(item)
        elif isinstance(item, dict):
            step = PlanStep(step_id=str(item.get("step_id") or f"step-{index}"),
                            title=str(item.get("title") or item.get("name") or ""),
                            status=str(item.get("status") or "pending"),
                            result=str(item.get("result") or ""))
        elif isinstance(item, str):
            step = PlanStep(step_id=f"step-{index}", title=item)
        else:
            raise ValueError(f"шаг плана #{index}: ожидается строка, dict или PlanStep")
        if not step.step_id or step.step_id in seen:
            raise ValueError(f"шаг плана #{index}: пустой или повторяющийся step_id")
        if not step.title.strip():
            raise ValueError(f"шаг плана #{index}: пустой title")
        if step.status not in STEP_STATUSES:
            raise ValueError(f"шаг плана #{index}: неизвестный статус {step.status!r}")
        if step.status != "pending":
            raise ValueError(
                f"шаг плана #{index}: до утверждения статус должен быть pending, "
                f"получен {step.status!r}")
        seen.add(step.step_id)
        steps.append(step)
    return steps


class TaskFSM:
    """Конечный автомат. Единственный способ изменить состояние — apply()."""

    def __init__(self, state: TaskState | None = None):
        self._state = copy.deepcopy(state) if state is not None else TaskState()
        self._state.validate()

    @property
    def state(self) -> TaskState:
        """Копия состояния: внешний код не может обойти переходы."""
        return copy.deepcopy(self._state)

    def snapshot(self) -> dict:
        return self._state.to_dict()

    def to_dict(self) -> dict:
        return self._state.to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> "TaskFSM":
        return cls(TaskState.from_dict(data))

    def validation_request(self) -> ValidationRequest:
        """Выдать валидатору запрос на проверку текущего шага.

        FSM не запускает проверки и не создаёт ValidationResult: это делает
        внешний валидатор. FSM только привязывает будущий результат к шагу и
        отправленному результату.
        """
        state = self._state
        if state.state != "validation":
            raise TransitionError(
                "validation_request доступен только в состоянии validation",
                state=state.state, event="validation_request")
        if state.current_step is None:
            raise TransitionError(
                "нет текущего шага", state=state.state, event="validation_request")
        attempts = sum(
            1 for item in state.transitions
            if item.event == "submit_step" and item.step_id == state.current_step)
        return ValidationRequest(
            step_id=state.current_step,
            subject=_result_fingerprint(state.last_result),
            attempt=attempts,
        )

    # ----- проверка -----

    def can_transition(self, event: str, *, actor: ActorContext | Actor | str,
                       reason: str = "",
                       plan: Iterable[Any] | None = None,
                       validation: ValidationResult | None = None,
                       step_result: str = "") -> tuple[bool, str]:
        """Проверить переход, ничего не меняя. → (разрешено, причина отказа)."""
        try:
            context = _normalize_actor(actor)
        except ValueError as exc:
            return False, str(exc)

        state = self._state

        if state.state not in STATES:
            return False, f"неизвестное состояние {state.state!r}"

        # done — терминальное состояние.
        if state.state == "done":
            return False, "из done переходы запрещены"

        if event in PRIVILEGED_EVENTS and not context.trusted:
            return False, (
                f"событие {event!r} требует доверенного ActorContext, "
                "назначенного TaskController")

        if state.paused:
            if event == "resume":
                if context.actor.value not in TRANSITIONS["resume"]["actors"]:
                    return False, f"resume не разрешён инициатору {context.actor.value!r}"
                return True, ""
            if event == "pause":
                return False, "задача уже на паузе"
            return False, "задача на паузе; сначала выполните resume"

        if event == "pause":
            if state.state not in TRANSITIONS["pause"]["from"]:
                return False, f"пауза не разрешена из состояния {state.state!r}"
            if context.actor.value not in TRANSITIONS["pause"]["actors"]:
                return False, f"пауза не разрешена инициатору {context.actor.value!r}"
            return True, ""

        if event == "resume":
            return False, "задача не на паузе"

        rule = TRANSITIONS.get(event)
        if rule is None:
            return False, f"неизвестное событие {event!r}"
        if state.state not in rule["from"]:
            return False, f"переход {event!r} не разрешён из состояния {state.state!r}"

        if context.actor.value not in rule["actors"]:
            if event == "approve_plan":
                return False, "утверждать план может только пользователь"
            return False, f"событие {event!r} не разрешено инициатору {context.actor.value!r}"

        if event == "propose_plan":
            try:
                _normalize_plan(plan)
            except ValueError as exc:
                return False, str(exc)
            return True, ""

        if event == "approve_plan":
            if state.plan_approved:
                return False, "план уже утверждён"
            try:
                normalized = _normalize_plan(state.plan)
            except ValueError as exc:
                return False, str(exc)
            if not any(step.status == "pending" for step in normalized):
                return False, "в плане нет шага для активации"
            return True, ""

        if event == "submit_step":
            if state.current_step is None:
                return False, "нет текущего шага"
            return True, ""

        if event in ("validation_passed", "validation_failed"):
            if state.current_step is None:
                return False, "нет шага для валидации"
            if validation is None:
                return False, "нужен ValidationResult"
            if not isinstance(validation, ValidationResult):
                return False, "нужен ValidationResult"
            if validation.step_id != state.current_step:
                return False, (
                    f"результат относится к шагу {validation.step_id!r}, "
                    f"а текущий шаг {state.current_step!r}")
            expected_subject = _result_fingerprint(state.last_result)
            if validation.subject != expected_subject:
                return False, (
                    "ValidationResult не привязан к отправленному результату шага")
            if event == "validation_passed" and not validation.passed:
                return False, "validation_passed требует успешной валидации"
            if event == "validation_failed" and validation.passed:
                return False, "validation_failed требует failed-валидации"
            return True, ""

        return False, f"неизвестное событие {event!r}"

    # ----- применение -----

    def apply(self, event: str, *, actor: ActorContext | Actor | str,
              reason: str = "",
              plan: Iterable[Any] | None = None,
              validation: ValidationResult | None = None,
              step_result: str = "", at: str | None = None) -> Transition:
        """Проверить и применить переход. При отказе состояние не меняется."""
        try:
            context = _normalize_actor(actor)
        except ValueError as exc:
            raise TransitionError(str(exc), state=self._state.state, event=event) from exc

        if plan is not None and not isinstance(plan, (list, tuple)):
            plan = list(plan)
        allowed, problem = self.can_transition(
            event, actor=context, reason=reason, plan=plan, validation=validation,
            step_result=step_result)
        if not allowed:
            raise TransitionError(problem, state=self._state.state, event=event)

        old = self._state
        new = copy.deepcopy(old)
        timestamp = at or _now()
        from_state = old.state
        step_for_history = old.current_step

        if event == "propose_plan":
            new.plan = _normalize_plan(plan)
            new.plan_approved = False
            new.plan_approved_by = None
            new.plan_approved_at = None
            new.current_step = None
            new.validation = None
            new.last_result = ""
            new.expected_action = "approve_plan"

        elif event == "approve_plan":
            new.plan_approved = True
            new.plan_approved_by = context.actor.value
            new.plan_approved_at = timestamp
            new.state = "execution"
            new.current_step = self._activate_first_pending(new)
            if new.current_step is None:
                raise TransitionError("в плане нет шагов", state=from_state, event=event)
            new.expected_action = f"execute_step:{new.current_step}"

        elif event == "submit_step":
            new.last_result = step_result
            new.validation = None
            new.state = "validation"
            new.expected_action = f"validate_step:{new.current_step}"

        elif event == "validation_passed":
            if validation is None:
                raise TransitionError("нужен ValidationResult", state=from_state, event=event)
            new.validation = validation.as_dict()
            step_for_history = old.current_step
            next_step = self._finish_current_and_next(new)
            if next_step:
                new.state = "execution"
                new.current_step = next_step
                new.expected_action = f"execute_step:{next_step}"
            else:
                new.state = "done"
                new.current_step = None
                new.expected_action = "none"

        elif event == "validation_failed":
            if validation is None:
                raise TransitionError("нужен ValidationResult", state=from_state, event=event)
            new.validation = validation.as_dict()
            new.state = "execution"
            new.expected_action = f"execute_step:{new.current_step}"

        elif event == "pause":
            new.paused = True
            new.paused_at = timestamp
            new.paused_state = from_state
            new.paused_step = old.current_step
            new.paused_expected_action = old.expected_action
            new.expected_action = "resume"

        elif event == "resume":
            new.paused = False
            if new.paused_state:
                new.state = new.paused_state
            new.current_step = new.paused_step
            new.expected_action = new.paused_expected_action or new.expected_action
            new.paused_at = None
            new.paused_state = None
            new.paused_step = None
            new.paused_expected_action = None

        transition = Transition(
            at=timestamp, event=event, actor=context.actor.value, reason=reason,
            from_state=from_state, to_state=new.state,
            step_id=step_for_history if event != "approve_plan" else new.current_step,
            validation=validation.as_dict() if validation else None,
            actor_source=context.source,
            actor_trusted=context.trusted,
        )
        new.transitions.append(transition)
        new.version += 1
        new.updated_at = timestamp
        new.validate()
        self._state = new
        return transition

    # ----- удобные методы -----

    def propose_plan(self, plan: Iterable[Any], *, actor: ActorContext | Actor | str = _UNTRUSTED_ASSISTANT,
                     reason: str = "", at: str | None = None) -> Transition:
        return self.apply("propose_plan", actor=actor, reason=reason, plan=plan, at=at)

    def approve_plan(self, *, actor: ActorContext | Actor | str, reason: str = "",
                     at: str | None = None) -> Transition:
        return self.apply("approve_plan", actor=actor, reason=reason, at=at)

    def submit_step(self, result: str = "", *, actor: ActorContext | Actor | str,
                    reason: str = "", at: str | None = None) -> Transition:
        return self.apply("submit_step", actor=actor, reason=reason, step_result=result, at=at)

    def apply_validation(self, result: ValidationResult, *, actor: ActorContext | Actor | str,
                         reason: str = "", at: str | None = None) -> Transition:
        """Применить уже созданный валидатором результат. FSM его не создаёт."""
        event = "validation_passed" if result.passed else "validation_failed"
        return self.apply(event, actor=actor, reason=reason, validation=result, at=at)

    def pause(self, *, actor: ActorContext | Actor | str, reason: str = "",
              at: str | None = None) -> Transition:
        return self.apply("pause", actor=actor, reason=reason, at=at)

    def resume(self, *, actor: ActorContext | Actor | str, reason: str = "",
               at: str | None = None) -> Transition:
        return self.apply("resume", actor=actor, reason=reason, at=at)

    # ----- внутренние -----

    @staticmethod
    def _activate_first_pending(state: TaskState) -> str | None:
        for step in state.plan:
            if step.status == "pending":
                step.status = "active"
                return step.step_id
        return None

    @staticmethod
    def _finish_current_and_next(state: TaskState) -> str | None:
        current = state.current_step
        for step in state.plan:
            if step.step_id == current:
                step.status = "done"
                break
        for step in state.plan:
            if step.status == "pending":
                step.status = "active"
                return step.step_id
        return None
