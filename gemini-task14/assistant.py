"""Конвейер ассистента с обязательной проверкой инвариантов.

Ассистент получает только read-only интерфейс ``InvariantChecker``
(обычно ``InvariantValidator``) и не импортирует ``invariants.commands``.
Изменение инвариантов остаётся отдельной CLI-операцией.

Гарантии конвейера:
- запрос проверяется до вызова модели;
- ответ модели (``candidate``) не отдаётся пользователю до проверки;
- запрещённый candidate не попадает в результат, историю или логи;
- streaming не используется: клиент обязан вернуть ответ целиком;
- любая ошибка validator/store/клиента приводит к фиксированному отказу.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from invariants.models import Violation
from invariants.validator import ValidationDecision

STATUS_OK = "ok"
STATUS_REFUSED = "refused"

#: Тот же контракт, что и у InvariantRule.id в models.py.
_INVARIANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

REQUEST_REFUSAL = "Запрос отклонён: нарушен инвариант {ids}."
RESPONSE_REFUSAL = "Ответ отклонён: нарушен инвариант {ids}."
UNAVAILABLE_REFUSAL = "Запрос отклонён: проверка инвариантов недоступна."


class InvariantChecker(Protocol):
    """Read-only интерфейс проверки инвариантов."""

    def check_request(self, text: str) -> ValidationDecision: ...

    def check_response(self, text: str) -> ValidationDecision: ...


class ModelClient(Protocol):
    """Клиент модели с единственным не-streaming вызовом."""

    def generate(self, request: str) -> str: ...


@dataclass(frozen=True)
class AssistantResult:
    """Результат обработки запроса.

    Для отказа ``text`` содержит только фиксированный текст и ID инвариантов;
    candidate и ``Violation.message`` сюда не попадают.
    """

    status: str
    text: str
    invariant_ids: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


@dataclass(frozen=True)
class HistoryTurn:
    """Успешный ход диалога. Отказы в историю не записываются."""

    request: str
    response: str


class AssistantPipeline:
    """Проверяет запрос, вызывает модель и проверяет её ответ."""

    __slots__ = ("_validator", "_client", "_history")

    def __init__(self, validator: InvariantChecker, client: ModelClient) -> None:
        self._validator = validator
        self._client = client
        self._history: list[HistoryTurn] = []

    @property
    def history(self) -> tuple[HistoryTurn, ...]:
        return tuple(self._history)

    def handle(self, request: str) -> AssistantResult:
        """Обработать запрос с fail-closed семантикой."""
        try:
            return self._handle(request)
        except Exception:
            # Не раскрываем текст исключения: он может содержать candidate.
            return AssistantResult(STATUS_REFUSED, UNAVAILABLE_REFUSAL)

    def _handle(self, request: str) -> AssistantResult:
        if not isinstance(request, str):
            return AssistantResult(STATUS_REFUSED, UNAVAILABLE_REFUSAL)

        request_ids = self._denied_ids(
            self._validator.check_request(request))
        if request_ids is not None:
            return self._refusal(REQUEST_REFUSAL, request_ids)

        candidate = self._client.generate(request)
        if not isinstance(candidate, str):
            return AssistantResult(STATUS_REFUSED, UNAVAILABLE_REFUSAL)

        response_ids = self._denied_ids(
            self._validator.check_response(candidate))
        if response_ids is not None:
            return self._refusal(RESPONSE_REFUSAL, response_ids)

        self._history.append(HistoryTurn(request=request, response=candidate))
        return AssistantResult(STATUS_OK, candidate)

    @staticmethod
    def _denied_ids(decision: object) -> tuple[str, ...] | None:
        """Вернуть ID нарушений или None, если решение разрешает текст.

        Пустой кортеж означает «отказ без пригодных ID» и трактуется как
        fail-closed. Некорректное решение тоже приводит к пустому кортежу.
        """
        if not isinstance(decision, ValidationDecision):
            return ()
        if decision.error:
            return ()
        raw_ids = [
            getattr(item, "invariant_id", None)
            for item in decision.violations
            if isinstance(item, Violation)
        ]
        if any(
            not isinstance(value, str)
            or _INVARIANT_ID_RE.fullmatch(value) is None
            for value in raw_ids
        ):
            # Некорректный ID не раскрываем: общий fail-closed отказ.
            return ()
        ids = tuple(sorted(set(raw_ids)))
        if decision.allowed is True:
            # Противоречивое решение: нарушения есть, но allowed=True.
            return ids if ids else None
        if decision.allowed is False:
            return ids if ids else ()
        return ()

    @staticmethod
    def _refusal(template: str, ids: tuple[str, ...]) -> AssistantResult:
        if not ids:
            return AssistantResult(STATUS_REFUSED, UNAVAILABLE_REFUSAL)
        return AssistantResult(
            STATUS_REFUSED,
            template.format(ids=", ".join(ids)),
            ids,
        )


__all__ = [
    "AssistantPipeline",
    "AssistantResult",
    "HistoryTurn",
    "InvariantChecker",
    "ModelClient",
    "REQUEST_REFUSAL",
    "RESPONSE_REFUSAL",
    "STATUS_OK",
    "STATUS_REFUSED",
    "UNAVAILABLE_REFUSAL",
]
