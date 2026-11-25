"""HTTP-клиент JSONPlaceholder и модель задачи."""

from __future__ import annotations

from typing import Any

import httpx2
from pydantic import BaseModel, ConfigDict, Field, ValidationError

import config


class TaskNotFound(Exception):
    """Задача с указанным id не найдена."""


class TaskApiError(Exception):
    """Ошибка запроса к JSONPlaceholder или некорректный ответ."""


class Task(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user_id: int = Field(alias="userId")
    id: int
    title: str
    completed: bool


class TaskApi:
    """Минимальный клиент JSONPlaceholder для одной операции get task."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = config.TASKS_API_TIMEOUT,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or config.TASKS_API_BASE).rstrip("/")
        self.timeout = timeout
        self.transport = transport

    async def fetch_task(self, task_id: int) -> Task:
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 1:
            raise TaskApiError("task_id должен быть целым числом больше нуля.")

        url = f"{self.base_url}/todos/{task_id}"
        try:
            async with httpx2.AsyncClient(
                timeout=httpx2.Timeout(self.timeout),
                transport=self.transport,
            ) as client:
                response = await client.get(url)
        except httpx2.TimeoutException as exc:
            raise TaskApiError(f"JSONPlaceholder не ответил за {self.timeout:g} с.") from exc
        except httpx2.HTTPError as exc:
            raise TaskApiError(f"Ошибка сети при запросе к JSONPlaceholder: {exc}") from exc

        if response.status_code == 404:
            raise TaskNotFound(f"Задача с id={task_id} не найдена.")
        if response.status_code != 200:
            raise TaskApiError(
                f"JSONPlaceholder вернул HTTP {response.status_code} для /todos/{task_id}."
            )

        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise TaskApiError("JSONPlaceholder вернул некорректный JSON.") from exc

        try:
            return Task.model_validate(payload)
        except ValidationError as exc:
            raise TaskApiError("JSONPlaceholder вернул неожиданную структуру задачи.") from exc
