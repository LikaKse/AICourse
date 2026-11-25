"""Скриптованная модель для бесплатной демонстрации полного цикла.

Не обращается к Gemini: get_task(task_id=5) → MCP-сервер → JSONPlaceholder →
FunctionResponse → финальный текст.
"""

from __future__ import annotations

from typing import Any

from google.genai import types

import config


class ScriptedGeminiClient:
    """Минимальный async-клиент с интерфейсом client.aio.models.generate_content."""

    def __init__(self, task_id: int = config.TASK_ID) -> None:
        self.task_id = task_id
        self.step = 0
        self.aio = self
        self.models = self
        self.requests: list[dict] = []

    async def generate_content(self, *, model: str, contents, config: types.GenerateContentConfig):
        self.requests.append({"model": model, "contents": contents, "config": config})
        if self.step == 0:
            self.step += 1
            return _call_response(
                "get_task",
                {"task_id": self.task_id},
                call_id="scripted-task",
                signature=b"scripted-signature-task",
            )

        self.step += 1
        task = _last_task(contents)
        title = task.get("title", "без названия")
        completed = task.get("completed")
        status = "выполнена" if completed else "не выполнена"
        return _text_response(
            f"Задача {task.get('id', self.task_id)}: «{title}». Статус: {status}."
        )


def _last_task(contents) -> dict[str, Any]:
    for content in reversed(contents):
        for part in reversed(getattr(content, "parts", []) or []):
            response = getattr(part, "function_response", None)
            if not response:
                continue
            payload = response.response or {}
            if not isinstance(payload, dict):
                continue
            result = payload.get("result")
            if isinstance(result, dict) and "title" in result:
                return result
            structured = payload.get("structured")
            if isinstance(structured, dict) and "title" in structured:
                return structured
    raise AssertionError("В истории нет FunctionResponse с данными задачи.")


def _wrap(parts) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(
            content=types.Content(role="model", parts=parts),
            finish_reason="STOP",
        )],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100,
            candidates_token_count=20,
            thoughts_token_count=5,
        ),
    )


def _call_response(name: str, args: dict, *, call_id: str, signature: bytes):
    return _wrap([
        types.Part(
            function_call=types.FunctionCall(id=call_id, name=name, args=args),
            thought_signature=signature,
        )
    ])


def _text_response(text: str):
    return _wrap([types.Part(text=text, thought_signature=b"scripted-signature-final")])
