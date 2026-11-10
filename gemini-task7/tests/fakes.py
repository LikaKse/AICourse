"""Фейковый клиент Gemini для офлайн-тестов агента.

Возвращает заранее заданные объекты types.GenerateContentResponse (того же типа, что и
настоящий SDK) и запоминает, что агент отправил в каждом запросе.
Это проверяет ЛОГИКУ ЦИКЛА агента. Выбор инструментов реальной моделью проверяют только
сценарии main.py --scenarios.
"""

from __future__ import annotations

import copy

from google.genai import errors, types


def text_response(text: str, *, thought: str | None = None, signature: bytes | None = None):
    parts = []
    if thought:
        parts.append(types.Part(text=thought, thought=True))
    parts.append(types.Part(text=text, thought_signature=signature))
    return _wrap(parts)


def call_response(*calls: tuple, signature: bytes | None = b"sig", text: str | None = None):
    """calls: (name, args) или (name, args, id). Сигнатура — на первом function_call, как у Gemini 3."""
    parts = []
    if text:
        parts.append(types.Part(text=text))
    for i, call in enumerate(calls):
        name, args, *rest = call
        call_id = rest[0] if rest else f"call-{name}-{i}"
        parts.append(types.Part(
            function_call=types.FunctionCall(id=call_id, name=name, args=args),
            thought_signature=signature if i == 0 else None,
        ))
    return _wrap(parts)


def empty_response(finish_reason: str = "SAFETY"):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=None, finish_reason=finish_reason)]
    )


def api_error(code: int = 503, message: str = "The model is overloaded.", status: str = "UNAVAILABLE"):
    return errors.APIError(code, {"error": {"code": code, "message": message, "status": status}})


def _wrap(parts):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts), finish_reason="STOP")],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100, candidates_token_count=20, thoughts_token_count=5
        ),
    )


class _FakeModels:
    def __init__(self, script):
        self.script = list(script)
        self.requests: list[dict] = []

    def generate_content(self, *, model, contents, config):
        # deepcopy: агент дальше мутирует историю, а нам нужен снимок на момент запроса
        self.requests.append({"model": model, "contents": copy.deepcopy(contents), "config": config})
        if not self.script:
            raise AssertionError("Агент сделал больше запросов, чем предусмотрено сценарием")
        item = self.script.pop(0)
        if callable(item) and not isinstance(item, types.GenerateContentResponse):
            item = item()
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, script):
        self.models = _FakeModels(script)

    @property
    def requests(self):
        return self.models.requests
