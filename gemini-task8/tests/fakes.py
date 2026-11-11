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


def _wrap(parts, *, prompt=100, candidates=20, thoughts=5, cached=None):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts), finish_reason="STOP")],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt, candidates_token_count=candidates, thoughts_token_count=thoughts,
            cached_content_token_count=cached,
            total_token_count=prompt + candidates + thoughts,
        ),
    )


def with_usage(response, *, prompt=None, candidates=None, thoughts=None, cached=None):
    """Подменить usage_metadata у готового фейкового ответа."""
    meta = response.usage_metadata
    for name, value in (("prompt_token_count", prompt), ("candidates_token_count", candidates),
                        ("thoughts_token_count", thoughts), ("cached_content_token_count", cached)):
        if value is not None:
            setattr(meta, name, value)
    return response


def no_usage(response):
    response.usage_metadata = None
    return response


def fake_token_count(contents) -> int:
    """Детерминированный «подсчёт» ТОЛЬКО для офлайн-тестов (настоящий — client.models.count_tokens).

    Каждая часть стоит 3 токена + по токену на слово текста / на поле аргументов и ответа функции.
    Нужен, чтобы тесты проверяли логику агента: что считается, когда и с какими частями истории.
    """
    total = 0
    for content in contents:
        for part in content.parts or []:
            total += 3
            if part.text:
                total += len(part.text.split())
            if part.function_call:
                total += 5 + len(part.function_call.args or {})
            if part.function_response:
                total += 5 + len(str(part.function_response.response).split())
    return total


class _FakeModels:
    def __init__(self, script, *, count_fn=fake_token_count, count_error=None, input_token_limit=None):
        self.script = list(script)
        self.requests: list[dict] = []
        self.count_requests: list[list] = []
        self.count_fn = count_fn
        self.count_error = count_error          # исключение, которое бросит count_tokens
        self.input_token_limit = input_token_limit

    def count_tokens(self, *, model, contents, config=None):
        self.count_requests.append(copy.deepcopy(contents))
        if self.count_error is not None:
            raise self.count_error
        return types.CountTokensResponse(total_tokens=self.count_fn(contents))

    def get(self, *, model, config=None):
        if self.input_token_limit is None:
            raise api_error(404, f"models/{model} is not found", "NOT_FOUND")
        return types.Model(name=f"models/{model}", input_token_limit=self.input_token_limit, output_token_limit=65536)

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
    def __init__(self, script, **kwargs):
        self.models = _FakeModels(script, **kwargs)

    @property
    def requests(self):
        return self.models.requests
