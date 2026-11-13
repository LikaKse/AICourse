"""Фейковый клиент Gemini для офлайн-тестов (API-лимиты не расходуются).

Возвращает настоящие объекты types.GenerateContentResponse и запоминает все запросы.
Ответы задаются либо списком (script), либо функцией responder(contents, config) → ответ/исключение.
"""

from __future__ import annotations

import copy

from google.genai import errors, types

SYSTEM_TOKENS = 7    # «стоимость» system instruction в фейковом usage_metadata


def text_response(text: str, *, prompt: int = 100, candidates: int | None = None, thoughts: int = 0, usage=True):
    resp = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]),
                                    finish_reason="STOP")],
    )
    if usage:
        cand = len(text.split()) if candidates is None else candidates
        resp.usage_metadata = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt, candidates_token_count=cand, thoughts_token_count=thoughts,
            total_token_count=prompt + cand + thoughts)
    return resp


def empty_response(finish_reason: str = "SAFETY"):
    return types.GenerateContentResponse(candidates=[types.Candidate(content=None, finish_reason=finish_reason)])


def api_error(code: int = 503, message: str = "The model is overloaded.", status: str = "UNAVAILABLE"):
    return errors.APIError(code, {"error": {"code": code, "message": message, "status": status}})


def fake_token_count(contents) -> int:
    """Детерминированный подсчёт ТОЛЬКО для тестов: 3 токена на часть + по токену на слово."""
    return sum(3 + len((p.text or "").split()) for c in contents for p in c.parts or [])


def all_text(contents) -> str:
    return "\n".join(p.text or "" for c in contents for p in c.parts or [])


def is_summary_request(config) -> bool:
    return "сжимаешь историю" in str(getattr(config, "system_instruction", ""))


class _FakeModels:
    def __init__(self, script=None, responder=None, count_error=None):
        self.script = list(script or [])
        self.responder = responder
        self.count_error = count_error
        self.requests: list[dict] = []
        self.count_requests: list[list] = []

    def count_tokens(self, *, model, contents, config=None):
        self.count_requests.append(copy.deepcopy(contents))
        if self.count_error is not None:
            raise self.count_error
        return types.CountTokensResponse(total_tokens=fake_token_count(contents))

    def generate_content(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": copy.deepcopy(contents), "config": config})
        if self.responder is not None:
            item = self.responder(contents, config)
        elif self.script:
            item = self.script.pop(0)
        else:
            raise AssertionError("Сделано больше запросов, чем предусмотрено сценарием теста")
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, script=None, responder=None, **kwargs):
        self.models = _FakeModels(script, responder, **kwargs)

    @property
    def requests(self):
        return self.models.requests

    def summary_requests(self):
        return [r for r in self.requests if is_summary_request(r["config"])]

    def answer_requests(self):
        return [r for r in self.requests if not is_summary_request(r["config"])]


def lossless_summary(request_text: str) -> str:
    """Фейковое summary без потерь фактов: строки предыдущего summary («- …») + реплики пользователя."""
    lines = [l for l in request_text.splitlines() if l.startswith("- ")]
    lines += ["- " + l.split(": ", 1)[1] for l in request_text.splitlines() if "] Пользователь: " in l]
    return "\n".join(lines)


def echo_responder(summary_fn=None):
    """«Модель» для сквозных тестов экспериментов.

    * Запрос на summary → summary_fn(текст запроса) (по умолчанию lossless_summary: все факты
      сценария есть в репликах пользователя, поэтому сжатие без потерь).
    * Обычный вопрос → «ответ», содержащий весь текст контекста. Так проверка фактов зависит
      только от того, что реально попало в контекст запроса.
    """
    def respond(contents, config):
        text = all_text(contents)
        prompt = fake_token_count(contents) + SYSTEM_TOKENS
        if is_summary_request(config):
            summary = summary_fn(text) if summary_fn else lossless_summary(text)
            return text_response(summary, prompt=prompt, thoughts=2)
        context_only = all_text(contents[:-1])
        return text_response("Ответ по контексту: " + context_only, prompt=prompt, candidates=5)
    return respond
