"""Фейковый клиент Gemini для офлайн-тестов и --offline-experiment (API не вызывается).

Возвращает настоящие объекты types.GenerateContentResponse и запоминает все запросы.
Ответы задаются списком (script) или функцией responder(contents, config) → ответ/исключение.

* ``scripted_extractor`` — «менеджер памяти»: операции для каждого сообщения берутся из
  data/offline_extraction.json (там есть и намеренно неверные операции — чтобы проверить отказы).
  Применяет их всё равно код с проверками, как при реальном вызове.
* ``echo_responder`` — «ответ» содержит весь контекст запроса (без вопроса). Тогда проверка ответа
  по эталону зависит ТОЛЬКО от того, что реально попало в промпт.
"""

from __future__ import annotations

import copy
import json
import re

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


def is_memory_request(config) -> bool:
    return "менеджер памяти" in str(getattr(config, "system_instruction", ""))


def ops_json(*ops) -> str:
    """Ответ «модели» менеджера памяти: операции с заполненными по умолчанию полями."""
    full = [{"layer": "long_term", "op": "set", "key": "", "value": "", "kind": "user", "category": "other",
             "evidence": "", "proposal_evidence": "", "reason": "", **o} for o in ops]
    return json.dumps({"operations": full}, ensure_ascii=False)


def request_user_message(request_text: str) -> tuple[int, str]:
    """(номер, текст) нового сообщения пользователя из запроса менеджеру памяти."""
    m = re.search(r"Новое сообщение пользователя:\n\[(\d+)\] Пользователь: (.*?)\n\nВерни", request_text, re.S)
    if not m:
        raise AssertionError("в запросе нет блока «Новое сообщение пользователя»")
    return int(m.group(1)), m.group(2)


def scripted_extractor(by_message: dict[int, list[dict]]):
    """Операции «модели» по номеру сообщения пользователя; нет записи — пустой список."""
    def extract(request_text: str) -> str:
        n, _ = request_user_message(request_text)
        return ops_json(*by_message.get(n, []))
    return extract


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

    def memory_requests(self):
        return [r for r in self.requests if is_memory_request(r["config"])]

    def answer_requests(self):
        return [r for r in self.requests if not is_memory_request(r["config"])]


def echo_responder(extract_fn=None):
    """«Модель» для сквозных офлайн-прогонов.

    * Запрос менеджеру памяти → extract_fn(текст запроса) (по умолчанию — пустой список операций).
    * Обычный вопрос → «ответ», содержащий весь текст контекста без вопроса.
    """
    def respond(contents, config):
        text = all_text(contents)
        prompt = fake_token_count(contents) + SYSTEM_TOKENS
        if is_memory_request(config):
            out = extract_fn(text) if extract_fn else ops_json()
            return text_response(out, prompt=prompt, candidates=11, thoughts=3)
        return text_response("Ответ по контексту: " + all_text(contents[:-1]), prompt=prompt, candidates=5)
    return respond
