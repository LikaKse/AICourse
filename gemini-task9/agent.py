"""Класс Agent: запросы к Gemini со сжатым контекстом и создание summary.

Схема одного вопроса пользователя (Agent.ask):

    context.maybe_compress()         ← если порог уже достигнут (например, прошлое сжатие упало)
    contents = summary + active + вопрос
    count_tokens(contents)           ← оценка до отправки (бесплатно)
    generate_content(contents)       ← usage_metadata: фактические токены
    remember=True  → пара «вопрос — ответ» в полную историю → maybe_compress()
    remember=False → история не меняется (контрольные вопросы экспериментов независимы)

Создание summary (Agent.summarize) — отдельный вызов Gemini со своей системной инструкцией:
предыдущее summary + новые сообщения → обновлённое summary. Его токены тоже учитываются.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from google.genai import errors, types

import config
from context_manager import (
    CompressionEvent, ContextManager, SummarizationError, SummaryOutput, content_text, text_content,
)
from token_stats import TokenCounter, Usage, compare_count_with_usage, usage_from_response
from utils import redact_secrets

logger = logging.getLogger("agent")

ROLE_NAMES = {"user": "Пользователь", "model": "Ассистент"}


class AgentAPIError(Exception):
    """Ошибка обращения к Gemini (уже с замаскированным ключом)."""


@dataclass
class AgentResult:
    text: str
    status: str                         # ok | api_error | empty_response
    usage: Usage = field(default_factory=Usage)
    request_tokens: int | None = None   # count_tokens(contents) до отправки
    request_count_error: str | None = None
    request_messages: int = 0           # сколько сообщений ушло в запрос
    compression: list[CompressionEvent] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def render_messages(messages: list[types.Content], start_index: int = 0) -> str:
    lines = []
    for i, content in enumerate(messages, start=start_index + 1):
        lines.append(f"[{i}] {ROLE_NAMES.get(content.role, content.role)}: {content_text(content)}")
    return "\n".join(lines)


def build_summary_request(previous_summary: str, messages: list[types.Content], start_index: int = 0) -> str:
    """Текст запроса на создание/обновление summary. Сюда попадают ТОЛЬКО summary и сообщения диалога."""
    return (
        "Предыдущее краткое содержание диалога:\n"
        f"{previous_summary.strip() or '(пока нет)'}\n\n"
        "Новые сообщения диалога, которые нужно добавить в краткое содержание:\n"
        f"{render_messages(messages, start_index)}\n\n"
        "Составь обновлённое краткое содержание всего диалога: объедини предыдущее краткое содержание "
        "с новыми сообщениями и сохрани все конкретные факты (имена, города, даты, сроки, бюджеты, "
        "числа, ограничения, модели, принятые решения)."
    )


class Agent:
    def __init__(
        self,
        client: Any,
        context: ContextManager | None = None,
        *,
        model: str = config.MODEL,
        system_instruction: str = config.SYSTEM_INSTRUCTION,
        summary_system_instruction: str = config.SUMMARY_SYSTEM_INSTRUCTION,
        answer_thinking_level: str | None = config.ANSWER_THINKING_LEVEL,
        summary_thinking_level: str | None = config.SUMMARY_THINKING_LEVEL,
        max_output_tokens: int = config.MAX_OUTPUT_TOKENS,
        summary_max_output_tokens: int = config.SUMMARY_MAX_OUTPUT_TOKENS,
        count_tokens: bool = config.COUNT_TOKENS,
    ) -> None:
        self.client = client
        self.model = model
        self.system_instruction = system_instruction
        self.summary_system_instruction = summary_system_instruction
        self.answer_thinking_level = answer_thinking_level
        self.summary_thinking_level = summary_thinking_level
        self.max_output_tokens = max_output_tokens
        self.summary_max_output_tokens = summary_max_output_tokens
        self.counter = TokenCounter(client, model, enabled=count_tokens)
        self.context = context if context is not None else ContextManager()
        if self.context.summarizer is None:
            self.context.summarizer = self.summarize

    # ----- низкоуровневый вызов -----

    def _config(self, system_instruction: str, thinking: str | None, max_out: int) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            thinking_config=types.ThinkingConfig(thinking_level=thinking) if thinking else None,
            max_output_tokens=max_out,
        )

    def _generate(self, contents: list[types.Content], gen_config: types.GenerateContentConfig) -> tuple[str | None, Usage, Any]:
        """(текст или None, usage, finish_reason). Ошибки API/сети → AgentAPIError с понятным текстом."""
        try:
            response = self.client.models.generate_content(model=self.model, contents=contents, config=gen_config)
        except errors.APIError as exc:
            msg = redact_secrets(f"{exc.code} {exc.status or ''}: {exc.message or exc}".strip())
            hint = {
                400: "некорректный запрос", 401: "неверный API-ключ", 403: "нет доступа (проверьте ключ)",
                404: f"модель {self.model} не найдена", 429: "превышен лимит запросов, подождите",
            }.get(exc.code, "временная ошибка сервера" if (exc.code or 0) >= 500 else "ошибка API")
            raise AgentAPIError(f"Ошибка Gemini API ({hint}): {msg}") from exc
        except Exception as exc:  # сеть, таймаут
            raise AgentAPIError(f"Не удалось связаться с Gemini: {redact_secrets(f'{type(exc).__name__}: {exc}')}") from exc

        usage = usage_from_response(response)
        candidate = response.candidates[0] if response.candidates else None
        content = candidate.content if candidate else None
        finish_reason = candidate.finish_reason if candidate else getattr(response, "prompt_feedback", None)
        text = content_text(content) if content and content.parts else ""
        return (text or None), usage, finish_reason

    # ----- summary -----

    def summarize(self, previous_summary: str, messages: list[types.Content]) -> SummaryOutput:
        """Создать/обновить summary одним вызовом Gemini. Ошибка → SummarizationError (с метаданными)."""
        start_index = self.context.summary.summarized_upto
        request = [text_content("user", build_summary_request(previous_summary, messages, start_index))]
        counted, count_err = self.counter.count(request)
        gen_config = self._config(self.summary_system_instruction, self.summary_thinking_level,
                                  self.summary_max_output_tokens)
        started = time.monotonic()
        meta = {"request_tokens": counted, "request_count_error": count_err,
                "previous_summary_chars": len(previous_summary)}
        try:
            text, usage, finish_reason = self._generate(request, gen_config)
        except AgentAPIError as exc:
            err = SummarizationError(str(exc))
            err.meta = meta
            raise err from exc
        meta.update({
            "usage": usage.to_dict(),
            "count_vs_usage": compare_count_with_usage(counted, usage),
            "finish_reason": str(finish_reason) if finish_reason is not None else None,
            "seconds": round(time.monotonic() - started, 2),
        })
        if not text:
            err = SummarizationError(f"модель вернула пустое summary (finish_reason={finish_reason})")
            err.meta = meta
            raise err
        meta["summary_tokens"], meta["summary_count_error"] = self.counter.count([text_content("user", text)])
        meta["summary_chars"] = len(text)
        return SummaryOutput(text=text, meta=meta)

    # ----- вопрос пользователя -----

    def ask(self, question: str, *, remember: bool = True) -> AgentResult:
        events: list[CompressionEvent] = []
        event = self.context.maybe_compress()
        if event:
            events.append(event)

        user = text_content("user", question)
        contents = self.context.build_contents(user)
        counted, count_err = self.counter.count(contents)
        gen_config = self._config(self.system_instruction, self.answer_thinking_level, self.max_output_tokens)

        def result(text: str, status: str, usage: Usage | None = None, error: str | None = None) -> AgentResult:
            return AgentResult(text, status, usage or Usage(), counted, count_err, len(contents), events, error)

        try:
            text, usage, finish_reason = self._generate(contents, gen_config)
        except AgentAPIError as exc:
            logger.error("%s", exc)
            return result(f"{exc}. Попробуйте ещё раз позже.", "api_error", error=str(exc))
        if not text:
            msg = f"Модель не вернула ответ (причина: {finish_reason})."
            return result(msg, "empty_response", usage, error=msg)

        if remember:
            events.extend(self.context.add_exchange(user, text_content("model", text)))
        return result(text, "ok", usage)
