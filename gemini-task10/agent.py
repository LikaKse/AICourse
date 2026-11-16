"""Класс Agent: запросы к Gemini через выбранную стратегию контекста (День 10).

Схема одного вопроса пользователя (Agent.ask):

    strategy.prepare()                     ← summary: отложенное сжатие (как в Дне 9)
    remember=True → strategy.before_user_message(вопрос)   ← facts: обновление памяти (вызов Gemini)
    contents = strategy.build_contents(вопрос)
    count_tokens(contents)                 ← оценка до отправки (бесплатно)
    generate_content(contents)             ← usage_metadata: фактические токены
    remember=True  → strategy.add_exchange(вопрос, ответ)  ← summary: сжатие
    remember=False → ни история, ни память не меняются (контрольные вопросы независимы)

Служебные вызовы (summary, facts) возвращаются в AgentResult.service_calls с их токенами.

Создание summary (Agent.summarize) — отдельный вызов Gemini со своей системной инструкцией:
предыдущее summary + новые сообщения → обновлённое summary. Его токены тоже учитываются.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from google.genai import errors, types

import config
from context_manager import (
    CompressionEvent, ContextManager, SummarizationError, SummaryOutput, content_text, text_content,
)
from facts_store import OPERATIONS_SCHEMA, FactsFormatError, parse_operations
from strategies import CallRecord, ContextStrategy, SummaryStrategy, make_strategy
from strategies.sticky_facts import FactsExtractionError
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
    compression: list[CompressionEvent] = field(default_factory=list)   # события сжатия (summary)
    error: str | None = None
    service_calls: list[CallRecord] = field(default_factory=list)       # все служебные вызовы Gemini

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def render_messages(messages: list[types.Content], start_index: int = 0) -> str:
    lines = []
    for i, content in enumerate(messages, start=start_index + 1):
        lines.append(f"[{i}] {ROLE_NAMES.get(content.role, content.role)}: {content_text(content)}")
    return "\n".join(lines)


def build_facts_request(facts_view: dict, assistant_text: str | None, user_text: str, message_no: int) -> str:
    """Запрос на обновление Sticky Facts. Сюда попадают ТОЛЬКО факты и два последних сообщения."""
    facts_json = json.dumps(facts_view, ensure_ascii=False, indent=1) if facts_view else "{}"
    prev = (f"[{message_no - 1}] Ассистент: {assistant_text}" if assistant_text else "(нет)")
    return (
        f"Текущие факты (JSON):\n{facts_json}\n\n"
        "Предыдущее сообщение ассистента (только контекст: его предложения — НЕ решения пользователя, "
        f"пока пользователь явно их не принял):\n{prev}\n\n"
        f"Новое сообщение пользователя:\n[{message_no}] Пользователь: {user_text}\n\n"
        "Верни операции изменения памяти по правилам (пустой список, если нового нет)."
    )


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
        context: ContextManager | ContextStrategy | None = None,
        *,
        model: str = config.MODEL,
        system_instruction: str | None = None,
        summary_system_instruction: str = config.SUMMARY_SYSTEM_INSTRUCTION,
        facts_system_instruction: str = config.FACTS_SYSTEM_INSTRUCTION,
        answer_thinking_level: str | None = config.ANSWER_THINKING_LEVEL,
        summary_thinking_level: str | None = config.SUMMARY_THINKING_LEVEL,
        facts_thinking_level: str | None = config.FACTS_THINKING_LEVEL,
        max_output_tokens: int = config.MAX_OUTPUT_TOKENS,
        summary_max_output_tokens: int = config.SUMMARY_MAX_OUTPUT_TOKENS,
        facts_max_output_tokens: int = config.FACTS_MAX_OUTPUT_TOKENS,
        count_tokens: bool = config.COUNT_TOKENS,
    ) -> None:
        """context — стратегия или ContextManager Дня 9 (оборачивается в SummaryStrategy).

        Без context — стратегия summary в памяти, как Agent(client) в Дне 9.
        system_instruction=None — инструкция, которую предлагает стратегия.
        """
        self.client = client
        self.model = model
        self.summary_system_instruction = summary_system_instruction
        self.facts_system_instruction = facts_system_instruction
        self.answer_thinking_level = answer_thinking_level
        self.summary_thinking_level = summary_thinking_level
        self.facts_thinking_level = facts_thinking_level
        self.max_output_tokens = max_output_tokens
        self.summary_max_output_tokens = summary_max_output_tokens
        self.facts_max_output_tokens = facts_max_output_tokens
        self.counter = TokenCounter(client, model, enabled=count_tokens)
        if context is None:
            context = make_strategy("summary")
        elif isinstance(context, ContextManager):
            context = SummaryStrategy(context=context)
        self.strategy: ContextStrategy = context
        self._system_instruction = system_instruction
        self.strategy.bind(self)

    @property
    def system_instruction(self) -> str:
        return self._system_instruction or self.strategy.system_instruction

    @property
    def context(self) -> ContextManager:
        """ContextManager Дня 9 (только для стратегии summary)."""
        if not isinstance(self.strategy, SummaryStrategy):
            raise AttributeError(f"У стратегии {self.strategy.name!r} нет ContextManager")
        return self.strategy.context

    def use_strategy(self, strategy: ContextStrategy) -> None:
        """Переключить стратегию (или ветку) у того же агента."""
        self.strategy = strategy
        strategy.bind(self)

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
        start_index = self.context.summary.summarized_upto if isinstance(self.strategy, SummaryStrategy) else 0
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

    # ----- Sticky Facts -----

    def extract_facts(self, facts_view: dict, assistant_text: str | None, user_text: str,
                      message_no: int) -> tuple[list[dict], dict]:
        """Один вызов Gemini: операции обновления facts (JSON). Ошибка → FactsExtractionError(meta)."""
        request = [text_content("user", build_facts_request(facts_view, assistant_text, user_text, message_no))]
        counted, count_err = self.counter.count(request)
        gen_config = self._config(self.facts_system_instruction, self.facts_thinking_level,
                                  self.facts_max_output_tokens)
        gen_config.response_mime_type = "application/json"
        gen_config.response_json_schema = OPERATIONS_SCHEMA
        started = time.monotonic()
        meta: dict = {"request_tokens": counted, "request_count_error": count_err}
        try:
            text, usage, finish_reason = self._generate(request, gen_config)
        except AgentAPIError as exc:
            raise FactsExtractionError(str(exc), meta) from exc
        meta.update({
            "usage": usage.to_dict(),
            "count_vs_usage": compare_count_with_usage(counted, usage),
            "finish_reason": str(finish_reason) if finish_reason is not None else None,
            "seconds": round(time.monotonic() - started, 2),
        })
        if not text:
            raise FactsExtractionError(f"модель вернула пустой ответ (finish_reason={finish_reason})", meta)
        try:
            ops = parse_operations(text)
        except FactsFormatError as exc:
            meta["raw"] = text[:500]
            raise FactsExtractionError(str(exc), meta) from exc
        return ops, meta

    # ----- вопрос пользователя -----

    def ask(self, question: str, *, remember: bool = True) -> AgentResult:
        calls: list[CallRecord] = list(self.strategy.prepare())

        user = text_content("user", question)
        if remember:
            calls.extend(self.strategy.before_user_message(user))
        contents = self.strategy.build_contents(user)
        counted, count_err = self.counter.count(contents)
        gen_config = self._config(self.system_instruction, self.answer_thinking_level, self.max_output_tokens)

        def result(text: str, status: str, usage: Usage | None = None, error: str | None = None) -> AgentResult:
            events = list(getattr(self.strategy, "last_events", []))
            return AgentResult(text, status, usage or Usage(), counted, count_err, len(contents), events, error,
                               list(calls))

        try:
            text, usage, finish_reason = self._generate(contents, gen_config)
        except AgentAPIError as exc:
            logger.error("%s", exc)
            return result(f"{exc}. Попробуйте ещё раз позже.", "api_error", error=str(exc))
        if not text:
            msg = f"Модель не вернула ответ (причина: {finish_reason})."
            return result(msg, "empty_response", usage, error=msg)

        if remember:
            calls.extend(self.strategy.add_exchange(user, text_content("model", text)))
        return result(text, "ok", usage)
