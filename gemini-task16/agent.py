"""Агент Дня 16: async-цикл Function Calling на основе Дня 6.

Логика та же, что в Дне 6:
    user → Gemini → function_call → local/MCP dispatch → FunctionResponse → Gemini → ...
Отличие только в async: MCP ClientSession и google-genai AsyncClient.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from google.genai import errors, types

import config

logger = logging.getLogger("agent")


def redact_secrets(text: str) -> str:
    """Заменяет значение API-ключа на *** (если ключ попал в текст ошибки)."""
    key = os.environ.get(config.API_KEY_ENV)
    if key and len(key) >= 8 and key in text:
        text = text.replace(key, "***")
    return text


def _short(value: Any, limit: int = config.LOG_PREVIEW_CHARS) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= limit else text[:limit] + f"… (+{len(text) - limit} симв.)"


@dataclass
class ToolCallRecord:
    iteration: int
    name: str
    args: dict
    response: dict
    source: str = "local"

    @property
    def ok(self) -> bool:
        return "error" not in self.response


@dataclass
class AgentResult:
    text: str
    status: str                     # ok | max_iterations | api_error | empty_response
    iterations: int
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class Agent:
    """Агент с ручным циклом Function Calling, локальными и MCP-инструментами."""

    def __init__(
        self,
        client: Any,
        model: str = config.MODEL,
        *,
        declarations: list[types.FunctionDeclaration] | None = None,
        dispatcher: Callable[[str, dict], Any] | None = None,
        system_instruction: str = config.SYSTEM_INSTRUCTION,
        max_iterations: int = config.MAX_ITERATIONS,
        thinking_level: str | None = config.THINKING_LEVEL,
        max_output_tokens: int | None = config.MAX_OUTPUT_TOKENS,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations должен быть ≥ 1")
        self.client = client
        self.model = model
        self.declarations = declarations or []
        self.dispatcher = dispatcher
        self.system_instruction = system_instruction
        self.max_iterations = max_iterations
        self.thinking_level = thinking_level
        self.max_output_tokens = max_output_tokens
        # Ответы модели хранятся как есть: thought_signature нельзя пересобирать.
        self.history: list[types.Content] = []

    def build_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            tools=[types.Tool(function_declarations=self.declarations)] if self.declarations else [],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.AUTO)
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=(
                types.ThinkingConfig(thinking_level=self.thinking_level) if self.thinking_level else None
            ),
            max_output_tokens=self.max_output_tokens,
        )

    def reset(self) -> None:
        self.history.clear()

    def request_preview(self, user_message: str | None = None) -> dict:
        """Тело запроса к модели со схемами инструментов, без вызова API."""
        contents = list(self.history)
        if user_message is not None:
            contents = contents + [types.Content(role="user", parts=[types.Part.from_text(text=user_message)])]
        return {
            "model": self.model,
            "contents": [
                item.model_dump(mode="json", by_alias=True, exclude_none=True)
                if hasattr(item, "model_dump")
                else item
                for item in contents
            ],
            "config": self.build_config().model_dump(mode="json", by_alias=True, exclude_none=True),
        }

    async def arun(self, user_message: str) -> AgentResult:
        """Обработать один запрос: цикл «модель → инструменты → модель»."""
        if self.dispatcher is None:
            raise ValueError("Не задан dispatcher инструментов.")
        checkpoint = len(self.history)
        self.history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))
        logger.info("Запрос пользователя: %s", _short(user_message))

        gen_config = self.build_config()
        calls_log: list[ToolCallRecord] = []
        in_tokens = out_tokens = 0

        def finish(text: str, status: str, iterations: int) -> AgentResult:
            if status != "ok":
                del self.history[checkpoint:]
            return AgentResult(text, status, iterations, calls_log, in_tokens, out_tokens)

        for iteration in range(1, self.max_iterations + 1):
            logger.info("Итерация %d/%d: запрос к модели %s", iteration, self.max_iterations, self.model)
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=self.history,
                    config=gen_config,
                )
            except errors.APIError as exc:
                msg = redact_secrets(f"{exc.code} {exc.status or ''}: {exc.message or exc}".strip())
                logger.error("Ошибка Gemini API: %s", msg)
                return finish(f"Ошибка Gemini API ({msg}). Попробуйте ещё раз позже.", "api_error", iteration)
            except Exception as exc:
                msg = redact_secrets(f"{type(exc).__name__}: {exc}")
                logger.error("Сбой запроса к Gemini: %s", msg)
                return finish(f"Не удалось связаться с Gemini ({msg}).", "api_error", iteration)

            usage = getattr(response, "usage_metadata", None)
            if usage:
                in_tokens += usage.prompt_token_count or 0
                out_tokens += (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)

            candidate = response.candidates[0] if response.candidates else None
            content = candidate.content if candidate else None
            if not content or not content.parts:
                reason = candidate.finish_reason if candidate else getattr(response, "prompt_feedback", None)
                logger.warning("Модель вернула пустой ответ (finish_reason=%s)", reason)
                return finish(f"Модель не вернула ответ (причина: {reason}).", "empty_response", iteration)

            if content.role is None:
                content.role = "model"
            self.history.append(content)
            signed = sum(1 for part in content.parts if part.thought_signature)
            if signed:
                logger.debug("В ответе %d част(ей) с thought_signature — сохранены в истории", signed)

            function_calls = [part.function_call for part in content.parts if part.function_call]
            if not function_calls:
                text = self._extract_text(content)
                logger.info("Финальный ответ после %d итерац.; вызовов: %d", iteration, len(calls_log))
                return finish(text, "ok", iteration)

            response_parts = []
            for fc in function_calls:
                args = dict(fc.args or {})
                logger.info("Вызов инструмента: %s(%s)", fc.name, _short(args))
                try:
                    raw_result = self.dispatcher(fc.name, args)
                    if inspect.isawaitable(raw_result):
                        raw_result = await raw_result
                except Exception as exc:
                    raw_result = {"error": f"Ошибка диспетчера {fc.name}: {type(exc).__name__}: {exc}"}
                payload, source = self._normalize_dispatch(raw_result)
                if "error" in payload:
                    logger.warning("Ошибка инструмента %s (%s): %s", fc.name, source, payload["error"])
                else:
                    logger.info("Результат %s (%s): %s", fc.name, source, _short(payload.get("result", payload)))
                calls_log.append(ToolCallRecord(iteration, fc.name, args, payload, source))
                response_parts.append(
                    types.Part(function_response=types.FunctionResponse(
                        id=fc.id,
                        name=fc.name,
                        response=payload,
                    ))
                )
            self.history.append(types.Content(role="user", parts=response_parts))

        logger.warning("Достигнут лимит итераций (%d) без финального ответа", self.max_iterations)
        return finish(
            f"Не удалось получить финальный ответ за {self.max_iterations} обращений к модели. "
            "Попробуйте переформулировать запрос.",
            "max_iterations",
            self.max_iterations,
        )

    @staticmethod
    def _normalize_dispatch(raw_result: Any) -> tuple[dict, str]:
        """Поддерживает DispatchResult реестра и обычный dict для тестов."""
        payload = getattr(raw_result, "payload", raw_result)
        source = getattr(raw_result, "source", "local")
        if not isinstance(payload, dict):
            payload = {"result": payload}
        return payload, str(source)

    @staticmethod
    def _extract_text(content: types.Content) -> str:
        texts = [part.text for part in content.parts if part.text and not part.thought]
        return "\n".join(texts).strip() or "(модель вернула пустой текст)"
