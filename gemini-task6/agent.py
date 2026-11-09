"""Класс Agent: вся логика общения с Gemini, вызова инструментов и повторения цикла.

Function Calling здесь ручной. Автоматическое выполнение функций в SDK (AFC) отключено,
и в запрос передаются только FunctionDeclaration, а не Python-функции. Поэтому SDK
ничего не вызывает сам: он возвращает function_call, а дальше работает цикл в Agent.run().

Схема одного запроса пользователя:

    user message ──► history
    loop (не больше max_iterations раз):
        generate_content(history, tools) ──► ответ модели
        ответ модели целиком ──► history (thought signatures сохраняются)
        есть function_call?  нет ──► вернуть текст (финальный ответ)
                             да  ──► dispatch() каждого вызова в Python
                                     FunctionResponse ──► history ──► следующая итерация
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from google.genai import errors, types

import config
import tools

logger = logging.getLogger("agent")


# ---------------------------------------------------------------------
# Защита от утечки ключа
# ---------------------------------------------------------------------

def redact_secrets(text: str) -> str:
    """Заменяет значение API-ключа на *** (на случай, если ключ попадёт в текст ошибки)."""
    key = os.environ.get(config.API_KEY_ENV)
    if key and len(key) >= 8 and key in text:
        text = text.replace(key, "***")
    return text


def _short(value: Any, limit: int = config.LOG_PREVIEW_CHARS) -> str:
    """JSON для лога с обрезкой длинных значений."""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= limit else text[:limit] + f"… (+{len(text) - limit} симв.)"


# ---------------------------------------------------------------------
# Результат работы агента
# ---------------------------------------------------------------------

@dataclass
class ToolCallRecord:
    iteration: int
    name: str
    args: dict
    response: dict          # то, что ушло модели: {"result": ...} или {"error": ...}

    @property
    def ok(self) -> bool:
        return "error" not in self.response


@dataclass
class AgentResult:
    text: str
    status: str                 # ok | max_iterations | api_error | empty_response
    iterations: int             # сколько раз агент обратился к модели
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0      # candidates + thoughts (thinking оплачивается как output)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


# ---------------------------------------------------------------------
# Агент
# ---------------------------------------------------------------------

class Agent:
    """Агент с ручным циклом Function Calling и историей диалога."""

    def __init__(
        self,
        client: Any,
        model: str = config.MODEL,
        *,
        declarations: list[types.FunctionDeclaration] | None = None,
        dispatcher: Callable[[str, dict], dict] = tools.dispatch,
        system_instruction: str = config.SYSTEM_INSTRUCTION,
        max_iterations: int = config.MAX_ITERATIONS,
        thinking_level: str | None = config.THINKING_LEVEL,
        max_output_tokens: int | None = config.MAX_OUTPUT_TOKENS,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations должен быть ≥ 1")
        self.client = client
        self.model = model
        self.declarations = declarations if declarations is not None else tools.FUNCTION_DECLARATIONS
        self.dispatcher = dispatcher
        self.system_instruction = system_instruction
        self.max_iterations = max_iterations
        self.thinking_level = thinking_level
        self.max_output_tokens = max_output_tokens
        # История диалога в формате Gemini. Ответы модели хранятся без изменений:
        # в их частях лежат thought_signature, без которых Gemini 3 не продолжит Function Calling.
        self.history: list[types.Content] = []

    # ----- конфигурация запроса -----

    def build_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            tools=[types.Tool(function_declarations=self.declarations)],
            # AUTO: модель сама решает, вызывать ли инструмент или сразу отвечать текстом.
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.AUTO)
            ),
            # Явно выключаем автоматическое выполнение функций SDK: цикл ниже — наш собственный.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=(
                types.ThinkingConfig(thinking_level=self.thinking_level) if self.thinking_level else None
            ),
            max_output_tokens=self.max_output_tokens,
        )

    def reset(self) -> None:
        """Очистить историю диалога."""
        self.history.clear()

    # ----- главный цикл -----

    def run(self, user_message: str) -> AgentResult:
        """Обработать один запрос пользователя: цикл «модель → инструменты → модель» до текстового ответа."""
        # Точка отката: если запрос не удастся довести до финального ответа, история вернётся
        # в состояние до него. Так в истории не останется «оборванных» function_call/function_response,
        # и следующий запрос не сломается.
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
            # 1–2. Отправляем всю историю и схемы инструментов.
            logger.info("Итерация %d/%d: запрос к модели %s", iteration, self.max_iterations, self.model)
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=self.history, config=gen_config
                )
            except errors.APIError as exc:
                msg = redact_secrets(f"{exc.code} {exc.status or ''}: {exc.message or exc}".strip())
                logger.error("Ошибка Gemini API: %s", msg)
                return finish(f"Ошибка Gemini API ({msg}). Попробуйте ещё раз позже.", "api_error", iteration)
            except Exception as exc:  # сеть, таймаут и т. п.
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

            # Ответ модели кладём в историю как есть, вместе с thought_signature.
            if content.role is None:
                content.role = "model"
            self.history.append(content)
            signed = sum(1 for p in content.parts if p.thought_signature)
            if signed:
                logger.debug("В ответе %d част(ей) с thought_signature — сохранены в истории", signed)

            # 3. Есть ли вызовы функций?
            function_calls = [p.function_call for p in content.parts if p.function_call]
            if not function_calls:
                text = self._extract_text(content)
                logger.info("Финальный ответ после %d итерац.; вызовов инструментов: %d", iteration, len(calls_log))
                return finish(text, "ok", iteration)

            # 4–5. Python выполняет функции, результаты уходят модели как FunctionResponse.
            response_parts = []
            for fc in function_calls:
                args = dict(fc.args or {})
                logger.info("Вызов инструмента: %s(%s)", fc.name, _short(args))
                result = self.dispatcher(fc.name, args)
                if "error" in result:
                    logger.warning("Ошибка инструмента %s: %s", fc.name, result["error"])
                else:
                    logger.info("Результат %s: %s", fc.name, _short(result["result"]))
                calls_log.append(ToolCallRecord(iteration, fc.name, args, result))
                response_parts.append(
                    types.Part(function_response=types.FunctionResponse(
                        id=fc.id,          # связывает ответ с конкретным вызовом (важно при параллельных вызовах)
                        name=fc.name,
                        response=result,
                    ))
                )
            # Все ответы на вызовы одного хода — одним сообщением, в том же порядке.
            self.history.append(types.Content(role="user", parts=response_parts))
            # 6–7. Следующая итерация снова проверит ответ модели на function_call.

        # 8. Защита от бесконечного цикла.
        logger.warning("Достигнут лимит итераций (%d) без финального ответа", self.max_iterations)
        return finish(
            f"Не удалось получить финальный ответ за {self.max_iterations} обращений к модели. "
            "Попробуйте переформулировать запрос.",
            "max_iterations",
            self.max_iterations,
        )

    @staticmethod
    def _extract_text(content: types.Content) -> str:
        """Текст ответа без «мыслей» модели (thought=True)."""
        texts = [p.text for p in content.parts if p.text and not p.thought]
        return "\n".join(texts).strip() or "(модель вернула пустой текст)"
