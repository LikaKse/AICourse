"""Класс Agent: вся логика общения с Gemini, вызова инструментов и повторения цикла.

Function Calling здесь ручной. Автоматическое выполнение функций в SDK (AFC) отключено,
и в запрос передаются только FunctionDeclaration, а не Python-функции. Поэтому SDK
ничего не вызывает сам: он возвращает function_call, а дальше работает цикл в Agent.run().

Схема одного запроса пользователя:

    user message ──► history;  count_tokens(сообщение)            ← День 8
    loop (не больше max_iterations раз):
        count_tokens(history) ──► > лимита? ──► context_overflow   ← День 8
        generate_content(history, tools) ──► ответ модели + usage_metadata
        ответ модели целиком ──► history (thought signatures сохраняются)
        есть function_call?  нет ──► вернуть текст (финальный ответ)
                             да  ──► dispatch() каждого вызова в Python
                                     FunctionResponse ──► history ──► следующая итерация
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from google.genai import errors, types

import config
import tools
from token_stats import CallStats, SessionStats, TurnStats, history_shape, usage_from_response

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


# Признаки ошибки «вход больше контекстного окна» в тексте ошибки Gemini API (HTTP 400).
_CONTEXT_OVERFLOW_RE = re.compile(
    r"input token count|exceeds the maximum number of tokens|token limit|context (window|length)|too many tokens",
    re.IGNORECASE,
)


def is_context_overflow_error(exc: BaseException) -> bool:
    """True, если ошибка API означает превышение лимита входных токенов."""
    if not isinstance(exc, errors.APIError) or exc.code != 400:
        return False
    return bool(_CONTEXT_OVERFLOW_RE.search(f"{exc.message or ''} {exc}"))


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
    status: str                 # ok | max_iterations | api_error | empty_response | context_overflow
    iterations: int             # сколько раз агент обратился к модели
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0      # candidates + thoughts (thinking оплачивается как output)
    stats: TurnStats = field(default_factory=TurnStats)   # подробная статистика токенов (День 8)

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
        count_tokens: bool = config.COUNT_TOKENS,
        input_token_limit: int | None = None,
        enforce_context_limit: bool = config.ENFORCE_CONTEXT_LIMIT,
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
        # День 8: подсчёт токенов.
        self.count_tokens_enabled = count_tokens
        self.input_token_limit = input_token_limit      # из models.get(); None — неизвестен
        self.enforce_context_limit = enforce_context_limit
        self.session_stats = SessionStats()              # расход за запуск; reset() его не обнуляет

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
        """Очистить историю диалога (накопленный расход токенов сессии остаётся)."""
        self.history.clear()

    # ----- подсчёт токенов -----

    def count_tokens(self, contents: list[types.Content]) -> tuple[int | None, str | None]:
        """Официальный подсчёт Gemini API: (число токенов, None) или (None, текст ошибки).

        В режиме Gemini Developer API SDK 2.34.0 считает только contents (без system instruction
        и схем инструментов — их SDK в этом режиме не принимает). Ошибка подсчёта не прерывает
        работу агента: статистика покажет «н/д».
        """
        if not self.count_tokens_enabled or not contents:
            return None, None
        try:
            response = self.client.models.count_tokens(model=self.model, contents=contents)
        except errors.APIError as exc:
            msg = redact_secrets(f"{exc.code} {exc.status or ''}: {exc.message or exc}".strip())
        except Exception as exc:  # сеть, таймаут, клиент без count_tokens
            msg = redact_secrets(f"{type(exc).__name__}: {exc}")
        else:
            if response.total_tokens is None:
                return None, "count_tokens вернул пустой total_tokens"
            return int(response.total_tokens), None
        logger.warning("Не удалось посчитать токены: %s", msg)
        return None, msg

    def count_history_tokens(self) -> int | None:
        """Токены всей текущей истории (например, после восстановления из файла)."""
        return self.count_tokens(self.history)[0]

    # ----- главный цикл -----

    def run(self, user_message: str) -> AgentResult:
        """Обработать один запрос пользователя: цикл «модель → инструменты → модель» до текстового ответа."""
        # Точка отката: если запрос не удастся довести до финального ответа, история вернётся
        # в состояние до него. Так в истории не останется «оборванных» function_call/function_response,
        # и следующий запрос не сломается.
        checkpoint = len(self.history)
        user_content = types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
        self.history.append(user_content)
        logger.info("Запрос пользователя: %s", _short(user_message))

        gen_config = self.build_config()
        calls_log: list[ToolCallRecord] = []
        stats = TurnStats(input_token_limit=self.input_token_limit)
        stats.request_tokens, err = self.count_tokens([user_content])
        if err:
            stats.count_errors.append(f"запрос: {err}")

        def finish(text: str, status: str, iterations: int) -> AgentResult:
            stats.status = status
            if status == "ok" and calls_log:
                # Сколько токенов в истории занимают вызовы инструментов и их результаты этого хода.
                exchange = [c for c in self.history[checkpoint + 1:-1]]
                stats.tool_exchange_tokens, tool_err = self.count_tokens(exchange)
                if tool_err:
                    stats.count_errors.append(f"инструменты: {tool_err}")
            if status != "ok":
                del self.history[checkpoint:]
            self.session_stats.add(stats)   # токены потрачены, даже если ход не удался
            logger.info("Токены за ход: вход %d, выход %d, обращений %d",
                        stats.input_tokens, stats.output_tokens, stats.model_calls)
            return AgentResult(text, status, iterations, calls_log, stats.input_tokens, stats.output_tokens, stats)

        for iteration in range(1, self.max_iterations + 1):
            # 0. День 8: считаем всю историю (вместе с function_call/function_response) до отправки.
            history_tokens, err = self.count_tokens(self.history)
            if err:
                stats.count_errors.append(f"история (обращение {iteration}): {err}")
            call = CallStats(iteration=iteration, history_tokens=history_tokens, shape=history_shape(self.history))
            logger.info("Итерация %d: история %s ток., %d сообщ.", iteration,
                        "н/д" if history_tokens is None else history_tokens, len(self.history))
            if (self.enforce_context_limit and self.input_token_limit and history_tokens is not None
                    and history_tokens > self.input_token_limit):
                stats.blocked_locally = True
                logger.warning("История %d ток. больше лимита контекста %d — запрос не отправлен",
                               history_tokens, self.input_token_limit)
                return finish(
                    f"Запрос не отправлен: история занимает {history_tokens} токенов, а лимит контекста "
                    f"модели {self.input_token_limit}. Сократите сообщение или очистите историю (reset).",
                    "context_overflow", iteration - 1,
                )

            # 1–2. Отправляем всю историю и схемы инструментов.
            logger.info("Итерация %d/%d: запрос к модели %s", iteration, self.max_iterations, self.model)
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=self.history, config=gen_config
                )
            except errors.APIError as exc:
                msg = redact_secrets(f"{exc.code} {exc.status or ''}: {exc.message or exc}".strip())
                logger.error("Ошибка Gemini API: %s", msg)
                call.error = msg
                stats.calls.append(call)
                if is_context_overflow_error(exc):
                    return finish(f"Превышен лимит контекста модели: Gemini API отклонил запрос ({msg}). "
                                  "Сократите сообщение или очистите историю (reset).", "context_overflow", iteration)
                return finish(f"Ошибка Gemini API ({msg}). Попробуйте ещё раз позже.", "api_error", iteration)
            except Exception as exc:  # сеть, таймаут и т. п.
                msg = redact_secrets(f"{type(exc).__name__}: {exc}")
                logger.error("Сбой запроса к Gemini: %s", msg)
                call.error = msg
                stats.calls.append(call)
                return finish(f"Не удалось связаться с Gemini ({msg}).", "api_error", iteration)

            call.usage = usage_from_response(response)
            stats.calls.append(call)

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
            call.function_calls_in_response = len(function_calls)
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
