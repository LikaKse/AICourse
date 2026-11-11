"""Учёт токенов (День 8): статистика каждого обращения к модели, хода пользователя и сессии.

Откуда берутся числа (приблизительного подсчёта по символам здесь нет):

* ``client.models.count_tokens(contents=...)`` — официальный подсчёт Gemini API ДО отправки.
  В google-genai 2.34.0 в режиме Gemini Developer API (ключ GEMINI_API_KEY) этот метод
  принимает только ``contents``: ``CountTokensConfig(system_instruction / tools / generation_config)``
  SDK отклоняет с ValueError (они поддерживаются только в Vertex AI). Поэтому count_tokens
  считает именно историю диалога: текст, function_call, function_response, thought-части.
* ``response.usage_metadata`` — фактические токены обращения ПОСЛЕ ответа:
  ``prompt_token_count`` (весь вход: история + system instruction + схемы инструментов),
  ``candidates_token_count`` (ответ), ``thoughts_token_count`` (thinking, оплачивается как output),
  ``cached_content_token_count``, ``tool_use_prompt_token_count``, ``total_token_count``.

Разница ``prompt_token_count − count_tokens(history)`` — это служебные токены запроса
(системная инструкция + FunctionDeclaration), которые count_tokens в этом режиме не видит.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import config


# ---------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------

@dataclass
class Usage:
    """Поля usage_metadata одного ответа (None из SDK → 0)."""
    prompt_tokens: int = 0
    candidates_tokens: int = 0
    thoughts_tokens: int = 0
    cached_tokens: int = 0
    tool_use_prompt_tokens: int = 0
    total_tokens: int = 0
    present: bool = False          # было ли usage_metadata в ответе вообще

    @property
    def input_tokens(self) -> int:
        return self.prompt_tokens + self.tool_use_prompt_tokens

    @property
    def output_tokens(self) -> int:
        # thinking оплачивается как output
        return self.candidates_tokens + self.thoughts_tokens


def usage_from_response(response: Any) -> Usage:
    """Достаёт usage_metadata из GenerateContentResponse. Нет метаданных → Usage(present=False)."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return Usage()

    def get(name: str) -> int:
        return int(getattr(meta, name, None) or 0)

    return Usage(
        prompt_tokens=get("prompt_token_count"),
        candidates_tokens=get("candidates_token_count"),
        thoughts_tokens=get("thoughts_token_count"),
        cached_tokens=get("cached_content_token_count"),
        tool_use_prompt_tokens=get("tool_use_prompt_token_count"),
        total_tokens=get("total_token_count"),
        present=True,
    )


@dataclass
class HistoryShape:
    """Из чего состоит история, отправляемая модели."""
    messages: int = 0
    user_text_parts: int = 0
    model_text_parts: int = 0
    thought_parts: int = 0
    function_call_parts: int = 0
    function_response_parts: int = 0


def history_shape(history: list) -> HistoryShape:
    shape = HistoryShape(messages=len(history))
    for content in history:
        for part in content.parts or []:
            if part.function_call:
                shape.function_call_parts += 1
            elif part.function_response:
                shape.function_response_parts += 1
            elif part.thought:
                shape.thought_parts += 1
            elif part.text is not None:
                if content.role == "model":
                    shape.model_text_parts += 1
                else:
                    shape.user_text_parts += 1
    return shape


@dataclass
class CallStats:
    """Одно обращение к generate_content внутри хода."""
    iteration: int
    history_tokens: int | None          # count_tokens(history) перед отправкой; None — подсчёт не удался
    shape: HistoryShape
    usage: Usage = field(default_factory=Usage)
    function_calls_in_response: int = 0
    error: str | None = None            # ошибка API этого обращения

    @property
    def overhead_tokens(self) -> int | None:
        """Служебные токены: system instruction + схемы инструментов (по факту usage_metadata)."""
        if self.history_tokens is None or not self.usage.present or not self.usage.prompt_tokens:
            return None
        return self.usage.prompt_tokens - self.history_tokens


@dataclass
class TurnStats:
    """Все обращения к модели, вызванные одним сообщением пользователя."""
    request_tokens: int | None = None       # count_tokens(только текущее сообщение пользователя)
    calls: list[CallStats] = field(default_factory=list)
    tool_exchange_tokens: int | None = None  # count_tokens(function_call + function_response этого хода)
    input_token_limit: int | None = None
    status: str = "ok"
    count_errors: list[str] = field(default_factory=list)
    blocked_locally: bool = False            # запрос не отправлен: count_tokens > лимита

    # --- суммы за ход (по фактическим usage_metadata) ---
    @property
    def input_tokens(self) -> int:
        return sum(c.usage.input_tokens for c in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.usage.output_tokens for c in self.calls)

    @property
    def thoughts_tokens(self) -> int:
        return sum(c.usage.thoughts_tokens for c in self.calls)

    @property
    def cached_tokens(self) -> int:
        return sum(c.usage.cached_tokens for c in self.calls)

    @property
    def model_calls(self) -> int:
        """Сколько запросов generate_content реально ушло (включая завершившиеся ошибкой)."""
        return len(self.calls)

    @property
    def history_tokens(self) -> int | None:
        """История перед ПОСЛЕДНИМ обращением хода — максимальный объём контекста за ход."""
        for call in reversed(self.calls):
            if call.history_tokens is not None:
                return call.history_tokens
        return None

    @property
    def response_tokens(self) -> int:
        """Токены финального ответа модели (последнее обращение)."""
        return self.calls[-1].usage.output_tokens if self.calls else 0

    @property
    def last_prompt_tokens(self) -> int | None:
        for call in reversed(self.calls):
            if call.usage.present and call.usage.prompt_tokens:
                return call.usage.prompt_tokens
        return None

    @property
    def cost_usd(self) -> float | None:
        return estimate_cost(self.input_tokens, self.output_tokens)

    def to_dict(self) -> dict:
        data = asdict(self)
        data.update(
            input_tokens=self.input_tokens, output_tokens=self.output_tokens,
            history_tokens=self.history_tokens, response_tokens=self.response_tokens,
            cost_usd=self.cost_usd,
        )
        for call, raw in zip(self.calls, data["calls"]):
            raw["overhead_tokens"] = call.overhead_tokens
        return data


@dataclass
class SessionStats:
    """Накопленный расход за запуск программы (reset истории его не обнуляет: токены уже потрачены)."""
    turns: int = 0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thoughts_tokens: int = 0
    cached_tokens: int = 0

    def add(self, turn: TurnStats) -> None:
        self.turns += 1
        self.model_calls += turn.model_calls
        self.input_tokens += turn.input_tokens
        self.output_tokens += turn.output_tokens
        self.thoughts_tokens += turn.thoughts_tokens
        self.cached_tokens += turn.cached_tokens

    @property
    def cost_usd(self) -> float | None:
        return estimate_cost(self.input_tokens, self.output_tokens)


# ---------------------------------------------------------------------
# Стоимость
# ---------------------------------------------------------------------

def model_pricing(model: str = config.MODEL) -> dict | None:
    """Тариф модели из config.PRICING или None, если тариф для неё не проверен."""
    return config.PRICING.get(model)


def estimate_cost(input_tokens: int, output_tokens: int, model: str = config.MODEL) -> float | None:
    """Оценка стоимости в USD по тарифу Standard (paid tier). None — тариф неизвестен.

    Кэшированные токены считаются по полной цене входа (оценка сверху): скидку за кэш
    для этой модели мы не проверяли.
    """
    price = model_pricing(model)
    if price is None:
        return None
    return (input_tokens * price["input_per_1m"] + output_tokens * price["output_per_1m"]) / 1_000_000


# ---------------------------------------------------------------------
# Вывод
# ---------------------------------------------------------------------

def _n(value: int | None) -> str:
    return "н/д" if value is None else f"{value:,}".replace(",", " ")


def _usd(value: float | None) -> str:
    return "тариф неизвестен" if value is None else f"≈ ${value:.6f}"


def format_context(tokens: int | None, limit: int | None) -> str:
    if tokens is None:
        return "н/д"
    if not limit:
        return f"{_n(tokens)} / лимит неизвестен"
    return f"{_n(tokens)} / {_n(limit)} ({tokens / limit:.2%})"


def format_turn(turn: TurnStats, session: SessionStats | None = None) -> str:
    lines = ["─── Токены ─────────────────────────────────────────"]
    lines.append(f"Запрос пользователя:     {_n(turn.request_tokens)}  (count_tokens)")
    for call in turn.calls:
        s = call.shape
        tools_note = ""
        if s.function_call_parts or s.function_response_parts:
            tools_note = f", function_call: {s.function_call_parts}, function_response: {s.function_response_parts}"
        lines.append(
            f"Обращение {call.iteration}: история {_n(call.history_tokens)} ток. "
            f"({s.messages} сообщ.{tools_note}) → "
            + (f"ОШИБКА: {call.error}" if call.error else
               f"вход {_n(call.usage.input_tokens)}, ответ {_n(call.usage.output_tokens)} "
               f"(текст/вызовы {_n(call.usage.candidates_tokens)} + thinking {_n(call.usage.thoughts_tokens)})"
               + (f", вызовов инструментов: {call.function_calls_in_response}" if call.function_calls_in_response else ""))
        )
    if turn.blocked_locally:
        lines.append("Запрос НЕ отправлен в модель: история по count_tokens больше лимита контекста.")
    if turn.tool_exchange_tokens is not None:
        lines.append(f"Вызовы инструментов и их результаты в этом ходе: {_n(turn.tool_exchange_tokens)} ток. (count_tokens)")
    overhead = next((c.overhead_tokens for c in reversed(turn.calls) if c.overhead_tokens is not None), None)
    if overhead is not None:
        lines.append(f"Служебные (system instruction + схемы инструментов): {_n(overhead)} "
                     "= prompt_token_count − count_tokens(история)")
    lines.append(f"История перед отправкой: {_n(turn.history_tokens)}")
    lines.append(f"Ответ модели (финальный): {_n(turn.response_tokens)}")
    lines.append(f"Итого за ход ({turn.model_calls} обращ.): вход {_n(turn.input_tokens)}, "
                 f"выход {_n(turn.output_tokens)} (из них thinking {_n(turn.thoughts_tokens)})"
                 + (f", кэш {_n(turn.cached_tokens)}" if turn.cached_tokens else ""))
    lines.append(f"Контекст (последний вход): {format_context(turn.last_prompt_tokens, turn.input_token_limit)}")
    lines.append(f"Стоимость хода: {_usd(turn.cost_usd)}")
    if turn.count_errors:
        lines.append("count_tokens недоступен: " + "; ".join(turn.count_errors))
    if session is not None:
        lines.append(format_session(session))
    lines.append("────────────────────────────────────────────────────")
    return "\n".join(lines)


def format_session(session: SessionStats) -> str:
    return (f"Сессия: ходов {session.turns}, обращений {session.model_calls}, "
            f"вход {_n(session.input_tokens)}, выход {_n(session.output_tokens)}, {_usd(session.cost_usd)}")


def pricing_note(model: str = config.MODEL) -> str:
    price = model_pricing(model)
    if price is None:
        return f"Тариф для {model} не проверен — стоимость не оценивается."
    return (f"Тариф {model}: ${price['input_per_1m']} / ${price['output_per_1m']} за 1M токенов "
            f"(input / output, {price['tier']}; источник {price['source']}, проверено {price['checked']}). "
            "На бесплатном уровне фактическая стоимость $0.")
