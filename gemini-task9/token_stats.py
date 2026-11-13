"""Подсчёт токенов: count_tokens до отправки и usage_metadata после ответа.

Два источника чисел (приблизительного подсчёта по символам здесь нет):

* ``client.models.count_tokens(contents=...)`` — официальный подсчёт Gemini API ДО отправки.
  В google-genai 2.34.0 в режиме Gemini Developer API он принимает только ``contents``
  (system instruction не учитывается — см. День 8).
* ``response.usage_metadata`` — фактические токены вызова ПОСЛЕ ответа:
  ``prompt_token_count`` (весь вход, включая system instruction), ``candidates_token_count``,
  ``thoughts_token_count`` (thinking, оплачивается как output), ``total_token_count``.

Разница ``prompt_token_count − count_tokens(contents)`` ≈ токены system instruction.
Экономия считается отдельно по каждому источнику, чтобы фактическое (usage_metadata)
не путалось с оценкой до отправки (count_tokens).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from google.genai import errors

import config
from utils import redact_secrets

logger = logging.getLogger("tokens")


@dataclass
class Usage:
    """Поля usage_metadata одного ответа (None из SDK → 0)."""
    prompt_tokens: int = 0
    candidates_tokens: int = 0
    thoughts_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    present: bool = False          # было ли usage_metadata в ответе вообще

    @property
    def input_tokens(self) -> int:
        return self.prompt_tokens

    @property
    def output_tokens(self) -> int:
        return self.candidates_tokens + self.thoughts_tokens

    def to_dict(self) -> dict:
        return {**asdict(self), "input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


def usage_from_response(response: Any) -> Usage:
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
        total_tokens=get("total_token_count"),
        present=True,
    )


class TokenCounter:
    """Обёртка над client.models.count_tokens. Ошибка подсчёта не роняет работу: (None, текст ошибки)."""

    def __init__(self, client: Any, model: str = config.MODEL, enabled: bool = config.COUNT_TOKENS) -> None:
        self.client = client
        self.model = model
        self.enabled = enabled
        self.calls = 0

    def count(self, contents: list) -> tuple[int | None, str | None]:
        if not contents:
            return 0, None
        if not self.enabled:
            return None, "count_tokens отключён"
        self.calls += 1
        try:
            response = self.client.models.count_tokens(model=self.model, contents=contents)
        except errors.APIError as exc:
            msg = redact_secrets(f"{exc.code} {exc.status or ''}: {exc.message or exc}".strip())
        except Exception as exc:  # сеть, таймаут
            msg = redact_secrets(f"{type(exc).__name__}: {exc}")
        else:
            if response.total_tokens is None:
                return None, "count_tokens вернул пустой total_tokens"
            return int(response.total_tokens), None
        logger.warning("Не удалось посчитать токены: %s", msg)
        return None, msg


# ---------------------------------------------------------------------
# Агрегаты и сравнение
# ---------------------------------------------------------------------

def sum_usage(items: Iterable[Usage]) -> dict:
    """Сумма usage_metadata по нескольким вызовам. complete=False — у части вызовов метаданных не было."""
    items = list(items)
    present = [u for u in items if u.present]
    return {
        "calls": len(items),
        "calls_with_usage_metadata": len(present),
        "complete": len(present) == len(items),
        "input_tokens": sum(u.input_tokens for u in present),
        "candidates_tokens": sum(u.candidates_tokens for u in present),
        "thoughts_tokens": sum(u.thoughts_tokens for u in present),
        "output_tokens": sum(u.output_tokens for u in present),
        "total_tokens": sum(u.input_tokens + u.output_tokens for u in present),
    }


def sum_known(values: Iterable[int | None]) -> int | None:
    """Сумма, если известны все значения; иначе None (не выдаём неполную сумму за полную)."""
    values = list(values)
    if any(v is None for v in values):
        return None
    return sum(values)


def savings_percent(baseline: int | float | None, compressed: int | float | None) -> float | None:
    """Экономия относительно baseline в процентах (отрицательная — перерасход)."""
    if baseline is None or compressed is None or baseline <= 0:
        return None
    return round((baseline - compressed) / baseline * 100, 2)


def compare_count_with_usage(counted: int | None, usage: Usage) -> dict:
    """Сверка count_tokens(contents) с usage_metadata.prompt_token_count одного вызова."""
    if counted is None or not usage.present:
        return {"count_tokens": counted, "prompt_token_count": usage.prompt_tokens if usage.present else None,
                "difference": None}
    return {"count_tokens": counted, "prompt_token_count": usage.prompt_tokens,
            "difference": usage.prompt_tokens - counted}


def estimate_cost(input_tokens: int | None, output_tokens: int | None, model: str = config.MODEL) -> float | None:
    """Расчётная стоимость по тарифу из config.PRICING (не счёт от Google). Нет тарифа → None."""
    price = config.PRICING.get(model)
    if price is None or input_tokens is None or output_tokens is None:
        return None
    return round(input_tokens / 1e6 * price["input_per_1m"] + output_tokens / 1e6 * price["output_per_1m"], 8)
