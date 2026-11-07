"""Стоимость запроса по тарифам из config.PRICING.

Формула (USD), цены за 1 000 000 токенов:
    cost = (input - cached) * price_input + cached * price_cached_input + output * price_output
где output = candidates_token_count + thoughts_token_count (thinking оплачивается как выходные).

Если стоимость определить нельзя, возвращается cost_usd = None и причина в cost_note.
"""

from datetime import date

import config

PER_TOKENS = 1_000_000


def price_for(model, on_date):
    """Тариф модели на дату (date или 'YYYY-MM-DD'). None — если тарифа нет."""
    if isinstance(on_date, str):
        on_date = date.fromisoformat(on_date[:10])
    for period in config.PRICING.get(model, []):
        start = date.fromisoformat(period["valid_from"]) if period["valid_from"] else date.min
        end = date.fromisoformat(period["valid_to"]) if period["valid_to"] else date.max
        if start <= on_date <= end:
            return period
    return None


def billable_output_tokens(candidates, thoughts):
    """Оплачиваемые выходные токены = ответ + рассуждения. None, если неизвестен ответ."""
    if candidates is None:
        return None
    return candidates + (thoughts or 0)


def max_output_tokens_for(model):
    """Сколько выходных токенов (вместе с thinking) модель может выдать за один запрос."""
    if config.MAX_OUTPUT_TOKENS is not None:
        return config.MAX_OUTPUT_TOKENS
    return config.OUTPUT_TOKEN_LIMITS[model]


def max_call_cost(model, on_date, input_tokens=None, output_tokens=None):
    """Максимальная стоимость ОДНОГО запроса: вход по оценке сверху + выход на весь лимит."""
    price = price_for(model, on_date)
    if price is None:
        return None
    input_tokens = config.MAX_INPUT_TOKENS_ESTIMATE if input_tokens is None else input_tokens
    output_tokens = max_output_tokens_for(model) if output_tokens is None else output_tokens
    return (input_tokens * price["input"] + output_tokens * price["output"]) / PER_TOKENS


def max_run_cost(models, n_tasks, repeats, on_date, attempts_per_call=1):
    """Максимальная стоимость эксперимента: {model: cost}, total.

    attempts_per_call=1 — оплачивается только итоговая попытка (ответы с ошибкой API без
    генерации не тарифицируются). attempts_per_call=3 — абсолютный худший случай: каждая
    попытка (например, клиентский таймаут после генерации) оплачена полностью.
    """
    per_model = {}
    for model in models:
        per_call = max_call_cost(model, on_date)
        per_model[model] = None if per_call is None else per_call * n_tasks * repeats * attempts_per_call
    known = [c for c in per_model.values() if c is not None]
    total = sum(known) if len(known) == len(per_model) else None
    return per_model, total


def compute_cost(model, on_date, prompt_tokens, candidates_tokens, thoughts_tokens, cached_tokens=None):
    """dict: cost_usd, input_cost_usd, output_cost_usd, price_input, price_output, cost_note."""
    out = {"cost_usd": None, "input_cost_usd": None, "output_cost_usd": None,
           "price_input_per_1m": None, "price_output_per_1m": None, "cost_note": None}
    price = price_for(model, on_date)
    if price is None:
        out["cost_note"] = f"нет тарифа для {model} на {on_date}"
        return out
    out["price_input_per_1m"] = price["input"]
    out["price_output_per_1m"] = price["output"]
    output_tokens = billable_output_tokens(candidates_tokens, thoughts_tokens)
    if prompt_tokens is None or output_tokens is None:
        out["cost_note"] = "API не вернул usage_metadata (prompt/candidates) — стоимость неизвестна"
        return out
    cached = cached_tokens or 0
    input_cost = ((prompt_tokens - cached) * price["input"] + cached * price["cached_input"]) / PER_TOKENS
    output_cost = output_tokens * price["output"] / PER_TOKENS
    out.update(input_cost_usd=input_cost, output_cost_usd=output_cost, cost_usd=input_cost + output_cost)
    notes = []
    if thoughts_tokens is None:
        notes.append("thoughts_token_count не вернулся — считаем 0")
    if cached:
        notes.append(f"{cached} токенов из кэша по цене cached_input")
    out["cost_note"] = "; ".join(notes) or None
    return out
