"""Стратегии управления контекстом и фабрика для переключения через config / CLI."""

from __future__ import annotations

import os

import config
from strategies.base import CallRecord, ContextStrategy
from strategies.sliding_window import SlidingWindowStrategy
from strategies.summary import SummaryStrategy


def strategy_class(name: str) -> type[ContextStrategy]:
    if name == "sliding":
        return SlidingWindowStrategy
    if name == "summary":
        return SummaryStrategy
    if name == "facts":
        from strategies.sticky_facts import StickyFactsStrategy
        return StickyFactsStrategy
    raise ValueError(f"Неизвестная стратегия {name!r}; допустимо: {', '.join(config.STRATEGIES)}")


def make_strategy(name: str = config.STRATEGY, state_dir: str | os.PathLike | None = None, **kwargs) -> ContextStrategy:
    """Создать стратегию по имени. kwargs — параметры конкретной стратегии (window_size, keep_last, …)."""
    return strategy_class(name)(state_dir, **kwargs)


__all__ = ["CallRecord", "ContextStrategy", "SlidingWindowStrategy", "SummaryStrategy",
           "make_strategy", "strategy_class"]
