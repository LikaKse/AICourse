"""Проверка цитат и чисел (перенесено из facts_store Дня 10) + правило «вопрос — не факт»."""

from __future__ import annotations

import re

from utils import normalize_text


def plain(text: str) -> str:
    """Для сравнения цитат: нормализация + без пунктуации и кавычек."""
    text = re.sub(r"[^\w\s:/+.-]", " ", normalize_text(text))
    text = re.sub(r"(?<!\w)[:/+.-]+|[:/+.-]+(?!\w)", " ", text)     # знаки только внутри слов
    return re.sub(r"\s+", " ", text).strip()


def quote_in(quote: str, text: str | None) -> bool:
    q = plain(quote)
    return bool(text) and len(q) >= 3 and q in plain(text)


def numbers_supported(value: str, *sources: str | None) -> bool:
    """Каждое число значения встречается в источнике (после нормализации «18 500 000» → «18500000»)."""
    source = " ".join(normalize_text(s) for s in sources if s)
    return all(n in source for n in re.findall(r"\d+", normalize_text(value)))


_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def quote_is_question(quote: str, text: str | None) -> bool:
    """Цитата целиком взята из вопросительного предложения («Какой фреймворк посоветуешь?»)."""
    return any(quote_in(quote, s) and s.strip().endswith("?") for s in _SENTENCE_END.split(text or ""))


def same_value(a: str, b: str) -> bool:
    return plain(a) == plain(b)
