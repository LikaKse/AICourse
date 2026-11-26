"""Детерминированное резюме без LLM."""

from __future__ import annotations

from models import Post, Summary


class SummarizerError(Exception):
    """Не удалось построить резюме."""


class Summarizer:
    def summarize(self, items: list[Post]) -> Summary:
        if not items:
            raise SummarizerError("Нельзя суммировать пустой список публикаций.")

        titles = [item.title for item in items[:3]]
        first = items[0]
        body = " ".join(first.body.split())
        if len(body) > 160:
            body = body[:157] + "..."
        summary = (
            f"Найдено публикаций: {len(items)}. "
            f"Заголовки: " + "; ".join(f"«{title}»" for title in titles) + ". "
            f"Первая публикация: «{first.title}» — {body}"
        )
        return Summary(source_count=len(items), summary=summary)
