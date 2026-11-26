"""Поиск по JSONPlaceholder без API-ключа."""

from __future__ import annotations

from typing import Any

import httpx2

import config
from models import Post, SearchResult


class SearchApiError(Exception):
    """JSONPlaceholder недоступен или вернул неожиданные данные."""


class SearchApi:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = config.SEARCH_TIMEOUT,
        max_results: int = config.SEARCH_MAX_RESULTS,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or config.SEARCH_API_BASE).rstrip("/")
        self.timeout = timeout
        self.max_results = max_results
        self.transport = transport

    async def search(self, query: str) -> SearchResult:
        if not isinstance(query, str):
            raise SearchApiError("query должен быть строкой.")
        query = query.strip()
        if not query:
            return SearchResult(query="", items=[])

        url = f"{self.base_url}/posts"
        try:
            async with httpx2.AsyncClient(
                timeout=httpx2.Timeout(self.timeout),
                transport=self.transport,
            ) as client:
                response = await client.get(url)
        except httpx2.TimeoutException as exc:
            raise SearchApiError(f"JSONPlaceholder не ответил за {self.timeout:g} с.") from exc
        except httpx2.HTTPError as exc:
            raise SearchApiError(f"Ошибка сети при запросе к JSONPlaceholder: {exc}") from exc

        if response.status_code != 200:
            raise SearchApiError(f"JSONPlaceholder вернул HTTP {response.status_code}.")

        try:
            payload: Any = response.json()
            if not isinstance(payload, list):
                raise ValueError("ожидался массив")
            needle = query.casefold()
            items = [
                Post.model_validate(item)
                for item in payload
                if needle in str(item.get("title", "")).casefold()
                or needle in str(item.get("body", "")).casefold()
            ][: self.max_results]
            return SearchResult(query=query, items=items)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise SearchApiError("JSONPlaceholder вернул неожиданную структуру данных.") from exc
