"""Тесты поиска JSONPlaceholder без сети."""

import unittest

import httpx2

from search_api import SearchApi, SearchApiError
from tests.fakes import POSTS


def api_with(handler) -> SearchApi:
    return SearchApi(base_url="https://example.test", transport=httpx2.MockTransport(handler))


class SearchApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_filter_and_limit(self):
        def handler(request):
            return httpx2.Response(200, json=POSTS)

        result = await api_with(handler).search("qui est esse")
        self.assertEqual(result.query, "qui est esse")
        self.assertEqual([item.id for item in result.items], [2])
        self.assertEqual(result.items[0].title, "qui est esse")

    async def test_empty_result(self):
        def handler(request):
            return httpx2.Response(200, json=POSTS)

        result = await api_with(handler).search("zzzz-no-match")
        self.assertEqual(result.items, [])

    async def test_http_error(self):
        def handler(request):
            return httpx2.Response(500, text="boom")

        with self.assertRaises(SearchApiError) as ctx:
            await api_with(handler).search("qui")
        self.assertIn("HTTP 500", str(ctx.exception))

    async def test_timeout(self):
        def handler(request):
            raise httpx2.ReadTimeout("slow", request=request)

        with self.assertRaises(SearchApiError):
            await api_with(handler).search("qui")

    async def test_bad_payload(self):
        def handler(request):
            return httpx2.Response(200, json={"not": "a list"})

        with self.assertRaises(SearchApiError):
            await api_with(handler).search("qui")

    async def test_empty_query_returns_empty(self):
        result = await api_with(lambda request: httpx2.Response(200, json=POSTS)).search("   ")
        self.assertEqual(result.items, [])


if __name__ == "__main__":
    unittest.main()
