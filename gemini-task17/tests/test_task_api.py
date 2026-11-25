"""Тесты HTTP-клиента JSONPlaceholder без сети."""

import unittest

import httpx2

from task_api import TaskApi, TaskApiError, TaskNotFound

TASK = {
    "userId": 1,
    "id": 5,
    "title": "laboriosam mollitia et enim quasi adipisci quia provident illum",
    "completed": False,
}


def api_with(handler) -> TaskApi:
    return TaskApi(base_url="https://example.test", transport=httpx2.MockTransport(handler))


class TaskApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_success_and_url(self):
        seen = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            seen.append(str(request.url))
            return httpx2.Response(200, json=TASK)

        task = await api_with(handler).fetch_task(5)
        self.assertEqual(task.id, 5)
        self.assertEqual(task.user_id, 1)
        self.assertEqual(task.title, TASK["title"])
        self.assertFalse(task.completed)
        self.assertEqual(seen, ["https://example.test/todos/5"])

    async def test_404(self):
        def handler(request):
            return httpx2.Response(404, json={})

        with self.assertRaises(TaskNotFound):
            await api_with(handler).fetch_task(999999)

    async def test_500(self):
        def handler(request):
            return httpx2.Response(500, text="boom")

        with self.assertRaises(TaskApiError) as ctx:
            await api_with(handler).fetch_task(5)
        self.assertIn("HTTP 500", str(ctx.exception))

    async def test_invalid_json(self):
        def handler(request):
            return httpx2.Response(200, text="not json")

        with self.assertRaises(TaskApiError) as ctx:
            await api_with(handler).fetch_task(5)
        self.assertIn("JSON", str(ctx.exception))

    async def test_unexpected_structure(self):
        def handler(request):
            return httpx2.Response(200, json={"id": 5})

        with self.assertRaises(TaskApiError):
            await api_with(handler).fetch_task(5)

    async def test_timeout(self):
        def handler(request):
            raise httpx2.ReadTimeout("slow", request=request)

        with self.assertRaises(TaskApiError) as ctx:
            await api_with(handler).fetch_task(5)
        self.assertIn("не ответил", str(ctx.exception))

    async def test_invalid_task_id(self):
        for value in (0, -1, True, "5"):
            with self.subTest(value=value):
                with self.assertRaises(TaskApiError):
                    await api_with(lambda request: httpx2.Response(200, json=TASK)).fetch_task(value)


if __name__ == "__main__":
    unittest.main()
