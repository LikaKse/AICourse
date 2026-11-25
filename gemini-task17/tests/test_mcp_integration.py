"""Интеграционные тесты: настоящий stdio MCP-сервер + локальная HTTP-заглушка.

По умолчанию пропускаются:
    RUN_MCP_INTEGRATION=1 .venv/bin/python -m unittest tests.test_mcp_integration -v

Gemini не используется.
"""

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config
from mcp_client import TaskMCPClient
from task_api import TaskApi

RUN = os.environ.get("RUN_MCP_INTEGRATION") == "1"
LIVE = os.environ.get("RUN_LIVE_API") == "1"


class _Handler(BaseHTTPRequestHandler):
    seen: list[str] = []

    def do_GET(self):  # noqa: N802
        self.seen.append(self.path)
        if self.path == "/todos/5":
            body = json.dumps({
                "userId": 1,
                "id": 5,
                "title": "laboriosam mollitia et enim quasi adipisci quia provident illum",
                "completed": False,
            }).encode()
            self.send_response(200)
        else:
            body = b"{}"
            self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def start_stub():
    _Handler.seen = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, f"http://127.0.0.1:{httpd.server_port}"


@unittest.skipUnless(RUN, "нужен RUN_MCP_INTEGRATION=1")
class RealTaskMCPTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd, cls.api_base = start_stub()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    async def test_full_stdio_cycle(self):
        self.assertTrue(config.PYTHON_BIN.is_file(), f"Нет Python: {config.PYTHON_BIN}")
        self.assertTrue(config.SERVER_PY.is_file(), f"Нет server.py: {config.SERVER_PY}")
        async with TaskMCPClient(api_base=self.api_base) as client:
            self.assertEqual(client.protocol_version, "2025-11-25")
            self.assertIn(config.SERVER_NAME, client.server_info or "")
            self.assertEqual([tool.name for tool in client.tools], ["get_task"])
            schema = client.tools[0].input_schema
            self.assertEqual(schema["required"], ["task_id"])
            self.assertEqual(schema["properties"]["task_id"]["type"], "integer")

            payload = await client.call_tool_for_model("get_task", {"task_id": 5})
            self.assertEqual(payload["source"], "mcp")
            self.assertEqual(payload["result"]["id"], 5)
            self.assertEqual(payload["result"]["userId"], 1)
            self.assertFalse(payload["result"]["completed"])
            self.assertIn("/todos/5", _Handler.seen)

    async def test_not_found_is_mcp_error(self):
        async with TaskMCPClient(api_base=self.api_base) as client:
            payload = await client.call_tool_for_model("get_task", {"task_id": 404})
        self.assertIn("не найдена", payload["error"])


@unittest.skipUnless(LIVE, "нужен RUN_LIVE_API=1")
class LiveJSONPlaceholderTest(unittest.IsolatedAsyncioTestCase):
    async def test_live_api(self):
        task = await TaskApi().fetch_task(5)
        self.assertEqual(task.id, 5)
        self.assertEqual(task.user_id, 1)
        self.assertFalse(task.completed)
        self.assertTrue(task.title)


if __name__ == "__main__":
    unittest.main()
