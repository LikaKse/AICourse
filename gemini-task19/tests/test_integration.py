"""Интеграция: реальный stdio MCP-сервер + локальная заглушка JSONPlaceholder.

По умолчанию пропускается:
    RUN_MCP_INTEGRATION=1 .venv/bin/python -m unittest tests.test_integration -v
"""

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from chain_agent import ChainAgent
from mcp_client import ChainMCPClient
from tests.fakes import POSTS

RUN = os.environ.get("RUN_MCP_INTEGRATION") == "1"


class _Handler(BaseHTTPRequestHandler):
    seen: list[str] = []

    def do_GET(self):  # noqa: N802
        self.seen.append(self.path)
        if self.path.startswith("/posts"):
            body = json.dumps(POSTS).encode()
            self.send_response(200)
        else:
            body = b"[]"
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
class RealChainTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd, cls.api_base = start_stub()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    async def test_success_chain_saves_exact_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output"
            async with ChainMCPClient(
                api_base=self.api_base,
                output_dir=output_dir,
                fail_summarize=False,
            ) as client:
                result = await ChainAgent(client).run("qui est esse", "summary.txt")
            self.assertEqual(result.status, "ok")
            self.assertEqual([step.name for step in result.steps], ["search", "summarize", "save_to_file"])
            self.assertEqual(result.steps[0].status, "ok")
            self.assertEqual(result.steps[1].status, "ok")
            self.assertEqual(result.steps[2].status, "ok")
            path = Path(tmp) / "output" / "summary.txt"
            self.assertTrue(path.is_file())
            self.assertEqual(path.read_text(encoding="utf-8"), result.summary)
            self.assertIn("qui est esse", result.summary)

    async def test_summarize_error_stops_and_saves_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output"
            async with ChainMCPClient(
                api_base=self.api_base,
                output_dir=output_dir,
                fail_summarize=True,
            ) as client:
                result = await ChainAgent(client).run("qui est esse", "should_not_exist.txt")
            self.assertEqual(result.status, "stopped")
            self.assertEqual(result.stopped_at, "summarize")
            self.assertEqual(result.completed_steps, ["search"])
            self.assertFalse((Path(tmp) / "output" / "should_not_exist.txt").exists())
            self.assertIn("Тестовая ошибка summarize", result.reason)

    async def test_empty_search_stops_and_saves_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output"
            async with ChainMCPClient(
                api_base=self.api_base,
                output_dir=output_dir,
            ) as client:
                result = await ChainAgent(client).run("zzzz-no-match", "should_not_exist.txt")
            self.assertEqual(result.status, "stopped")
            self.assertEqual(result.stopped_at, "search")
            self.assertEqual(result.completed_steps, [])
            self.assertFalse((Path(tmp) / "output" / "should_not_exist.txt").exists())


if __name__ == "__main__":
    unittest.main()
