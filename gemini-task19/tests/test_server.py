"""Тесты MCP-сервера Дня 19 через in-memory Client."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mcp import Client

import config
import server
from models import Post, SearchResult
from tests.fakes import POSTS


class FakeSearchApi:
    def __init__(self, items=None, error=None):
        self.items = items if items is not None else POSTS
        self.error = error
        self.calls = []

    async def search(self, query):
        self.calls.append(query)
        if self.error:
            raise self.error
        return SearchResult(query=query, items=[Post.model_validate(item) for item in self.items])


class ServerTest(unittest.IsolatedAsyncioTestCase):
    async def test_three_tool_schemas(self):
        tools = await server.mcp.list_tools()
        names = {tool.name for tool in tools}
        self.assertEqual(names, {"search", "summarize", "save_to_file"})
        by_name = {tool.name: tool for tool in tools}
        self.assertEqual(by_name["search"].input_schema["required"], ["query"])
        self.assertEqual(by_name["summarize"].input_schema["required"], ["items"])
        self.assertEqual(by_name["save_to_file"].input_schema["required"], ["content", "filename"])

    async def test_success_chain_and_file_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeSearchApi()
            with mock.patch.object(server, "search_api", fake), \
                 mock.patch.object(config, "OUTPUT_DIR", Path(tmp)):
                async with Client(server.mcp) as client:
                    found = await client.call_tool("search", {"query": "qui"})
                    self.assertFalse(found.is_error)
                    items = found.structured_content["items"]
                    self.assertEqual(len(items), 2)

                    summarized = await client.call_tool("summarize", {"items": items})
                    self.assertFalse(summarized.is_error)
                    summary = summarized.structured_content["summary"]
                    self.assertIn("qui est esse", summary)

                    saved = await client.call_tool(
                        "save_to_file",
                        {"content": summary, "filename": "summary.txt"},
                    )
                    self.assertFalse(saved.is_error)
                    path = Path(tmp) / "summary.txt"
                    self.assertTrue(path.is_file())
                    self.assertEqual(path.read_text(encoding="utf-8"), summary)
                    self.assertEqual(saved.structured_content["bytes_written"], len(summary.encode()))

    async def test_summarize_failure_hook(self):
        with mock.patch.object(server, "FAIL_SUMMARIZE", True):
            async with Client(server.mcp) as client:
                result = await client.call_tool("summarize", {"items": POSTS})
        self.assertTrue(result.is_error)
        text = "\n".join(block.text for block in result.content if getattr(block, "type", None) == "text")
        self.assertIn("Тестовая ошибка summarize", text)

    async def test_save_rejects_empty_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(config, "OUTPUT_DIR", Path(tmp)):
            async with Client(server.mcp) as client:
                empty = await client.call_tool("save_to_file", {"content": " ", "filename": "x.txt"})
                bad_name = await client.call_tool(
                    "save_to_file", {"content": "text", "filename": "../x.txt"}
                )
        self.assertTrue(empty.is_error)
        self.assertTrue(bad_name.is_error)


if __name__ == "__main__":
    unittest.main()
