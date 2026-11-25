"""Тесты регистрации и вызова инструмента на собственном MCP-сервере."""

import unittest
from unittest import mock

from mcp import Client

import server
from task_api import TaskNotFound
from tests.fakes import FakeTaskApi


class ServerRegistrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_tool_schema(self):
        tools = await server.mcp.list_tools()
        self.assertEqual([tool.name for tool in tools], ["get_task"])
        tool = tools[0]
        self.assertIn("JSONPlaceholder", tool.description)
        self.assertEqual(tool.input_schema["type"], "object")
        self.assertEqual(tool.input_schema["required"], ["task_id"])
        self.assertEqual(tool.input_schema["properties"]["task_id"]["type"], "integer")
        self.assertIn("userId", tool.output_schema["properties"])

    async def test_call_tool_success_in_memory(self):
        fake = FakeTaskApi()
        with mock.patch.object(server, "api", fake):
            async with Client(server.mcp) as client:
                result = await client.call_tool("get_task", {"task_id": 5})
        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["id"], 5)
        self.assertEqual(result.structured_content["userId"], 1)
        self.assertEqual(fake.calls, [5])

    async def test_call_tool_not_found(self):
        fake = FakeTaskApi(error=TaskNotFound("Задача с id=999 не найдена."))
        with mock.patch.object(server, "api", fake):
            async with Client(server.mcp) as client:
                result = await client.call_tool("get_task", {"task_id": 999})
        self.assertTrue(result.is_error)
        text = "\n".join(block.text for block in result.content if getattr(block, "type", None) == "text")
        self.assertIn("не найдена", text)


if __name__ == "__main__":
    unittest.main()
