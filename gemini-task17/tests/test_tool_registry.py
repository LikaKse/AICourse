"""Тесты маршрутизации local/MCP в реестре."""

import unittest

from mcp_client import MCPToolError
from tests.fakes import FakeMCPClient, make_tool
from tool_registry import ToolRegistry


def build_registry(mcp):
    return ToolRegistry(mcp_client=mcp, mcp_tools=mcp.exposed_tools())


class RegistryTest(unittest.IsolatedAsyncioTestCase):
    def test_declarations_include_local_and_get_task(self):
        registry = build_registry(FakeMCPClient())
        names = {declaration.name for declaration in registry.declarations}
        self.assertIn("calculate", names)
        self.assertIn("get_task", names)

    async def test_local_calculate_does_not_touch_mcp(self):
        mcp = FakeMCPClient()
        registry = build_registry(mcp)
        result = await registry.adispatch("calculate", {"expression": "2+2"})
        self.assertEqual(result.source, "local")
        self.assertEqual(result.payload["result"]["result"], 4)
        self.assertEqual(mcp.calls, [])

    async def test_get_task_routed_to_mcp(self):
        mcp = FakeMCPClient()
        registry = build_registry(mcp)
        result = await registry.adispatch("get_task", {"task_id": 5})
        self.assertEqual(result.source, "mcp")
        self.assertEqual(result.payload["result"]["id"], 5)
        self.assertEqual(mcp.calls, [("get_task", {"task_id": 5})])

    async def test_mcp_error_payload(self):
        mcp = FakeMCPClient(payload={"error": "Задача не найдена"})
        registry = build_registry(mcp)
        result = await registry.adispatch("get_task", {"task_id": 999})
        self.assertEqual(result.source, "mcp")
        self.assertIn("не найдена", result.payload["error"])

    async def test_mcp_exception_payload(self):
        mcp = FakeMCPClient(error=MCPToolError("server crashed"))
        registry = build_registry(mcp)
        result = await registry.adispatch("get_task", {"task_id": 5})
        self.assertIn("server crashed", result.payload["error"])

    async def test_unknown_tool(self):
        registry = build_registry(FakeMCPClient())
        result = await registry.adispatch("delete_task", {})
        self.assertIn("Неизвестный инструмент", result.payload["error"])

    async def test_name_collision_prefix(self):
        mcp = FakeMCPClient(tools=[make_tool("calculate")])
        registry = build_registry(mcp)
        self.assertIn("mcp_fs_calculate", registry.bindings)
        result = await registry.adispatch("mcp_fs_calculate", {"task_id": 5})
        self.assertEqual(result.source, "mcp")


if __name__ == "__main__":
    unittest.main()
