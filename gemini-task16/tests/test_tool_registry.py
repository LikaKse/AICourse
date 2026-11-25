"""Тесты объединённого реестра: маршрутизация local/MCP, ошибки, коллизии."""

import unittest

from mcp_client import MCPToolError
from tests.fakes import FakeMCPClient, make_tool
from tool_registry import ToolRegistry


def build_registry(mcp):
    return ToolRegistry(mcp_client=mcp, mcp_tools=mcp.exposed_tools())


class RegistryTest(unittest.IsolatedAsyncioTestCase):
    def test_declarations_include_local_and_mcp(self):
        registry = build_registry(FakeMCPClient())
        names = {declaration.name for declaration in registry.declarations}
        self.assertIn("calculate", names)
        self.assertIn("read_text_file", names)
        self.assertNotIn("read_file", names)

    async def test_local_call_does_not_touch_mcp(self):
        mcp = FakeMCPClient()
        registry = build_registry(mcp)
        result = await registry.adispatch("calculate", {"expression": "2+2"})
        self.assertEqual(result.source, "local")
        self.assertEqual(result.payload["result"]["result"], 4)
        self.assertEqual(mcp.calls, [])

    async def test_mcp_call_is_routed_to_client(self):
        mcp = FakeMCPClient(payloads={
            "read_text_file": {"source": "mcp", "tool": "read_text_file", "content": "quantity=12"}
        })
        registry = build_registry(mcp)
        result = await registry.adispatch("read_text_file", {"path": "numbers.txt"})
        self.assertEqual(result.source, "mcp")
        self.assertEqual(result.payload["content"], "quantity=12")
        self.assertEqual(mcp.calls, [("read_text_file", {"path": "numbers.txt"})])

    async def test_mcp_error_becomes_error_payload(self):
        mcp = FakeMCPClient(payloads={"read_text_file": {"error": "Access denied"}})
        registry = build_registry(mcp)
        result = await registry.adispatch("read_text_file", {"path": "../secret"})
        self.assertEqual(result.source, "mcp")
        self.assertIn("Access denied", result.payload["error"])

    async def test_mcp_exception_becomes_error_payload(self):
        mcp = FakeMCPClient(error=MCPToolError("session lost"))
        registry = build_registry(mcp)
        result = await registry.adispatch("read_text_file", {"path": "numbers.txt"})
        self.assertIn("session lost", result.payload["error"])

    async def test_unknown_and_hidden_tool(self):
        registry = build_registry(FakeMCPClient())
        for name in ("write_file", "unknown"):
            with self.subTest(name=name):
                result = await registry.adispatch(name, {})
                self.assertIn("Неизвестный инструмент", result.payload["error"])

    async def test_name_collision_gets_mcp_prefix(self):
        mcp = FakeMCPClient(tools=[make_tool(name="calculate")])
        registry = build_registry(mcp)
        self.assertIn("calculate", registry.bindings)
        self.assertIn("mcp_fs_calculate", registry.bindings)
        self.assertEqual(registry.bindings["calculate"].source, "local")
        self.assertEqual(registry.bindings["mcp_fs_calculate"].source, "mcp")

        result = await registry.adispatch("mcp_fs_calculate", {"expression": "1+1"})
        self.assertEqual(result.source, "mcp")
        self.assertEqual(mcp.calls, [("calculate", {"expression": "1+1"})])

    async def test_local_dispatcher_exception_is_caught(self):
        def broken(name, args):
            raise RuntimeError("boom")

        registry = ToolRegistry(local_dispatcher=broken, mcp_client=FakeMCPClient(), mcp_tools=[])
        result = await registry.adispatch("calculate", {"expression": "1+1"})
        self.assertEqual(result.source, "local")
        self.assertIn("boom", result.payload["error"])


if __name__ == "__main__":
    unittest.main()
