"""Интеграционный тест с реальным MCP Filesystem 2.4.0.

По умолчанию ПРОПУСКАЕТСЯ. Запуск:
    RUN_MCP_INTEGRATION=1 .venv/bin/python -m unittest tests.test_mcp_integration -v

Gemini API здесь не используется.
"""

import os
import unittest

import config
from mcp_client import MCPFilesystemClient, MCPToolError
from mcp_schema import to_gemini_declaration

RUN = os.environ.get("RUN_MCP_INTEGRATION") == "1"
WRITE_TOOLS = {
    "write_file", "append_file", "edit_file", "create_directory", "move_file", "copy_file",
}


@unittest.skipUnless(RUN, "нужен RUN_MCP_INTEGRATION=1")
class RealFilesystemMCPTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.assertTrue(
            config.NODE_BIN.is_file(),
            f"Нет project-local Node: {config.NODE_BIN}. Запустите bash scripts/setup_mcp_env.sh",
        )
        self.assertTrue(
            config.MCP_SERVER_JS.is_file(),
            f"Нет MCP-сервера: {config.MCP_SERVER_JS}. Запустите bash scripts/setup_mcp_env.sh",
        )

    async def test_connection_tools_and_read_only_workspace(self):
        async with MCPFilesystemClient() as client:
            self.assertIn("2.4.0", client.server_info or "")
            self.assertEqual(client.protocol_version, "2025-11-25")
            self.assertTrue(client.connected)
            self.assertIn(str(config.WORKSPACE), client.build_server_parameters().args)
            self.assertIn("--read-only", client.build_server_parameters().args)
            self.assertIn("--confine-roots", client.build_server_parameters().args)

            names = {tool.name for tool in client.tools}
            self.assertIn("read_text_file", names)
            self.assertIn("read_file", names)  # deprecated alias есть на сервере
            self.assertFalse(WRITE_TOOLS & names, f"read-only сервер отдал запись: {WRITE_TOOLS & names}")

            exposed = {tool.name for tool in client.exposed_tools()}
            self.assertIn("read_text_file", exposed)
            self.assertNotIn("read_file", exposed)
            self.assertNotIn("read_media_file", exposed)

            payload = await client.call_tool_for_model("read_text_file", {"path": "numbers.txt"})
            self.assertIn("quantity=12", payload["content"])
            self.assertIn("unit_price=25000", payload["content"])
            self.assertIn("markup_percent=10", payload["content"])

            allowed = await client.call_tool_for_model("list_allowed_directories", {})
            self.assertIn(str(config.WORKSPACE), allowed["content"])

            declaration = to_gemini_declaration(
                next(tool for tool in client.tools if tool.name == "read_text_file")
            )
            self.assertEqual(declaration.name, "read_text_file")
            self.assertNotIn("$schema", declaration.parameters_json_schema)

        self.assertFalse(client.connected)

    async def test_outside_workspace_is_blocked_twice(self):
        async with MCPFilesystemClient() as client:
            # 1. Клиентский guard.
            with self.assertRaises(MCPToolError):
                await client.call_tool("read_text_file", {"path": "/etc/hosts"})

            # 2. Серверный guard: обходим клиентскую нормализацию напрямую.
            raw = await client._session.call_tool("read_text_file", {"path": "/etc/hosts"})
            self.assertTrue(raw.is_error)
            text = "\n".join(
                block.text for block in raw.content if getattr(block, "type", None) == "text"
            )
            self.assertIn("Access denied", text)


if __name__ == "__main__":
    unittest.main()
