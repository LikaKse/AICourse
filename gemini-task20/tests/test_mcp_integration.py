"""Реальные stdio-интеграционные тесты Дня 20.

Запуск:
    RUN_MCP_INTEGRATION=1 .venv/bin/python -m unittest tests.test_mcp_integration -v
"""

import os
import unittest

import config
from flow_agent import FlowAgent
from manager import MultiMCPManager
from server_specs import default_specs

RUN = os.environ.get("RUN_MCP_INTEGRATION") == "1"


@unittest.skipUnless(RUN, "нужен RUN_MCP_INTEGRATION=1")
class RealMultiServerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.assertTrue(config.NODE_BIN.is_file(), f"Нет Node: {config.NODE_BIN}")
        self.assertTrue(config.FS_SERVER_JS.is_file(), f"Нет Filesystem MCP: {config.FS_SERVER_JS}")
        self.assertTrue(config.CONTEXT_FILE.is_file(), f"Нет контекста: {config.CONTEXT_FILE}")

    async def test_full_flow_through_three_servers(self):
        manager = MultiMCPManager(default_specs())
        await manager.start()
        try:
            self.assertEqual(manager.available_servers, ["catalog", "notes", "filesystem"])
            self.assertEqual(manager.unavailable, [])
            self.assertIn("catalog__search", manager.routes)
            self.assertIn("notes__search", manager.routes)
            self.assertEqual(manager.routes["catalog__search"].original_name, "search")
            self.assertEqual(manager.routes["notes__search"].original_name, "search")

            allowed = await manager.call_tool("filesystem__list_allowed_directories", {})
            self.assertFalse(allowed.is_error)
            self.assertIn(str(config.WORKSPACE), allowed.content)
            self.assertNotIn(str(config.PROJECT_ROOT / "servers"), allowed.content)

            result = await FlowAgent(manager).run(config.DEMO_QUERY)
            self.assertEqual(result.status, "ok")
            self.assertEqual(result.completed_tools, [
                "filesystem__read_text_file",
                "catalog__search",
                "catalog__get_item",
                "notes__search",
                "notes__get_note",
                "notes__summarize",
            ])
            self.assertIn("Laptop Pro 14", result.final_text)
            self.assertIn("Laptop battery review", result.final_text)
            self.assertIn("Контекст закупки", result.final_text)
        finally:
            await manager.aclose()

    async def test_real_mcp_is_error_and_empty_search_contract(self):
        manager = MultiMCPManager(default_specs())
        await manager.start()
        try:
            missing = await manager.call_tool("notes__get_note", {"note_id": 999})
            self.assertTrue(missing.is_error)
            self.assertIn("не найдена", missing.content)
            self.assertIsNone(missing.structured)

            empty = await manager.call_tool("catalog__search", {"query": "zzzz-no-match"})
            self.assertFalse(empty.is_error)
            self.assertEqual(empty.structured["items"], [])
        finally:
            await manager.aclose()

    async def test_catalog_down_continues_notes_and_filesystem(self):
        manager = MultiMCPManager(default_specs(catalog_command="/definitely/missing/python"))
        await manager.start()
        try:
            self.assertEqual(manager.available_servers, ["notes", "filesystem"])
            self.assertEqual([item.alias for item in manager.unavailable], ["catalog"])
            result = await FlowAgent(manager).run(config.DEMO_QUERY)
            self.assertEqual(result.status, "partial")
            self.assertEqual(result.unavailable, ["catalog"])
            public_names = [step.public_name for step in result.steps]
            self.assertNotIn("catalog__search", public_names)
            self.assertNotIn("catalog__get_item", public_names)
            self.assertIn("filesystem__read_text_file", public_names)
            self.assertIn("notes__search", public_names)
            self.assertIn("notes__summarize", public_names)
            self.assertIn("Laptop battery review", result.final_text)
            # В заметке может упоминаться модель; проверяем, что каталог её не вернул.
            self.assertNotIn("Товар: Laptop Pro 14", result.final_text)
            self.assertIn("catalog", result.final_text)
        finally:
            await manager.aclose()


if __name__ == "__main__":
    unittest.main()
