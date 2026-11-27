"""Тесты MultiMCPManager: объединение, маршруты, частичный отказ, закрытие."""

import unittest

from manager import MultiMCPManager, ServerSpec
from tests.fakes import FakeConnection


def spec(alias: str) -> ServerSpec:
    return ServerSpec(alias=alias, command="fake", args=[])


def manager(*aliases: str) -> MultiMCPManager:
    return MultiMCPManager(
        [spec(alias) for alias in aliases],
        connection_factory=FakeConnection,
    )


class ManagerTest(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        FakeConnection.failing_aliases = set()

    async def test_three_servers_and_union_of_tools(self):
        m = manager("catalog", "notes", "filesystem")
        await m.start()
        self.assertEqual(m.available_servers, ["catalog", "notes", "filesystem"])
        self.assertEqual(
            {tool.public_name for tool in m.tools},
            {
                "catalog__search", "catalog__get_item",
                "notes__search", "notes__get_note", "notes__summarize",
                "filesystem__read_text_file",
            },
        )
        await m.aclose()

    async def test_duplicate_original_names_are_distinct_and_routed(self):
        m = manager("catalog", "notes", "filesystem")
        await m.start()
        self.assertIn("catalog__search", m.routes)
        self.assertIn("notes__search", m.routes)
        self.assertEqual(m.routes["catalog__search"].server_alias, "catalog")
        self.assertEqual(m.routes["notes__search"].server_alias, "notes")
        result1 = await m.call_tool("catalog__search", {"query": "laptop"})
        result2 = await m.call_tool("notes__search", {"query": "laptop"})
        self.assertEqual(result1.server_alias, "catalog")
        self.assertEqual(result2.server_alias, "notes")
        await m.aclose()

    async def test_partial_start_failure_keeps_others_and_closes(self):
        FakeConnection.failing_aliases = {"catalog"}
        m = manager("catalog", "notes", "filesystem")
        await m.start()
        self.assertEqual(m.available_servers, ["notes", "filesystem"])
        self.assertEqual([item.alias for item in m.unavailable], ["catalog"])
        result = await m.call_tool("catalog__search", {"query": "laptop"})
        self.assertTrue(result.is_error)
        self.assertIn("catalog", result.content)
        notes = await m.call_tool("notes__search", {"query": "laptop"})
        self.assertFalse(notes.is_error)
        await m.aclose()
        self.assertTrue(all(connection.closed for connection in m.connections))

    async def test_unknown_tool_is_error(self):
        m = manager("notes")
        await m.start()
        result = await m.call_tool("catalog__search", {"query": "x"})
        self.assertTrue(result.is_error)
        self.assertIn("Неизвестный инструмент", result.content)
        await m.aclose()

    async def test_duplicate_alias_rejected(self):
        with self.assertRaises(ValueError):
            MultiMCPManager([spec("notes"), spec("notes")], connection_factory=FakeConnection)

    async def test_close_is_idempotent(self):
        m = manager("notes")
        await m.start()
        await m.aclose()
        await m.aclose()
        self.assertTrue(m.connections[0].closed)


if __name__ == "__main__":
    unittest.main()
