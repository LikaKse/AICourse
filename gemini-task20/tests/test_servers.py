"""Тесты собственных MCP-серверов catalog и notes."""

import unittest

from mcp import Client

from servers import catalog_server, notes_server


class CatalogServerTest(unittest.IsolatedAsyncioTestCase):
    async def test_tools_and_schemas(self):
        tools = await catalog_server.mcp.list_tools()
        self.assertEqual({tool.name for tool in tools}, {"search", "get_item"})
        by_name = {tool.name: tool for tool in tools}
        self.assertEqual(by_name["search"].input_schema["required"], ["query"])
        self.assertEqual(by_name["get_item"].input_schema["required"], ["item_id"])

    async def test_search_and_get_item(self):
        async with Client(catalog_server.mcp) as client:
            found = await client.call_tool("search", {"query": "laptop"})
            self.assertFalse(found.is_error)
            self.assertEqual(len(found.structured_content["items"]), 2)
            item = await client.call_tool("get_item", {"item_id": 1})
            self.assertFalse(item.is_error)
            self.assertEqual(item.structured_content["name"], "Laptop Pro 14")

    async def test_empty_search_is_valid(self):
        async with Client(catalog_server.mcp) as client:
            found = await client.call_tool("search", {"query": "zzzz-no-match"})
        self.assertFalse(found.is_error)
        self.assertEqual(found.structured_content["items"], [])


class NotesServerTest(unittest.IsolatedAsyncioTestCase):
    async def test_tools_and_duplicate_search_name(self):
        tools = await notes_server.mcp.list_tools()
        self.assertEqual({tool.name for tool in tools}, {"search", "get_note", "summarize"})
        self.assertIn("search", {tool.name for tool in tools})

    async def test_search_get_and_summarize(self):
        async with Client(notes_server.mcp) as client:
            found = await client.call_tool("search", {"query": "laptop"})
            notes = found.structured_content["items"]
            self.assertEqual(len(notes), 1)
            note = await client.call_tool("get_note", {"note_id": 1})
            self.assertFalse(note.is_error)
            summary = await client.call_tool("summarize", {"note_ids": [1]})
            self.assertFalse(summary.is_error)
            self.assertIn("Laptop battery review", summary.structured_content["summary"])

    async def test_empty_search_is_valid(self):
        async with Client(notes_server.mcp) as client:
            found = await client.call_tool("search", {"query": "zzzz"})
        self.assertFalse(found.is_error)
        self.assertEqual(found.structured_content["items"], [])


if __name__ == "__main__":
    unittest.main()
