"""Тесты state-machine FlowAgent."""

import unittest

from flow_agent import FlowAgent, classify_intent, search_term
from manager import ToolCallResult
from tests.fakes import FakeManager, item, note, success_responses


class IntentTest(unittest.TestCase):
    def test_full_query(self):
        intent = classify_intent(
            "Найди ноутбук в каталоге, связанную заметку и контекст закупки"
        )
        self.assertTrue(intent.context)
        self.assertTrue(intent.catalog)
        self.assertTrue(intent.notes)

    def test_only_catalog(self):
        intent = classify_intent("Найди товар ноутбук в каталоге")
        self.assertFalse(intent.context)
        self.assertTrue(intent.catalog)
        self.assertFalse(intent.notes)

    def test_only_notes(self):
        intent = classify_intent("Найди заметку про laptop")
        self.assertFalse(intent.context)
        self.assertFalse(intent.catalog)
        self.assertTrue(intent.notes)

    def test_search_term(self):
        self.assertEqual(search_term("Найди ноутбук"), "laptop")
        self.assertEqual(search_term("Найди заметку про laptop"), "laptop")
        self.assertEqual(search_term("monitor"), "monitor")


class FlowSuccessTest(unittest.IsolatedAsyncioTestCase):
    async def test_full_flow_uses_three_servers_in_order(self):
        manager = FakeManager(responses=success_responses())
        result = await FlowAgent(manager).run(
            "Найди ноутбук в каталоге, связанную заметку и контекст закупки, затем сводку."
        )
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
        self.assertIn("Контекст", result.final_text)

    async def test_only_catalog_does_not_call_notes_or_filesystem(self):
        manager = FakeManager(responses=success_responses())
        result = await FlowAgent(manager).run("Найди товар ноутбук в каталоге")
        self.assertEqual(result.status, "ok")
        self.assertEqual([name for name, _ in manager.calls], [
            "catalog__search", "catalog__get_item",
        ])
        self.assertNotIn("filesystem__read_text_file", [name for name, _ in manager.calls])

    async def test_only_notes_does_not_call_catalog_or_filesystem(self):
        manager = FakeManager(responses=success_responses())
        result = await FlowAgent(manager).run("Найди заметку про laptop")
        self.assertEqual(result.status, "ok")
        self.assertEqual([name for name, _ in manager.calls], [
            "notes__search", "notes__get_note", "notes__summarize",
        ])

    async def test_empty_file_is_valid_result(self):
        responses = success_responses()
        responses["filesystem__read_text_file"] = {"content": ""}
        manager = FakeManager(responses=responses)
        result = await FlowAgent(manager).run("Прочитай контекст из файла")
        self.assertEqual(result.status, "ok")
        self.assertIn("(файл пуст)", result.final_text)

    async def test_exact_data_transfer(self):
        manager = FakeManager(responses=success_responses())
        await FlowAgent(manager).run("Найди ноутбук и заметку")
        calls = dict(manager.calls)
        self.assertEqual(calls["catalog__get_item"]["item_id"], 1)
        self.assertEqual(calls["notes__get_note"]["note_id"], 1)
        self.assertEqual(calls["notes__summarize"]["note_ids"], [1])


class FlowFailureTest(unittest.IsolatedAsyncioTestCase):
    async def test_catalog_unavailable_continues_notes_and_filesystem(self):
        manager = FakeManager(
            aliases=("notes", "filesystem"),
            responses=success_responses(),
            unavailable=("catalog",),
        )
        result = await FlowAgent(manager).run(
            "Найди ноутбук в каталоге, связанную заметку и контекст закупки"
        )
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.unavailable, ["catalog"])
        self.assertEqual([name for name, _ in manager.calls], [
            "filesystem__read_text_file", "notes__search", "notes__get_note", "notes__summarize",
        ])
        self.assertIn("Laptop battery review", result.final_text)
        self.assertNotIn("Laptop Pro 14", result.final_text)
        self.assertIn("catalog", result.final_text)

    async def test_catalog_is_error_stops_catalog_but_continues_notes(self):
        responses = success_responses()
        responses["catalog__search"] = ToolCallResult(
            public_name="catalog__search",
            server_alias="catalog",
            original_name="search",
            content="catalog search failed",
            structured=None,
            is_error=True,
        )
        manager = FakeManager(responses=responses)
        result = await FlowAgent(manager).run("Найди ноутбук и заметку")
        calls = [name for name, _ in manager.calls]
        self.assertNotIn("catalog__get_item", calls)
        self.assertIn("notes__search", calls)
        self.assertIn("catalog search failed", result.final_text)
        self.assertNotIn("Laptop Pro 14", result.final_text)

    async def test_empty_catalog_skips_get_item_but_continues_notes(self):
        responses = success_responses()
        responses["catalog__search"] = {"query": "laptop", "items": []}
        manager = FakeManager(responses=responses)
        result = await FlowAgent(manager).run("Найди ноутбук и заметку")
        calls = [name for name, _ in manager.calls]
        self.assertIn("catalog__search", calls)
        self.assertNotIn("catalog__get_item", calls)
        self.assertIn("notes__search", calls)
        self.assertIn("товары не найдены", result.final_text)

    async def test_empty_notes_skips_note_and_summary_but_continues_catalog(self):
        responses = success_responses()
        responses["notes__search"] = {"query": "laptop", "items": []}
        manager = FakeManager(responses=responses)
        result = await FlowAgent(manager).run("Найди ноутбук и заметку")
        calls = [name for name, _ in manager.calls]
        self.assertIn("catalog__get_item", calls)
        self.assertNotIn("notes__get_note", calls)
        self.assertNotIn("notes__summarize", calls)
        self.assertIn("Laptop Pro 14", result.final_text)

    async def test_summarize_error_does_not_continue(self):
        responses = success_responses()
        responses["notes__summarize"] = ToolCallResult(
            public_name="notes__summarize",
            server_alias="notes",
            original_name="summarize",
            content="summarize failed",
            structured=None,
            is_error=True,
        )
        manager = FakeManager(responses=responses)
        result = await FlowAgent(manager).run("Найди ноутбук и заметку")
        self.assertEqual(result.status, "partial")
        self.assertIn("summarize failed", result.final_text)
        self.assertNotIn("Сводка:", result.final_text)

    async def test_invalid_result_stops_dependent_branch(self):
        responses = success_responses()
        responses["catalog__search"] = {"query": "laptop", "items": [{"id": 1}]}
        manager = FakeManager(responses=responses)
        result = await FlowAgent(manager).run("Найди ноутбук и заметку")
        calls = [name for name, _ in manager.calls]
        self.assertNotIn("catalog__get_item", calls)
        self.assertIn("notes__search", calls)
        self.assertIn("некорректный ответ", result.final_text)

    async def test_no_data_is_failed_and_not_invented(self):
        manager = FakeManager(aliases=(), responses={}, unavailable=("catalog", "notes", "filesystem"))
        result = await FlowAgent(manager).run("Найди ноутбук и заметку")
        self.assertEqual(result.status, "failed")
        self.assertIn("Нет доступных данных", result.final_text)
        self.assertNotIn("Laptop Pro", result.final_text)


if __name__ == "__main__":
    unittest.main()
