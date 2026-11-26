"""Тесты детерминированного ChainAgent."""

import unittest

from chain_agent import ChainAgent
from tests.fakes import FakeChainClient, POSTS, error, ok

SUMMARY_TEXT = "Найдено публикаций: 2. Заголовки: «qui est esse»."


def success_responses():
    return {
        "search": ok({"query": "qui", "items": POSTS}),
        "summarize": ok({"source_count": 2, "summary": SUMMARY_TEXT}),
        "save_to_file": ok({"path": "output/summary.txt", "bytes_written": len(SUMMARY_TEXT.encode())}),
    }


class ChainSuccessTest(unittest.IsolatedAsyncioTestCase):
    async def test_exact_data_transfer(self):
        client = FakeChainClient(success_responses())
        result = await ChainAgent(client).run("qui", "summary.txt")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.saved_path, "output/summary.txt")
        self.assertEqual([step.name for step in result.steps], ["search", "summarize", "save_to_file"])
        self.assertEqual([step.status for step in result.steps], ["ok", "ok", "ok"])

        summarize_args = client.calls[1][1]
        self.assertEqual(summarize_args["items"], POSTS)
        save_args = client.calls[2][1]
        self.assertEqual(save_args["content"], SUMMARY_TEXT)
        self.assertEqual(save_args["filename"], "summary.txt")

    async def test_no_extra_calls(self):
        client = FakeChainClient(success_responses())
        await ChainAgent(client).run("qui", "summary.txt")
        self.assertEqual([name for name, _ in client.calls], ["search", "summarize", "save_to_file"])


class ChainStopTest(unittest.IsolatedAsyncioTestCase):
    async def test_summarize_error_stops_before_save(self):
        responses = success_responses()
        responses["summarize"] = error("summarize упал")
        client = FakeChainClient(responses)
        result = await ChainAgent(client).run("qui", "summary.txt")
        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.stopped_at, "summarize")
        self.assertIn("summarize упал", result.reason)
        self.assertEqual(result.completed_steps, ["search"])
        self.assertEqual([name for name, _ in client.calls], ["search", "summarize"])

    async def test_empty_search_stops_before_summarize(self):
        responses = success_responses()
        responses["search"] = ok({"query": "none", "items": []})
        client = FakeChainClient(responses)
        result = await ChainAgent(client).run("none", "summary.txt")
        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.stopped_at, "search")
        self.assertIn("пустой", result.reason)
        self.assertEqual(result.completed_steps, [])
        self.assertEqual([name for name, _ in client.calls], ["search"])

    async def test_invalid_search_payload_stops(self):
        responses = success_responses()
        responses["search"] = ok({"query": "qui", "items": "not-a-list"})
        client = FakeChainClient(responses)
        result = await ChainAgent(client).run("qui", "summary.txt")
        self.assertEqual(result.stopped_at, "search")
        self.assertEqual(result.steps[0].status, "invalid")
        self.assertEqual([name for name, _ in client.calls], ["search"])

    async def test_search_item_missing_required_field_stops(self):
        responses = success_responses()
        responses["search"] = ok({
            "query": "qui",
            "items": [{"id": 2, "title": "t", "body": "b"}],  # нет userId
        })
        client = FakeChainClient(responses)
        result = await ChainAgent(client).run("qui", "summary.txt")
        self.assertEqual(result.stopped_at, "search")
        self.assertEqual(result.steps[0].status, "invalid")
        self.assertEqual([name for name, _ in client.calls], ["search"])

    async def test_invalid_summarize_payload_stops(self):
        responses = success_responses()
        responses["summarize"] = ok({"source_count": 2, "summary": 123})
        client = FakeChainClient(responses)
        result = await ChainAgent(client).run("qui", "summary.txt")
        self.assertEqual(result.stopped_at, "summarize")
        self.assertEqual(result.steps[1].status, "invalid")
        self.assertNotIn("save_to_file", [name for name, _ in client.calls])

    async def test_summarize_missing_field_stops(self):
        responses = success_responses()
        responses["summarize"] = ok({"source_count": 2})
        client = FakeChainClient(responses)
        result = await ChainAgent(client).run("qui", "summary.txt")
        self.assertEqual(result.stopped_at, "summarize")
        self.assertEqual(result.steps[1].status, "invalid")
        self.assertNotIn("save_to_file", [name for name, _ in client.calls])

    async def test_summary_count_mismatch_stops(self):
        responses = success_responses()
        responses["summarize"] = ok({"source_count": 1, "summary": SUMMARY_TEXT})
        client = FakeChainClient(responses)
        result = await ChainAgent(client).run("qui", "summary.txt")
        self.assertEqual(result.stopped_at, "summarize")
        self.assertIn("source_count=1", result.reason)
        self.assertNotIn("save_to_file", [name for name, _ in client.calls])

    async def test_empty_summary_stops(self):
        responses = success_responses()
        responses["summarize"] = ok({"source_count": 2, "summary": "   "})
        client = FakeChainClient(responses)
        result = await ChainAgent(client).run("qui", "summary.txt")
        self.assertEqual(result.stopped_at, "summarize")
        self.assertEqual(result.steps[1].status, "empty")
        self.assertNotIn("save_to_file", [name for name, _ in client.calls])

    async def test_invalid_save_payload_stops(self):
        responses = success_responses()
        responses["save_to_file"] = ok({"path": "output/summary.txt", "bytes_written": 1})
        client = FakeChainClient(responses)
        result = await ChainAgent(client).run("qui", "summary.txt")
        self.assertEqual(result.stopped_at, "save_to_file")
        self.assertEqual(result.steps[2].status, "invalid")
        self.assertEqual(result.completed_steps, ["search", "summarize"])

    async def test_save_wrong_type_stops(self):
        responses = success_responses()
        responses["save_to_file"] = ok({"path": "output/summary.txt", "bytes_written": "4"})
        client = FakeChainClient(responses)
        result = await ChainAgent(client).run("qui", "summary.txt")
        self.assertEqual(result.stopped_at, "save_to_file")
        self.assertEqual(result.steps[2].status, "invalid")

    async def test_mcp_exception_on_summarize_stops(self):
        responses = success_responses()
        responses["summarize"] = RuntimeError("connection lost")
        client = FakeChainClient(responses)
        result = await ChainAgent(client).run("qui", "summary.txt")
        self.assertEqual(result.stopped_at, "summarize")
        self.assertIn("connection lost", result.reason)
        self.assertNotIn("save_to_file", [name for name, _ in client.calls])


if __name__ == "__main__":
    unittest.main()
