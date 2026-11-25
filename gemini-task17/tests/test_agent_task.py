"""Офлайн-тест полного цикла агента с fake Gemini и fake MCP."""

import unittest

from agent import Agent
from tests.fakes import FakeGeminiClient, FakeMCPClient, api_error, call_response, text_response
from tool_registry import ToolRegistry


def build_registry(mcp):
    return ToolRegistry(mcp_client=mcp, mcp_tools=mcp.exposed_tools())


class AgentTaskTest(unittest.IsolatedAsyncioTestCase):
    async def test_agent_calls_mcp_and_uses_result(self):
        mcp = FakeMCPClient()
        registry = build_registry(mcp)

        def second_step():
            last = client.requests[-1]["contents"][-1]
            response = [
                part.function_response
                for part in last.parts
                if part.function_response
            ][0]
            self.assertEqual(response.name, "get_task")
            self.assertEqual(response.response["result"]["id"], 5)
            return text_response(
                "Задача 5: «laboriosam mollitia et enim quasi adipisci quia provident illum». "
                "Статус: не выполнена."
            )

        client = FakeGeminiClient([
            call_response(("get_task", {"task_id": 5}, "task-call"), signature=b"SIG-1"),
            second_step,
        ])
        agent = Agent(client, declarations=registry.declarations, dispatcher=registry.adispatch)
        result = await agent.arun("Что за задача с id 5?")

        self.assertEqual(result.status, "ok", result.text)
        self.assertEqual([(c.source, c.name) for c in result.tool_calls], [("mcp", "get_task")])
        self.assertEqual(result.tool_calls[0].args, {"task_id": 5})
        self.assertEqual(result.tool_calls[0].response["result"]["title"],
                         "laboriosam mollitia et enim quasi adipisci quia provident illum")
        self.assertIn("не выполнена", result.text)
        self.assertEqual(mcp.calls, [("get_task", {"task_id": 5})])

        second = client.requests[1]["contents"]
        self.assertEqual([c.role for c in second], ["user", "model", "user"])
        self.assertEqual(second[1].parts[0].thought_signature, b"SIG-1")
        function_response = [
            part.function_response for part in second[2].parts if part.function_response
        ][0]
        self.assertEqual(function_response.id, "task-call")
        self.assertEqual(function_response.response["result"]["id"], 5)

    async def test_mcp_error_is_returned_to_model(self):
        mcp = FakeMCPClient(payload={"error": "Задача с id=999 не найдена."})
        registry = build_registry(mcp)
        client = FakeGeminiClient([
            call_response(("get_task", {"task_id": 999})),
            text_response("Задача с id 999 не найдена."),
        ])
        result = await Agent(client, declarations=registry.declarations, dispatcher=registry.adispatch).arun(
            "Что за задача 999?"
        )
        self.assertEqual(result.status, "ok")
        self.assertFalse(result.tool_calls[0].ok)
        self.assertIn("не найдена", result.tool_calls[0].response["error"])

    async def test_api_error_rolls_back(self):
        mcp = FakeMCPClient()
        registry = build_registry(mcp)
        client = FakeGeminiClient([
            text_response("Первый"),
            api_error(503),
            text_response("После ошибки"),
        ])
        agent = Agent(client, declarations=registry.declarations, dispatcher=registry.adispatch)
        await agent.arun("первый")
        failed = await agent.arun("второй")
        self.assertEqual(failed.status, "api_error")
        self.assertEqual(len(agent.history), 2)
        self.assertEqual((await agent.arun("третий")).status, "ok")

    def test_request_preview_contains_local_and_mcp(self):
        registry = build_registry(FakeMCPClient())
        agent = Agent(client=None, declarations=registry.declarations, dispatcher=registry.adispatch)
        preview = str(agent.request_preview("Что за задача 5?"))
        self.assertIn("calculate", preview)
        self.assertIn("get_task", preview)
        self.assertIn("functionDeclarations", preview)


if __name__ == "__main__":
    unittest.main()
