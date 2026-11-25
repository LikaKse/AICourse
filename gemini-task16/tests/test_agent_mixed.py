"""Офлайн-тесты смешанного цикла: fake Gemini + local/MCP dispatch."""

import logging
import unittest

from google.genai import types

import config
from agent import Agent
from tests.fakes import (
    FakeGeminiClient,
    FakeMCPClient,
    api_error,
    call_response,
    make_tool,
    text_response,
)
from tool_registry import ToolRegistry

FILE_TEXT = "quantity=12\nunit_price=25000\nmarkup_percent=10\n"


def build(mcp=None, *, max_iterations=config.MAX_ITERATIONS):
    registry = ToolRegistry(mcp_client=mcp, mcp_tools=(mcp.exposed_tools() if mcp else []))
    return registry


def function_responses(content: types.Content):
    return [part.function_response for part in content.parts if part.function_response]


class MixedLoopTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        logging.getLogger("agent").setLevel(logging.INFO)

    async def test_mcp_read_then_local_calculate(self):
        mcp = FakeMCPClient(payloads={
            "read_text_file": {"source": "mcp", "tool": "read_text_file", "content": FILE_TEXT}
        })
        registry = build(mcp)

        def second_step():
            last = client.requests[-1]["contents"][-1]
            response = function_responses(last)[0].response
            self.assertIn("quantity=12", response["content"])
            return call_response(
                ("calculate", {"expression": "12 * 25000 * 1.1"}, "id-calc"),
                signature=b"SIG-2",
            )

        client = FakeGeminiClient([
            call_response(("read_text_file", {"path": "numbers.txt"}, "id-read"), signature=b"SIG-1"),
            second_step,
            text_response("Итоговая стоимость: 330 000."),
        ])
        agent = Agent(client, declarations=registry.declarations, dispatcher=registry.adispatch)
        result = await agent.arun(config.DEMO_PROMPT)

        self.assertEqual(result.status, "ok", result.text)
        self.assertEqual(result.iterations, 3)
        self.assertEqual([(c.source, c.name) for c in result.tool_calls],
                         [("mcp", "read_text_file"), ("local", "calculate")])
        self.assertEqual(result.tool_calls[1].response["result"]["result"], 330_000)
        self.assertEqual(mcp.calls, [("read_text_file", {"path": "numbers.txt"})])

        # Второй запрос: user → model(function_call + signature) → user(function_response)
        second = client.requests[1]["contents"]
        self.assertEqual([c.role for c in second], ["user", "model", "user"])
        self.assertEqual(second[1].parts[0].thought_signature, b"SIG-1")
        (fr,) = function_responses(second[2])
        self.assertEqual((fr.id, fr.name), ("id-read", "read_text_file"))
        self.assertEqual(fr.response["content"], FILE_TEXT)

        # Третий запрос содержит результат calculate
        (calc_response,) = function_responses(client.requests[2]["contents"][-1])
        self.assertEqual(calc_response.id, "id-calc")
        self.assertEqual(calc_response.response["result"]["result"], 330_000)

    async def test_local_only_calculate(self):
        registry = build(FakeMCPClient())
        client = FakeGeminiClient([
            call_response(("calculate", {"expression": "2+2"})),
            text_response("4"),
        ])
        result = await Agent(client, declarations=registry.declarations, dispatcher=registry.adispatch).arun("2+2?")
        self.assertEqual(result.status, "ok")
        self.assertEqual([c.source for c in result.tool_calls], ["local"])

    async def test_mcp_error_is_returned_to_model(self):
        mcp = FakeMCPClient(payloads={
            "read_text_file": {"error": "Access denied"}
        })
        registry = build(mcp)
        client = FakeGeminiClient([
            call_response(("read_text_file", {"path": "secret.txt"})),
            call_response(("calculate", {"expression": "1+1"})),
            text_response("Файл недоступен, но 2 посчитано."),
        ])
        result = await Agent(client, declarations=registry.declarations, dispatcher=registry.adispatch).arun("x")
        self.assertEqual(result.status, "ok")
        self.assertFalse(result.tool_calls[0].ok)
        self.assertIn("Access denied", result.tool_calls[0].response["error"])
        error_payload = function_responses(client.requests[1]["contents"][-1])[0].response
        self.assertIn("Access denied", error_payload["error"])

    async def test_dispatcher_exception_is_returned_to_model(self):
        async def broken(name, args):
            raise RuntimeError("dispatcher exploded")

        client = FakeGeminiClient([
            call_response(("calculate", {"expression": "1+1"})),
            text_response("Инструмент сломался."),
        ])
        result = await Agent(client, declarations=[], dispatcher=broken).arun("x")
        self.assertEqual(result.status, "ok")
        self.assertFalse(result.tool_calls[0].ok)
        self.assertIn("dispatcher exploded", result.tool_calls[0].response["error"])

    async def test_api_error_rolls_back_history(self):
        registry = build(FakeMCPClient())
        client = FakeGeminiClient([
            text_response("Первый ответ"),
            call_response(("calculate", {"expression": "2+2"})),
            api_error(503),
            text_response("После ошибки"),
        ])
        agent = Agent(client, declarations=registry.declarations, dispatcher=registry.adispatch)
        await agent.arun("первый")
        result = await agent.arun("второй")
        self.assertEqual(result.status, "api_error")
        self.assertEqual(len(agent.history), 2)
        result3 = await agent.arun("третий")
        self.assertEqual(result3.status, "ok")

    async def test_max_iterations(self):
        registry = build(FakeMCPClient())
        client = FakeGeminiClient([call_response(("calculate", {"expression": "1+1"}))] * 5)
        result = await Agent(
            client, declarations=registry.declarations, dispatcher=registry.adispatch, max_iterations=3
        ).arun("loop")
        self.assertEqual(result.status, "max_iterations")
        self.assertEqual(len(result.tool_calls), 3)

    def test_request_preview_contains_both_schemas(self):
        mcp = FakeMCPClient(tools=[make_tool("read_text_file")])
        registry = build(mcp)
        agent = Agent(client=None, declarations=registry.declarations, dispatcher=registry.adispatch)
        preview = agent.request_preview(config.DEMO_PROMPT)
        text = str(preview)
        self.assertIn("calculate", text)
        self.assertIn("read_text_file", text)
        self.assertIn("systemInstruction", text)
        self.assertIn("functionDeclarations", text)


if __name__ == "__main__":
    unittest.main()
