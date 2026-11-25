"""Тесты TaskMCPClient без реального сервера."""

import unittest
from pathlib import Path

from mcp_client import (
    MCPConnectionError,
    MCPTimeoutError,
    MCPToolError,
    TaskMCPClient,
)
from tests.fakes import FakeCallResult, FakeSession, FakeTransport, make_tool


def make_client(session, transport=None, **kwargs):
    transport = transport or FakeTransport()
    return TaskMCPClient(
        python_bin=Path("/fake/python"),
        server_py=Path("/fake/server.py"),
        transport_factory=lambda params, errlog: transport,
        session_factory=lambda read, write, read_timeout_seconds=None: session,
        validate_environment=False,
        **kwargs,
    )


class ServerCommandTest(unittest.TestCase):
    def test_command_is_python_stdio(self):
        client = TaskMCPClient(
            api_base="https://example.test",
            python_bin=Path("/fake/python"),
            server_py=Path("/fake/server.py"),
            validate_environment=False,
        )
        params = client.build_server_parameters()
        self.assertEqual(params.command, "/fake/python")
        self.assertEqual(params.args, ["-u", "/fake/server.py"])
        self.assertEqual(params.env["TASKS_API_BASE"], "https://example.test")
        self.assertEqual(params.cwd, str(Path(__file__).resolve().parent.parent))


class LifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def test_connect_list_and_call(self):
        session = FakeSession(call_result=FakeCallResult(structured={
            "userId": 1, "id": 5,
            "title": "laboriosam mollitia et enim quasi adipisci quia provident illum",
            "completed": False,
        }))
        client = make_client(session)
        async with client:
            self.assertTrue(client.connected)
            self.assertEqual(client.protocol_version, "2025-11-25")
            self.assertIn("jsonplaceholder-tasks", client.server_info)
            self.assertEqual([t.name for t in client.exposed_tools()], ["get_task"])
            payload = await client.call_tool_for_model("get_task", {"task_id": 5})
        self.assertFalse(client.connected)
        self.assertTrue(session.exited)
        self.assertEqual(payload["source"], "mcp")
        self.assertEqual(payload["result"]["id"], 5)

    async def test_connection_error(self):
        client = make_client(FakeSession(), transport=FakeTransport(fail=True))
        with self.assertRaises(MCPConnectionError):
            async with client:
                pass

    async def test_initialize_timeout(self):
        client = make_client(FakeSession(initialize_delay=0.1), connect_timeout=0.01)
        with self.assertRaises(MCPTimeoutError):
            async with client:
                pass

    async def test_call_timeout(self):
        client = make_client(FakeSession(call_delay=0.1), call_timeout=0.01)
        async with client:
            with self.assertRaises(MCPTimeoutError):
                await client.call_tool("get_task", {"task_id": 5})

    async def test_tool_error_payload(self):
        session = FakeSession(call_result=FakeCallResult("Задача не найдена", is_error=True))
        client = make_client(session)
        async with client:
            payload = await client.call_tool_for_model("get_task", {"task_id": 999})
        self.assertEqual(payload, {"error": "Задача не найдена"})

    async def test_hidden_tool_rejected(self):
        session = FakeSession(tools=[make_tool("get_task")])
        client = make_client(session)
        async with client:
            with self.assertRaises(MCPToolError):
                await client.call_tool("write_file", {})

    async def test_missing_required_tool(self):
        session = FakeSession(tools=[make_tool("other")])
        client = make_client(session)
        with self.assertRaises(MCPConnectionError):
            async with client:
                pass

    async def test_missing_python_environment(self):
        client = TaskMCPClient(
            python_bin=Path("/definitely/missing/python"),
            server_py=Path("/definitely/missing/server.py"),
            validate_environment=True,
        )
        with self.assertRaises(MCPConnectionError):
            async with client:
                pass


if __name__ == "__main__":
    unittest.main()
