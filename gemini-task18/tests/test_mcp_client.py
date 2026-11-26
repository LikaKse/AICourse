"""Тесты WeatherMCPClient без реального сервера."""

import unittest
from pathlib import Path

from mcp_client import (
    MCPConnectionError,
    MCPTimeoutError,
    MCPToolError,
    WeatherMCPClient,
)
from tests.fakes import FakeCallResult, FakeSession, FakeTransport, make_tool


def make_client(session, transport=None, **kwargs):
    return WeatherMCPClient(
        db_path=Path("/tmp/test.db"),
        api_base="https://example.test",
        python_bin=Path("/fake/python"),
        server_py=Path("/fake/server.py"),
        transport_factory=lambda params, errlog: transport or FakeTransport(),
        session_factory=lambda read, write, read_timeout_seconds=None: session,
        validate_environment=False,
        **kwargs,
    )


class CommandTest(unittest.TestCase):
    def test_command_contains_db_and_api_env(self):
        client = WeatherMCPClient(
            db_path=Path("/tmp/history.db"),
            api_base="https://example.test",
            python_bin=Path("/fake/python"),
            server_py=Path("/fake/server.py"),
            validate_environment=False,
        )
        params = client.build_server_parameters()
        self.assertEqual(params.command, "/fake/python")
        self.assertEqual(params.args, ["-u", "/fake/server.py"])
        self.assertEqual(params.env["TASK18_DB_PATH"], "/tmp/history.db")
        self.assertEqual(params.env["WEATHER_API_BASE"], "https://example.test")


class LifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def test_connect_and_call_structured(self):
        session = FakeSession(call_result=FakeCallResult(structured={"temperature_c": 1.5}))
        client = make_client(session)
        async with client:
            self.assertTrue(client.connected)
            self.assertEqual(client.protocol_version, "2025-11-25")
            self.assertEqual(len(client.exposed_tools()), 4)
            result = await client.call_tool_structured("fetch_weather", {})
        self.assertFalse(client.connected)
        self.assertTrue(session.exited)
        self.assertEqual(result["temperature_c"], 1.5)

    async def test_error_payload_raises(self):
        session = FakeSession(call_result=FakeCallResult("boom", is_error=True))
        client = make_client(session)
        async with client:
            with self.assertRaises(MCPToolError):
                await client.call_tool_structured("fetch_weather", {})

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
                await client.call_tool("fetch_weather", {})

    async def test_missing_required_tool(self):
        session = FakeSession(tools=[make_tool("fetch_weather")])
        client = make_client(session)
        with self.assertRaises(MCPConnectionError):
            async with client:
                pass

    async def test_hidden_tool(self):
        client = make_client(FakeSession())
        async with client:
            with self.assertRaises(MCPToolError):
                await client.call_tool("delete_db", {})


if __name__ == "__main__":
    unittest.main()
