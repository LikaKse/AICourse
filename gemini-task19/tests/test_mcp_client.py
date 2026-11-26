"""Тесты ChainMCPClient без реального сервера."""

import unittest
from pathlib import Path

from mcp_client import ChainMCPClient, MCPConnectionError, MCPTimeoutError, MCPToolError
from tests.fakes import FakeCallResult, FakeSession, FakeTransport, make_tool


def make_client(session, transport=None, **kwargs):
    return ChainMCPClient(
        api_base="https://example.test",
        output_dir=Path("/tmp/output"),
        python_bin=Path("/fake/python"),
        server_py=Path("/fake/server.py"),
        transport_factory=lambda params, errlog: transport or FakeTransport(),
        session_factory=lambda read, write, read_timeout_seconds=None: session,
        validate_environment=False,
        **kwargs,
    )


class CommandTest(unittest.TestCase):
    def test_env_default_has_no_fail_hook(self):
        client = ChainMCPClient(
            api_base="https://example.test",
            output_dir=Path("/tmp/output"),
            python_bin=Path("/fake/python"),
            server_py=Path("/fake/server.py"),
            validate_environment=False,
        )
        params = client.build_server_parameters()
        self.assertEqual(params.command, "/fake/python")
        self.assertEqual(params.args, ["-u", "/fake/server.py"])
        self.assertEqual(params.env["TASK19_API_BASE"], "https://example.test")
        self.assertEqual(params.env["TASK19_OUTPUT_DIR"], str(Path("/tmp/output").resolve()))
        self.assertNotIn("TASK19_FAIL_SUMMARIZE", params.env)

    def test_fail_hook_only_when_requested(self):
        client = ChainMCPClient(
            api_base="https://example.test",
            output_dir=Path("/tmp/output"),
            fail_summarize=True,
            python_bin=Path("/fake/python"),
            server_py=Path("/fake/server.py"),
            validate_environment=False,
        )
        self.assertEqual(client.build_server_parameters().env["TASK19_FAIL_SUMMARIZE"], "1")


class LifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def test_connect_and_call(self):
        session = FakeSession(call_result=FakeCallResult(structured={"query": "x", "items": []}))
        client = make_client(session)
        async with client:
            self.assertTrue(client.connected)
            self.assertEqual(client.protocol_version, "2025-11-25")
            self.assertEqual(len(client.exposed_tools()), 3)
            result = await client.call_tool("search", {"query": "x"})
        self.assertFalse(client.connected)
        self.assertTrue(session.exited)
        self.assertEqual(result.structured["query"], "x")

    async def test_mcp_is_error_preserved(self):
        session = FakeSession(call_result=FakeCallResult("boom", is_error=True))
        client = make_client(session)
        async with client:
            result = await client.call_tool("summarize", {"items": []})
        self.assertTrue(result.is_error)
        self.assertEqual(result.content, "boom")

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
                await client.call_tool("search", {"query": "x"})

    async def test_missing_tool(self):
        client = make_client(FakeSession(tools=[make_tool("search")]))
        with self.assertRaises(MCPConnectionError):
            async with client:
                pass

    async def test_hidden_tool(self):
        client = make_client(FakeSession())
        async with client:
            with self.assertRaises(MCPToolError):
                await client.call_tool("delete_file", {})


if __name__ == "__main__":
    unittest.main()
