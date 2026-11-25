"""Тесты MCP-клиента без реального сервера: lifecycle, команда, ошибки, таймауты."""

import tempfile
import unittest
from pathlib import Path

from mcp_client import (
    MCPConnectionError,
    MCPFilesystemClient,
    MCPTimeoutError,
    MCPToolError,
)
from tests.fakes import FakeCallResult, FakeSession, FakeTransport, make_tool


def make_client(session, transport=None, workspace=None, **kwargs):
    transport = transport or FakeTransport()
    verify_workspace = kwargs.pop("verify_workspace", False)
    return MCPFilesystemClient(
        workspace=workspace,
        node_bin=Path("/fake/node"),
        server_js=Path("/fake/server.js"),
        transport_factory=lambda params, errlog: transport,
        session_factory=lambda read, write, read_timeout_seconds=None: session,
        validate_environment=False,
        verify_workspace=verify_workspace,
        **kwargs,
    )


class ServerCommandTest(unittest.TestCase):
    def test_command_is_read_only_and_confined(self):
        client = MCPFilesystemClient(
            workspace=Path("/tmp/demo_workspace"),
            node_bin=Path("/tmp/node"),
            server_js=Path("/tmp/server.js"),
            validate_environment=False,
            verify_workspace=False,
        )
        params = client.build_server_parameters()
        self.assertEqual(params.command, "/tmp/node")
        self.assertEqual(params.args[0], "/tmp/server.js")
        self.assertIn("--read-only", params.args)
        self.assertIn("--confine-roots", params.args)
        self.assertEqual(params.args[-1], str(client.workspace))
        self.assertEqual(params.cwd, str(Path(__file__).resolve().parent.parent))


class LifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def test_connect_and_list_tools(self):
        session = FakeSession(tools=[make_tool("read_text_file"), make_tool("list_allowed_directories")])
        client = make_client(session)
        async with client as connected:
            self.assertTrue(connected.connected)
            self.assertEqual(connected.protocol_version, "2025-11-25")
            self.assertIn("fake-filesystem", connected.server_info)
            self.assertEqual([t.name for t in connected.exposed_tools()], ["read_text_file", "list_allowed_directories"])
            self.assertTrue(session.entered)
        self.assertFalse(client.connected)
        self.assertTrue(session.exited)

    async def test_session_and_transport_are_closed_on_connect_error(self):
        session = FakeSession()
        transport = FakeTransport(fail=True)
        client = make_client(session, transport=transport)
        with self.assertRaises(MCPConnectionError):
            async with client:
                pass
        self.assertTrue(client._stack is None)

    async def test_initialize_timeout(self):
        session = FakeSession(initialize_delay=0.1)
        client = make_client(session, connect_timeout=0.01)
        with self.assertRaises(MCPTimeoutError):
            async with client:
                pass

    async def test_call_timeout(self):
        session = FakeSession(call_delay=0.1)
        client = make_client(session, call_timeout=0.01)
        async with client:
            with self.assertRaises(MCPTimeoutError):
                await client.call_tool("read_text_file", {"path": "numbers.txt"})


class ToolCallTest(unittest.IsolatedAsyncioTestCase):
    async def test_read_text_file_returns_mcp_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            (workspace / "numbers.txt").write_text("quantity=12\n", encoding="utf-8")
            session = FakeSession(call_results={
                "read_text_file": FakeCallResult(
                    "quantity=12\n",
                    structured={"content": "quantity=12\n"},
                )
            })
            client = make_client(session, workspace=workspace)
            async with client:
                payload = await client.call_tool_for_model("read_text_file", {"path": "numbers.txt"})
            self.assertEqual(payload["source"], "mcp")
            self.assertEqual(payload["content"], "quantity=12\n")
            self.assertNotIn("structured", payload)  # duplicate content не дублируется
            self.assertEqual(session.calls[0][0], "read_text_file")
            self.assertEqual(session.calls[0][1]["path"], str(workspace / "numbers.txt"))

    async def test_mcp_is_error_becomes_error_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            (workspace / "numbers.txt").write_text("x", encoding="utf-8")
            session = FakeSession(call_results={
                "read_text_file": FakeCallResult("Access denied", is_error=True)
            })
            client = make_client(session, workspace=workspace)
            async with client:
                payload = await client.call_tool_for_model("read_text_file", {"path": "numbers.txt"})
            self.assertEqual(payload, {"error": "Access denied"})

    async def test_path_outside_workspace_rejected_before_mcp(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            session = FakeSession()
            client = make_client(session, workspace=workspace)
            async with client:
                with self.assertRaises(MCPToolError):
                    await client.call_tool("read_text_file", {"path": "/etc/hosts"})
            self.assertEqual(session.calls, [])

    async def test_hidden_tool_rejected(self):
        session = FakeSession(tools=[make_tool("read_text_file")])
        client = make_client(session)
        async with client:
            with self.assertRaises(MCPToolError):
                await client.call_tool("write_file", {"path": "x", "content": "y"})


class EnvironmentAndWorkspaceTest(unittest.TestCase):
    def test_missing_node_is_clear_error(self):
        client = MCPFilesystemClient(
            workspace=Path(__file__).resolve().parent.parent / "demo_workspace",
            node_bin=Path("/definitely/missing/node"),
            server_js=Path("/definitely/missing/server.js"),
            validate_environment=True,
            verify_workspace=False,
        )
        with self.assertRaises(MCPConnectionError) as ctx:
            client._check_environment()
        self.assertIn("setup_mcp_env.sh", str(ctx.exception))

    def test_node_22_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            node = root / "node"
            node.write_text("#!/bin/sh\necho v22.13.1\n", encoding="utf-8")
            node.chmod(0o755)
            server = root / "server.js"
            server.write_text("// fake", encoding="utf-8")
            workspace = root / "workspace"
            workspace.mkdir()
            client = MCPFilesystemClient(
                workspace=workspace,
                node_bin=node,
                server_js=server,
                validate_environment=True,
                verify_workspace=False,
            )
            with self.assertRaises(MCPConnectionError) as ctx:
                client._check_environment()
            self.assertIn("Node.js ≥ 24", str(ctx.exception))

    def test_node_24_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            node = root / "node"
            node.write_text("#!/bin/sh\necho v24.22.3\n", encoding="utf-8")
            node.chmod(0o755)
            server = root / "server.js"
            server.write_text("// fake", encoding="utf-8")
            workspace = root / "workspace"
            workspace.mkdir()
            client = MCPFilesystemClient(
                workspace=workspace,
                node_bin=node,
                server_js=server,
                validate_environment=True,
                verify_workspace=False,
            )
            client._check_environment()  # не должно бросить


class WorkspaceVerificationTest(unittest.IsolatedAsyncioTestCase):
    async def test_allowed_directories_must_match_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            session = FakeSession(allowed_text=f"Allowed directories:\n{workspace}")
            client = make_client(session, workspace=workspace, verify_workspace=True)
            async with client:
                self.assertTrue(client.connected)

    async def test_other_allowed_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve() / "workspace"
            workspace.mkdir()
            session = FakeSession(allowed_text="Allowed directories:\n/tmp/other")
            client = make_client(session, workspace=workspace, verify_workspace=True)
            with self.assertRaises(MCPConnectionError):
                async with client:
                    pass


if __name__ == "__main__":
    unittest.main()
