"""Фейковые MCP-объекты и данные для тестов Дня 19."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from mcp.types import TextContent

from mcp_client import MCPCallResult, MCPToolInfo

POSTS = [
    {
        "userId": 1,
        "id": 2,
        "title": "qui est esse",
        "body": "est rerum tempore vitae sequi sint nihil reprehenderit dolor beatae ea dolores",
    },
    {
        "userId": 1,
        "id": 3,
        "title": "ea molestias quasi exercitationem",
        "body": "et iusto sed quo iure voluptatem occaecati",
    },
]

TOOLS = ("search", "summarize", "save_to_file")


def make_tool(name: str) -> MCPToolInfo:
    schemas = {
        "search": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        "summarize": {
            "type": "object",
            "properties": {"items": {"type": "array"}},
            "required": ["items"],
        },
        "save_to_file": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "filename": {"type": "string"},
            },
            "required": ["content", "filename"],
        },
    }
    return MCPToolInfo(name=name, description=f"Fake {name}", input_schema=schemas[name])


class FakeCallResult:
    def __init__(self, text: str = "", *, is_error: bool = False, structured=None):
        self.content = [TextContent(type="text", text=text)]
        self.structured_content = structured
        self.is_error = is_error


class FakeSession:
    def __init__(
        self,
        *,
        tools=None,
        initialize_delay: float = 0,
        call_delay: float = 0,
        call_result: FakeCallResult | None = None,
        protocol_version: str = "2025-11-25",
    ) -> None:
        self.tools = tools or [make_tool(name) for name in TOOLS]
        self.initialize_delay = initialize_delay
        self.call_delay = call_delay
        self.call_result = call_result or FakeCallResult(structured={"ok": True})
        self.protocol_version = protocol_version
        self.exited = False
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True
        return False

    async def initialize(self):
        if self.initialize_delay:
            await asyncio.sleep(self.initialize_delay)
        return SimpleNamespace(
            protocol_version=self.protocol_version,
            server_info=SimpleNamespace(name="content-chain", version="0.1.0"),
        )

    async def list_tools(self):
        return SimpleNamespace(tools=self.tools)

    async def call_tool(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        if self.call_delay:
            await asyncio.sleep(self.call_delay)
        return self.call_result


class FakeTransport:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.exited = False

    async def __aenter__(self):
        if self.fail:
            raise OSError("cannot spawn")
        return object(), object()

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True
        return False


class FakeChainClient:
    """Подмена ChainMCPClient для тестов ChainAgent."""

    def __init__(self, responses: dict):
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, dict(arguments)))
        response = self.responses[name]
        if callable(response):
            response = response(arguments)
        if isinstance(response, Exception):
            raise response
        return response


def ok(structured) -> MCPCallResult:
    return MCPCallResult(tool="", content="", structured=structured, is_error=False)


def error(message: str) -> MCPCallResult:
    return MCPCallResult(tool="", content=message, structured=None, is_error=True)
