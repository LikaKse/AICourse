"""Фейковые MCP-транспорт/сессия для тестов."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from mcp.types import TextContent

from mcp_client import MCPToolInfo

TOOLS = ("fetch_weather", "save_observation", "read_observations", "build_report")


def make_tool(name: str) -> MCPToolInfo:
    schemas = {
        "fetch_weather": {"type": "object", "properties": {}},
        "save_observation": {
            "type": "object",
            "properties": {
                "observed_at": {"type": "string"},
                "temperature_c": {"type": "number"},
                "wind_speed_kmh": {"type": "number"},
            },
            "required": ["observed_at", "temperature_c", "wind_speed_kmh"],
        },
        "read_observations": {
            "type": "object",
            "properties": {
                "period_start": {"type": "string"},
                "period_end": {"type": "string"},
            },
            "required": ["period_start", "period_end"],
        },
        "build_report": {
            "type": "object",
            "properties": {
                "period_start": {"type": "string"},
                "period_end": {"type": "string"},
            },
            "required": ["period_start", "period_end"],
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
        self.entered = False
        self.exited = False
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True
        return False

    async def initialize(self):
        if self.initialize_delay:
            await asyncio.sleep(self.initialize_delay)
        return SimpleNamespace(
            protocol_version=self.protocol_version,
            server_info=SimpleNamespace(name="weather-history", version="0.1.0"),
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
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        if self.fail:
            raise OSError("cannot spawn")
        self.entered = True
        return object(), object()

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True
        return False
