"""Фейковые соединения и менеджеры для офлайн-тестов."""

from __future__ import annotations

from types import SimpleNamespace

from mcp.types import TextContent

from manager import (
    ServerSpec,
    ToolCallResult,
    ToolDescriptor,
    ToolRoute,
    UnavailableServer,
)

TOOL_MAP = {
    "catalog": ["search", "get_item"],
    "notes": ["search", "get_note", "summarize"],
    "filesystem": ["read_text_file"],
}


def descriptor(alias: str, tool: str) -> ToolDescriptor:
    return ToolDescriptor(
        public_name=f"{alias}__{tool}",
        server_alias=alias,
        original_name=tool,
        description=f"{alias}/{tool}",
        input_schema={"type": "object", "properties": {}},
    )


class FakeCallResult:
    def __init__(self, text: str = "", *, is_error: bool = False, structured=None):
        self.content = [TextContent(type="text", text=text)]
        self.structured_content = structured
        self.is_error = is_error


class FakeConnection:
    failing_aliases: set[str] = set()

    def __init__(self, spec: ServerSpec, **kwargs):
        self.spec = spec
        self.connected = False
        self.error: str | None = None
        self.tools = [descriptor(spec.alias, name) for name in TOOL_MAP.get(spec.alias, [])]
        self.closed = False
        self.calls: list[tuple[str, dict]] = []
        self.responses: dict = kwargs.get("responses", {})

    async def start(self):
        if self.spec.alias in self.failing_aliases:
            self.error = "fake failure"
            return
        self.connected = True

    async def call_tool(self, original_name, arguments=None):
        self.calls.append((original_name, dict(arguments or {})))
        response = self.responses.get(original_name, {})
        if isinstance(response, Exception):
            raise response
        return FakeCallResult(structured=response)

    async def close(self):
        self.connected = False
        self.closed = True


class FakeManager:
    def __init__(self, aliases=("catalog", "notes", "filesystem"), responses=None, unavailable=()):
        self.routes = {}
        self.tools = []
        for alias in aliases:
            for name in TOOL_MAP[alias]:
                tool = descriptor(alias, name)
                self.tools.append(tool)
                self.routes[tool.public_name] = ToolRoute(
                    public_name=tool.public_name,
                    server_alias=alias,
                    original_name=name,
                )
        self.responses = responses or {}
        self.calls: list[tuple[str, dict]] = []
        self.unavailable = [UnavailableServer(alias, "fake unavailable") for alias in unavailable]

    async def call_tool(self, public_name, arguments=None):
        arguments = dict(arguments or {})
        self.calls.append((public_name, arguments))
        route = self.routes.get(public_name)
        if route is None:
            return ToolCallResult(
                public_name=public_name,
                server_alias=None,
                original_name=None,
                content=f"Неизвестный инструмент {public_name}",
                structured=None,
                is_error=True,
            )
        response = self.responses.get(public_name, {})
        if isinstance(response, ToolCallResult):
            return response
        if isinstance(response, Exception):
            return ToolCallResult(
                public_name=public_name,
                server_alias=route.server_alias,
                original_name=route.original_name,
                content=str(response),
                structured=None,
                is_error=True,
            )
        return ToolCallResult(
            public_name=public_name,
            server_alias=route.server_alias,
            original_name=route.original_name,
            content="",
            structured=response,
            is_error=False,
        )


def item(id=1, name="Laptop Pro 14"):
    return {
        "id": id,
        "name": name,
        "category": "laptop",
        "description": "Ноутбук для разработки.",
        "price_usd": 1800,
    }


def note(id=1, title="Laptop battery review"):
    return {
        "id": id,
        "title": title,
        "text": "Держит заряд весь день.",
    }


def success_responses():
    return {
        "filesystem__read_text_file": {"content": "Контекст: нужен ноутбук."},
        "catalog__search": {"query": "laptop", "items": [item(1), item(2, "Laptop Air 13")]},
        "catalog__get_item": item(1),
        "notes__search": {"query": "laptop", "items": [note(1)]},
        "notes__get_note": note(1),
        "notes__summarize": {"note_ids": [1], "summary": "Laptop battery review: держит заряд."},
    }
