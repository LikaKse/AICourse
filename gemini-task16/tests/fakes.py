"""Фейковые Gemini/MCP-объекты для офлайн-тестов."""

from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace

from google.genai import errors, types
from mcp.types import TextContent

from mcp_client import MCPToolInfo


# ---------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------

def text_response(text: str, *, thought: str | None = None, signature: bytes | None = None):
    parts = []
    if thought:
        parts.append(types.Part(text=thought, thought=True))
    parts.append(types.Part(text=text, thought_signature=signature))
    return _wrap(parts)


def call_response(*calls: tuple, signature: bytes | None = b"sig", text: str | None = None):
    """calls: (name, args) или (name, args, id)."""
    parts = []
    if text:
        parts.append(types.Part(text=text))
    for index, call in enumerate(calls):
        name, args, *rest = call
        call_id = rest[0] if rest else f"call-{name}-{index}"
        parts.append(types.Part(
            function_call=types.FunctionCall(id=call_id, name=name, args=args),
            thought_signature=signature if index == 0 else None,
        ))
    return _wrap(parts)


def empty_response(finish_reason: str = "SAFETY"):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=None, finish_reason=finish_reason)]
    )


def api_error(code: int = 503, message: str = "The model is overloaded.", status: str = "UNAVAILABLE"):
    return errors.APIError(code, {"error": {"code": code, "message": message, "status": status}})


def _wrap(parts):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts), finish_reason="STOP")],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100, candidates_token_count=20, thoughts_token_count=5
        ),
    )


class _FakeAsyncModels:
    def __init__(self, script):
        self.script = list(script)
        self.requests: list[dict] = []

    async def generate_content(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": copy.deepcopy(contents), "config": config})
        if not self.script:
            raise AssertionError("Агент сделал больше запросов, чем предусмотрено сценарием")
        item = self.script.pop(0)
        if callable(item) and not isinstance(item, types.GenerateContentResponse):
            item = item()
        if isinstance(item, Exception):
            raise item
        return item


class FakeGeminiClient:
    def __init__(self, script):
        self.models = _FakeAsyncModels(script)
        self.aio = SimpleNamespace(models=self.models)

    @property
    def requests(self):
        return self.models.requests


# ---------------------------------------------------------------------
# MCP: реестр/агент
# ---------------------------------------------------------------------

def make_tool(name: str = "read_text_file", schema: dict | None = None) -> MCPToolInfo:
    return MCPToolInfo(
        name=name,
        description=f"Fake {name}",
        input_schema=schema or {
            "type": "object",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )


class FakeMCPClient:
    """Подмена MCPFilesystemClient для тестов реестра и агента."""

    def __init__(self, tools=None, payloads=None, error: Exception | None = None):
        self.tools = tools or [make_tool()]
        self.payloads = payloads or {}
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    def exposed_tools(self):
        return self.tools

    async def call_tool_for_model(self, name: str, arguments: dict):
        self.calls.append((name, dict(arguments)))
        if self.error is not None:
            raise self.error
        payload = self.payloads.get(name)
        if callable(payload):
            payload = payload(arguments)
        return copy.deepcopy(payload) if payload is not None else {
            "source": "mcp",
            "tool": name,
            "content": "quantity=12\nunit_price=25000\nmarkup_percent=10\n",
        }


# ---------------------------------------------------------------------
# MCP: транспорт/сессия для MCPFilesystemClient
# ---------------------------------------------------------------------

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
        allowed_text: str = "",
        call_results: dict | None = None,
        protocol_version: str = "2025-11-25",
    ) -> None:
        self.tools = tools or [make_tool()]
        self.initialize_delay = initialize_delay
        self.call_delay = call_delay
        self.allowed_text = allowed_text
        self.call_results = call_results or {}
        self.protocol_version = protocol_version
        self.entered = False
        self.exited = False
        self.initialized = False
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
        self.initialized = True
        return SimpleNamespace(
            protocol_version=self.protocol_version,
            server_info=SimpleNamespace(name="fake-filesystem", version="2.4.0"),
        )

    async def list_tools(self):
        return SimpleNamespace(tools=self.tools)

    async def call_tool(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        if self.call_delay:
            await asyncio.sleep(self.call_delay)
        if name == "list_allowed_directories" and not self.call_results.get(name):
            return FakeCallResult(self.allowed_text or "Allowed directories:\n/tmp")
        result = self.call_results.get(name)
        if callable(result):
            result = result(arguments)
        return result if result is not None else FakeCallResult("ok")


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
