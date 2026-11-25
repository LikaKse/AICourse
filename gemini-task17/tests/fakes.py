"""Фейковые объекты для офлайн-тестов Дня 17."""

from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace

from google.genai import errors, types
from mcp.types import TextContent

from mcp_client import MCPToolInfo
from task_api import Task

TASK_DATA = {
    "userId": 1,
    "id": 5,
    "title": "laboriosam mollitia et enim quasi adipisci quia provident illum",
    "completed": False,
}


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


def api_error(code: int = 503, message: str = "overloaded", status: str = "UNAVAILABLE"):
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
            raise AssertionError("Агент сделал больше запросов, чем предусмотрено")
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
# MCP для агента/реестра
# ---------------------------------------------------------------------

def make_tool(name: str = "get_task") -> MCPToolInfo:
    return MCPToolInfo(
        name=name,
        description="Fake get_task",
        input_schema={
            "type": "object",
            "properties": {"task_id": {"type": "integer"}},
            "required": ["task_id"],
        },
    )


class FakeMCPClient:
    def __init__(self, payload=None, *, error: Exception | None = None, tools=None):
        self.tools = tools or [make_tool()]
        self.payload = payload or {"source": "mcp", "tool": "get_task", "result": copy.deepcopy(TASK_DATA)}
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    def exposed_tools(self):
        return self.tools

    async def call_tool_for_model(self, name: str, arguments: dict):
        self.calls.append((name, dict(arguments)))
        if self.error is not None:
            raise self.error
        return copy.deepcopy(self.payload)


# ---------------------------------------------------------------------
# MCP transport/session для TaskMCPClient
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
        call_result: FakeCallResult | None = None,
        protocol_version: str = "2025-11-25",
    ) -> None:
        self.tools = tools or [make_tool()]
        self.initialize_delay = initialize_delay
        self.call_delay = call_delay
        self.call_result = call_result or FakeCallResult(
            structured=copy.deepcopy(TASK_DATA)
        )
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
            server_info=SimpleNamespace(name="jsonplaceholder-tasks", version="0.1.0"),
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


class FakeTaskApi:
    def __init__(self, task=None, error: Exception | None = None):
        self.task = task or copy.deepcopy(TASK_DATA)
        self.error = error
        self.calls: list[int] = []

    async def fetch_task(self, task_id: int):
        self.calls.append(task_id)
        if self.error is not None:
            raise self.error
        data = dict(self.task)
        data["id"] = task_id
        return Task.model_validate(data)
