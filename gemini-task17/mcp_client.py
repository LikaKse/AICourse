"""MCP-клиент для собственного сервера Дня 17.

Сервер запускается через stdio:
    .venv/bin/python -u server.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

import config

logger = logging.getLogger("mcp")


class MCPError(Exception):
    """Базовая ошибка MCP-клиента."""


class MCPConnectionError(MCPError):
    """Не удалось запустить сервер или установить MCP-сессию."""


class MCPTimeoutError(MCPError):
    """Истёк таймаут подключения или вызова."""


class MCPToolError(MCPError):
    """Ошибка конкретного MCP-инструмента."""


@dataclass(frozen=True)
class MCPToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, Any] | None = None

    @classmethod
    def from_mcp(cls, tool: Any) -> "MCPToolInfo":
        annotations = getattr(tool, "annotations", None)
        if annotations is not None and hasattr(annotations, "model_dump"):
            annotations = annotations.model_dump(mode="json", exclude_none=True)
        return cls(
            name=tool.name,
            description=tool.description or tool.name,
            input_schema=dict(getattr(tool, "input_schema", None) or {}),
            annotations=annotations if isinstance(annotations, dict) else None,
        )


@dataclass(frozen=True)
class MCPCallResult:
    tool: str
    content: str
    structured: Any
    is_error: bool

    def to_model_payload(self) -> dict:
        if self.is_error:
            return {"error": self.content or f"MCP-инструмент {self.tool} завершился с ошибкой."}
        payload: dict[str, Any] = {"source": "mcp", "tool": self.tool}
        if self.structured is not None:
            payload["result"] = self.structured
        elif self.content:
            payload["result"] = self.content
        return payload


def _preview(value: Any, limit: int = config.LOG_PREVIEW_CHARS) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= limit else text[:limit] + f"… (+{len(text) - limit} симв.)"


def _content_text(result: Any) -> str:
    chunks: list[str] = []
    for block in getattr(result, "content", None) or []:
        if isinstance(block, TextContent) or getattr(block, "type", None) == "text":
            chunks.append(block.text)
        elif hasattr(block, "model_dump"):
            chunks.append(json.dumps(block.model_dump(mode="json"), ensure_ascii=False, default=str))
        else:
            chunks.append(str(block))
    text = "\n".join(chunks)
    return text if text.strip() else ""


class TaskMCPClient:
    """Один долгоживущий stdio-сеанс с собственным MCP-сервером."""

    def __init__(
        self,
        api_base: str | None = None,
        *,
        python_bin: Path | None = None,
        server_py: Path | None = None,
        connect_timeout: float | None = None,
        call_timeout: float | None = None,
        transport_factory: Callable[..., Any] | None = None,
        session_factory: Callable[..., Any] | None = None,
        validate_environment: bool = True,
        errlog: TextIO | None = None,
    ) -> None:
        self.api_base = (api_base or config.TASKS_API_BASE).rstrip("/")
        self.python_bin = Path(python_bin or config.PYTHON_BIN)
        self.server_py = Path(server_py or config.SERVER_PY)
        self.connect_timeout = config.MCP_CONNECT_TIMEOUT if connect_timeout is None else connect_timeout
        self.call_timeout = config.MCP_CALL_TIMEOUT if call_timeout is None else call_timeout
        self._transport_factory = transport_factory or stdio_client
        self._session_factory = session_factory or ClientSession
        self._validate_environment = validate_environment
        self._errlog = errlog if errlog is not None else sys.stderr

        self._stack: AsyncExitStack | None = None
        self._session: Any | None = None
        self.tools: list[MCPToolInfo] = []
        self.server_info: str | None = None
        self.protocol_version: str | None = None
        self.connected = False

    def build_server_parameters(self) -> StdioServerParameters:
        return StdioServerParameters(
            command=str(self.python_bin),
            args=["-u", str(self.server_py)],
            cwd=str(config.PROJECT_ROOT),
            env={
                "TASKS_API_BASE": self.api_base,
                "PYTHONUNBUFFERED": "1",
            },
        )

    def _check_environment(self) -> None:
        if not self.python_bin.is_file():
            raise MCPConnectionError(
                f"Не найден Python для MCP-сервера: {self.python_bin}. "
                "Создайте .venv: ../venv/bin/python -m venv .venv"
            )
        if not self.server_py.is_file():
            raise MCPConnectionError(f"Не найден MCP-сервер: {self.server_py}")
        logger.info("MCP Python: %s", self.python_bin)

    async def __aenter__(self) -> "TaskMCPClient":
        self._stack = AsyncExitStack()
        try:
            if self._validate_environment:
                self._check_environment()
            params = self.build_server_parameters()
            logger.info("Запуск MCP: %s %s", params.command, " ".join(params.args))
            read, write = await self._stack.enter_async_context(
                self._transport_factory(params, errlog=self._errlog)
            )
            self._session = await self._stack.enter_async_context(
                self._session_factory(read, write, read_timeout_seconds=self.call_timeout)
            )
            init = await self._with_timeout(self._session.initialize(), self.connect_timeout, "initialize")
            info = getattr(init, "server_info", None)
            if info is not None:
                self.server_info = f"{getattr(info, 'name', '?')} {getattr(info, 'version', '')}".strip()
            self.protocol_version = getattr(init, "protocol_version", None)

            result = await self._with_timeout(self._session.list_tools(), self.connect_timeout, "tools/list")
            self.tools = [MCPToolInfo.from_mcp(tool) for tool in result.tools]
            names = {tool.name for tool in self.tools}
            if config.MCP_REQUIRED_TOOL not in names:
                raise MCPConnectionError(
                    f"MCP не отдал обязательный инструмент {config.MCP_REQUIRED_TOOL!r}. "
                    f"Доступны: {', '.join(names) or '—'}"
                )
            self.connected = True
            logger.info(
                "MCP подключён: server=%s protocol=%s tools=%d",
                self.server_info or "?",
                self.protocol_version or "?",
                len(self.tools),
            )
            return self
        except MCPError:
            await self.aclose()
            raise
        except Exception as exc:
            await self.aclose()
            raise MCPConnectionError(
                f"Не удалось подключиться к MCP-серверу: {type(exc).__name__}: {exc}"
            ) from exc

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        await self.aclose()
        return False

    async def aclose(self) -> None:
        self.connected = False
        self._session = None
        stack, self._stack = self._stack, None
        if stack is not None:
            try:
                await stack.aclose()
            except Exception as exc:
                logger.warning("Ошибка при закрытии MCP: %s", exc)

    async def _with_timeout(self, coro, timeout: float | None, label: str):
        if timeout is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout)
        except TimeoutError as exc:
            raise MCPTimeoutError(f"Таймаут MCP ({label}): {timeout:g} с.") from exc

    def exposed_tools(self) -> list[MCPToolInfo]:
        by_name = {tool.name: tool for tool in self.tools}
        return [by_name[name] for name in config.MCP_EXPOSED_TOOLS if name in by_name]

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> MCPCallResult:
        if not self.connected or self._session is None:
            raise MCPConnectionError("MCP-сессия не подключена.")
        if name not in {tool.name for tool in self.exposed_tools()}:
            raise MCPToolError(f"MCP-инструмент {name!r} не разрешён для модели.")

        args = dict(arguments or {})
        logger.info("MCP вызов: %s(%s)", name, _preview(args))
        try:
            raw = await self._with_timeout(
                self._session.call_tool(name, args),
                self.call_timeout,
                f"tools/call {name}",
            )
        except MCPError:
            raise
        except Exception as exc:
            raise MCPToolError(f"Ошибка MCP-вызова {name}: {type(exc).__name__}: {exc}") from exc

        content = _content_text(raw)
        structured = getattr(raw, "structured_content", None)
        is_error = bool(getattr(raw, "is_error", False))
        result = MCPCallResult(tool=name, content=content, structured=structured, is_error=is_error)
        if is_error:
            logger.warning("MCP инструмент %s вернул ошибку: %s", name, _preview(content))
        else:
            logger.info("MCP результат %s: %s", name, _preview(structured if structured is not None else content))
        return result

    async def call_tool_for_model(self, name: str, arguments: dict[str, Any]) -> dict:
        result = await self.call_tool(name, arguments)
        return result.to_model_payload()
