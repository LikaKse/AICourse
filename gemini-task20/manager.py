"""Менеджер нескольких MCP-серверов и точная маршрутизация инструментов."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TextIO

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

import config

logger = logging.getLogger("manager")


class MCPError(Exception):
    pass


class MCPConnectionError(MCPError):
    pass


class MCPTimeoutError(MCPError):
    pass


@dataclass(frozen=True)
class ServerSpec:
    alias: str
    command: str
    args: list[str]
    cwd: str | None = None
    env: dict[str, str] | None = None
    allowed_tools: frozenset[str] | None = None
    verify_allowed_directory: Path | None = None


@dataclass(frozen=True)
class ToolRoute:
    public_name: str
    server_alias: str
    original_name: str


@dataclass(frozen=True)
class ToolDescriptor:
    public_name: str
    server_alias: str
    original_name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class UnavailableServer:
    alias: str
    reason: str


@dataclass
class ToolCallResult:
    public_name: str
    server_alias: str | None
    original_name: str | None
    content: str
    structured: Any
    is_error: bool

    @property
    def error_text(self) -> str:
        return self.content or "MCP-инструмент завершился с ошибкой."


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


class ServerConnection:
    """Одно независимое stdio-соединение с MCP-сервером."""

    def __init__(
        self,
        spec: ServerSpec,
        *,
        connect_timeout: float = config.MCP_CONNECT_TIMEOUT,
        call_timeout: float = config.MCP_CALL_TIMEOUT,
        errlog: TextIO | None = None,
    ) -> None:
        self.spec = spec
        self.connect_timeout = connect_timeout
        self.call_timeout = call_timeout
        self._errlog = errlog if errlog is not None else sys.stderr
        self._stack: AsyncExitStack | None = None
        self._session: Any | None = None
        self.tools: list[ToolDescriptor] = []
        self.connected = False
        self.error: str | None = None

    async def start(self) -> None:
        stack = AsyncExitStack()
        try:
            params = StdioServerParameters(
                command=self.spec.command,
                args=list(self.spec.args),
                cwd=self.spec.cwd or str(config.PROJECT_ROOT),
                env=self.spec.env or {},
            )
            logger.info("Запуск сервера %s: %s %s", self.spec.alias, params.command, " ".join(params.args))
            read, write = await stack.enter_async_context(
                stdio_client(params, errlog=self._errlog)
            )
            session = await stack.enter_async_context(
                ClientSession(read, write, read_timeout_seconds=self.call_timeout)
            )
            await self._with_timeout(session.initialize(), self.connect_timeout, "initialize")
            result = await self._with_timeout(session.list_tools(), self.connect_timeout, "tools/list")
            raw_tools = list(result.tools)
            if self.spec.allowed_tools is not None:
                raw_tools = [tool for tool in raw_tools if tool.name in self.spec.allowed_tools]
            self.tools = [
                ToolDescriptor(
                    public_name=f"{self.spec.alias}__{tool.name}",
                    server_alias=self.spec.alias,
                    original_name=tool.name,
                    description=tool.description or tool.name,
                    input_schema=dict(getattr(tool, "input_schema", None) or {}),
                )
                for tool in raw_tools
            ]
            if self.spec.verify_allowed_directory is not None:
                await self._verify_allowed_directory(session, self.spec.verify_allowed_directory)
            self._stack = stack
            self._session = session
            self.connected = True
            logger.info("Сервер %s подключён: tools=%d", self.spec.alias, len(self.tools))
        except Exception as exc:
            await stack.aclose()
            self.connected = False
            self.error = f"{type(exc).__name__}: {exc}"
            logger.error("Сервер %s недоступен: %s", self.spec.alias, self.error)

    async def _verify_allowed_directory(self, session: Any, expected: Path) -> None:
        raw = await self._with_timeout(
            session.call_tool("list_allowed_directories", {}),
            self.connect_timeout,
            "list_allowed_directories",
        )
        if getattr(raw, "is_error", False):
            raise MCPConnectionError(_content_text(raw) or "list_allowed_directories failed")
        text = _content_text(raw)
        allowed = [
            str(Path(line.strip()).resolve())
            for line in text.splitlines()
            if line.strip().startswith("/")
        ]
        expected_resolved = str(expected.resolve())
        if allowed != [expected_resolved]:
            raise MCPConnectionError(
                f"Ожидалась одна разрешённая папка {expected_resolved}, получено: {allowed}"
            )

    async def _with_timeout(self, coro, timeout: float, label: str):
        try:
            return await asyncio.wait_for(coro, timeout)
        except TimeoutError as exc:
            raise MCPTimeoutError(f"Таймаут {self.spec.alias}/{label}: {timeout:g} с.") from exc

    async def call_tool(self, original_name: str, arguments: dict[str, Any] | None = None) -> Any:
        if not self.connected or self._session is None:
            raise MCPConnectionError(f"Сервер {self.spec.alias} недоступен.")
        return await self._with_timeout(
            self._session.call_tool(original_name, dict(arguments or {})),
            self.call_timeout,
            f"tools/call {original_name}",
        )

    async def close(self) -> None:
        self.connected = False
        self._session = None
        stack, self._stack = self._stack, None
        if stack is not None:
            try:
                await stack.aclose()
            except Exception as exc:
                logger.warning("Ошибка закрытия %s: %s", self.spec.alias, exc)


class MultiMCPManager:
    """Управляет независимыми соединениями и объединённым каталогом."""

    def __init__(
        self,
        specs: list[ServerSpec],
        *,
        connection_factory: Callable[..., ServerConnection] = ServerConnection,
        connect_timeout: float = config.MCP_CONNECT_TIMEOUT,
        call_timeout: float = config.MCP_CALL_TIMEOUT,
        errlog: TextIO | None = None,
    ) -> None:
        aliases = [spec.alias for spec in specs]
        if len(aliases) != len(set(aliases)):
            raise ValueError("Алиасы MCP-серверов должны быть уникальными.")
        self.specs = specs
        self.connections = [
            connection_factory(
                spec,
                connect_timeout=connect_timeout,
                call_timeout=call_timeout,
                errlog=errlog,
            )
            for spec in specs
        ]
        self.routes: dict[str, ToolRoute] = {}
        self.tools: list[ToolDescriptor] = []
        self.unavailable: list[UnavailableServer] = []
        self._by_alias: dict[str, ServerConnection] = {}
        self._started = False

    async def start(self) -> "MultiMCPManager":
        # Последовательный запуск: AsyncExitStack каждого соединения входит и
        # выходит в одной и той же задаче (важно для anyio cancel scopes).
        for connection in self.connections:
            try:
                await connection.start()
            except Exception as exc:
                connection.connected = False
                connection.error = f"{type(exc).__name__}: {exc}"
                logger.error("Сервер %s не запущен: %s", connection.spec.alias, connection.error)
        self._by_alias = {connection.spec.alias: connection for connection in self.connections}
        self.unavailable = [
            UnavailableServer(connection.spec.alias, connection.error or "unknown")
            for connection in self.connections
            if not connection.connected
        ]
        for connection in self.connections:
            if not connection.connected:
                continue
            for tool in connection.tools:
                if tool.public_name in self.routes:
                    raise ValueError(f"Дублирующееся публичное имя инструмента: {tool.public_name}")
                self.routes[tool.public_name] = ToolRoute(
                    public_name=tool.public_name,
                    server_alias=tool.server_alias,
                    original_name=tool.original_name,
                )
                self.tools.append(tool)
        self._started = True
        return self

    @property
    def available_servers(self) -> list[str]:
        return [connection.spec.alias for connection in self.connections if connection.connected]

    def route_for(self, public_name: str) -> ToolRoute | None:
        return self.routes.get(public_name)

    def descriptor_for(self, public_name: str) -> ToolDescriptor | None:
        return next((tool for tool in self.tools if tool.public_name == public_name), None)

    async def call_tool(self, public_name: str, arguments: dict[str, Any] | None = None) -> ToolCallResult:
        route = self.routes.get(public_name)
        if route is None:
            unavailable = ", ".join(item.alias for item in self.unavailable) or "нет"
            return ToolCallResult(
                public_name=public_name,
                server_alias=None,
                original_name=None,
                content=f"Неизвестный инструмент {public_name!r}. Недоступные серверы: {unavailable}.",
                structured=None,
                is_error=True,
            )
        connection = self._by_alias.get(route.server_alias)
        if connection is None or not connection.connected:
            reason = next(
                (item.reason for item in self.unavailable if item.alias == route.server_alias),
                "соединение не открыто",
            )
            return ToolCallResult(
                public_name=public_name,
                server_alias=route.server_alias,
                original_name=route.original_name,
                content=f"Сервер {route.server_alias!r} недоступен: {reason}",
                structured=None,
                is_error=True,
            )
        logger.info("MCP вызов %s → %s/%s", public_name, route.server_alias, route.original_name)
        try:
            raw = await connection.call_tool(route.original_name, arguments)
        except Exception as exc:
            return ToolCallResult(
                public_name=public_name,
                server_alias=route.server_alias,
                original_name=route.original_name,
                content=f"Ошибка MCP-вызова {public_name}: {type(exc).__name__}: {exc}",
                structured=None,
                is_error=True,
            )
        result = ToolCallResult(
            public_name=public_name,
            server_alias=route.server_alias,
            original_name=route.original_name,
            content=_content_text(raw),
            structured=getattr(raw, "structured_content", None),
            is_error=bool(getattr(raw, "is_error", False)),
        )
        if not result.is_error:
            logger.info("MCP результат %s: %s", public_name, _preview(result.structured))
        else:
            logger.warning("MCP ошибка %s: %s", public_name, result.content)
        return result

    async def aclose(self) -> None:
        # Закрываем в обратном порядке открытия: anyio cancel scopes должны
        # выходить LIFO. Ошибка одного close не мешает закрыть остальные.
        for connection in reversed(self.connections):
            try:
                await connection.close()
            except Exception as exc:
                logger.warning("Не удалось закрыть %s: %s", connection.spec.alias, exc)
        self.routes.clear()
        self.tools.clear()
        self._by_alias.clear()
        self._started = False


def _preview(value: Any, limit: int = config.LOG_PREVIEW_CHARS) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= limit else text[:limit] + f"… (+{len(text) - limit} симв.)"
