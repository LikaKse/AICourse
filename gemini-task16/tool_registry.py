"""Единый реестр локальных и MCP-инструментов.

Модель видит один список FunctionDeclaration. Реестр по имени определяет источник
(local/mcp), выполняет вызов и всегда возвращает {"result": ...} или {"error": ...}.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from google.genai import types

import local_tools
import mcp_schema
from mcp_client import MCPError, MCPFilesystemClient, MCPToolInfo

logger = logging.getLogger("registry")


@dataclass(frozen=True)
class ToolBinding:
    public_name: str
    source: str                 # "local" | "mcp"
    original_name: str
    declaration: types.FunctionDeclaration


@dataclass
class DispatchResult:
    payload: dict
    source: str


class ToolRegistry:
    """Объединяет локальные функции Дня 6 и инструменты MCP Filesystem."""

    def __init__(
        self,
        local_declarations: Iterable[types.FunctionDeclaration] | None = None,
        local_dispatcher: Callable[[str, dict], dict] | None = None,
        mcp_client: MCPFilesystemClient | Any | None = None,
        mcp_tools: Iterable[MCPToolInfo] | None = None,
    ) -> None:
        self.local_dispatcher = local_dispatcher or local_tools.dispatch
        self.mcp_client = mcp_client
        self.bindings: dict[str, ToolBinding] = {}
        self._declarations: list[types.FunctionDeclaration] = []

        used: set[str] = set()
        for declaration in local_declarations if local_declarations is not None else local_tools.FUNCTION_DECLARATIONS:
            if declaration.name in used:
                raise ValueError(f"Повторяющееся имя локального инструмента: {declaration.name}")
            used.add(declaration.name)
            self.bindings[declaration.name] = ToolBinding(
                public_name=declaration.name,
                source="local",
                original_name=declaration.name,
                declaration=declaration,
            )
            self._declarations.append(declaration)

        for tool in mcp_tools or []:
            public_name = mcp_schema.unique_public_name(tool.name, used)
            used.add(public_name)
            declaration = mcp_schema.to_gemini_declaration(tool, public_name=public_name)
            self.bindings[public_name] = ToolBinding(
                public_name=public_name,
                source="mcp",
                original_name=tool.name,
                declaration=declaration,
            )
            self._declarations.append(declaration)

    @property
    def declarations(self) -> list[types.FunctionDeclaration]:
        return list(self._declarations)

    @property
    def names(self) -> list[str]:
        return [binding.public_name for binding in self.bindings.values()]

    def source_of(self, name: str) -> str | None:
        binding = self.bindings.get(name)
        return binding.source if binding else None

    def local_declarations(self) -> list[types.FunctionDeclaration]:
        return [b.declaration for b in self.bindings.values() if b.source == "local"]

    def mcp_declarations(self) -> list[types.FunctionDeclaration]:
        return [b.declaration for b in self.bindings.values() if b.source == "mcp"]

    async def adispatch(self, name: str, args: dict | None = None) -> DispatchResult:
        binding = self.bindings.get(name)
        if binding is None:
            available = ", ".join(self.bindings) or "—"
            return DispatchResult(
                {"error": f"Неизвестный инструмент {name!r}. Доступны: {available}."},
                "unknown",
            )

        args = dict(args or {})
        if binding.source == "local":
            return self._dispatch_local(binding, args)
        return await self._dispatch_mcp(binding, args)

    def _dispatch_local(self, binding: ToolBinding, args: dict) -> DispatchResult:
        try:
            result = self.local_dispatcher(binding.original_name, args)
            if inspect.isawaitable(result):
                # Локальные инструменты синхронные; это защита от неверной инъекции.
                raise TypeError("Локальный диспетчер не должен быть async")
            if not isinstance(result, dict):
                result = {"result": result}
        except Exception as exc:
            result = {"error": f"Внутренняя ошибка локального инструмента {binding.public_name}: {exc}"}
        return DispatchResult(result, "local")

    async def _dispatch_mcp(self, binding: ToolBinding, args: dict) -> DispatchResult:
        if self.mcp_client is None:
            return DispatchResult({"error": "MCP-клиент не подключён."}, "mcp")
        try:
            payload = await self.mcp_client.call_tool_for_model(binding.original_name, args)
            if not isinstance(payload, dict):
                payload = {"result": payload}
        except MCPError as exc:
            payload = {"error": str(exc)}
        except Exception as exc:
            payload = {"error": f"Ошибка MCP-инструмента {binding.original_name}: {type(exc).__name__}: {exc}"}
        return DispatchResult(payload, "mcp")

    def describe(self) -> list[dict]:
        return [
            {
                "name": b.public_name,
                "source": b.source,
                "original_name": b.original_name,
            }
            for b in self.bindings.values()
        ]
