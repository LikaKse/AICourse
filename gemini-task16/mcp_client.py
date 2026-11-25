"""MCP-клиент Filesystem через stdio и низкоуровневый ClientSession.

Сервер запускается project-local Node 24:
    node .tools/.../dist/index.js --read-only --confine-roots demo_workspace

Никаких глобальных Node/npm-изменений нет. Сервер сам не отдаёт инструменты записи,
а клиент дополнительно проверяет allowlist и пути.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
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
    """Минимальная информация об инструменте, нужная для схемы Gemini."""

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
    """Нормализованный результат tools/call."""

    tool: str
    content: str
    structured: Any
    is_error: bool

    def to_model_payload(self) -> dict:
        if self.is_error:
            return {"error": self.content or f"MCP-инструмент {self.tool} завершился с ошибкой."}
        payload: dict[str, Any] = {"source": "mcp", "tool": self.tool}
        if self.content:
            payload["content"] = self.content
        # Filesystem возвращает и text, и structuredContent={"content": text}.
        # Не дублируем одно и то же в FunctionResponse.
        duplicate = (
            isinstance(self.structured, dict)
            and set(self.structured) == {"content"}
            and self.structured.get("content") == self.content
        )
        if self.structured is not None and not duplicate:
            payload["structured"] = self.structured
        return payload


def _preview(value: Any, limit: int = config.LOG_PREVIEW_CHARS) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= limit else text[:limit] + f"… (+{len(text) - limit} симв.)"


def _content_text(result: Any) -> str:
    """Собирает текст из content blocks MCP."""
    chunks: list[str] = []
    for block in getattr(result, "content", None) or []:
        if isinstance(block, TextContent) or getattr(block, "type", None) == "text":
            chunks.append(block.text)
        else:
            if hasattr(block, "model_dump"):
                chunks.append(json.dumps(block.model_dump(mode="json"), ensure_ascii=False, default=str))
            else:
                chunks.append(str(block))
    text = "\n".join(chunks)
    return text if text.strip() else ""


class MCPFilesystemClient:
    """Асинхронный клиент Filesystem MCP с одним долгоживущим соединением."""

    def __init__(
        self,
        workspace: Path | None = None,
        node_bin: Path | None = None,
        server_js: Path | None = None,
        *,
        connect_timeout: float | None = None,
        call_timeout: float | None = None,
        transport_factory: Callable[..., Any] | None = None,
        session_factory: Callable[..., Any] | None = None,
        validate_environment: bool = True,
        verify_workspace: bool = True,
        errlog: TextIO | None = None,
    ) -> None:
        self.workspace = Path(workspace or config.WORKSPACE).expanduser().resolve()
        self.node_bin = Path(node_bin or config.NODE_BIN)
        self.server_js = Path(server_js or config.MCP_SERVER_JS)
        self.connect_timeout = config.MCP_CONNECT_TIMEOUT if connect_timeout is None else connect_timeout
        self.call_timeout = config.MCP_CALL_TIMEOUT if call_timeout is None else call_timeout
        self._transport_factory = transport_factory or stdio_client
        self._session_factory = session_factory or ClientSession
        self._validate_environment = validate_environment
        self._verify_workspace = verify_workspace
        self._errlog = errlog if errlog is not None else sys.stderr

        self._stack: AsyncExitStack | None = None
        self._session: Any | None = None
        self.tools: list[MCPToolInfo] = []
        self.server_info: str | None = None
        self.protocol_version: str | None = None
        self.connected = False

    # ------------------------------------------------------------------
    # Команда запуска
    # ------------------------------------------------------------------

    def build_server_parameters(self) -> StdioServerParameters:
        args: list[str] = [str(self.server_js)]
        if config.MCP_READ_ONLY:
            args.append("--read-only")
        if config.MCP_CONFINE_ROOTS:
            args.append("--confine-roots")
        args.append(str(self.workspace))
        return StdioServerParameters(
            command=str(self.node_bin),
            args=args,
            cwd=str(config.PROJECT_ROOT),
        )

    def _check_environment(self) -> None:
        if not self.workspace.is_dir():
            raise MCPConnectionError(
                f"Папка demo_workspace не найдена: {self.workspace}. "
                "Создайте её и положите numbers.txt."
            )
        if not self.node_bin.is_file():
            raise MCPConnectionError(
                f"Не найден project-local Node: {self.node_bin}. "
                "Запустите: bash scripts/setup_mcp_env.sh"
            )
        if not self.server_js.is_file():
            raise MCPConnectionError(
                f"Не найден MCP Filesystem server: {self.server_js}. "
                "Запустите: bash scripts/setup_mcp_env.sh"
            )
        try:
            completed = subprocess.run(
                [str(self.node_bin), "-v"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise MCPConnectionError(f"Не удалось запустить Node: {exc}") from exc
        version = (completed.stdout or completed.stderr or "").strip()
        if not version.startswith("v"):
            raise MCPConnectionError(f"Не удалось определить версию Node: {version!r}")
        try:
            major = int(version[1:].split(".", 1)[0])
        except (ValueError, IndexError):
            raise MCPConnectionError(f"Не удалось разобрать версию Node: {version!r}") from None
        if major < config.NODE_MAJOR_MIN:
            raise MCPConnectionError(
                f"Нужен Node.js ≥ {config.NODE_MAJOR_MIN}, найден {version}. "
                "Старый MCP-сервер ради Node 22 не используется. "
                "Запустите: bash scripts/setup_mcp_env.sh"
            )
        logger.info("Node: %s (%s)", version, self.node_bin)

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "MCPFilesystemClient":
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
            logger.info(
                "MCP подключён: server=%s protocol=%s tools=%d",
                self.server_info or "?",
                self.protocol_version or "?",
                len(self.tools),
            )
            # call_tool проверяет self.connected, поэтому ставим флаг до проверки папок.
            self.connected = True
            if self._verify_workspace:
                await self._check_allowed_directories()
            return self
        except MCPError:
            await self.aclose()
            raise
        except Exception as exc:
            await self.aclose()
            raise MCPConnectionError(
                f"Не удалось подключиться к MCP Filesystem: {type(exc).__name__}: {exc}"
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
            except Exception as exc:  # cleanup не должен маскировать исходную ошибку
                logger.warning("Ошибка при закрытии MCP: %s", exc)

    async def _with_timeout(self, coro, timeout: float | None, label: str):
        if timeout is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout)
        except TimeoutError as exc:
            raise MCPTimeoutError(f"Таймаут MCP ({label}): {timeout:g} с.") from exc

    async def _check_allowed_directories(self) -> None:
        result = await self.call_tool("list_allowed_directories", {}, check_exposed=False)
        if result.is_error:
            raise MCPConnectionError(f"MCP не сообщил разрешённые папки: {result.content}")
        allowed: list[str] = []
        for line in result.content.splitlines():
            line = line.strip()
            if not line or line.lower().startswith("allowed"):
                continue
            if line.startswith("/"):
                allowed.append(line)
        resolved = {str(Path(item).resolve()) for item in allowed}
        if str(self.workspace) not in resolved:
            raise MCPConnectionError(
                f"MCP разрешил не только demo_workspace. Ожидалось {self.workspace}, получено: {allowed}"
            )

    # ------------------------------------------------------------------
    # Инструменты
    # ------------------------------------------------------------------

    def exposed_tools(self) -> list[MCPToolInfo]:
        by_name = {tool.name: tool for tool in self.tools}
        missing = [name for name in config.MCP_EXPOSED_TOOLS if name not in by_name]
        if missing:
            logger.warning("MCP не отдал ожидаемые инструменты: %s", ", ".join(missing))
        if config.MCP_REQUIRED_TOOL not in by_name:
            raise MCPConnectionError(
                f"MCP не отдал обязательный инструмент {config.MCP_REQUIRED_TOOL!r}. "
                f"Доступны: {', '.join(by_name) or '—'}"
            )
        return [by_name[name] for name in config.MCP_EXPOSED_TOOLS if name in by_name]

    def _is_exposed(self, name: str) -> bool:
        return any(tool.name == name for tool in self.exposed_tools())

    def resolve_workspace_path(self, raw: str) -> Path:
        """Разрешает путь относительно demo_workspace и запрещает выход наружу."""
        if not isinstance(raw, str) or not raw.strip():
            raise MCPToolError("Путь должен быть непустой строкой.")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            candidates = [self.workspace / path]
            project_candidate = config.PROJECT_ROOT / path
            if project_candidate.exists():
                candidates.append(project_candidate)
            path = next((item for item in candidates if item.exists()), candidates[0])
        resolved = path.resolve(strict=False)
        if resolved != self.workspace and self.workspace not in resolved.parents:
            raise MCPToolError(
                f"Доступ запрещён: путь {raw!r} находится вне demo_workspace ({self.workspace})."
            )
        return resolved

    def _normalize_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(arguments)
        if isinstance(normalized.get("path"), str):
            normalized["path"] = str(self.resolve_workspace_path(normalized["path"]))
        if isinstance(normalized.get("paths"), list):
            normalized["paths"] = [
                str(self.resolve_workspace_path(item)) if isinstance(item, str) else item
                for item in normalized["paths"]
            ]
        return normalized

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        check_exposed: bool = True,
    ) -> MCPCallResult:
        if not self.connected or self._session is None:
            raise MCPConnectionError("MCP-сессия не подключена.")
        if check_exposed and not self._is_exposed(name):
            raise MCPToolError(f"MCP-инструмент {name!r} не разрешён для модели.")
        try:
            args = self._normalize_arguments(dict(arguments or {}))
        except MCPToolError:
            raise
        except Exception as exc:
            raise MCPToolError(f"Неверные аргументы MCP {name}: {exc}") from exc

        logger.info("MCP вызов: %s(%s)", name, _preview(args))
        try:
            raw = await self._with_timeout(
                self._session.call_tool(name, args),
                self.call_timeout,
                f"tools/call {name}",
            )
        except MCPError:
            raise
        except TimeoutError:
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
            logger.info("MCP результат %s: %s", name, _preview(content))
        return result

    async def call_tool_for_model(self, name: str, arguments: dict[str, Any]) -> dict:
        """Возвращает dict для FunctionResponse: {"result": ...} или {"error": ...}."""
        result = await self.call_tool(name, arguments)
        return result.to_model_payload()
