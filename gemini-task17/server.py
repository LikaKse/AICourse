"""Собственный MCP-сервер Дня 17.

Инструмент get_task(task_id) обращается к JSONPlaceholder и возвращает Task.
Запуск: python -u server.py  (stdio)
"""

from __future__ import annotations

import logging
import sys

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import config
from task_api import Task, TaskApi, TaskApiError, TaskNotFound

mcp = MCPServer(
    config.SERVER_NAME,
    version=config.SERVER_VERSION,
    instructions="Сервер задач JSONPlaceholder. Используйте get_task для получения задачи по id.",
    log_level="WARNING",
)

api = TaskApi()


@mcp.tool(description="Возвращает задачу JSONPlaceholder по её идентификатору.")
async def get_task(task_id: int) -> Task:
    """Возвращает задачу JSONPlaceholder по её идентификатору."""
    try:
        return await api.fetch_task(task_id)
    except TaskNotFound as exc:
        raise ToolError(str(exc)) from exc
    except TaskApiError as exc:
        raise ToolError(str(exc)) from exc


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-7s [mcp-server] %(message)s",
    )
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    mcp.run(transport="stdio")
