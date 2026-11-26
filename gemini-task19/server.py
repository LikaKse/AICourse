"""Собственный MCP-сервер Дня 19: search → summarize → save_to_file."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import config
from models import Post, SavedFile, SearchResult, Summary
from search_api import SearchApi, SearchApiError
from summarizer import Summarizer, SummarizerError

mcp = MCPServer(
    config.SERVER_NAME,
    version=config.SERVER_VERSION,
    instructions="Сервер цепочки: поиск публикаций, резюме и сохранение в файл.",
    log_level="WARNING",
)

search_api = SearchApi()
summarizer = Summarizer()

# Тестовый hook: включается только при запуске сервера с TASK19_FAIL_SUMMARIZE=1.
FAIL_SUMMARIZE = os.environ.get("TASK19_FAIL_SUMMARIZE") == "1"


@mcp.tool(description="Ищет публикации JSONPlaceholder по вхождению в title/body.")
async def search(query: str) -> SearchResult:
    try:
        return await search_api.search(query)
    except SearchApiError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool(description="Строит краткое детерминированное резюме по публикациям.")
async def summarize(items: list[Post]) -> Summary:
    if FAIL_SUMMARIZE:
        raise ToolError("Тестовая ошибка summarize (TASK19_FAIL_SUMMARIZE=1).")
    try:
        return summarizer.summarize(items)
    except SummarizerError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool(description="Сохраняет непустой текст в файл внутри output/.")
async def save_to_file(content: str, filename: str) -> SavedFile:
    if not isinstance(content, str) or not content.strip():
        raise ToolError("content должен быть непустой строкой.")
    name = filename.strip() if isinstance(filename, str) else ""
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ToolError("filename должен быть простым именем файла без пути.")

    try:
        output_dir = Path(config.OUTPUT_DIR).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        target = (output_dir / name).resolve()
        if output_dir not in target.parents:
            raise ToolError("Путь сохранения выходит за пределы output/.")
        target.write_text(content, encoding="utf-8")
        try:
            path = target.relative_to(config.PROJECT_ROOT).as_posix()
        except ValueError:
            path = str(target)
        return SavedFile(path=path, bytes_written=len(content.encode("utf-8")))
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Не удалось сохранить файл: {exc}") from exc


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-7s [mcp-server] %(message)s",
    )
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    mcp.run(transport="stdio")
