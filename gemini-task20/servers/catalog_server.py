"""MCP-сервер каталога товаров."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

from models import CatalogItem, CatalogSearch  # noqa: E402
from servers.catalog_data import CATALOG  # noqa: E402

mcp = MCPServer(
    "catalog",
    version="0.1.0",
    instructions="Локальный каталог товаров.",
    log_level="WARNING",
)


@mcp.tool(description="Ищет товары по вхождению в name/category/description.")
async def search(query: str) -> CatalogSearch:
    needle = query.strip().casefold()
    items = [
        CatalogItem(**item)
        for item in CATALOG
        if needle in item["name"].casefold()
        or needle in item["category"].casefold()
        or needle in item["description"].casefold()
    ][:5]
    return CatalogSearch(query=query, items=items)


@mcp.tool(description="Возвращает товар по id.")
async def get_item(item_id: int) -> CatalogItem:
    for item in CATALOG:
        if item["id"] == item_id:
            return CatalogItem(**item)
    raise ToolError(f"Товар с id={item_id} не найден.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-7s [catalog] %(message)s",
    )
    mcp.run(transport="stdio")
