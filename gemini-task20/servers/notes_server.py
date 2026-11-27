"""MCP-сервер заметок."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

from models import Note, NotesSearch, NotesSummary  # noqa: E402
from servers.notes_data import NOTES  # noqa: E402

mcp = MCPServer(
    "notes",
    version="0.1.0",
    instructions="Локальные заметки.",
    log_level="WARNING",
)


@mcp.tool(description="Ищет заметки по вхождению в title/text.")
async def search(query: str) -> NotesSearch:
    needle = query.strip().casefold()
    items = [
        Note(**note)
        for note in NOTES
        if needle in note["title"].casefold() or needle in note["text"].casefold()
    ][:5]
    return NotesSearch(query=query, items=items)


@mcp.tool(description="Возвращает заметку по id.")
async def get_note(note_id: int) -> Note:
    for note in NOTES:
        if note["id"] == note_id:
            return Note(**note)
    raise ToolError(f"Заметка с id={note_id} не найдена.")


@mcp.tool(description="Строит краткое резюме по выбранным заметкам.")
async def summarize(note_ids: list[int]) -> NotesSummary:
    if not note_ids:
        raise ToolError("note_ids не должен быть пустым.")
    by_id = {note["id"]: note for note in NOTES}
    missing = [note_id for note_id in note_ids if note_id not in by_id]
    if missing:
        raise ToolError(f"Заметки не найдены: {missing}.")
    selected = [by_id[note_id] for note_id in note_ids]
    summary = " ".join(
        f"{note['title']}: {note['text']}" for note in selected
    )
    if not summary.strip():
        raise ToolError("Не удалось построить непустое резюме.")
    return NotesSummary(note_ids=list(note_ids), summary=summary)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-7s [notes] %(message)s",
    )
    mcp.run(transport="stdio")
