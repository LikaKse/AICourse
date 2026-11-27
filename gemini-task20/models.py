"""Pydantic-модели данных и MCP-ответов."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CatalogItem(BaseModel):
    model_config = ConfigDict(strict=True)

    id: int
    name: str
    category: str
    description: str
    price_usd: int = Field(ge=0)


class CatalogSearch(BaseModel):
    model_config = ConfigDict(strict=True)

    query: str
    items: list[CatalogItem]


class Note(BaseModel):
    model_config = ConfigDict(strict=True)

    id: int
    title: str
    text: str


class NotesSearch(BaseModel):
    model_config = ConfigDict(strict=True)

    query: str
    items: list[Note]


class NotesSummary(BaseModel):
    model_config = ConfigDict(strict=True)

    note_ids: list[int]
    summary: str


class FileContent(BaseModel):
    model_config = ConfigDict(strict=True)

    content: str
