"""Pydantic-модели MCP-инструментов и цепочки."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Post(BaseModel):
    model_config = ConfigDict(populate_by_name=True, strict=True)

    user_id: int = Field(alias="userId")
    id: int
    title: str
    body: str


class SearchResult(BaseModel):
    model_config = ConfigDict(strict=True)

    query: str
    items: list[Post]


class Summary(BaseModel):
    model_config = ConfigDict(strict=True)

    source_count: int = Field(ge=1)
    summary: str


class SavedFile(BaseModel):
    model_config = ConfigDict(strict=True)

    path: str
    bytes_written: int = Field(ge=0)
