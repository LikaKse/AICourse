"""Общий интерфейс слоя памяти: своё хранилище (один файл), load / save / clear, блок для промпта."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from google.genai import types

from memory.storage import check_header, content_text, read_json, save_layer_file


@dataclass
class PromptBlock:
    """Часть промпта, пришедшая из одного слоя (для подсчёта токенов по слоям)."""
    layer: str
    contents: list[types.Content]

    @property
    def text(self) -> str:
        return "\n".join(content_text(c) for c in self.contents)


class MemoryLayer(ABC):
    """Слой памяти. Слои ничего не знают друг о друге; связывает их только MemoryManager."""

    name: str = ""

    def __init__(self, path: str | os.PathLike | None) -> None:
        self.path = Path(path) if path is not None else None

    # ----- хранилище -----

    def _read(self) -> dict | None:
        """Тело файла слоя (заголовок проверен) или None, если файла нет / слой в памяти."""
        if self.path is None or not self.path.is_file():
            return None
        return check_header(read_json(self.path), self.name)

    def save(self) -> None:
        if self.path is not None:
            save_layer_file(self.path, self.name, self.to_dict())

    def clear(self) -> None:
        """Очистить ТОЛЬКО этот слой (и его файл). Остальные слои не затрагиваются."""
        self.reset_state()
        self.save()

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def reset_state(self) -> None: ...

    @abstractmethod
    def to_dict(self) -> dict: ...

    # ----- чтение -----

    @abstractmethod
    def prompt_contents(self) -> list[types.Content]:
        """Что этот слой добавляет в промпт по своим правилам чтения (пусто — ничего)."""

    @abstractmethod
    def is_empty(self) -> bool: ...

    @abstractmethod
    def stats(self) -> dict: ...

    @abstractmethod
    def dump(self) -> dict:
        """Содержимое слоя в компактном виде (для отчёта и сравнения после перезапуска)."""
