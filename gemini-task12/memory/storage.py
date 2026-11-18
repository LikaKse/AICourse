"""Общее для хранилищ слоёв: заголовок файла, чтение JSON, (де)сериализация сообщений.

Каждый слой пишет СВОЙ файл с заголовком ``{"format_version": 1, "layer": "<имя слоя>", "saved_at": ...}``.
Поле ``layer`` проверяется при загрузке: файл одного слоя нельзя по ошибке загрузить в другой.
Запись атомарная (tmp-файл + os.replace), значение API-ключа маскируется.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pydantic
from google.genai import types
from google.genai import version as genai_version

from utils import atomic_write_text, redact_secrets

FORMAT_VERSION = 1
VALID_ROLES = {"user", "model"}


class MemoryLoadError(Exception):
    """Файл слоя памяти повреждён или несовместим с форматом."""


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> object:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise MemoryLoadError(f"{Path(path).name}: файл не в UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise MemoryLoadError(f"{Path(path).name}: некорректный JSON (строка {exc.lineno}): {exc.msg}") from exc


def check_header(payload: object, layer: str) -> dict:
    if not isinstance(payload, dict):
        raise MemoryLoadError(f"{layer}: ожидался JSON-объект верхнего уровня")
    if payload.get("format_version") != FORMAT_VERSION:
        raise MemoryLoadError(f"{layer}: неподдерживаемая версия формата {payload.get('format_version')!r}")
    if payload.get("layer") != layer:
        raise MemoryLoadError(f"{layer}: файл принадлежит слою {payload.get('layer')!r}")
    return payload


def save_layer_file(path: str | os.PathLike, layer: str, body: dict) -> None:
    data = {"format_version": FORMAT_VERSION, "layer": layer, "saved_at": now(), **body}
    atomic_write_text(path, redact_secrets(json.dumps(data, ensure_ascii=False, indent=2)))


# ---------------------------------------------------------------------
# Сообщения
# ---------------------------------------------------------------------

def text_content(role: str, text: str) -> types.Content:
    return types.Content(role=role, parts=[types.Part.from_text(text=text)])


def content_text(content: types.Content | None) -> str:
    """Текст сообщения без «мыслей» модели."""
    if content is None:
        return ""
    return "\n".join(p.text for p in content.parts or [] if p.text and not p.thought).strip()


def dump_contents(contents: list[types.Content]) -> list[dict]:
    return [c.model_dump(mode="json", exclude_none=True) for c in contents]


def load_contents(items: object, where: str) -> list[types.Content]:
    if not isinstance(items, list):
        raise MemoryLoadError(f"{where}: поле 'messages' отсутствует или не список")
    out = []
    for i, item in enumerate(items):
        try:
            content = types.Content.model_validate(item)
        except pydantic.ValidationError as exc:
            raise MemoryLoadError(f"{where}: сообщение #{i} несовместимо с google-genai "
                                  f"{genai_version.__version__}: {exc.errors()[0]['msg']}") from exc
        if content.role not in VALID_ROLES or not content.parts:
            raise MemoryLoadError(f"{where}: сообщение #{i}: неверная роль {content.role!r} или пустые parts")
        out.append(content)
    return out
