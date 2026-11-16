"""Общие мелочи: маскировка ключа и атомарная запись файлов."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import config


def redact_secrets(text: str) -> str:
    """Заменяет значение API-ключа на *** (на случай, если ключ попадёт в текст ошибки или файл)."""
    key = os.environ.get(config.API_KEY_ENV)
    if key and len(key) >= 8 and key in text:
        text = text.replace(key, "***")
    return text


def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    """Записать файл атомарно: сначала во временный файл, затем os.replace.

    Если процесс упадёт во время записи, старый файл останется целым.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def write_json(path: str | os.PathLike, data: Any) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    atomic_write_text(path, redact_secrets(text))


def normalize_text(text: str) -> str:
    """Нормализация для сравнения по шаблонам: нижний регистр, ё→е, без * и `,
    разделители тысяч убраны («18 500 000» → «18500000»), пробелы схлопнуты."""
    import re

    text = (text or "").lower().replace("ё", "е").replace("\u00a0", " ").replace("\u202f", " ")
    text = text.replace("*", "").replace("`", "")
    text = re.sub(r"(?<=\d)[ .,](?=\d{3}(?!\d))", "", text)
    return re.sub(r"\s+", " ", text).strip()
