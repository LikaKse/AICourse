"""Хранение истории диалога агента в JSON между перезапусками программы.

История сохраняется в формате Gemini: каждый types.Content сериализуется штатным методом
pydantic-модели SDK (model_dump(mode="json")) и восстанавливается через Content.model_validate().
Поэтому без потерь сохраняются все поля: текст, function_call (с id и args), function_response,
thought / thought_signature (bytes → base64) и любые другие поля, которые знает SDK.

Формат файла:
    {
      "format_version": 1,
      "sdk_version": "2.34.0",
      "saved_at": "2026-11-10T12:00:00",
      "history": [ {"role": "user", "parts": [...]}, {"role": "model", "parts": [...]}, ... ]
    }

В файл попадает только история диалога: ни ключа API, ни конфигурации клиента там нет.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pydantic
from google.genai import types
from google.genai import version as genai_version

from agent import redact_secrets

FORMAT_VERSION = 1
VALID_ROLES = {"user", "model"}


class HistoryLoadError(Exception):
    """Файл истории повреждён или несовместим с текущей версией SDK / формата."""


def serialize_history(history: list[types.Content]) -> dict:
    """История → JSON-совместимый словарь (без None-полей)."""
    return {
        "format_version": FORMAT_VERSION,
        "sdk_version": genai_version.__version__,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "history": [content.model_dump(mode="json", exclude_none=True) for content in history],
    }


def deserialize_history(payload: object) -> list[types.Content]:
    """JSON-словарь → список types.Content. При любой несовместимости — HistoryLoadError."""
    if not isinstance(payload, dict):
        raise HistoryLoadError("ожидался JSON-объект верхнего уровня")
    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise HistoryLoadError(f"неподдерживаемая версия формата: {version!r} (ожидалась {FORMAT_VERSION})")
    items = payload.get("history")
    if not isinstance(items, list):
        raise HistoryLoadError("поле 'history' отсутствует или не является списком")

    history: list[types.Content] = []
    for index, item in enumerate(items):
        try:
            content = types.Content.model_validate(item)
        except pydantic.ValidationError as exc:
            raise HistoryLoadError(
                f"сообщение #{index} несовместимо с google-genai {genai_version.__version__}: "
                f"{exc.error_count()} ошибк(и) валидации, первая: {exc.errors()[0]['msg']}"
            ) from exc
        if content.role not in VALID_ROLES or not content.parts:
            raise HistoryLoadError(f"сообщение #{index}: неверная роль {content.role!r} или пустые parts")
        history.append(content)
    return history


class HistoryStore:
    """Чтение и запись истории диалога в один JSON-файл."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.is_file()

    def load(self) -> list[types.Content]:
        """Загрузить историю. Нет файла → пустая история. Файл повреждён → HistoryLoadError."""
        if not self.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except UnicodeDecodeError as exc:
            raise HistoryLoadError(f"файл не в UTF-8: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise HistoryLoadError(f"некорректный JSON (строка {exc.lineno}, столбец {exc.colno}): {exc.msg}") from exc
        return deserialize_history(payload)

    def save(self, history: list[types.Content]) -> None:
        """Сохранить историю атомарно: сначала во временный файл, затем замена.

        Если процесс упадёт во время записи, старый файл останется целым.
        """
        text = json.dumps(serialize_history(history), ensure_ascii=False, indent=2)
        text = redact_secrets(text)  # на случай, если пользователь вставил ключ прямо в сообщение
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                tmp.write(text)
            os.replace(tmp_name, self.path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def clear(self) -> None:
        """Очистить сохранённую историю (файл остаётся, но с пустым списком)."""
        self.save([])

    def backup_corrupted(self) -> Path:
        """Переименовать повреждённый файл, чтобы не потерять его и начать с чистой истории."""
        backup = self.path.with_name(f"{self.path.stem}.corrupted_{datetime.now():%Y%m%d_%H%M%S}{self.path.suffix}")
        self.path.rename(backup)
        return backup
