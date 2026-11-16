"""Хранение полной истории диалога и summary в JSON (раздельно) между перезапусками.

Два независимых файла:

* ``full_history.json`` — ПОЛНАЯ история (types.Content). Сжатые сообщения отсюда не удаляются:
  она нужна для анализа и восстановления. Формат как в Дне 8:
  ``{"format_version", "sdk_version", "saved_at", "history": [...]}``.
* ``summary.json`` — текущее summary и указатель ``summarized_upto`` (индекс первого
  НЕсжатого сообщения полной истории) + журнал всех обновлений summary.
  Активная история = ``full_history[summarized_upto:]``.

Обе записи атомарные (tmp-файл + os.replace), ключ API в файлы не попадает.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import pydantic
from google.genai import types
from google.genai import version as genai_version

from utils import atomic_write_text, redact_secrets

FORMAT_VERSION = 1
VALID_ROLES = {"user", "model"}


class HistoryLoadError(Exception):
    """Файл истории или summary повреждён или несовместим с форматом."""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise HistoryLoadError(f"{path.name}: файл не в UTF-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise HistoryLoadError(
            f"{path.name}: некорректный JSON (строка {exc.lineno}, столбец {exc.colno}): {exc.msg}"
        ) from exc


def _backup(path: Path) -> Path:
    backup = path.with_name(f"{path.stem}.corrupted_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
    path.rename(backup)
    return backup


# ---------------------------------------------------------------------
# Полная история
# ---------------------------------------------------------------------

def serialize_history(history: list[types.Content]) -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "sdk_version": genai_version.__version__,
        "saved_at": _now(),
        "history": [content.model_dump(mode="json", exclude_none=True) for content in history],
    }


def deserialize_history(payload: object) -> list[types.Content]:
    if not isinstance(payload, dict):
        raise HistoryLoadError("ожидался JSON-объект верхнего уровня")
    if payload.get("format_version") != FORMAT_VERSION:
        raise HistoryLoadError(
            f"неподдерживаемая версия формата {payload.get('format_version')!r} (ожидалась {FORMAT_VERSION})"
        )
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
                f"{exc.errors()[0]['msg']}"
            ) from exc
        if content.role not in VALID_ROLES or not content.parts:
            raise HistoryLoadError(f"сообщение #{index}: неверная роль {content.role!r} или пустые parts")
        history.append(content)
    return history


class HistoryStore:
    """Полная история диалога в одном JSON-файле."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.is_file()

    def load(self) -> list[types.Content]:
        """Нет файла → пустая история. Файл повреждён → HistoryLoadError."""
        if not self.exists():
            return []
        return deserialize_history(_read_json(self.path))

    def save(self, history: list[types.Content]) -> None:
        text = json.dumps(serialize_history(history), ensure_ascii=False, indent=2)
        atomic_write_text(self.path, redact_secrets(text))

    def backup_corrupted(self) -> Path:
        return _backup(self.path)


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

@dataclass
class SummaryState:
    """Текущее summary и граница сжатия.

    summarized_upto — сколько первых сообщений полной истории уже вошли в summary.
    Указатель только растёт, поэтому одно и то же сообщение не суммируется повторно.
    updates — журнал обновлений: какие сообщения [start, end) вошли в каждую версию.
    """
    text: str = ""
    summarized_upto: int = 0
    version: int = 0
    updated_at: str | None = None
    updates: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"format_version": FORMAT_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, payload: object) -> "SummaryState":
        if not isinstance(payload, dict):
            raise HistoryLoadError("summary: ожидался JSON-объект верхнего уровня")
        if payload.get("format_version") != FORMAT_VERSION:
            raise HistoryLoadError(f"summary: неподдерживаемая версия формата {payload.get('format_version')!r}")
        text = payload.get("text", "")
        upto = payload.get("summarized_upto", 0)
        version = payload.get("version", 0)
        updates = payload.get("updates", [])
        if not isinstance(text, str):
            raise HistoryLoadError("summary: поле 'text' должно быть строкой")
        if not isinstance(upto, int) or isinstance(upto, bool) or upto < 0:
            raise HistoryLoadError("summary: 'summarized_upto' должно быть целым числом ≥ 0")
        if not isinstance(version, int) or version < 0 or not isinstance(updates, list):
            raise HistoryLoadError("summary: некорректные поля 'version' / 'updates'")
        if upto > 0 and not text.strip():
            raise HistoryLoadError("summary: сообщения помечены сжатыми, но текст summary пуст")
        return cls(text=text, summarized_upto=upto, version=version,
                   updated_at=payload.get("updated_at"), updates=updates)


class SummaryStore:
    """Summary в отдельном JSON-файле (не смешивается с полной историей)."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.is_file()

    def load(self) -> SummaryState:
        if not self.exists():
            return SummaryState()
        return SummaryState.from_dict(_read_json(self.path))

    def save(self, state: SummaryState) -> None:
        state.updated_at = _now()
        text = json.dumps(state.to_dict(), ensure_ascii=False, indent=2)
        atomic_write_text(self.path, redact_secrets(text))

    def backup_corrupted(self) -> Path:
        return _backup(self.path)
