"""Атомарное JSON-хранилище памяти Дня 15.

Файл: ``state_dir/memory.json``. Активные данные и журнал лежат в одном
атомарно заменяемом файле, поэтому snapshot и история не могут разойтись.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .errors import (
    MemoryCorrupt,
    MemoryStoreError,
    MemoryValidationError,
    MemoryWriteError,
)
from .models import MemorySnapshot


class MemoryStore:
    """Snapshot + журнал в одном атомарном файле с межпроцессной блокировкой."""

    def __init__(self, state_dir: str | os.PathLike[str]):
        self.state_dir = Path(state_dir)
        self.memory_path = self.state_dir / "memory.json"
        self.lock_path = self.state_dir / ".memory.lock"

    # ----- блокировка -----

    @contextmanager
    def locked(self) -> Iterator[None]:
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise MemoryStoreError(
                f"не удалось создать каталог памяти: {exc}") from exc
        try:
            lock_file = self.lock_path.open("a+", encoding="utf-8")
        except OSError as exc:
            raise MemoryStoreError(
                f"не удалось открыть lock-файл: {exc}") from exc

        with lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            except OSError as exc:
                raise MemoryStoreError(
                    f"не удалось получить блокировку памяти: {exc}") from exc
            try:
                yield
            finally:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass

    # ----- чтение -----

    def load(self) -> MemorySnapshot:
        with self.locked():
            return self._load_unlocked()

    def _load_unlocked(self) -> MemorySnapshot:
        if not self.memory_path.exists():
            return MemorySnapshot()
        try:
            raw = self.memory_path.read_bytes()
        except OSError as exc:
            raise MemoryStoreError(
                f"не удалось прочитать memory.json: {exc}") from exc
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MemoryCorrupt(
                "memory.json: файл не в UTF-8") from exc
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MemoryCorrupt(
                f"memory.json: некорректный JSON (строка {exc.lineno}): "
                f"{exc.msg}") from exc
        if not isinstance(data, dict):
            raise MemoryCorrupt("memory.json: ожидался JSON-объект")
        try:
            snapshot = MemorySnapshot.from_dict(data)
            self._validate_snapshot_consistency(snapshot)
            return snapshot
        except MemoryValidationError as exc:
            raise MemoryCorrupt(
                f"memory.json: повреждённая структура: {exc}") from exc
        except Exception as exc:
            # Любая структурная ошибка загрузки — fail-closed.
            raise MemoryCorrupt(
                f"memory.json: повреждённая структура: {exc}") from exc

    @staticmethod
    def _validate_snapshot_consistency(snapshot: MemorySnapshot) -> None:
        """Проверить транзакционную согласованность snapshot и журнала."""
        if snapshot.revision == 0:
            if (snapshot.journal
                    or snapshot.mutable
                    or snapshot.cumulative
                    or snapshot.temporary
                    or snapshot.mutable_tombstones
                    or snapshot.temporary_tombstones
                    or snapshot.completed_tasks):
                raise MemoryValidationError(
                    "revision 0 допускает только пустую память")
            return

        if len(snapshot.journal) != snapshot.revision:
            raise MemoryValidationError(
                "длина журнала не совпадает с revision snapshot")
        revisions = [event.revision for event in snapshot.journal]
        if revisions != list(range(1, snapshot.revision + 1)):
            raise MemoryValidationError(
                "revision журнала должны образовывать 1..revision")
        if snapshot.updated_at is None:
            raise MemoryValidationError(
                "updated_at обязателен при revision > 0")

    # ----- запись -----

    def save(self, snapshot: MemorySnapshot) -> None:
        with self.locked():
            self._save_unlocked(snapshot)

    def _save_unlocked(self, snapshot: MemorySnapshot) -> None:
        """Атомарно заменить memory.json. Вызывающий держит ``locked()``."""
        if not isinstance(snapshot, MemorySnapshot):
            raise MemoryWriteError("save ожидает MemorySnapshot")
        try:
            self._validate_snapshot_consistency(snapshot)
        except MemoryValidationError as exc:
            raise MemoryWriteError(
                f"snapshot не согласован: {exc}") from exc

        # Защита от отката revision и потерянных обновлений.
        if self.memory_path.exists():
            current = self._load_unlocked()
            if snapshot.revision <= current.revision:
                raise MemoryWriteError(
                    "нельзя сохранить snapshot без увеличения revision")

        try:
            payload = json.dumps(
                snapshot.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise MemoryWriteError(
                f"не удалось сериализовать память: {exc}") from exc

        fd = -1
        tmp_name = ""
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=".memory-", suffix=".tmp", dir=self.state_dir)
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.memory_path)
            tmp_name = ""
        except OSError as exc:
            raise MemoryWriteError(
                f"не удалось атомарно сохранить memory.json: {exc}") from exc
        finally:
            if fd != -1:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if tmp_name:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass


__all__ = ["MemoryStore"]
