"""Отдельное хранилище инвариантов, журнал транзакций и append-only история.

Файлы в state-dir:
- ``invariants.json`` — snapshot;
- ``invariant_history.jsonl`` — append-only история;
- ``invariant_transaction.json`` — журнал незавершённой операции;
- ``.invariants.lock`` — межпроцессная блокировка.

Транзакция: journal → snapshot → history → удаление journal. Если процесс
погибает между шагами, следующий ``load()``/``read_history()``/команда
выполняет восстановление и доводит операцию до согласованного состояния.

Формат journal (единственный допустимый набор полей):

    {
      "format_version": 1,
      "status": "prepared",
      "transaction_id": "tx-<32 lowercase hex>",
      "old_revision": <int >= 0>,
      "new_snapshot": {<snapshot>},
      "change": {<InvariantChange>},
      "history_before_size": <int >= 0>
    }

Неизвестные поля, неизвестный status, неверный transaction_id и
несогласованные revision отклоняются как InvariantStoreCorrupt.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .errors import (
    InvariantStoreCorrupt,
    InvariantStoreError,
    InvariantStoreWriteError,
)
from .models import InvariantChange, InvariantSnapshot

#: Единственный допустимый набор полей журнала. Лишние поля означают
#: несовместимый/повреждённый журнал и отклоняются fail-closed.
_JOURNAL_FIELDS = frozenset({
    "format_version",
    "status",
    "transaction_id",
    "old_revision",
    "new_snapshot",
    "change",
    "history_before_size",
})
_TRANSACTION_ID_RE = re.compile(r"^tx-[0-9a-f]{32}$")


class InvariantStore:
    """Snapshot + история + журнал незавершённой транзакции."""

    def __init__(self, state_dir: str | os.PathLike[str]):
        self.state_dir = Path(state_dir)
        self.snapshot_path = self.state_dir / "invariants.json"
        self.history_path = self.state_dir / "invariant_history.jsonl"
        self.journal_path = self.state_dir / "invariant_transaction.json"
        self.lock_path = self.state_dir / ".invariants.lock"

    # ----- блокировка и восстановление -----

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Эксклюзивная критическая секция для процессов и потоков."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                self._recover_unlocked()
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    # ----- чтение -----

    def load(self) -> InvariantSnapshot:
        # Решение о пустом хранилище принимается уже под блокировкой.
        with self.locked():
            return self._load_unlocked()

    def _load_unlocked(self) -> InvariantSnapshot:
        exists_snapshot = self.snapshot_path.exists()
        exists_history = self.history_path.exists()
        if not exists_snapshot:
            if exists_history and self.history_path.stat().st_size:
                raise InvariantStoreCorrupt(
                    "история существует, но invariants.json отсутствует")
            return InvariantSnapshot()

        snapshot = self._read_snapshot()
        history = self._read_history_unlocked()
        if history:
            last_revision = history[-1].store_revision
            if last_revision != snapshot.revision:
                raise InvariantStoreCorrupt(
                    "revision snapshot не совпадает с последней записью истории")
        elif snapshot.revision != 0:
            raise InvariantStoreCorrupt(
                "snapshot имеет revision > 0 без истории изменений")
        return snapshot

    def read_history(self) -> tuple[InvariantChange, ...]:
        with self.locked():
            return self._read_history_unlocked()

    def _read_history_unlocked(self) -> tuple[InvariantChange, ...]:
        if not self.history_path.exists():
            return ()
        try:
            raw = self.history_path.read_bytes()
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise InvariantStoreError(
                f"не удалось прочитать историю инвариантов: {exc}") from exc

        changes: list[InvariantChange] = []
        for line_no, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                raise InvariantStoreCorrupt(
                    f"история инвариантов: пустая строка {line_no}")
            try:
                parsed = json.loads(line)
                changes.append(InvariantChange.from_dict(parsed))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise InvariantStoreCorrupt(
                    f"история инвариантов: некорректная запись в строке "
                    f"{line_no}: {exc}") from exc
        self._validate_history_sequence(changes)
        return tuple(changes)

    @staticmethod
    def _validate_history_sequence(changes: list[InvariantChange]) -> None:
        """Проверить непрерывную последовательность revision и уникальность event_id."""
        seen: set[str] = set()
        for expected_revision, change in enumerate(changes, 1):
            if change.store_revision != expected_revision:
                raise InvariantStoreCorrupt(
                    "история инвариантов: пропуск/повтор/перестановка revision: "
                    f"ожидалась {expected_revision}, получена {change.store_revision}")
            if change.event_id in seen:
                raise InvariantStoreCorrupt(
                    f"история инвариантов: повторяющийся event_id {change.event_id!r}")
            seen.add(change.event_id)

    # ----- запись -----

    def _commit(self, snapshot: InvariantSnapshot, change: InvariantChange) -> None:
        """Публичная обёртка для тестов/внутренних вызовов."""
        with self.locked():
            self._commit_unlocked(snapshot, change)

    def _commit_unlocked(
        self, snapshot: InvariantSnapshot, change: InvariantChange
    ) -> None:
        """Выполнить транзакцию. Вызывающий обязан держать ``locked()``."""
        if change.store_revision != snapshot.revision:
            raise InvariantStoreWriteError(
                "store_revision события не совпадает с revision snapshot")
        if snapshot.revision < 1:
            raise InvariantStoreWriteError("snapshot revision должен быть >= 1")

        history_before_size = (
            self.history_path.stat().st_size if self.history_path.exists() else 0)
        journal = {
            "format_version": 1,
            "status": "prepared",
            "transaction_id": f"tx-{uuid.uuid4().hex}",
            "old_revision": snapshot.revision - 1,
            "new_snapshot": snapshot.to_dict(),
            "change": change.to_dict(),
            "history_before_size": history_before_size,
        }
        self._write_journal(journal)
        # После journal падение процесса оставляет журнал: следующий
        # load()/command завершит операцию.
        self._write_snapshot(snapshot)
        self._append_history_unlocked(change)
        self._remove_journal_unlocked()

    def _recover_unlocked(self) -> None:
        """Восстановить незавершённую транзакцию; идемпотентно."""
        if not self.journal_path.exists():
            return
        try:
            raw = self.journal_path.read_text(encoding="utf-8")
            journal = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvariantStoreCorrupt(
                f"журнал транзакции повреждён: {exc}") from exc

        try:
            if type(journal) is not dict:
                raise ValueError("журнал должен быть объектом")
            unknown = set(journal) - _JOURNAL_FIELDS
            missing = _JOURNAL_FIELDS - set(journal)
            if unknown or missing:
                raise ValueError(
                    f"неверный набор полей журнала: неизвестные {sorted(unknown)}, "
                    f"отсутствуют {sorted(missing)}")
            if type(journal["format_version"]) is not int or journal["format_version"] != 1:
                raise ValueError("неверный format_version журнала")
            if journal["status"] != "prepared":
                raise ValueError(f"неизвестный статус журнала {journal['status']!r}")
            transaction_id = journal["transaction_id"]
            if (not isinstance(transaction_id, str)
                    or not _TRANSACTION_ID_RE.fullmatch(transaction_id)):
                raise ValueError("неверный transaction_id журнала")
            old_revision = journal["old_revision"]
            history_before_size = journal["history_before_size"]
            if type(old_revision) is not int or old_revision < 0:
                raise ValueError("old_revision должен быть >= 0")
            if type(history_before_size) is not int or history_before_size < 0:
                raise ValueError("history_before_size должен быть >= 0")
            new_snapshot = InvariantSnapshot.from_dict(journal["new_snapshot"])
            change = InvariantChange.from_dict(journal["change"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InvariantStoreCorrupt(
                f"журнал транзакции имеет неверную структуру: {exc}") from exc

        if new_snapshot.revision != old_revision + 1:
            raise InvariantStoreCorrupt(
                "журнал: new_snapshot.revision должна быть old_revision + 1")
        if change.store_revision != new_snapshot.revision:
            raise InvariantStoreCorrupt(
                "журнал: revision события не совпадает с new_snapshot")

        current_snapshot: InvariantSnapshot | None = None
        if self.snapshot_path.exists():
            current_snapshot = self._read_snapshot()
            if current_snapshot.revision not in (old_revision, new_snapshot.revision):
                raise InvariantStoreCorrupt(
                    "журнал: текущий snapshot не соответствует old/new revision")

        data = b""
        if self.history_path.exists():
            try:
                data = self.history_path.read_bytes()
            except OSError as exc:
                raise InvariantStoreError(
                    f"не удалось прочитать историю для восстановления: {exc}") from exc
        if len(data) < history_before_size:
            raise InvariantStoreCorrupt(
                "журнал: история короче зафиксированного префикса")

        prefix = data[:history_before_size]
        tail = data[history_before_size:]
        if prefix and not prefix.endswith(b"\n"):
            raise InvariantStoreCorrupt(
                "журнал: префикс истории не заканчивается переводом строки")

        expected_line = self._history_line(change)
        applied = False
        if tail:
            if tail == expected_line:
                applied = True
            elif expected_line.startswith(tail):
                # Оборванная запись является префиксом именно ожидаемой
                # записи из journal; её можно безопасно заменить полной.
                applied = False
            else:
                raise InvariantStoreCorrupt(
                    "журнал: хвост истории не является частью ожидаемой записи")

        if not applied:
            self._write_bytes_atomic(self.history_path, prefix + expected_line)

        if current_snapshot is None:
            if old_revision != 0:
                raise InvariantStoreCorrupt(
                    "журнал: snapshot отсутствует при old_revision > 0")
            self._write_snapshot(new_snapshot)
        elif current_snapshot.revision == old_revision:
            self._write_snapshot(new_snapshot)
        # Если revision уже new — snapshot записан до падения.

        # Полная проверка после восстановления.
        recovered_snapshot = self._read_snapshot()
        recovered_history = self._read_history_unlocked()
        if (not recovered_history
                or recovered_history[-1].store_revision != recovered_snapshot.revision
                or recovered_history[-1].event_id != change.event_id):
            raise InvariantStoreCorrupt(
                "после восстановления snapshot и history не согласованы")
        self._remove_journal_unlocked()

    # ----- низкоуровневые операции -----

    def _read_snapshot(self) -> InvariantSnapshot:
        try:
            text = self.snapshot_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise InvariantStoreError(
                f"не удалось прочитать invariants.json: {exc}") from exc
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InvariantStoreCorrupt(
                f"invariants.json: некорректный JSON в строке {exc.lineno}: "
                f"{exc.msg}") from exc
        try:
            return InvariantSnapshot.from_dict(raw)
        except (TypeError, ValueError) as exc:
            raise InvariantStoreCorrupt(
                f"invariants.json: некорректная структура: {exc}") from exc

    def _write_snapshot(self, snapshot: InvariantSnapshot) -> None:
        payload = json.dumps(
            snapshot.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        self._write_bytes_atomic(
            self.snapshot_path, (payload + "\n").encode("utf-8"))

    def _write_journal(self, journal: dict[str, Any]) -> None:
        payload = json.dumps(
            journal, ensure_ascii=False, indent=2, sort_keys=True)
        self._write_bytes_atomic(
            self.journal_path, (payload + "\n").encode("utf-8"))

    def _remove_journal_unlocked(self) -> None:
        try:
            self.journal_path.unlink(missing_ok=True)
        except OSError as exc:
            raise InvariantStoreWriteError(
                f"не удалось удалить журнал транзакции: {exc}") from exc

    @staticmethod
    def _history_line(change: InvariantChange) -> bytes:
        return (
            json.dumps(change.to_dict(), ensure_ascii=False,
                       separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("utf-8")

    def _append_history_unlocked(self, change: InvariantChange) -> None:
        line = self._history_line(change)
        try:
            with self.history_path.open("ab") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise InvariantStoreWriteError(
                f"не удалось записать историю инвариантов: {exc}") from exc

    def _write_bytes_atomic(self, target: Path, data: bytes) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            fd, temporary = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=self.state_dir)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            temporary = None
        except OSError as exc:
            raise InvariantStoreWriteError(
                f"не удалось атомарно записать {target.name}: {exc}") from exc
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass


__all__ = ["InvariantStore"]
