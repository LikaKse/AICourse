"""Типизированные ошибки памяти Дня 15."""


class MemoryError(Exception):
    """Базовая ошибка подсистемы памяти."""


class MemoryValidationError(MemoryError, ValueError):
    """Некорректная модель памяти или её сериализация."""


class MemoryStoreError(MemoryError):
    """Ошибка доступа к хранилищу памяти."""


class MemoryCorrupt(MemoryStoreError):
    """memory.json отсутствует в ожидаемой структуре или повреждён."""


class MemoryWriteError(MemoryStoreError):
    """Не удалось атомарно сохранить memory.json."""


__all__ = [
    "MemoryCorrupt",
    "MemoryError",
    "MemoryStoreError",
    "MemoryValidationError",
    "MemoryWriteError",
]
