"""Типизированные ошибки хранилища и команд инвариантов."""


class InvariantError(Exception):
    """Базовая ошибка подсистемы инвариантов."""


class InvariantValidationError(InvariantError):
    """Некорректные данные правила/проверки/снимка."""


class InvariantCommandError(InvariantError):
    """Команда изменения инварианта отклонена."""


class InvariantDuplicateError(InvariantCommandError):
    """Правило с таким id уже существует."""


class InvariantNotFoundError(InvariantCommandError):
    """Правило с таким id не найдено."""


class InvariantVersionConflict(InvariantCommandError):
    """Версия правила не совпала с expected_version."""


class InvariantRevisionConflict(InvariantCommandError):
    """Revision хранилища не совпала с ожидаемой."""


class InvariantStoreError(InvariantError):
    """Хранилище недоступно или повреждено."""


class InvariantStoreCorrupt(InvariantStoreError):
    """JSON/история не соответствуют ожидаемой структуре."""


class InvariantStoreWriteError(InvariantStoreError):
    """Не удалось атомарно записать snapshot или историю."""


__all__ = [
    "InvariantCommandError",
    "InvariantDuplicateError",
    "InvariantError",
    "InvariantNotFoundError",
    "InvariantRevisionConflict",
    "InvariantStoreCorrupt",
    "InvariantStoreError",
    "InvariantStoreWriteError",
    "InvariantValidationError",
    "InvariantVersionConflict",
]
