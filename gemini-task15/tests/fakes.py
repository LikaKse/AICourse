"""Тестовые часы и вспомогательные значения."""

from datetime import datetime, timezone

UTC = timezone.utc


def at(day=1, hour=0, minute=0, second=0):
    return datetime(2026, 12, day, hour, minute, second, tzinfo=UTC)


class FrozenClock:
    def __init__(self, moment):
        self.moment = moment

    def now(self):
        return self.moment

    def set(self, moment):
        self.moment = moment
