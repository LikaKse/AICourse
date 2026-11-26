"""Pydantic-модели и работа с UTC-временем."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_iso(value: str, field: str = "timestamp") -> str:
    """ISO-строка → UTC ISO с offset; naive время считается UTC."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} должен быть непустой ISO-строкой.")
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field} имеет неверный ISO-формат: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds")


class WeatherReading(BaseModel):
    observed_at: str
    temperature_c: float
    wind_speed_kmh: float
    source: str = "open-meteo"


class SavedObservation(BaseModel):
    id: int
    observed_at: str
    fetched_at: str
    temperature_c: float
    wind_speed_kmh: float
    source: str


class Observation(BaseModel):
    id: int
    observed_at: str
    fetched_at: str
    temperature_c: float
    wind_speed_kmh: float
    source: str


class Observations(BaseModel):
    items: list[Observation]


class Report(BaseModel):
    report_id: int
    generated_at: str
    period_start: str
    period_end: str
    record_count: int
    temperature_avg: float | None
    temperature_min: float | None
    temperature_max: float | None
    wind_speed_avg: float | None
    observation_ids: list[int]
