"""SQLite-хранилище наблюдений и агрегированных отчётов.

Периоды всегда фильтруются по fetched_at — времени получения данных приложением.
observed_at сохраняется отдельно как исходное время API.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable

from models import (
    Observation,
    Report,
    SavedObservation,
    normalize_iso,
    utc_now_iso,
)


class ObservationStore:
    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        clock: Callable[[], str] = utc_now_iso,
    ) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.initialize()

    def initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            observed_at TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            source TEXT NOT NULL,
            temperature_c REAL NOT NULL,
            wind_speed_kmh REAL NOT NULL,
            raw_json TEXT NOT NULL DEFAULT ''
        );

        CREATE INDEX IF NOT EXISTS idx_observations_fetched_at
            ON observations(fetched_at);

        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            generated_at TEXT NOT NULL,
            period_start TEXT NOT NULL,
            period_end TEXT NOT NULL,
            record_count INTEGER NOT NULL,
            temperature_avg REAL,
            temperature_min REAL,
            temperature_max REAL,
            wind_speed_avg REAL,
            observation_ids TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_reports_generated_at
            ON reports(generated_at);
        """
        with self._lock:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            with self._conn:
                self._conn.executescript(schema)

    def save_observation(
        self,
        observed_at: str,
        temperature_c: float,
        wind_speed_kmh: float,
        source: str = "open-meteo",
        raw_json: str = "",
    ) -> SavedObservation:
        observed = normalize_iso(observed_at, "observed_at")
        fetched = self._clock()
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO observations
                    (observed_at, fetched_at, source, temperature_c, wind_speed_kmh, raw_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (observed, fetched, source, float(temperature_c), float(wind_speed_kmh), raw_json or ""),
            )
            observation_id = int(cursor.lastrowid)
        return SavedObservation(
            id=observation_id,
            observed_at=observed,
            fetched_at=fetched,
            temperature_c=float(temperature_c),
            wind_speed_kmh=float(wind_speed_kmh),
            source=source,
        )

    def _period(self, period_start: str, period_end: str) -> tuple[str, str]:
        start = normalize_iso(period_start, "period_start")
        end = normalize_iso(period_end, "period_end")
        if start > end:
            raise ValueError("period_start не может быть позже period_end.")
        return start, end

    def read_observations(self, period_start: str, period_end: str) -> list[Observation]:
        start, end = self._period(period_start, period_end)
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, observed_at, fetched_at, temperature_c, wind_speed_kmh, source
                FROM observations
                WHERE fetched_at >= ? AND fetched_at <= ?
                ORDER BY fetched_at, id
                """,
                (start, end),
            ).fetchall()
        return [
            Observation(
                id=row["id"],
                observed_at=row["observed_at"],
                fetched_at=row["fetched_at"],
                temperature_c=row["temperature_c"],
                wind_speed_kmh=row["wind_speed_kmh"],
                source=row["source"],
            )
            for row in rows
        ]

    def build_report(self, period_start: str, period_end: str) -> Report:
        start, end = self._period(period_start, period_end)
        with self._lock, self._conn:
            aggregate = self._conn.execute(
                """
                SELECT
                    COUNT(*) AS record_count,
                    AVG(temperature_c) AS temperature_avg,
                    MIN(temperature_c) AS temperature_min,
                    MAX(temperature_c) AS temperature_max,
                    AVG(wind_speed_kmh) AS wind_speed_avg
                FROM observations
                WHERE fetched_at >= ? AND fetched_at <= ?
                """,
                (start, end),
            ).fetchone()
            ids = [
                row["id"]
                for row in self._conn.execute(
                    """
                    SELECT id FROM observations
                    WHERE fetched_at >= ? AND fetched_at <= ?
                    ORDER BY fetched_at, id
                    """,
                    (start, end),
                ).fetchall()
            ]
            record_count = int(aggregate["record_count"])
            generated_at = self._clock()
            temperature_avg = _round(aggregate["temperature_avg"])
            temperature_min = _round(aggregate["temperature_min"])
            temperature_max = _round(aggregate["temperature_max"])
            wind_speed_avg = _round(aggregate["wind_speed_avg"])
            cursor = self._conn.execute(
                """
                INSERT INTO reports
                    (generated_at, period_start, period_end, record_count,
                     temperature_avg, temperature_min, temperature_max, wind_speed_avg,
                     observation_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    generated_at, start, end, record_count,
                    temperature_avg, temperature_min, temperature_max, wind_speed_avg,
                    json.dumps(ids),
                ),
            )
            report_id = int(cursor.lastrowid)
        return Report(
            report_id=report_id,
            generated_at=generated_at,
            period_start=start,
            period_end=end,
            record_count=record_count,
            temperature_avg=temperature_avg,
            temperature_min=temperature_min,
            temperature_max=temperature_max,
            wind_speed_avg=wind_speed_avg,
            observation_ids=ids,
        )

    def read_reports(self) -> list[Report]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, generated_at, period_start, period_end, record_count,
                       temperature_avg, temperature_min, temperature_max,
                       wind_speed_avg, observation_ids
                FROM reports ORDER BY id
                """
            ).fetchall()
        return [
            Report(
                report_id=row["id"],
                generated_at=row["generated_at"],
                period_start=row["period_start"],
                period_end=row["period_end"],
                record_count=row["record_count"],
                temperature_avg=row["temperature_avg"],
                temperature_min=row["temperature_min"],
                temperature_max=row["temperature_max"],
                wind_speed_avg=row["wind_speed_avg"],
                observation_ids=json.loads(row["observation_ids"]),
            )
            for row in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _round(value: Any) -> float | None:
    return None if value is None else round(float(value), 3)
