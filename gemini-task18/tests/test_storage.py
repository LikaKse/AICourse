"""Тесты SQLite: fetched_at, агрегаты и сохранение между перезапусками."""

import itertools
import tempfile
import unittest
from pathlib import Path

from storage import ObservationStore

START = "2026-11-26T09:00:00+00:00"
END = "2026-11-26T13:00:00+00:00"


class StorageTest(unittest.TestCase):
    def test_filter_by_fetched_at_not_observed_at(self):
        ticks = itertools.count(1)
        store = ObservationStore(
            ":memory:",
            clock=lambda: f"2026-11-26T10:00:{next(ticks):02d}+00:00",
        )
        # observed_at одинаковый, но это три разных запуска.
        for temp in (10.0, 20.0, 30.0):
            store.save_observation("2026-11-26T08:00:00", temp, 5.0)
        rows = store.read_observations(START, END)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({row.observed_at for row in rows}), 1)
        self.assertEqual(len({row.fetched_at for row in rows}), 3)

        partial = store.read_observations(
            "2026-11-26T10:00:01+00:00",
            "2026-11-26T10:00:02+00:00",
        )
        self.assertEqual([row.temperature_c for row in partial], [10.0, 20.0])
        store.close()

    def test_report_aggregates_all_records(self):
        ticks = itertools.count(1)
        store = ObservationStore(
            ":memory:",
            clock=lambda: f"2026-11-26T10:00:{next(ticks):02d}+00:00",
        )
        for temp in (10.0, 20.0, 30.0):
            store.save_observation("2026-11-26T08:00:00", temp, 5.0)

        report = store.build_report(START, END)
        self.assertEqual(report.record_count, 3)
        self.assertEqual(report.observation_ids, [1, 2, 3])
        self.assertEqual(report.temperature_avg, 20.0)
        self.assertEqual(report.temperature_min, 10.0)
        self.assertEqual(report.temperature_max, 30.0)
        self.assertEqual(report.wind_speed_avg, 5.0)
        self.assertEqual(len(store.read_reports()), 1)
        store.close()

    def test_empty_period(self):
        store = ObservationStore(":memory:")
        report = store.build_report(START, END)
        self.assertEqual(report.record_count, 0)
        self.assertEqual(report.observation_ids, [])
        self.assertIsNone(report.temperature_avg)
        store.close()

    def test_invalid_period(self):
        store = ObservationStore(":memory:")
        with self.assertRaises(ValueError):
            store.read_observations(END, START)
        store.close()

    def test_persistence_across_restarts_two_plus_two_runs(self):
        """2 запуска → закрыть БД → открыть → ещё 2 → отчёт по всем 4."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.db"
            ticks = itertools.count(1)
            clock = lambda: f"2026-11-26T10:00:{next(ticks):02d}+00:00"

            store1 = ObservationStore(path, clock=clock)
            store1.save_observation("2026-11-26T08:00:00", 10.0, 5.0)
            store1.save_observation("2026-11-26T08:00:00", 20.0, 6.0)
            store1.close()

            store2 = ObservationStore(path, clock=clock)
            self.assertEqual(len(store2.read_observations(START, END)), 2)
            store2.save_observation("2026-11-26T08:00:00", 30.0, 7.0)
            store2.save_observation("2026-11-26T08:00:00", 40.0, 8.0)
            rows = store2.read_observations(START, END)
            report = store2.build_report(START, END)
            store2.close()

        self.assertEqual(len(rows), 4)
        self.assertEqual(report.record_count, 4)
        self.assertEqual(report.observation_ids, [1, 2, 3, 4])
        self.assertEqual(report.temperature_avg, 25.0)
        self.assertEqual(report.temperature_min, 10.0)
        self.assertEqual(report.temperature_max, 40.0)


if __name__ == "__main__":
    unittest.main()
