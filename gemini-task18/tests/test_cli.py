"""Тесты CLI/конфигурации планировщика без сети."""

import json
import tempfile
import unittest
from pathlib import Path

from scheduler import Scheduler, build_parser
from tests.test_scheduler import FakeSchedulerClient, no_sleep


class ParserTest(unittest.TestCase):
    def test_demo_and_overrides(self):
        args = build_parser().parse_args([
            "--demo", "--poll-interval", "0.1", "--report-every", "2",
            "--max-observations", "4",
        ])
        self.assertTrue(args.demo)
        self.assertEqual(args.poll_interval, 0.1)
        self.assertEqual(args.report_every, 2)
        self.assertEqual(args.max_observations, 4)

    def test_default_is_infinite(self):
        args = build_parser().parse_args([])
        self.assertFalse(args.demo)
        self.assertFalse(args.once)
        self.assertIsNone(args.max_observations)


class SchedulerConfigTest(unittest.TestCase):
    def test_invalid_values(self):
        client = FakeSchedulerClient()
        for kwargs in (
            {"poll_interval": -1},
            {"report_every": 0},
            {"report_period": 0},
            {"max_observations": -1},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    Scheduler(client, **kwargs)


class ReportFileTest(unittest.IsolatedAsyncioTestCase):
    async def test_report_json_written(self):
        client = FakeSchedulerClient()
        with tempfile.TemporaryDirectory() as tmp:
            scheduler = Scheduler(
                client,
                poll_interval=0,
                report_every=1,
                report_period=60,
                max_observations=1,
                reports_dir=Path(tmp),
                now_fn=lambda: "2026-11-26T10:02:00+00:00",
                sleep_fn=no_sleep,
            )
            await scheduler.run()
            files = list(Path(tmp).glob("report_*.json"))
            self.assertEqual(len(files), 1)
            data = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(data["record_count"], 1)


if __name__ == "__main__":
    unittest.main()
