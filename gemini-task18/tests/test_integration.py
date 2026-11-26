"""Интеграция: реальный stdio MCP, SQLite и локальная заглушка Open-Meteo.

Ключевой сценарий:
    2 запуска → закрыть MCP/сервер/SQLite → 2 запуска → отчёт по всем 4.

По умолчанию пропускается:
    RUN_MCP_INTEGRATION=1 .venv/bin/python -m unittest tests.test_integration -v
"""

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from mcp_client import WeatherMCPClient
from scheduler import Scheduler
from storage import ObservationStore
from weather_api import WeatherApi

RUN = os.environ.get("RUN_MCP_INTEGRATION") == "1"
LIVE = os.environ.get("RUN_LIVE_API") == "1"


class _Handler(BaseHTTPRequestHandler):
    seen: list[str] = []

    def do_GET(self):  # noqa: N802
        self.seen.append(self.path)
        if self.path.startswith("/v1/forecast"):
            body = json.dumps({
                "current": {
                    "time": "2026-11-26T10:00",
                    "temperature_2m": 1.5,
                    "wind_speed_10m": 9.0,
                }
            }).encode()
            self.send_response(200)
        else:
            body = b"{}"
            self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def start_stub():
    _Handler.seen = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, f"http://127.0.0.1:{httpd.server_port}"


async def no_sleep(_seconds):
    return None


@unittest.skipUnless(RUN, "нужен RUN_MCP_INTEGRATION=1")
class PersistenceIntegrationTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd, cls.api_base = start_stub()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    async def _run_phase(self, db_path, reports_dir, max_observations):
        async with WeatherMCPClient(db_path=db_path, api_base=self.api_base) as client:
            scheduler = Scheduler(
                client,
                poll_interval=0,
                report_every=2,
                report_period=60,
                max_observations=max_observations,
                reports_dir=reports_dir,
                sleep_fn=no_sleep,
            )
            await scheduler.run()
            return scheduler

    async def test_two_plus_two_runs_accumulate_all_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "history.db"
            reports_dir = tmp_path / "reports"

            first = await self._run_phase(db_path, reports_dir, 2)
            self.assertEqual(first.saved_count, 2)
            self.assertEqual(first.report_count, 1)

            # Новый MCP-клиент/сервер/SQLite-соединение — имитация перезапуска.
            second = await self._run_phase(db_path, reports_dir, 2)
            self.assertEqual(second.saved_count, 2)
            self.assertEqual(second.report_count, 1)

            store = ObservationStore(db_path)
            rows = store.read_observations(
                "2026-11-26T00:00:00+00:00",
                "2026-11-26T23:59:59+00:00",
            )
            reports = store.read_reports()
            store.close()

        self.assertEqual(len(rows), 4)
        self.assertEqual([row.observed_at for row in rows], ["2026-11-26T10:00:00+00:00"] * 4)
        self.assertEqual(reports[-1].record_count, 4)
        self.assertEqual(reports[-1].observation_ids, [1, 2, 3, 4])
        self.assertEqual(len(_Handler.seen), 4)


@unittest.skipUnless(LIVE, "нужен RUN_LIVE_API=1")
class LiveOpenMeteoTest(unittest.IsolatedAsyncioTestCase):
    async def test_live_api(self):
        reading = await WeatherApi().fetch_current()
        self.assertIsInstance(reading.temperature_c, float)
        self.assertIsInstance(reading.wind_speed_kmh, float)
        self.assertTrue(reading.observed_at)


if __name__ == "__main__":
    unittest.main()
