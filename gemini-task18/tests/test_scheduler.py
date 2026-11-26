"""Тесты планировщика без реального MCP: несколько запусков и агрегация."""

import unittest

from scheduler import Scheduler


class FakeSchedulerClient:
    def __init__(self):
        self.saved = []
        self.reports = []
        self.calls = []
        self.next_id = 1

    async def call_tool_structured(self, name, arguments=None):
        arguments = arguments or {}
        self.calls.append((name, dict(arguments)))
        if name == "fetch_weather":
            n = self.next_id
            return {
                "observed_at": "2026-11-26T08:00:00+00:00",
                "temperature_c": float(n * 10),
                "wind_speed_kmh": float(n),
                "source": "open-meteo",
            }
        if name == "save_observation":
            saved = {
                "id": self.next_id,
                "observed_at": arguments["observed_at"],
                "fetched_at": f"2026-11-26T10:00:0{self.next_id}+00:00",
                "temperature_c": arguments["temperature_c"],
                "wind_speed_kmh": arguments["wind_speed_kmh"],
                "source": arguments.get("source", "open-meteo"),
            }
            self.saved.append(saved)
            self.next_id += 1
            return saved
        if name == "read_observations":
            return {"items": list(self.saved)}
        if name == "build_report":
            values = [row["temperature_c"] for row in self.saved]
            report = {
                "report_id": len(self.reports) + 1,
                "generated_at": "2026-11-26T10:01:00+00:00",
                "period_start": arguments["period_start"],
                "period_end": arguments["period_end"],
                "record_count": len(self.saved),
                "temperature_avg": sum(values) / len(values) if values else None,
                "temperature_min": min(values) if values else None,
                "temperature_max": max(values) if values else None,
                "wind_speed_avg": sum(r["wind_speed_kmh"] for r in self.saved) / len(self.saved),
                "observation_ids": [row["id"] for row in self.saved],
            }
            self.reports.append(report)
            return report
        raise AssertionError(f"unexpected tool {name}")


async def no_sleep(_seconds):
    return None


class SchedulerTest(unittest.IsolatedAsyncioTestCase):
    async def test_four_runs_and_reports_all_records(self):
        client = FakeSchedulerClient()
        scheduler = Scheduler(
            client,
            poll_interval=0,
            report_every=2,
            report_period=60,
            max_observations=4,
            reports_dir=None,
            now_fn=lambda: "2026-11-26T10:02:00+00:00",
            sleep_fn=no_sleep,
        )
        count = await scheduler.run()
        self.assertEqual(count, 4)
        self.assertEqual(scheduler.report_count, 2)
        self.assertEqual(len(client.saved), 4)
        self.assertEqual(client.reports[-1]["record_count"], 4)
        self.assertEqual(client.reports[-1]["observation_ids"], [1, 2, 3, 4])

        names = [name for name, _ in client.calls]
        self.assertEqual(names.count("fetch_weather"), 4)
        self.assertEqual(names.count("save_observation"), 4)
        self.assertEqual(names.count("read_observations"), 2)
        self.assertEqual(names.count("build_report"), 2)

    async def test_final_report_when_count_not_multiple(self):
        client = FakeSchedulerClient()
        scheduler = Scheduler(
            client,
            poll_interval=0,
            report_every=2,
            report_period=60,
            max_observations=3,
            reports_dir=None,
            now_fn=lambda: "2026-11-26T10:02:00+00:00",
            sleep_fn=no_sleep,
        )
        await scheduler.run()
        self.assertEqual(scheduler.report_count, 2)
        self.assertEqual(client.reports[-1]["record_count"], 3)
        self.assertEqual(client.reports[-1]["observation_ids"], [1, 2, 3])

    async def test_report_calls_read_and_report_via_mcp(self):
        client = FakeSchedulerClient()
        scheduler = Scheduler(
            client,
            poll_interval=0,
            report_every=1,
            report_period=60,
            max_observations=1,
            reports_dir=None,
            now_fn=lambda: "2026-11-26T10:02:00+00:00",
            sleep_fn=no_sleep,
        )
        await scheduler.run()
        names = [name for name, _ in client.calls]
        self.assertEqual(names, [
            "fetch_weather",
            "save_observation",
            "read_observations",
            "build_report",
        ])


if __name__ == "__main__":
    unittest.main()
