"""Тесты MCP-сервера: регистрация, схемы и вызовы через in-memory Client."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mcp import Client

import server
from models import WeatherReading
from storage import ObservationStore
from weather_api import WeatherApiError


class FakeWeatherApi:
    def __init__(self, reading=None, error=None):
        self.reading = reading or WeatherReading(
            observed_at="2026-11-26T10:00:00+00:00",
            temperature_c=1.5,
            wind_speed_kmh=9.0,
        )
        self.error = error
        self.calls = 0

    async def fetch_current(self):
        self.calls += 1
        if self.error:
            raise self.error
        return self.reading


class ServerTest(unittest.IsolatedAsyncioTestCase):
    async def test_tools_and_schemas(self):
        tools = await server.mcp.list_tools()
        names = {tool.name for tool in tools}
        self.assertEqual(names, {
            "fetch_weather", "save_observation", "read_observations", "build_report",
        })
        by_name = {tool.name: tool for tool in tools}
        save_schema = by_name["save_observation"].input_schema
        self.assertEqual(save_schema["required"], ["observed_at", "temperature_c", "wind_speed_kmh"])
        report_schema = by_name["build_report"].input_schema
        self.assertEqual(report_schema["required"], ["period_start", "period_end"])
        self.assertEqual(by_name["fetch_weather"].input_schema.get("required", []), [])

    async def test_full_tool_cycle_in_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ObservationStore(Path(tmp) / "test.db")
            fake_api = FakeWeatherApi()
            with mock.patch.object(server, "store", store), mock.patch.object(server, "api", fake_api):
                async with Client(server.mcp) as client:
                    reading = await client.call_tool("fetch_weather", {})
                    self.assertFalse(reading.is_error)
                    data = reading.structured_content
                    self.assertEqual(data["temperature_c"], 1.5)

                    saved = await client.call_tool("save_observation", {
                        "observed_at": data["observed_at"],
                        "temperature_c": data["temperature_c"],
                        "wind_speed_kmh": data["wind_speed_kmh"],
                    })
                    self.assertFalse(saved.is_error)
                    self.assertEqual(saved.structured_content["id"], 1)

                    history = await client.call_tool("read_observations", {
                        "period_start": "2026-11-26T09:00:00+00:00",
                        "period_end": "2026-11-26T12:00:00+00:00",
                    })
                    self.assertEqual(len(history.structured_content["items"]), 1)

                    report = await client.call_tool("build_report", {
                        "period_start": "2026-11-26T09:00:00+00:00",
                        "period_end": "2026-11-26T12:00:00+00:00",
                    })
                    self.assertEqual(report.structured_content["record_count"], 1)
                    self.assertEqual(report.structured_content["observation_ids"], [1])
            store.close()

    async def test_fetch_error_is_mcp_error(self):
        fake_api = FakeWeatherApi(error=WeatherApiError("API down"))
        with mock.patch.object(server, "api", fake_api):
            async with Client(server.mcp) as client:
                result = await client.call_tool("fetch_weather", {})
        self.assertTrue(result.is_error)
        text = "\n".join(block.text for block in result.content if getattr(block, "type", None) == "text")
        self.assertIn("API down", text)


if __name__ == "__main__":
    unittest.main()
