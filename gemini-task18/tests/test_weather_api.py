"""Тесты клиента Open-Meteo без сети."""

import unittest

import httpx2

from weather_api import WeatherApi, WeatherApiError

PAYLOAD = {
    "current": {
        "time": "2026-11-26T10:00",
        "temperature_2m": 0.6,
        "wind_speed_10m": 10.0,
    }
}


def api_with(handler) -> WeatherApi:
    return WeatherApi(base_url="https://example.test", transport=httpx2.MockTransport(handler))


class WeatherApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_success_and_request(self):
        seen = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            seen.append((str(request.url), dict(request.url.params)))
            return httpx2.Response(200, json=PAYLOAD)

        reading = await api_with(handler).fetch_current()
        self.assertEqual(reading.temperature_c, 0.6)
        self.assertEqual(reading.wind_speed_kmh, 10.0)
        self.assertEqual(reading.observed_at, "2026-11-26T10:00:00+00:00")
        url, params = seen[0]
        self.assertIn("/v1/forecast", url)
        self.assertEqual(params["current"], "temperature_2m,wind_speed_10m")
        self.assertEqual(params["timezone"], "UTC")

    async def test_http_error(self):
        def handler(request):
            return httpx2.Response(500, text="boom")

        with self.assertRaises(WeatherApiError) as ctx:
            await api_with(handler).fetch_current()
        self.assertIn("HTTP 500", str(ctx.exception))

    async def test_timeout(self):
        def handler(request):
            raise httpx2.ReadTimeout("slow", request=request)

        with self.assertRaises(WeatherApiError) as ctx:
            await api_with(handler).fetch_current()
        self.assertIn("не ответил", str(ctx.exception))

    async def test_bad_structure(self):
        def handler(request):
            return httpx2.Response(200, json={"current": {"time": "2026-11-26T10:00"}})

        with self.assertRaises(WeatherApiError):
            await api_with(handler).fetch_current()


if __name__ == "__main__":
    unittest.main()
