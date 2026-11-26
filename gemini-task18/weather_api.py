"""Клиент Open-Meteo без API-ключа."""

from __future__ import annotations

from typing import Any

import httpx2

import config
from models import WeatherReading, normalize_iso


class WeatherApiError(Exception):
    """Open-Meteo недоступен или вернул неожиданные данные."""


class WeatherApi:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        latitude: float = config.WEATHER_LATITUDE,
        longitude: float = config.WEATHER_LONGITUDE,
        timeout: float = config.WEATHER_TIMEOUT,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or config.WEATHER_API_BASE).rstrip("/")
        self.latitude = latitude
        self.longitude = longitude
        self.timeout = timeout
        self.transport = transport

    async def fetch_current(self) -> WeatherReading:
        url = f"{self.base_url}/v1/forecast"
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "current": "temperature_2m,wind_speed_10m",
            "timezone": "UTC",
        }
        try:
            async with httpx2.AsyncClient(
                timeout=httpx2.Timeout(self.timeout),
                transport=self.transport,
            ) as client:
                response = await client.get(url, params=params)
        except httpx2.TimeoutException as exc:
            raise WeatherApiError(f"Open-Meteo не ответил за {self.timeout:g} с.") from exc
        except httpx2.HTTPError as exc:
            raise WeatherApiError(f"Ошибка сети при запросе к Open-Meteo: {exc}") from exc

        if response.status_code != 200:
            raise WeatherApiError(f"Open-Meteo вернул HTTP {response.status_code}.")

        try:
            payload: Any = response.json()
            current = payload["current"]
            return WeatherReading(
                observed_at=normalize_iso(current["time"], "current.time"),
                temperature_c=float(current["temperature_2m"]),
                wind_speed_kmh=float(current["wind_speed_10m"]),
                source="open-meteo",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise WeatherApiError("Open-Meteo вернул неожиданную структуру данных.") from exc
