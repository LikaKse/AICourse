"""Собственный MCP-сервер Дня 18.

Инструменты:
    fetch_weather() -> WeatherReading
    save_observation(...) -> SavedObservation
    read_observations(period_start, period_end) -> Observations
    build_report(period_start, period_end) -> Report
"""

from __future__ import annotations

import logging
import sys

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import config
from models import Observations, Report, SavedObservation, WeatherReading
from storage import ObservationStore
from weather_api import WeatherApi, WeatherApiError

mcp = MCPServer(
    config.SERVER_NAME,
    version=config.SERVER_VERSION,
    instructions="Сервер погодных наблюдений Open-Meteo с историей в SQLite.",
    log_level="WARNING",
)

api = WeatherApi()
store = ObservationStore(config.DATABASE_PATH)


@mcp.tool(description="Возвращает текущую температуру и скорость ветра из Open-Meteo.")
async def fetch_weather() -> WeatherReading:
    try:
        return await api.fetch_current()
    except WeatherApiError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool(description="Сохраняет одно успешное наблюдение; fetched_at ставится сервером.")
async def save_observation(
    observed_at: str,
    temperature_c: float,
    wind_speed_kmh: float,
    source: str = "open-meteo",
    raw_json: str = "",
) -> SavedObservation:
    try:
        return store.save_observation(
            observed_at=observed_at,
            temperature_c=temperature_c,
            wind_speed_kmh=wind_speed_kmh,
            source=source,
            raw_json=raw_json,
        )
    except Exception as exc:
        raise ToolError(f"Не удалось сохранить наблюдение: {exc}") from exc


@mcp.tool(description="Читает все наблюдения за период по fetched_at.")
async def read_observations(period_start: str, period_end: str) -> Observations:
    try:
        return Observations(items=store.read_observations(period_start, period_end))
    except Exception as exc:
        raise ToolError(f"Не удалось прочитать наблюдения: {exc}") from exc


@mcp.tool(description="Строит и сохраняет агрегированный отчёт по всем записям периода.")
async def build_report(period_start: str, period_end: str) -> Report:
    try:
        return store.build_report(period_start, period_end)
    except Exception as exc:
        raise ToolError(f"Не удалось построить отчёт: {exc}") from exc


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-7s [mcp-server] %(message)s",
    )
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    mcp.run(transport="stdio")
