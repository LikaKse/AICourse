"""Настройки Дня 18: планировщик, Open-Meteo и SQLite.

При импорте ничего не запускается и сетевые запросы не выполняются.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------- Источник данных ----------
WEATHER_API_BASE = os.environ.get("WEATHER_API_BASE", "https://api.open-meteo.com").rstrip("/")
WEATHER_LATITUDE = 52.52
WEATHER_LONGITUDE = 13.41
WEATHER_TIMEOUT = 10.0

# ---------- SQLite ----------
DATABASE_PATH = Path(
    os.environ.get("TASK18_DB_PATH", PROJECT_ROOT / "data" / "history.db")
)
REPORTS_DIR = PROJECT_ROOT / "reports"

# ---------- MCP ----------
SERVER_NAME = "weather-history"
SERVER_VERSION = "0.1.0"
SERVER_PY = PROJECT_ROOT / "server.py"
_DEFAULT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
PYTHON_BIN = Path(
    os.environ.get(
        "MCP_PYTHON_BIN",
        str(_DEFAULT_PYTHON if _DEFAULT_PYTHON.is_file() else sys.executable),
    )
)
MCP_EXPOSED_TOOLS = (
    "fetch_weather",
    "save_observation",
    "read_observations",
    "build_report",
)
MCP_REQUIRED_TOOLS = frozenset(MCP_EXPOSED_TOOLS)
MCP_CONNECT_TIMEOUT = 30.0
MCP_CALL_TIMEOUT = 30.0

# ---------- Планировщик ----------
POLL_INTERVAL_SECONDS = 60.0
REPORT_EVERY_N_OBSERVATIONS = 5
REPORT_PERIOD_SECONDS = 300.0
MAX_OBSERVATIONS = 0  # 0 = работать бесконечно

DEMO_POLL_INTERVAL = 1.0
DEMO_REPORT_EVERY = 2
DEMO_REPORT_PERIOD = 60.0
DEMO_MAX_OBSERVATIONS = 4
DEMO_DB_PATH = PROJECT_ROOT / "data" / "demo.db"

# ---------- Логи ----------
LOG_DIR = "logs"
LOG_PREVIEW_CHARS = 500
