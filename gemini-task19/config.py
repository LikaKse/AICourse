"""Настройки Дня 19: MCP-цепочка search → summarize → save_to_file."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------- Источник данных ----------
SEARCH_API_BASE = os.environ.get(
    "TASK19_API_BASE", "https://jsonplaceholder.typicode.com"
).rstrip("/")
SEARCH_TIMEOUT = 10.0
SEARCH_MAX_RESULTS = 10

# ---------- MCP ----------
SERVER_NAME = "content-chain"
SERVER_VERSION = "0.1.0"
SERVER_PY = PROJECT_ROOT / "server.py"
_DEFAULT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
PYTHON_BIN = Path(
    os.environ.get(
        "MCP_PYTHON_BIN",
        str(_DEFAULT_PYTHON if _DEFAULT_PYTHON.is_file() else sys.executable),
    )
)
MCP_EXPOSED_TOOLS = ("search", "summarize", "save_to_file")
MCP_REQUIRED_TOOLS = frozenset(MCP_EXPOSED_TOOLS)
MCP_CONNECT_TIMEOUT = 30.0
MCP_CALL_TIMEOUT = 30.0

# ---------- Файлы ----------
OUTPUT_DIR = Path(os.environ.get("TASK19_OUTPUT_DIR", PROJECT_ROOT / "output")).resolve()
DEFAULT_FILENAME = "summary.txt"
ERROR_FILENAME = "should_not_exist.txt"

# ---------- Демонстрация ----------
DEMO_QUERY = "qui est esse"
EMPTY_QUERY = "zzzz-no-match"

# ---------- Логи ----------
LOG_DIR = "logs"
LOG_PREVIEW_CHARS = 500
