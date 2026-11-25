"""Настройки проекта «День 17. Первый инструмент MCP».

Только константы и пути: при импорте ничего не запускается и API не вызывается.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------- Gemini ----------
MODEL = "gemini-3.5-flash-lite"
API_KEY_ENV = "GEMINI_API_KEY"
THINKING_LEVEL = "LOW"
MAX_OUTPUT_TOKENS = 2048
MAX_ITERATIONS = 10

SYSTEM_INSTRUCTION = (
    "Ты — полезный ассистент. Отвечай на русском языке, кратко и по делу.\n"
    "У тебя есть локальные инструменты и собственный MCP-инструмент get_task.\n"
    "- Для вопросов о задачах JSONPlaceholder вызывай MCP get_task и передавай task_id.\n"
    "- Не придумывай данные задачи: используй только результат get_task.\n"
    "- Для арифметики вызывай локальный calculate.\n"
    "- Если инструмент вернул ошибку, объясни её пользователю.\n"
)

# ---------- Локальные инструменты (День 6) ----------
TIMEZONE = "Asia/Ho_Chi_Minh"
USD_TO_VND_RATE = 25_000
SUPPORTED_CURRENCIES = ("USD", "VND")
CALC_MAX_EXPRESSION_LENGTH = 200
CALC_MAX_POWER_EXPONENT = 100
CALC_MAX_ABS_RESULT = 1e30

# ---------- Собственный MCP-сервер ----------
TASKS_API_BASE = os.environ.get(
    "TASKS_API_BASE", "https://jsonplaceholder.typicode.com"
).rstrip("/")
TASKS_API_TIMEOUT = 10.0
TASK_ID = 5

SERVER_NAME = "jsonplaceholder-tasks"
SERVER_VERSION = "0.1.0"
SERVER_PY = PROJECT_ROOT / "server.py"

# Запускаем сервер интерпретатором из .venv, чтобы у него был установлен mcp.
_DEFAULT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
PYTHON_BIN = Path(
    os.environ.get(
        "MCP_PYTHON_BIN",
        str(_DEFAULT_PYTHON if _DEFAULT_PYTHON.is_file() else sys.executable),
    )
)

MCP_EXPOSED_TOOLS = ("get_task",)
MCP_REQUIRED_TOOL = "get_task"
MCP_CONNECT_TIMEOUT = 30.0
MCP_CALL_TIMEOUT = 30.0

# ---------- Демонстрация ----------
DEMO_PROMPT = "Что за задача с id 5 на JSONPlaceholder? Кратко опиши её."

# ---------- Gemini network ----------
REQUEST_TIMEOUT_MS = 60_000
RETRY_ATTEMPTS = 3
RETRY_STATUS_CODES = (429, 500, 502, 503, 504)

# ---------- Логи ----------
LOG_DIR = "logs"
LOG_PREVIEW_CHARS = 500
