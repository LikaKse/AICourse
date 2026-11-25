"""Настройки проекта «День 16. Подключение MCP».

Здесь только константы и пути: при импорте ничего не запускается и API не вызывается.
Node.js и MCP-сервер лежат в отдельной папке .tools и не меняют системное окружение.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------- Модель Gemini ----------
MODEL = "gemini-3.5-flash-lite"
API_KEY_ENV = "GEMINI_API_KEY"

# LOW — как в Днях 5–7: достаточно планирования цепочки инструментов, но недорого.
THINKING_LEVEL = "LOW"
MAX_OUTPUT_TOKENS = 2048

# Максимум обращений к модели на один запрос пользователя.
MAX_ITERATIONS = 10

SYSTEM_INSTRUCTION = (
    "Ты — полезный ассистент. Отвечай на русском языке, кратко и по делу.\n"
    "У тебя есть локальные инструменты и инструменты MCP-сервера Filesystem.\n"
    "- Для любых вычислений вызывай локальный calculate, а не считай в уме.\n"
    "- Для чтения файлов используй MCP read_text_file. Пути можно передавать абсолютные "
    "или относительные к папке demo_workspace.\n"
    "- Сначала читай файл, дожидайся результата, и только потом используй прочитанные "
    "числа в calculate. Не придумывай содержимое файла.\n"
    "- Проценты в calculate записывай через умножение: 10 процентов — это * 1.1.\n"
    "- Если инструмент вернул ошибку, объясни её пользователю или исправь аргументы "
    "и попробуй снова.\n"
    "- Не пытайся изменять файлы: сервер запущен в режиме только для чтения."
)

# ---------- Локальные инструменты (День 6) ----------
TIMEZONE = "Asia/Ho_Chi_Minh"
USD_TO_VND_RATE = 25_000
SUPPORTED_CURRENCIES = ("USD", "VND")

CALC_MAX_EXPRESSION_LENGTH = 200
CALC_MAX_POWER_EXPONENT = 100
CALC_MAX_ABS_RESULT = 1e30

# ---------- MCP Filesystem ----------
WORKSPACE = (PROJECT_ROOT / "demo_workspace").resolve()
DEMO_FILE = WORKSPACE / "numbers.txt"

# Версии зафиксированы: Node 24 нужен серверу Filesystem 2.4.0.
NODE_VERSION = "24.22.3"
NODE_MAJOR_MIN = 24
MCP_SERVER_VERSION = "2.4.0"

TOOLS_DIR = PROJECT_ROOT / ".tools"


def _node_platform_tag() -> str:
    """Тег официального tarball Node.js для текущей ОС и архитектуры."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    os_tag = {"darwin": "darwin", "linux": "linux"}.get(system)
    if os_tag is None:
        raise RuntimeError(f"Неподдерживаемая ОС для project-local Node: {platform.system()}")
    arch_tag = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x64", "amd64": "x64"}.get(machine)
    if arch_tag is None:
        raise RuntimeError(f"Неподдерживаемая архитектура для project-local Node: {machine}")
    return f"{os_tag}-{arch_tag}"


NODE_PLATFORM_TAG = _node_platform_tag()
NODE_DIR = TOOLS_DIR / f"node-v{NODE_VERSION}-{NODE_PLATFORM_TAG}"
NODE_BIN = Path(os.environ.get("MCP_NODE_BIN", NODE_DIR / "bin" / "node"))

MCP_SERVER_DIR = TOOLS_DIR / "mcp-server"
MCP_SERVER_JS = Path(
    os.environ.get(
        "MCP_SERVER_JS",
        MCP_SERVER_DIR
        / "node_modules"
        / "@modelcontextprotocol"
        / "server-filesystem"
        / "dist"
        / "index.js",
    )
)

# Серверный read-only режим: сервер сам не отдаёт инструменты записи.
MCP_READ_ONLY = True
# Дополнительно запрещаем клиентским Roots расширять доступ за пределы WORKSPACE.
MCP_CONFINE_ROOTS = True

MCP_CONNECT_TIMEOUT = 120.0
MCP_CALL_TIMEOUT = 30.0

# Какие MCP-инструменты разрешено показывать модели.
# read_file — deprecated-алиас read_text_file, read_media_file возвращает base64.
MCP_EXPOSED_TOOLS = (
    "read_text_file",
    "read_multiple_files",
    "list_directory",
    "list_directory_with_sizes",
    "directory_tree",
    "search_files",
    "get_file_info",
    "list_allowed_directories",
)
MCP_REQUIRED_TOOL = "read_text_file"

# ---------- Смешанный сценарий ----------
DEMO_PROMPT = (
    "В папке demo_workspace есть файл numbers.txt с количеством, ценой за единицу "
    "и процентом наценки. Рассчитай итоговую стоимость с наценкой."
)
DEMO_EXPECTED_TOTAL = 330_000

# ---------- Сеть и повторы Gemini ----------
REQUEST_TIMEOUT_MS = 60_000
RETRY_ATTEMPTS = 3
RETRY_STATUS_CODES = (429, 500, 502, 503, 504)

# ---------- Логи и результаты ----------
LOG_DIR = "logs"
LOG_PREVIEW_CHARS = 500
RESULTS_DIR = PROJECT_ROOT / "results"

# ---------- Тарифы (для оценки стоимости, не для автоматических тестов) ----------
PRICE_INPUT_PER_1M = 0.30
PRICE_OUTPUT_PER_1M = 2.50
