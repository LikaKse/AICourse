"""Настройки Дня 20: три MCP-сервера и оркестратор."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_BIN = Path(
    os.environ.get(
        "MCP_PYTHON_BIN",
        str(
            (PROJECT_ROOT / ".venv" / "bin" / "python")
            if (PROJECT_ROOT / ".venv" / "bin" / "python").is_file()
            else sys.executable
        ),
    )
)

# ---------- Собственные серверы ----------
SERVERS_DIR = PROJECT_ROOT / "servers"
CATALOG_SERVER = SERVERS_DIR / "catalog_server.py"
NOTES_SERVER = SERVERS_DIR / "notes_server.py"

# ---------- Сторонний Filesystem MCP ----------
NODE_VERSION = "24.22.3"
FILESYSTEM_SERVER_VERSION = "2.4.0"
TOOLS_DIR = PROJECT_ROOT / ".tools"
_MACHINE = platform.machine().lower()
_ARCH = "arm64" if _MACHINE in {"arm64", "aarch64"} else "x64"
NODE_PLATFORM = f"{platform.system().lower()}-{_ARCH}"
NODE_BIN = Path(
    os.environ.get(
        "FS_NODE_BIN",
        TOOLS_DIR / f"node-v{NODE_VERSION}-{NODE_PLATFORM}" / "bin" / "node",
    )
)
FS_SERVER_JS = Path(
    os.environ.get(
        "FS_SERVER_JS",
        TOOLS_DIR
        / "mcp-server"
        / "node_modules"
        / "@modelcontextprotocol"
        / "server-filesystem"
        / "dist"
        / "index.js",
    )
)
WORKSPACE = (PROJECT_ROOT / "workspace").resolve()
CONTEXT_FILE = WORKSPACE / "context.txt"

# ---------- MCP ----------
MCP_CONNECT_TIMEOUT = 30.0
MCP_CALL_TIMEOUT = 30.0

# ---------- Демонстрация ----------
DEMO_QUERY = (
    "Найди ноутбук в каталоге, связанную заметку и контекст закупки, "
    "затем подготовь сводку."
)

# ---------- Логи ----------
LOG_DIR = "logs"
LOG_PREVIEW_CHARS = 500
