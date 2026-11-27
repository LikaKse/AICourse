#!/usr/bin/env bash
# Воспроизводимая демонстрация Дня 20.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== MCP-серверы и объединённый каталог ==="
.venv/bin/python main.py --list-tools

echo
echo "=== Полный флоу и catalog-down ==="
.venv/bin/python main.py --demo
