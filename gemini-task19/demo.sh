#!/usr/bin/env bash
# Воспроизводимая демонстрация Дня 19. Очищает только output/, кроме .gitkeep.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

find output -mindepth 1 ! -name '.gitkeep' -delete

echo "=== Три сценария цепочки ==="
.venv/bin/python main.py --demo

echo
echo "=== Файлы output/ ==="
find output -maxdepth 1 -type f -exec basename {} \; | sort
echo
echo "=== Содержимое output/summary.txt ==="
if [ -f output/summary.txt ]; then
  cat output/summary.txt
  echo
else
  echo "output/summary.txt отсутствует" >&2
  exit 1
fi

if [ -e output/should_not_exist.txt ]; then
  echo "ОШИБКА: файл ошибочного сценария существует" >&2
  exit 1
fi
echo
echo "Проверено: should_not_exist.txt не создан."
