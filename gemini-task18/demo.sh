#!/usr/bin/env bash
# Воспроизводимая демонстрация Дня 18. Удаляет только data/demo.db.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

rm -f data/demo.db data/demo.db-wal data/demo.db-shm

echo "=== 1. MCP-инструменты ==="
.venv/bin/python scheduler.py --list-tools

echo
echo "=== 2. Четыре наблюдения и два отчёта ==="
.venv/bin/python scheduler.py --demo --db data/demo.db

echo
echo "=== 3. Все наблюдения в SQLite ==="
.venv/bin/python - <<'PY'
from storage import ObservationStore

store = ObservationStore("data/demo.db")
rows = store.read_observations(
    "2000-01-01T00:00:00+00:00",
    "2100-01-01T00:00:00+00:00",
)
for row in rows:
    print(
        f"id={row.id} observed_at={row.observed_at} fetched_at={row.fetched_at} "
        f"temperature={row.temperature_c} wind={row.wind_speed_kmh}"
    )
reports = store.read_reports()
print(
    f"reports={len(reports)} last_count={reports[-1].record_count} "
    f"last_ids={reports[-1].observation_ids}"
)
store.close()
PY
