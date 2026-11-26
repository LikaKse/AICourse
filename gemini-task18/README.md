# День 18. Планировщик и фоновые задачи

Фоновый планировщик снимает текущую погоду из Open-Meteo, сохраняет каждое
успешное наблюдение в SQLite и периодически строит агрегированный отчёт через
собственный MCP-сервер. Gemini не используется.

## Полный путь данных

```text
scheduler.py
  │  каждые POLL_INTERVAL_SECONDS
  ▼
WeatherMCPClient (ClientSession, stdio)
  │
  ├── call_tool("fetch_weather")
  │       ▼
  │   server.py → WeatherApi → Open-Meteo
  │       ▼
  │   WeatherReading
  │
  ├── call_tool("save_observation", {...})
  │       ▼
  │   server.py → ObservationStore → SQLite INSERT observations
  │
  └── каждые REPORT_EVERY_N_OBSERVATIONS
          ├── call_tool("read_observations", period_start, period_end)
          └── call_tool("build_report", period_start, period_end)
                  ▼
              SQLite SELECT + INSERT reports
                  ▼
              Report {record_count, min, max, avg, observation_ids}
```

Планировщик не импортирует `storage`, `weather_api` и не обращается к SQLite
напрямую: все операции идут через MCP-клиент.

## Важно: `fetched_at`

- `observed_at` — исходное время наблюдения, которое вернул Open-Meteo.
- `fetched_at` — время получения/сохранения данных приложением, ставится
  MCP-сервером.
- Фильтрация периода в `read_observations` и `build_report` идёт **по
  `fetched_at`**.
- Поэтому каждый запуск планировщика учитывается отдельно, даже если API
  возвращает один и тот же `observed_at`.

## MCP-инструменты

| Инструмент | Входные параметры | Результат |
|---|---|---|
| `fetch_weather` | нет | `WeatherReading` |
| `save_observation` | `observed_at`, `temperature_c`, `wind_speed_kmh`, `source`, `raw_json` | `SavedObservation` с `id` и `fetched_at` |
| `read_observations` | `period_start`, `period_end` | `Observations {items: [...]}` |
| `build_report` | `period_start`, `period_end` | `Report` и запись в `reports` |

Схемы строятся MCP SDK из type hints. Сервер запускается через stdio:

```text
.venv/bin/python -u server.py
```

## SQLite

Файл по умолчанию: `data/history.db`.

```sql
CREATE TABLE observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    source TEXT NOT NULL,
    temperature_c REAL NOT NULL,
    wind_speed_kmh REAL NOT NULL,
    raw_json TEXT NOT NULL DEFAULT ''
);

CREATE INDEX idx_observations_fetched_at ON observations(fetched_at);

CREATE TABLE reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generated_at TEXT NOT NULL,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    record_count INTEGER NOT NULL,
    temperature_avg REAL,
    temperature_min REAL,
    temperature_max REAL,
    wind_speed_avg REAL,
    observation_ids TEXT NOT NULL
);
```

`observation_ids` — JSON-массив id всех записей периода, по которым построен
отчёт. Это позволяет доказать, что отчёт включает все записи, а не только
последнюю.

## Планировщик

`asyncio`-цикл без внешних зависимостей.

| Параметр | По умолчанию | Смысл |
|---|---|---|
| `POLL_INTERVAL_SECONDS` | 60 | интервал наблюдений |
| `REPORT_EVERY_N_OBSERVATIONS` | 5 | отчёт каждые N наблюдений |
| `REPORT_PERIOD_SECONDS` | 300 | окно агрегации `[now - period, now]` по `fetched_at` |
| `MAX_OBSERVATIONS` | 0 | 0 — бесконечно; N — остановиться после N |
| `TASK18_DB_PATH` | `data/history.db` | SQLite |
| `WEATHER_API_BASE` | Open-Meteo | URL API/заглушки |

После последнего наблюдения финальный отчёт строится даже если число
наблюдений не кратно `report_every`.

## Установка

```bash
cd ~/work/AICourse/gemini-task18
../venv/bin/python -m venv .venv
.venv/bin/pip install -r requirements.lock
```

Отдельное окружение; общий `AICourse/venv` и предыдущие проекты не изменяются.

## Команды

```bash
# MCP-инструменты и схемы
.venv/bin/python scheduler.py --list-tools

# один запуск: наблюдение + отчёт
.venv/bin/python scheduler.py --once --db data/once.db

# короткая демонстрация: 4 наблюдения, 2 отчёта
.venv/bin/python scheduler.py --demo --db data/demo.db

# фоновый режим
.venv/bin/python scheduler.py \
  --poll-interval 60 \
  --report-every 5 \
  --report-period 300

# другой интервал для теста
.venv/bin/python scheduler.py \
  --poll-interval 2 \
  --report-every 2 \
  --report-period 60 \
  --max-observations 4 \
  --db data/test.db
```

## Тесты

### Офлайн

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
```

Результат текущего прогона:

```text
Ran 34 tests
OK (skipped=2)
```

Пропущены интеграционный сценарий и живой Open-Meteo.

Проверяется:

- Open-Meteo через `httpx2.MockTransport`;
- SQLite: фильтрация по `fetched_at`, агрегаты, пустой период;
- 2 запуска → закрытие БД → 2 запуска → отчёт по всем 4 записям и
  `observation_ids == [1, 2, 3, 4]`;
- схемы и вызовы 4 MCP-инструментов через in-memory `Client`;
- MCP-клиент: lifecycle, таймауты, ошибки, закрытие;
- планировщик: 4 запуска, отчёты после 2-го и 4-го, отчёт по всем id;
- планировщик вызывает `read_observations` и `build_report` именно через MCP;
- отсутствие Gemini-зависимости и ключей.

### Интеграционный

```bash
RUN_MCP_INTEGRATION=1 .venv/bin/python -m unittest tests.test_integration -v
```

Результат текущего прогона:

```text
Ran 2 tests
OK (skipped=1)
```

Сценарий:

1. Локальная HTTP-заглушка Open-Meteo.
2. 2 запуска через реальный stdio MCP-сервер.
3. Закрытие MCP-клиента, сервера и SQLite.
4. Повторное открытие той же базы, ещё 2 запуска.
5. Отчёт: `record_count=4`, `observation_ids=[1,2,3,4]`.
6. Все `observed_at` одинаковые — накопление всё равно по `fetched_at`.

### Живая проверка Open-Meteo

```bash
RUN_LIVE_API=1 .venv/bin/python -m unittest tests.test_integration.LiveOpenMeteoTest -v
```

## Демонстрация

```bash
./demo.sh
```

Скрипт очищает только `data/demo.db`, запускает 4 наблюдения с интервалом 1 с,
показывает два отчёта и затем печатает содержимое SQLite.

Подробный сценарий для видео — в `DEMO.md`.

## Ограничения

- Open-Meteo — публичный демонстрационный API; данные не для продакшена.
- Нет retry и кэша: каждый цикл делает новый HTTP-запрос.
- Планировщик не является системным cron/launchd; это долгоживущий процесс.
  Для автозапуска можно использовать `nohup` или `launchd`, но это вне кода.
- Отчёт охватывает настраиваемое окно `REPORT_PERIOD_SECONDS`, а не всю историю
  безусловно.
- Реальные отчёты сохраняются в SQLite; JSON-копии — в `reports/`.
- Gemini не используется, ключей нет.
- Коммит не создавался.
