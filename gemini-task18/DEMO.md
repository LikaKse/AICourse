# Сценарий видео Дня 18

Ориентировочная длительность: 3–4 минуты.
Gemini не используется, API-ключи не нужны.

## Подготовка

```bash
cd ~/work/AICourse/gemini-task18
.venv/bin/python --version
```

Показать, что окружение отдельное:

```bash
.venv/bin/pip list --format=freeze | grep -E '^(mcp|httpx2|pydantic)='
```

## 1. MCP-инструменты

```bash
.venv/bin/python scheduler.py --list-tools
```

Что показать:

- сервер `weather-history 0.1.0`;
- протокол `2025-11-25`;
- четыре инструмента:
  - `fetch_weather`;
  - `save_observation`;
  - `read_observations`;
  - `build_report`;
- схемы параметров.

## 2. Демонстрационный запуск

```bash
./demo.sh
```

Что показать:

- четыре строки `run #1..#4`;
- одинаковый `observed_at` от Open-Meteo;
- разные `fetched_at` — приложение учитывает каждый запуск;
- `report #1` с двумя записями;
- `report #2` с четырьмя записями;
- `ObservationStore`: 4 строки и последний отчёт `record_count=4`,
  `observation_ids=[1, 2, 3, 4]`.

## 3. Пояснение

Проговорить:

1. `scheduler.py` не обращается к SQLite и Open-Meteo напрямую.
2. Все вызовы идут через `WeatherMCPClient` и stdio MCP-сервер.
3. `fetch_weather` получает данные.
4. `save_observation` сохраняет каждую успешную итерацию.
5. `read_observations` читает период.
6. `build_report` агрегирует все записи периода и сохраняет отчёт.
7. Фильтрация идёт по `fetched_at`, поэтому одинаковый `observed_at` не
   приводит к потере запусков.

## 4. Проверка накопления

```bash
.venv/bin/python scheduler.py --once --db data/demo.db
```

Показать, что в базе появляется пятая запись, а новый отчёт строится уже по
пяти записям.

## 5. Тесты

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
RUN_MCP_INTEGRATION=1 .venv/bin/python -m unittest tests.test_integration -v
```

Показать:

- офлайн-тесты;
- интеграционный сценарий 2 + 2 запуска;
- отчёт `record_count=4`, `observation_ids=[1,2,3,4]`.
