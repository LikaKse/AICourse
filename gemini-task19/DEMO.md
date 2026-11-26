# Сценарий видео Дня 19

Ориентировочная длительность: 3–4 минуты.
Gemini не используется. Автоматическая композиция выполняется **кодом**, а не LLM.

## Подготовка

```bash
cd ~/work/AICourse/gemini-task19
.venv/bin/python --version
.venv/bin/pip list --format=freeze | grep -E '^(mcp|httpx2|pydantic)='
```

## 1. MCP-инструменты

```bash
.venv/bin/python main.py --list-tools
```

Показать:

- сервер `content-chain 0.1.0`;
- протокол `2025-11-25`;
- три инструмента: `search`, `summarize`, `save_to_file`;
- их схемы.

## 2. Успешная цепочка

```bash
.venv/bin/python main.py --scenario success
```

Показать:

- `[1/3] search` → найдены публикации;
- `[2/3] summarize` → резюме;
- `[3/3] save_to_file` → `output/summary.txt`;
- `chain: OK`.

Затем:

```bash
cat output/summary.txt
```

## 3. Ошибка summarize

```bash
.venv/bin/python main.py --scenario summarize-error
```

Показать:

- `[1/3] search` успешен;
- `[2/3] summarize` — ошибка `TASK19_FAIL_SUMMARIZE`;
- `chain: STOPPED at summarize`;
- `save_to_file` отсутствует в выполненных шагах;
- файл `should_not_exist.txt` не создан.

## 4. Пустой search

```bash
.venv/bin/python main.py --scenario empty-search
```

Показать:

- `[1/3] search` → `items=0`;
- `chain: STOPPED at search`;
- `summarize` и `save_to_file` не вызывались;
- файл не создан.

## 5. Всё сразу

```bash
./demo.sh
```

Показать:

- три сценария подряд;
- `output/summary.txt` существует;
- `should_not_exist.txt` отсутствует.

## 6. Пояснение

Проговорить:

1. `ChainAgent` — детерминированный оркестратор, не LLM.
2. Он передаёт `items` из `search` в `summarize` и `summary` из
   `summarize` в `save_to_file`.
3. Ответы MCP валидируются Pydantic-моделями.
4. При ошибке, пустом или некорректном ответе цепочка останавливается.
5. `save_to_file` вызывается только после успешного непустого summary.
6. `TASK19_FAIL_SUMMARIZE` — только тестовый hook.

## 7. Тесты

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
RUN_MCP_INTEGRATION=1 .venv/bin/python -m unittest tests.test_integration -v
```

Показать:

- 45 офлайн-тестов;
- 3 интеграционных сценария через реальный stdio;
- проверку, что последующие инструменты не вызываются.
