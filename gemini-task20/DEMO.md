# Сценарий видео Дня 20

Ориентировочная длительность: 4–5 минут.
Gemini не используется. Оркестрация выполняется **детерминированным кодом**, а не LLM.

## Подготовка

```bash
cd ~/work/AICourse/gemini-task20
.venv/bin/python --version
.venv/bin/pip list --format=freeze | grep -E '^(mcp|pydantic)='
bash scripts/setup_filesystem_env.sh
```

Показать, что Node project-local:

```bash
.tools/node-v24.22.3-darwin-x64/bin/node -v
```

## 1. Три сервера и объединённый каталог

```bash
.venv/bin/python main.py --list-tools
```

Показать:

- доступные серверы `catalog`, `notes`, `filesystem`;
- публичные имена `catalog__search` и `notes__search`;
- одинаковое оригинальное имя `search` у двух серверов;
- Filesystem-инструменты;
- отсутствие write-инструментов у Filesystem.

## 2. Полный флоу

```bash
.venv/bin/python main.py --scenario full
```

Показать порядок:

```text
[1] filesystem/read_text_file
[2] catalog/search
[3] catalog/get_item
[4] notes/search      ← другой сервер, то же имя search
[5] notes/get_note
[6] notes/summarize
status: ok
```

Показать итоговый текст: контекст, товар, заметка, сводка.

## 3. catalog недоступен

```bash
.venv/bin/python main.py --scenario catalog-down
```

Показать:

- `catalog UNAVAILABLE`;
- шаги каталога отсутствуют;
- `notes` и `filesystem` продолжают работать;
- в итоге есть контекст/заметка/сводка;
- нет раздела `Товар: ...`, отсутствующий товар не выдуман;
- `status: partial`, `unavailable: catalog`.

## 4. Пояснение

Проговорить:

1. `MultiMCPManager` держит три независимых соединения.
2. `ToolRoute` — точная таблица public name → server/original tool.
3. `catalog__search` и `notes__search` неоднозначности не создают.
4. `FlowAgent` — state machine: выбор зависит от запроса, доступности и
   предыдущих результатов.
5. Пустой search — валидный результат, зависимые шаги пропускаются.
6. `is_error`, `structured_content` и `content` обрабатываются по контракту MCP SDK.
7. Filesystem запускается с `--read-only --confine-roots <workspace>`.

## 5. Тесты

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
RUN_MCP_INTEGRATION=1 .venv/bin/python -m unittest tests.test_mcp_integration -v
```

Показать:

- офлайн-тесты маршрутизации, состояния и частичных отказов;
- реальные stdio-тесты трёх серверов;
- сценарий `catalog-down`;
- корректное закрытие соединений.
