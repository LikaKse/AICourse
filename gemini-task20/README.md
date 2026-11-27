# День 20. Orchestration MCP

Один агент работает с тремя MCP-серверами:

| Alias | Сервер | Инструменты |
|---|---|---|
| `catalog` | собственный Python | `search`, `get_item` |
| `notes` | собственный Python | `search`, `get_note`, `summarize` |
| `filesystem` | официальный `@modelcontextprotocol/server-filesystem@2.4.0` | read-only инструменты, включая `read_text_file` |

**Автоматическая оркестрация выполняется детерминированным кодом, а не LLM.**
`FlowAgent` — state machine: следующий инструмент выбирается по намерению запроса,
доступности серверов и результатам предыдущих вызовов. Gemini и API-ключи не
используются. Это сделано для предсказуемости и офлайн-тестируемости.

## Архитектура

```text
main.py
  │
  ▼
MultiMCPManager
  ├── ServerConnection("catalog")
  │      AsyncExitStack → ClientSession → stdio
  ├── ServerConnection("notes")
  └── ServerConnection("filesystem")
  │
  ▼
Unified tool catalog + exact ToolRoute map
  │
  ▼
FlowAgent (state machine)
  ├── filesystem__read_text_file
  ├── catalog__search / catalog__get_item
  └── notes__search / notes__get_note / notes__summarize
```

Каждый сервер имеет отдельное `AsyncExitStack` и `ClientSession`. Запуск
последовательный; при закрытии используются LIFO-порядок, чтобы корректно
выходить из anyio cancel scopes.

## Публичные имена и маршрутизация

Формат:

```text
<server_alias>__<original_tool_name>
```

Пример:

```text
catalog__search
catalog__get_item
notes__search
notes__get_note
notes__summarize
filesystem__read_text_file
```

`catalog__search` и `notes__search` имеют одинаковое оригинальное имя `search`.
Маршрут хранится в точной таблице:

```python
routes["catalog__search"] = ToolRoute("catalog__search", "catalog", "search")
routes["notes__search"] = ToolRoute("notes__search", "notes", "search")
```

Во время вызова менеджер берёт готовый `ToolRoute` из словаря. Он не парсит
публичное имя и не угадывает сервер. Поэтому одинаковые имена на разных
серверах не создают неоднозначности.

## State-machine FlowAgent

`classify_intent(query)` явно определяет:

- `context` — упоминание контекста/файла/закупки;
- `catalog` — упоминание каталога/товара/ноутбука;
- `notes` — упоминание заметки.

`choose_action(state, intent)` выбирает следующий шаг только при необходимости:

1. контекст, если он запрошен и ещё не прочитан;
2. каталог, если он запрошен и ещё не найден;
3. товар, если поиск вернул непустой список;
4. заметки, если они запрошены и ещё не найдены;
5. заметка, если поиск вернул непустой список;
6. резюме, если есть выбранная заметка.

Шесть вызовов — это пример полного запроса, а не жёстко зашитый цикл.
Если `catalog` недоступен, шаги каталога пропускаются, а `notes` и
`filesystem` продолжаются. Если `search` вернул пустой список, зависимые
`get_item`/`get_note`/`summarize` не вызываются.

## Filesystem MCP 2.4.0

Проверенная команда:

```text
.tools/node-v24.22.3-darwin-x64/bin/node
  .tools/mcp-server/node_modules/@modelcontextprotocol/server-filesystem/dist/index.js
  --read-only
  --confine-roots
  /абсолютный/путь/gemini-task20/workspace
```

`--confine-roots` поддерживается в 2.4.0. После подключения менеджер вызывает
`list_allowed_directories` и требует ровно один путь — `workspace`.
`--read-only` означает, что write-инструменты не отдаются.

Установка project-local Node и сервера:

```bash
cd ~/work/AICourse/gemini-task20
bash scripts/setup_filesystem_env.sh
```

Системный Node и другие проекты не меняются.

## MCP-ошибки и контракты

Используется фактический `CallToolResult` MCP SDK 2.3.0:

- `is_error` — ошибка инструмента;
- `content` — текстовый ответ/текст ошибки;
- `structured_content` — структурированный результат;
- таймаут/соединение — отдельная ошибка менеджера.

Правила:

- `catalog__search` / `notes__search`: `items=[]` — валидный результат, не ошибка;
- `filesystem__read_text_file`: пустой файл — валидный результат;
- `catalog__get_item`, `notes__get_note`: обязательный объект;
- `notes__summarize`: пустое резюме — нарушение контракта;
- отсутствие `structured_content` при `is_error=False` — некорректный ответ.

Ошибка или невалидный ответ останавливает только зависимую ветку; независимые
ветки продолжаются, если данные доступны.

## Структура

```
gemini-task20/
├── main.py
├── config.py
├── models.py
├── manager.py
├── flow_agent.py
├── server_specs.py
├── servers/
│   ├── catalog_server.py
│   ├── catalog_data.py
│   ├── notes_server.py
│   └── notes_data.py
├── workspace/context.txt
├── scripts/
│   ├── setup_filesystem_env.sh
│   ├── mcp-server-package.json
│   └── mcp-server-package-lock.json
├── requirements.txt
├── requirements.lock
├── .gitignore
├── README.md
├── DEMO.md
├── demo.sh
└── tests/
    ├── fakes.py
    ├── test_manager.py
    ├── test_flow_agent.py
    ├── test_servers.py
    ├── test_mcp_integration.py
    ├── test_cli.py
    └── test_no_secrets.py
```

## Установка

```bash
cd ~/work/AICourse/gemini-task20
../venv/bin/python -m venv .venv
.venv/bin/pip install -r requirements.lock
bash scripts/setup_filesystem_env.sh
```

Зависимости: `mcp==2.3.0`, `pydantic==2.14.4`; `google-genai` отсутствует.

## Команды

```bash
# все три сервера и объединённый каталог
.venv/bin/python main.py --list-tools

# полный успешный флоу
.venv/bin/python main.py --scenario full

# catalog недоступен; notes/filesystem работают
.venv/bin/python main.py --scenario catalog-down

# оба сценария
./demo.sh
```

## Тесты

### Офлайн

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
```

Проверяется:

- подключение трёх серверов и объединение инструментов;
- точная маршрутизация `catalog__search` / `notes__search`;
- частичный отказ и закрытие соединений;
- state-machine: полный флоу, только каталог, только заметки;
- пустой search не вызывает зависимые шаги;
- ошибки `is_error`, неверные структуры, пустой файл;
- отсутствие выдуманного товара при недоступном каталоге;
- CLI, отсутствие Gemini и ключей.

### Интеграционные, реальный stdio

```bash
RUN_MCP_INTEGRATION=1 .venv/bin/python -m unittest tests.test_mcp_integration -v
```

Проверяется:

1. все три настоящих сервера доступны;
2. `filesystem` разрешает только `workspace`;
3. полный флоу из шести вызовов;
4. реальный `is_error` от `notes__get_note`;
5. пустой `catalog__search` — валидный результат;
6. недоступный `catalog` не мешает `notes/filesystem`;
7. закрытие всех соединений.

## Ограничения

- `FlowAgent` детерминированный, без LLM; это осознанное требование
  предсказуемости.
- Filesystem MCP используется только для чтения.
- `catalog` и `notes` — локальные демонстрационные данные.
- Системный Node не меняется; project-local Node лежит в `.tools`.
- `TASK20` не использует Gemini и платные запросы.
- Коммит не создавался.
