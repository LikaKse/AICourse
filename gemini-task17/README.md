# День 17. Первый инструмент MCP

Собственный MCP-сервер на Python, который предоставляет инструмент
`get_task(task_id: int)` и обращается к бесплатному JSONPlaceholder API.
Агент на Gemini вызывает этот MCP-инструмент, получает задачу и использует
результат в итоговом ответе.

## Полный цикл

```text
Пользователь
  │
  ▼
Agent.arun()
  │  Gemini → function_call get_task({"task_id": 5})
  ▼
TaskMCPClient (ClientSession, stdio)
  │
  ▼
server.py  (MCPServer + @mcp.tool)
  │
  ▼
TaskApi → httpx2 → GET https://jsonplaceholder.typicode.com/todos/5
  │
  ▼
Task {userId, id, title, completed}
  │
  ▼
FunctionResponse → Gemini → финальный ответ
```

Сервер MCP общается с агентом через stdin/stdout. HTTP-порт не открывается.
В `stdout` идёт только JSON-RPC; логи сервера — в `stderr`.

## Структура

```
gemini-task17/
├── main.py              # CLI: --check-api, --list-tools, --dry-run, --demo-scripted, --demo
├── config.py            # модель, API base, таймауты, пути
├── agent.py             # async-цикл Function Calling
├── local_tools.py       # локальные инструменты Дня 6
├── task_api.py          # Task model + HTTP-клиент JSONPlaceholder
├── server.py            # собственный MCP-сервер (stdio)
├── mcp_client.py        # TaskMCPClient: ClientSession + stdio
├── mcp_schema.py        # MCP Tool → Gemini FunctionDeclaration
├── tool_registry.py     # local/MCP маршрутизация
├── scripted_client.py   # fake-модель для --demo-scripted
├── requirements.txt
├── requirements.lock
├── .env.example
├── .gitignore
├── README.md
└── tests/
    ├── fakes.py
    ├── test_task_api.py
    ├── test_server.py
    ├── test_mcp_client.py
    ├── test_tool_registry.py
    ├── test_agent_task.py
    ├── test_cli.py
    ├── test_mcp_integration.py
    ├── test_local_tools.py
    ├── test_mcp_schema.py
    └── test_no_secrets.py
```

## Версии

| Компонент | Версия |
|---|---|
| Python | 3.12.6 |
| `mcp` | 2.3.0 |
| `mcp-types` | 2.3.0 |
| `google-genai` | 2.34.0 |
| `httpx2` | 2.16.2 |
| `pydantic` | 2.14.4 |
| `httpx` | 0.28.1 (для google-genai) |

Node.js не требуется.

## Установка отдельного окружения

```bash
cd ~/work/AICourse/gemini-task17
../venv/bin/python -m venv .venv
.venv/bin/pip install -r requirements.lock
```

Общий `AICourse/venv` не изменяется.

## MCP-сервер

`server.py`:

```python
mcp = MCPServer(
    "jsonplaceholder-tasks",
    version="0.1.0",
    instructions="Сервер задач JSONPlaceholder.",
    log_level="WARNING",
)

api = TaskApi()

@mcp.tool(description="Возвращает задачу JSONPlaceholder по её идентификатору.")
async def get_task(task_id: int) -> Task:
    try:
        return await api.fetch_task(task_id)
    except TaskNotFound as exc:
        raise ToolError(str(exc)) from exc
    except TaskApiError as exc:
        raise ToolError(str(exc)) from exc

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

Входная схема строится MCP SDK автоматически из `task_id: int`:

```json
{
  "type": "object",
  "properties": {"task_id": {"type": "integer"}},
  "required": ["task_id"]
}
```

Возвращается Pydantic-модель `Task`, поэтому MCP отдаёт и текст, и
`structuredContent`:

```json
{
  "userId": 1,
  "id": 5,
  "title": "laboriosam mollitia et enim quasi adipisci quia provident illum",
  "completed": false
}
```

`TaskApi`:

- URL: `{TASKS_API_BASE}/todos/{task_id}`;
- `TASKS_API_BASE` по умолчанию `https://jsonplaceholder.typicode.com`;
- таймаут 10 секунд;
- 404 → `TaskNotFound` → MCP `ToolError`;
- 5xx, сеть, таймаут, невалидный JSON → `TaskApiError` → MCP `ToolError`.

## Запуск сервера

Агент запускает сервер сам через `StdioServerParameters`:

```python
StdioServerParameters(
    command=str(config.PYTHON_BIN),       # gemini-task17/.venv/bin/python
    args=["-u", str(config.SERVER_PY)],   # server.py
    cwd=str(config.PROJECT_ROOT),
    env={"TASKS_API_BASE": config.TASKS_API_BASE, "PYTHONUNBUFFERED": "1"},
)
```

Отдельный терминал не нужен. Процесс живёт в течение MCP-сессии и закрывается
через `AsyncExitStack`.

## Агент и реестр

`ToolRegistry` объединяет локальные инструменты Дня 6 и MCP `get_task`.
Модель видит один список `FunctionDeclaration` и сама выбирает инструмент.
Реестр по имени определяет источник:

- `local` → `local_tools.dispatch`;
- `mcp` → `TaskMCPClient.call_tool_for_model`.

MCP-результат передаётся модели как:

```json
{
  "source": "mcp",
  "tool": "get_task",
  "result": {
    "userId": 1,
    "id": 5,
    "title": "laboriosam mollitia et enim quasi adipisci quia provident illum",
    "completed": false
  }
}
```

Ошибка инструмента возвращается как `{"error": "..."}`; цикл агента продолжается.

## Команды

```bash
cd ~/work/AICourse/gemini-task17

# прямой бесплатный запрос к JSONPlaceholder
.venv/bin/python main.py --check-api

# MCP-сервер: список инструментов и схема get_task
.venv/bin/python main.py --list-tools

# MCP + локальные схемы, без Gemini и без API-вызова
.venv/bin/python main.py --dry-run

# полный цикл: fake-модель + настоящий MCP + JSONPlaceholder
.venv/bin/python main.py --demo-scripted

# реальная модель — только после отдельного разрешения
export GEMINI_API_KEY=...
.venv/bin/python main.py --demo
```

`--demo` проверяет наличие ключа и спрашивает подтверждение. Реальные вызовы
Gemini в автоматических тестах не выполняются.

## Демонстрация `--demo-scripted`

Ожидаемый вывод:

```text
[1] MCP   get_task  (итерация 1)
    args:   {"task_id": 5}
    result: {"source": "mcp", "tool": "get_task",
             "result": {"userId": 1, "id": 5,
                        "title": "laboriosam mollitia et enim quasi adipisci quia provident illum",
                        "completed": false}}

Агент: Задача 5: «laboriosam mollitia et enim quasi adipisci quia provident illum».
       Статус: не выполнена.

[PASS]
```

Это проверяет полный технический путь: агент → MCP-клиент → собственный
MCP-сервер → JSONPlaceholder → результат → ответ. Выбор инструмента реальной
моделью проверяется только `--demo`.

## Тесты

### Офлайн, без сети и Gemini

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
```

Проверяется:

- HTTP-клиент JSONPlaceholder на `httpx2.MockTransport`: 200, 404, 500,
  таймаут, невалидный JSON, неверный `task_id`;
- регистрация `get_task`, входная схема `task_id: integer`, output schema;
- вызов через in-memory MCP `Client`;
- `ToolError` → MCP `is_error=True`;
- lifecycle `TaskMCPClient`: список инструментов, вызов, таймауты, ошибки,
  закрытие сессии;
- маршрутизация local/MCP;
- fake Gemini → `get_task` → `FunctionResponse` → финальный ответ;
- локальный `calculate` не сломан;
- секретов нет в файлах.

### Интеграционный, с настоящим MCP-сервером

```bash
RUN_MCP_INTEGRATION=1 .venv/bin/python -m unittest tests.test_mcp_integration -v
```

Поднимается локальная HTTP-заглушка JSONPlaceholder, запускается настоящий
`server.py` через stdio. Проверяются handshake, схема `get_task`, полный вызов,
404 → MCP error. Внешняя сеть и Gemini не нужны.

### Опциональная живая проверка API

```bash
RUN_LIVE_API=1 .venv/bin/python -m unittest tests.test_mcp_integration.LiveJSONPlaceholderTest -v
```

## Ограничения

- Используется только один MCP-инструмент `get_task`.
- JSONPlaceholder — демонстрационный API; данные не предназначены для продакшена.
- Запрос к JSONPlaceholder выполняется на каждый вызов `get_task`; кэша нет.
- `TaskApi` не повторяет запросы при временных ошибках.
- Реальный `--demo` платный и не запускался.
- Коммит не создавался.
