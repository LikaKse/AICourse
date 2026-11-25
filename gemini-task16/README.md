# День 16. Подключение MCP

Самостоятельный агент на Gemini с **ручным Function Calling**, локальными инструментами Дня 6
и внешним **MCP Filesystem 2.4.0**. Модель сама выбирает, вызвать локальный инструмент,
MCP-инструмент или оба последовательно.

Проект полностью изолирован:

- системный Node.js не меняется — используется project-local **Node 24.22.3**;
- общий `AICourse/venv` не меняется — используется **`gemini-task16/.venv`**;
- MCP-сервер запускается с `--read-only --confine-roots demo_workspace`;
- предыдущие задания не изменяются.

## Структура

```
gemini-task16/
├── main.py                  # CLI: --list-tools, --dry-run, --demo-scripted, --demo
├── config.py                # модель, пути, версии, лимиты, системная инструкция
├── agent.py                 # async-цикл Function Calling (логика Дня 6)
├── local_tools.py           # get_current_time, calculate, convert_currency + dispatch
├── mcp_client.py            # stdio + ClientSession, read-only, allowlist, ошибки
├── mcp_schema.py            # MCP JSON Schema → FunctionDeclaration Gemini
├── tool_registry.py         # объединение local/MCP и автоматическая маршрутизация
├── scripted_client.py       # детерминированная «модель» для --demo-scripted
├── demo_workspace/
│   └── numbers.txt          # quantity=12, unit_price=25000, markup_percent=10
├── scripts/
│   ├── setup_mcp_env.sh     # project-local Node 24 + MCP server
│   ├── mcp-server-package.json
│   └── mcp-server-package-lock.json
├── tests/
│   ├── fakes.py
│   ├── test_local_tools.py
│   ├── test_mcp_schema.py
│   ├── test_tool_registry.py
│   ├── test_agent_mixed.py
│   ├── test_mcp_client.py
│   ├── test_mcp_integration.py
│   ├── test_cli.py
│   └── test_no_secrets.py
├── requirements.txt
├── requirements.lock
├── .env.example
├── .gitignore
└── README.md
```

## Версии

| Компонент | Версия |
|---|---|
| Python | 3.12.6 |
| Node.js | 24.22.3 LTS «Krypton» |
| npm | 11.19.1 |
| `@modelcontextprotocol/server-filesystem` | 2.4.0 |
| `@modelcontextprotocol/server` | 2.4.0 |
| `@modelcontextprotocol/node` | 2.1.2 |
| `mcp` (Python SDK) | 2.3.0 |
| `mcp-types` | 2.3.0 |
| `google-genai` | 2.34.0 |
| `pydantic` | 2.14.4 |
| `anyio` | 4.16.0 |
| `httpx` / `httpx2` | 0.28.1 / 2.16.2 |

Полный freeze Python-зависимостей — в `requirements.lock`.
Транзитивные npm-зависимости — в `scripts/mcp-server-package-lock.json`.

## Установка изолированных окружений

### 1. Node 24 и MCP Filesystem 2.4.0

```bash
cd ~/work/AICourse/gemini-task16
bash scripts/setup_mcp_env.sh
```

Скрипт:

- скачивает официальный Node v24.22.3 с `nodejs.org`;
- проверяет SHA-256;
- распаковывает в `.tools/` (в git не попадает);
- ставит `@modelcontextprotocol/server-filesystem@2.4.0` через `npm ci --ignore-scripts`;
- не меняет системный Node, Homebrew, nvm и `PATH`.

Если Node 24 не найден, MCP-режимы завершатся с понятной ошибкой. Отката на старый
`server-filesystem@1.0.0` нет.

### 2. Python-окружение

```bash
cd ~/work/AICourse/gemini-task16
../venv/bin/python -m venv .venv
.venv/bin/pip install -r requirements.lock
```

Общий `AICourse/venv` не изменяется. Проверка `pip install --dry-run mcp==2.3.0`
в общем venv показала только 14 новых пакетов и ни одного обновления/понижения
существующих, но отдельное окружение выбрано для полной изоляции.

## MCP-подключение

Используется низкоуровневый `mcp.ClientSession`:

```python
async with stdio_client(params) as (read, write):
    async with ClientSession(read, write, read_timeout_seconds=30) as session:
        await session.initialize()
        tools = await session.list_tools()
        result = await session.call_tool("read_text_file", {"path": "..."})
```

Команда запуска:

```text
.tools/node-v24.22.3-darwin-x64/bin/node
  .tools/mcp-server/node_modules/@modelcontextprotocol/server-filesystem/dist/index.js
  --read-only
  --confine-roots
  demo_workspace
```

Защита доступа:

1. **`--read-only`** — сервер сам не отдаёт инструменты записи. В этом режиме он
   возвращает только 10 read-only инструментов.
2. **`--confine-roots`** — клиентские Roots не могут расширить доступ за пределы
   `demo_workspace`.
3. **allowlist в `ToolRegistry`** — модели показываются только текстовые read-only
   инструменты.
4. **клиентская проверка пути** — путь вне `demo_workspace` отклоняется до вызова MCP.

Модели отдаются: `read_text_file`, `read_multiple_files`, `list_directory`,
`list_directory_with_sizes`, `directory_tree`, `search_files`, `get_file_info`,
`list_allowed_directories`.

`read_file` — deprecated-алиас `read_text_file`, он скрыт. `read_media_file`
скрыт, потому что возвращает base64. Write-инструментов в read-only режиме нет.

Схемы MCP преобразуются в `types.FunctionDeclaration` с `parameters_json_schema`;
из JSON Schema удаляется `$schema`, сохраняются `type`, `properties`, `required`,
`enum`, `minimum`, `maximum` и описания.

## Архитектура агента

```
Пользователь
   │
   ▼
Agent.arun()
   │  history += user
   │  loop 1..MAX_ITERATIONS:
   │     await client.aio.models.generate_content(history, tools=declarations)
   │     history += ответ модели целиком (thought_signature сохраняется)
   │     нет function_call → финальный текст
   │     для каждого function_call:
   │         ToolRegistry.adispatch(name, args)
   │             ├─ local → local_tools.dispatch()
   │             └─ mcp   → MCPFilesystemClient.call_tool_for_model()
   │         FunctionResponse(id, name, {result|error}) → history
   │  лимит итераций → max_iterations + откат истории
```

Ключевые свойства:

- `FunctionCallingConfigMode.AUTO` — модель сама выбирает инструмент;
- `AutomaticFunctionCallingConfig(disable=True)` — SDK не выполняет функции;
- история хранится как `list[types.Content]`, ответы модели не пересобираются;
- `FunctionResponse.id` совпадает с `FunctionCall.id`;
- ошибки инструментов возвращаются модели как `{"error": ...}`;
- при ошибке API/сети/лимите история откатывается.

## Смешанный сценарий

`demo_workspace/numbers.txt`:

```text
quantity=12
unit_price=25000
markup_percent=10
```

Запрос:

> В папке `demo_workspace` есть файл `numbers.txt` с количеством, ценой за единицу
> и процентом наценки. Рассчитай итоговую стоимость с наценкой.

Ожидаемая цепочка:

1. MCP `read_text_file` → `quantity=12`, `unit_price=25000`, `markup_percent=10`.
2. Локальный `calculate("12 * 25000 * (1 + 10 / 100)")` → `330000`.
3. Финальный ответ.

В консоль выводятся:

- версия Node, сервера и протокол;
- полный список инструментов и allowlist;
- тело запроса к Gemini со схемами `calculate` и `read_text_file`;
- каждый вызов: источник (`MCP`/`LOCAL`), аргументы, результат;
- итоговый ответ и PASS/FAIL.

## Запуск

```bash
cd ~/work/AICourse/gemini-task16

# бесплатно, без Gemini
.venv/bin/python main.py --list-tools
.venv/bin/python main.py --dry-run
.venv/bin/python main.py --demo-scripted

# реальная модель — только после отдельного разрешения
export GEMINI_API_KEY=...
.venv/bin/python main.py --demo
```

`--demo` сначала проверяет наличие ключа и спрашивает подтверждение. Ключ
читается только из окружения и нигде не сохраняется. Реальные вызовы Gemini
в автоматических тестах не выполняются.

## Тесты

### Офлайн, без Node и Gemini

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
```

Проверяется:

- локальный `calculate` и остальные инструменты Дня 6;
- схемы MCP → Gemini, удаление `$schema`, nullable-типы;
- маршрутизация local/MCP, коллизии имён, скрытые инструменты;
- смешанный цикл `read_text_file → calculate` на фейковой модели;
- возврат `FunctionResponse` с правильным `id`;
- ошибки MCP и локальных инструментов;
- таймауты, ошибка соединения, закрытие сессии;
- команда запуска содержит `--read-only` и `--confine-roots`;
- Node < 24 отклоняется;
- секретов нет в файлах, `.env`/`.venv`/`.tools` в `.gitignore`.

### Интеграционный, с реальным MCP, без Gemini

```bash
RUN_MCP_INTEGRATION=1 .venv/bin/python -m unittest tests.test_mcp_integration -v
```

Проверяется реальный сервер 2.4.0: handshake, протокол 2025-11-25, 10 read-only
инструментов, отсутствие write-инструментов, чтение `numbers.txt`, единственная
разрешённая папка `demo_workspace`, запрет чтения `/etc/hosts` и на клиенте,
и на сервере.

### Результаты текущего прогона

- офлайн: **75 tests OK, 3 skipped** (2 интеграционных + проверка реального ключа);
- интеграционный MCP: **2 tests OK**;
- `--list-tools`: сервер `secure-filesystem-server 2.4.0`, протокол `2025-11-25`, 10 инструментов;
- `--dry-run`: exit 0, в запросе есть `calculate` и `read_text_file`;
- `--demo-scripted`: **PASS**, `read_text_file → calculate → 330000`.

## Ограничения

- `calculate` не поддерживает переменные, функции и `%` как процент.
- MCP-сервер используется только для чтения; запись в `demo_workspace` запрещена.
- Первый запуск `setup_mcp_env.sh` требует сеть для скачивания Node и npm-пакета.
- `--demo-scripted` проверяет весь технический путь, но не выбор инструментов
  реальной моделью. Реальный выбор проверяется только `--demo`.
- Вызовы Gemini платные; автоматические тесты их не делают.
