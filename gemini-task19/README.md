# День 19. Композиция MCP-инструментов

Один MCP-сервер с тремя инструментами:

1. `search` — ищет публикации JSONPlaceholder.
2. `summarize` — строит краткое детерминированное резюме.
3. `save_to_file` — сохраняет непустое резюме в `output/`.

**Важно:** автоматическая композиция выполняется кодом `ChainAgent`, а не LLM.
Порядок вызовов, проверки и остановки заданы детерминированно. Gemini в проекте
не используется, API-ключи не нужны.

## Полный путь

```text
main.py
  │
  ▼
ChainAgent.run(query, filename)
  │
  ├── MCP search(query)
  │       ▼
  │   server.py → SearchApi → JSONPlaceholder /posts
  │       ▼
  │   SearchResult {query, items: [Post]}
  │       ▼
  │   валидация SearchResult
  │
  ├── items пуст → STOP, summarize/save не вызываются
  │
  ├── MCP summarize(items)
  │       ▼
  │   server.py → Summarizer
  │       ▼
  │   Summary {source_count, summary}
  │       ▼
  │   валидация Summary + source_count == len(items)
  │
  ├── ошибка/пусто/некорректно → STOP, save не вызывается
  │
  └── MCP save_to_file(content=summary, filename)
          ▼
      output/<filename>
          ▼
      SavedFile {path, bytes_written}
          ▼
      валидация SavedFile
```

Все три инструмента — на одном MCP-сервере. Оркестратор общается с ним через
`ChainMCPClient` (`ClientSession`, stdio).

## Валидация ответов

Каждый ответ MCP проверяется Pydantic-моделями в strict-режиме:

- `SearchResult`: `query: str`, `items: list[Post]`;
- `Post`: `userId: int`, `id: int`, `title: str`, `body: str`;
- `Summary`: `source_count: int`, `summary: str`;
- `SavedFile`: `path: str`, `bytes_written: int`.

Дополнительно:

- `summarize.source_count` должен совпадать с числом переданных items;
- `summary` не должен быть пустым;
- `saved.bytes_written` должен совпадать с длиной summary в UTF-8.

Любой некорректный ответ останавливает цепочку с понятной причиной.

## Остановки

| Ситуация | `stopped_at` | `save_to_file` |
|---|---|---|
| `search` вернул MCP-ошибку | `search` | не вызывается |
| `search` вернул пустой `items` | `search` | не вызывается |
| `search` вернул неверную структуру | `search` | не вызывается |
| `summarize` вернул MCP-ошибку | `summarize` | не вызывается |
| `summarize` вернул неверную структуру | `summarize` | не вызывается |
| `summarize` вернул пустой summary | `summarize` | не вызывается |
| `summarize.source_count` не совпал | `summarize` | не вызывается |
| `save_to_file` вернул ошибку/неверную структуру | `save_to_file` | файл не считается сохранённым |

`ChainResult` содержит:

- `status` — `ok` / `stopped`;
- `stopped_at` — шаг остановки;
- `reason` — причина;
- `steps` — все выполненные шаги с аргументами и результатами;
- `completed_steps` — только успешно завершённые шаги;
- `saved_path` — путь файла при успехе.

## MCP-сервер

```python
@mcp.tool(description="Ищет публикации JSONPlaceholder...")
async def search(query: str) -> SearchResult: ...

@mcp.tool(description="Строит краткое детерминированное резюме...")
async def summarize(items: list[Post]) -> Summary: ...

@mcp.tool(description="Сохраняет непустой текст в output/...")
async def save_to_file(content: str, filename: str) -> SavedFile: ...
```

`save_to_file`:

- запрещает пустой content;
- запрещает `../`, абсолютные пути и имена с `/`;
- пишет только внутрь `output/`;
- возвращает относительный путь и число записанных байт.

### Тестовый hook

`TASK19_FAIL_SUMMARIZE=1` включает ошибку `summarize` только в тестовом сценарии.
`ChainMCPClient` добавляет эту переменную в env сервера только при
`fail_summarize=True`; обычный запуск её не устанавливает.

## Установка

```bash
cd ~/work/AICourse/gemini-task19
../venv/bin/python -m venv .venv
.venv/bin/pip install -r requirements.lock
```

Отдельное окружение; `gemini-task17`, `gemini-task18` и общий `venv` не изменяются.
`google-genai` не нужен.

## Команды

```bash
# три MCP-инструмента и схемы
.venv/bin/python main.py --list-tools

# успешная цепочка
.venv/bin/python main.py --scenario success

# ошибка summarize; save_to_file не вызывается
.venv/bin/python main.py --scenario summarize-error

# пустой search; summarize/save_to_file не вызываются
.venv/bin/python main.py --scenario empty-search

# все три сценария подряд
./demo.sh
```

`demo.sh` очищает только `output/`, запускает три сценария и показывает, что
`output/summary.txt` создан, а файлы ошибочных сценариев отсутствуют.

## Тесты

### Офлайн

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
```

Результат:

```text
Ran 45 tests
OK (skipped=3)
```

Проверяется:

- поиск JSONPlaceholder через `httpx2.MockTransport`;
- детерминированное резюме;
- схемы и вызовы трёх MCP-инструментов;
- успешная цепочка и точная передача:
  - `summarize` получает именно `items` от `search`;
  - `save_to_file` получает именно `summary` от `summarize`;
- ошибка `summarize`: `save_to_file` не вызывается;
- пустой `search`: `summarize`/`save_to_file` не вызываются;
- некорректный ответ `search`/`summarize`/`save_to_file`;
- `source_count` mismatch;
- пустой summary;
- содержимое сохранённого файла;
- запрет path traversal и пустого content;
- MCP lifecycle, таймауты и ошибки;
- отсутствие Gemini-зависимостей.

### Интеграционный, реальный stdio

```bash
RUN_MCP_INTEGRATION=1 .venv/bin/python -m unittest tests.test_integration -v
```

Результат:

```text
Ran 3 tests
OK
```

Проверяется:

1. успешная цепочка через настоящий MCP-клиент/сервер и локальную
   HTTP-заглушку; файл `summary.txt` содержит именно summary;
2. ошибка `summarize` при `TASK19_FAIL_SUMMARIZE=1`; файл не создан;
3. пустой `search`; `summarize`/`save_to_file` не вызываются, файл не создан.

## Ограничения

- JSONPlaceholder — демонстрационный API; в тестах используется локальная заглушка.
- Резюме детерминированное, без LLM и без внешних AI-сервисов.
- Нет retry/кэша для JSONPlaceholder.
- `TASK19_FAIL_SUMMARIZE` — только тестовый hook, не для обычного запуска.
- Коммит не создавался.
