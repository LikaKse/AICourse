# День 15 — память агента: устаревание и конфликты

## Цель задания

Реализовать локальную память агента, которая:

- хранит долговременные факты, накопленные решения и временные факты задачи;
- разрешает конфликты версий по времени записи;
- не показывает агенту устаревшие значения, историю изменений и tombstones;
- очищает временную память при завершении задачи;
- демонстрирует всё это детерминированным офлайн-сценарием на fake `ModelClient`.

Реальный Gemini API, сеть и платные сервисы не используются.

---

## Архитектура

```text
gemini-task15/
├── demo.py                 # офлайн-демонстрация всего сценария
├── memory/
│   ├── errors.py           # MemoryError, MemoryValidationError, MemoryCorrupt, ...
│   ├── models.py           # Fact, Tombstone, TaskCompletion, MemoryEvent, MemorySnapshot
│   ├── store.py            # атомарный memory.json + flock + строгая валидация
│   ├── service.py          # единственный write-API: конфликты, удаление, complete/reopen
│   ├── context.py          # ContextBuilder: только активные факты → prompt
│   ├── reader.py           # MemoryReader: read-only доступ через store.load()
│   └── agent.py            # MemoryAwareAgent + SYSTEM_PROMPT + ModelClient
└── tests/
    ├── test_models.py
    ├── test_store.py
    ├── test_service.py
    ├── test_concurrency.py
    ├── test_context.py
    ├── test_reader.py
    ├── test_agent.py
    └── test_demo.py
```

Поток данных:

```text
MemoryService ──> MemoryStore (memory.json)
                        │
MemoryReader ──> ContextBuilder ──> MemoryAwareAgent ──> ModelClient
```

- **MemoryService** — единственный способ изменить память.
- **MemoryStore** — хранит snapshot и журнал в одном файле `memory.json`,
  пишет атомарно через временный файл и `os.replace`, использует
  межпроцессную блокировку `flock`.
- **MemoryReader** — read-only: вызывает штатный `MemoryStore.load()` с
  блокировкой и проверкой целостности; не читает файл напрямую и не имеет
  write-методов.
- **ContextBuilder** — строит детерминированный prompt только из активных
  фактов.
- **MemoryAwareAgent** — добавляет вопрос пользователя и вызывает
  `ModelClient.generate()`.

---

## Типы памяти

### Mutable (долговременные факты)

Один актуальный факт на ключ. Замена перезаписывает значение, но прежние
версии остаются только в журнале и в prompt не попадают.

Примеры: `project.deadline`, `project.status`, `stack.approach`.

### Cumulative (накопленные решения)

Решения только добавляются. Каждое получает `entry_id` и `recorded_at`.
Старые решения не перезаписываются и не удаляются через публичный API.
В prompt выводятся списком в порядке `recorded_at`, затем `entry_id`.

### Temporary (временные факты задачи)

Факты привязаны к `task_id`. В prompt попадают только факты текущей задачи
и только если задача не завершена. После `complete_task(task_id)` временные
факты задачи удаляются и заменяются tombstones.

---

## Правила разрешения конфликтов

Конфликт решается по `recorded_at` — времени, к которому относится факт,
а не по времени вызова метода.

Для `set_long_term` и `set_temporary`:

| Условие | Результат |
|---|---|
| `new.recorded_at > current.recorded_at` | `applied` — новое значение становится активным |
| `new.recorded_at < current.recorded_at` | `rejected_stale` — остаётся текущее значение |
| равно, значение совпадает | `unchanged` |
| равно, значение отличается | `rejected_equal` — остаётся текущее значение |
| запись после удаления с `recorded_at <= deleted_at` | `rejected_stale` / `rejected_equal` |
| запись в завершённую задачу (temporary) | `rejected_task_completed` |

Для `delete_long_term` и `delete_temporary`:

| Условие | Результат |
|---|---|
| `deleted_at > current.recorded_at` | `applied`, создаётся tombstone |
| `deleted_at < current.recorded_at` | `rejected_stale` |
| `deleted_at == current.recorded_at` | `rejected_equal` |

Отклонённые попытки не меняют активные данные, но фиксируются в журнале.
Tombstone запрещает повторно добавить значение не новее времени удаления;
более новое значение добавить можно.

`complete_task` требует, чтобы `completed_at` был строго позже всех
temporary-фактов и tombstones задачи. Временные факты удаляются, создаются
tombstones, задача помечается завершённой. `reopen_task` снимает отметку
завершения, но **не восстанавливает** старые временные факты: их tombstones
остаются в силе.

---

## Что попадает в prompt

```text
<SYSTEM_PROMPT>

[ACTIVE MEMORY]
long_term:
  key = "value" | recorded_at=... | source=...
decisions:
  - "решение" | recorded_at=... | source=...
task task-1:
  key = "value" | recorded_at=... | source=...

[USER]
вопрос пользователя
```

В prompt **не попадают**:

- старые значения mutable-фактов;
- журнал (`journal`) и `old_value`/`new_value`;
- tombstones;
- отклонённые попытки;
- temporary-факты других задач и завершённых задач.

Сортировка детерминирована: long-term по ключу, decisions по
`(recorded_at, entry_id)`, temporary по ключу. Значения рендерятся
каноническим JSON с сортировкой ключей.

---

## Очистка временной памяти

1. `set_temporary(task_id, key, value)` добавляет факт только указанной задаче.
2. `complete_task(task_id)`:
   - удаляет все temporary-факты этой задачи;
   - создаёт по tombstone на каждый удалённый ключ;
   - сохраняет `TaskCompletion`;
   - не трогает long-term и cumulative;
   - не трогает temporary других задач.
3. `reopen_task(task_id)` снимает завершение, но старые факты не возвращает.
   Новые temporary-факты можно добавлять только с `recorded_at` позже
   соответствующего tombstone.

---

## Запуск тестов

Из каталога `gemini-task15`:

```bash
../venv/bin/python -m pytest -q -p no:cacheprovider
../venv/bin/python -m pytest -q -W error -p no:cacheprovider
```

Проверка синтаксиса и diff:

```bash
../venv/bin/python -m compileall -q .
git diff --check
```

---

## Запуск демонстрации

```bash
../venv/bin/python demo.py
```

Демонстрация работает во временном каталоге `tempfile.TemporaryDirectory()`
и не пишет состояние в проект. Используются только fake-клиенты:

- `FactLookupClient` — извлекает актуальное значение запрошенного ключа из
  prompt и возвращает его либо `UNKNOWN`;
- `RecordingClient` — фиксирует prompt и возвращает заданный ответ.

### Сценарий

1. `DEADLINE-1` → `DEADLINE-2`.
2. `REQUIREMENT-OLD` → удаление требования.
3. `STATUS-1` → `STATUS-2`.
4. `MONOLITH` → `MICROSERVICES`.
5. Добавляются два решения: `Use PostgreSQL`, `Use Redis`.
6. Создаются temporary-факты `task-1` и отдельный факт `task-2`.
7. Выполняется `complete_task("task-1")`.

После каждого изменения печатаются вопрос, ответ агента, ожидаемое значение,
полный prompt и результат шага.

`stale_mentions` считается по заранее заданному набору устаревших значений:

```text
DEADLINE-1, REQUIREMENT-OLD, STATUS-1, MONOLITH, TEMP-1, TEMP-NOTE-1
```

Счётчики ведутся отдельно для prompt и для ответов агента. Ожидается `0` и `0`.

### Ожидаемый итог

```text
Long-term facts:
  project.deadline = "DEADLINE-2"
  project.status = "STATUS-2"
  stack.approach = "MICROSERVICES"
Cumulative decisions:
  1. "Use PostgreSQL"
  2. "Use Redis"
Remaining temporary:
  task-2.task.step = "TEMP-2"
Stale in prompt: 0
Stale in responses: 0
RESULT: PASS
```

Код возврата `demo.py`: `0` при PASS, `1` при FAIL.

---

## Результаты

- Все шаги demo проверяют **фактический ответ агента**, а не только наличие
  значения в prompt:
  - новый срок — `DEADLINE-2`;
  - после удаления требования — `UNKNOWN`;
  - новый статус — `STATUS-2`;
  - новый подход — `MICROSERVICES`.
- Временные факты `task-1` видны только в `task-1`, факты `task-2` — только
  в `task-2`.
- После `complete_task("task-1")` временные факты `task-1` исчезли,
  `task-2` сохранились.
- Long-term факты и оба решения после очистки сохранились.
- `stale_mentions` в prompt: `0`.
- `stale_mentions` в ответах: `0`.
- Итог: `PASS`.

---

## Выводы

- Время записи факта, а не порядок вызовов, определяет актуальность — это
  защищает от запоздалых обновлений.
- Единый атомарный файл с журналом не позволяет snapshot и истории разойтись.
- Tombstones предотвращают «воскрешение» удалённых данных.
- Read-only слой и отсутствие write-API у агента не дают модели случайно
  изменить память.
- Изоляция temporary по `task_id` и очистка при завершении не дают утечь
  контексту одной задачи в другую.
- Prompt содержит только активные данные, поэтому устаревшие значения не
  попадают к модели.

---

## Ограничения

- `ModelClient` в demo и тестах — fake; реальный Gemini API не вызывается.
- Prompt не имеет token-budget, truncation и приоритизации; при большом
  объёме памяти он будет расти линейно.
- Нет TTL для temporary-фактов: они живут до `complete_task`.
- История доступна только через `MemoryService.history()` и не передаётся
  агенту.
- Консервативное правило `rejected_equal`: при равном `recorded_at` и
  разном значении побеждает текущее значение.
- `reopen_task` не восстанавливает старые temporary-факты; после reopen
  отсутствует отдельный task-watermark, поэтому новые записи проверяются
  по tombstone каждого ключа.
- Вложенные JSON-значения в возвращаемых `Fact` следует считать read-only.
