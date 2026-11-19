# День 13. Task State Machine

В каталоге реализованы:

- чистое ядро конечного автомата задачи (`task_state.py`);
- доверенный `TaskController` и ограниченный интерфейс агента
  (`task_controller.py`);
- граница исполнения шага с `AgentReport`/`ExecutionReceipt`
  (`execution.py`);
- read-only адаптер `working_memory.json` этапа 2.1 (`task_store.py`);
- локальная демонстрация без Gemini API (`demo.py`).

Интеграция с `WorkingMemory`, профилем и CLI — следующий этап.
Криптографическая проверка конверта (этап 2.2, HMAC) **не реализована**:
`task_store.py` по-прежнему только структурно разбирает `_task_fsm`.

## Состояния

- `planning` — план создаётся и ожидает утверждения пользователем;
- `execution` — выполняется текущий шаг плана;
- `validation` — результат шага проверяется программно;
- `done` — терминальное состояние, переходы из него запрещены.

Пауза — не пятое состояние, а флаг `paused`. Она разрешена в `planning`,
`execution` и `validation`, сохраняет этап, текущий шаг и ожидаемое действие.
`resume` возвращает задачу в тот же этап.

## План

- `propose_plan()` принимает только шаги со статусом `pending`;
- план с заранее установленными `active`, `done` или `failed` отклоняется;
- `approve_plan()` активирует ровно первый шаг плана;
- `can_transition()` и `apply()` используют одну и ту же проверку, поэтому
  не могут разойтись в решении.

## Валидация

`ValidationResult` нельзя создать как «просто `passed=True`». Успех вычисляется
только как конъюнкция всех проверок:

```python
request = machine.validation_request()          # FSM выдаёт запрос валидатору
result = request.create_result(                  # результат создаёт валидатор
    checks=(
        {"name": "tests", "passed": True, "evidence": "pytest: 12 passed"},
    ),
    evidence="обязательные проверки пройдены",
)
machine.apply_validation(result, actor=validator)  # FSM только применяет результат
```

Каждая проверка обязана содержать `name`, `passed` и `evidence`. Результат
привязан к `step_id`, отпечатку отправленного результата и `attempt`, поэтому
старый результат другой попытки не применится к текущей.

`TaskFSM.validation_request()` выдаёт одноразовый capability; он проверяется
и в `TaskController`, и в `TaskFSM.can_transition()`. Результат без
действующего capability отклоняется даже при прямом вызове
`apply_validation()`. После `pause()`/`resume()` старый capability
инвалидируется; запрос нужно получать заново.

Capability — API-механизм, **не криптографическая аутентификация**. В одном
Python-процессе токен можно извлечь, а `TaskFSM`/`TaskController` доступны
через интроспекцию. Для настоящей изоляции валидатора нужен отдельный
процесс или внешний сервис.

## Архитектура: контроллер и агент

```
пользователь/UI
      │
      ▼
TaskController ──────────────► TaskFSM
  approve_plan                  TRANSITIONS
  pause/resume                  can_transition/apply
  validation_request
  confirm_validation
  run_current_step
      │
      ▼
AgentAdapter
  propose_plan
  report_step_result
  status

TaskController ──► CodeExecutor (LocalCodeExecutor)
                       │ реальный локальный runner
                       │ CODE-контекст создаётся здесь
                       ▼
                 ExecutionReceipt ──► submit_execution() ──► TaskFSM
```

- `AgentAdapter` — **API-разграничение для обычного интерфейса**, не защита
  от злонамеренного Python-кода. У него нет методов
  approve/validate/pause/resume/run; он хранит только три операции.
  Closure-ссылки на контроллер доступны через `__closure__`; для строгой
  изоляции нужен отдельный процесс или внешний сервис.
- `AgentReport` — советующее сообщение агента; он **никогда** не меняет FSM
  и не участвует в валидации.
- `ExecutionReceipt` создаётся только после реального локального действия.
- Только `LocalCodeExecutor` создаёт доверенный `Actor.CODE`.
- `TaskController` не создаёт CODE-контекст и не принимает строку агента как
  результат шага.
- `TaskFSM.apply()` возвращает копию `Transition`; история FSM не разделяет
  изменяемый `validation` с вызывающим кодом.
- Контроллер создаётся только через `new_task()`/`restore()`; внутренняя
  сборка — приватный класс-метод, который deepcopy-ит `TaskState`. Публичного
  builder/токена нет; интроспекция приватного метода всё равно возможна.

### Receipt: что проверяется

- `controller_id`, `run_seq`, `task_id`, `step_id`, `attempt` должны совпадать
  с активным `ExecutionRequest`;
- повторная отправка receipt отклоняется;
- `actor.actor is Actor.CODE`, `actor.trusted is True`,
  `actor.source == f"executor:{executor_id}"`;
- receipt/request — frozen dataclasses из неизменяемых примитивов.

Контракт `CodeExecutor.execute()`: исключение допустимо только **до**
отправки receipt. Если executor выбросил исключение после успешного
`submit_execution`, контроллер бросает `ExecutionAppliedError` с
`receipt_applied=True` и текущим состоянием, а не выдаёт ошибку за обычный
неуспех.

Проверка `actor.source`/`executor_id`/`evidence` — **дополнительная политика и
аудит, а не аутентификация**. Все эти поля self-declared и могут быть
подделаны кодом в том же процессе.

## Восстановление состояния

`TaskState.from_dict()` проверяет инварианты и при нарушении выбрасывает
`ValueError`. В частности, запрещено загружать:

- `done` без успешной валидации последнего шага и согласованной истории;
- `execution` или `validation` без утверждённого плана;
- `current_step`, которого нет в плане;
- `paused_state=done`;
- состояние, где история, статусы шагов и `expected_action` противоречат
  друг другу.

Инварианты **не подтверждают авторство истории**. `TaskController.restore`:

| Состояние | Без подтверждения | `allow_unverified=True` |
|---|---|---|
| чистый `planning` без привилегированных переходов | разрешено | разрешено |
| `execution` | `UntrustedStateError` | разрешено, можно запускать шаг, `history_trusted=False` |
| `validation` | `UntrustedStateError` | только инспекция и `restart_current_step()`; подтвердить старое validation нельзя |
| `paused` | `UntrustedStateError` | разрешено, `history_trusted=False` |
| `done` | `UntrustedStateError` | только для чтения, `history_trusted=False` |

Любое восстановленное состояние имеет `history_trusted=False`, даже после
подтверждения пользователем. Подтверждение — согласие на риск, не
доказательство подлинности.

Для восстановленного `validation` запрещены `validation_request()` и
`confirm_validation()`: старый результат исполнения не считается проверенным.
Штатный путь продолжения — `controller.restart_current_step()`; он фиксирует
failed-валидацию «требуется повторное выполнение» и возвращает задачу в
`execution`. Только после нового реального запуска можно снова запросить
валидацию и получить `done`.

## Демонстрация

```bash
cd gemini-task13
../venv/bin/python demo.py
```

Сценарий: создание задачи → предложение плана → AgentReport (не меняет FSM) →
попытка агента запустить шаг до утверждения (`AgentPermissionError`) →
отказ контроллера до утверждения (`ControllerError`) → утверждение
контроллером → пауза/продолжение → реальный локальный runner → отказ агенту
в подтверждении валидации → подтверждение валидатором → `done` → запрет
переходов после `done`.

`LocalCodeExecutor` вызывает только функцию из фиксированного реестра
`RUNNERS` (`build_weekly_report`). Нет `eval`/`exec`/`subprocess`, сети, Gemini
API и записи файлов.

## Матрица соответствия исходному заданию Дня 13

| Требование | Реализация | Тесты |
|---|---|---|
| Этап задачи | `TaskState.state`, `STATES` | `test_initial_state_is_planning` |
| Текущий шаг | `TaskState.current_step`, `_validate_current_step` | `test_propose_and_approve_plan`, `test_from_dict_rejects_current_step_missing_from_plan` |
| Ожидаемое действие | `TaskState.expected_action`, `_validate_expected_action` | `test_initial_state_is_planning`, `test_pause_resume_preserves_state_step_and_expected_action` |
| Допустимые состояния и переходы | `TRANSITIONS`, `can_transition`, `apply` | `test_can_transition_and_apply_agree`, `test_cannot_skip_stages` |
| Запрет реализации без утверждённого плана | `submit_step` только из `execution`, `_validate_approval` | `test_cannot_execute_without_approved_plan`, `test_from_dict_rejects_execution_without_approved_plan`, demo |
| Запрет завершения без валидации | `_validate_validation`, `_validate_history` | `test_done_requires_successful_validation`, `test_from_dict_rejects_forged_done_*` |
| Проверка перехода до применения | `apply` вызывает `can_transition` | `test_can_transition_is_pure`, `test_can_transition_and_apply_agree` |
| Пауза на любом этапе | `pause` из planning/execution/validation | `test_pause_resume_preserves_state_step_and_expected_action`, `test_pause_is_not_allowed_from_done` |
| Продолжение без повторного объяснения | `paused_state/paused_step/paused_expected_action` | `test_pause_resume_preserves_state_step_and_expected_action` |
| Обработка запрещённых переходов | `TransitionError` с `reason/state/event` | `test_any_transition_is_blocked_while_paused`, `test_done_is_terminal`, demo |
| Демонстрация попытки нарушения | `demo.py` | `tests/test_demo.py` |
| Разделение ролей | `TaskController`, `AgentAdapter` | `tests/test_task_controller.py` |
| Результат агента ≠ свидетельство | `AgentReport` vs `ExecutionReceipt` | `test_agent_report_never_changes_fsm`, `tests/test_execution.py` |

## Границы доверия и безопасности

- `actor` — политическая метка, а не доказательство личности.
- `ActorContext.controller()` **публичен**: любой код в процессе может создать
  доверенный контекст и обойти `AgentAdapter`. Это подтверждено тестом
  `test_public_actor_context_can_bypass_adapter_boundary` и является известной
  границей, а не защитой.
- `AgentAdapter` не хранит контроллер, но его closure-функции доступны через
  интроспекцию (`__closure__`). Это API-разграничение для обычного интерфейса,
  не sandbox; строгая изоляция требует отдельного процесса/сервиса.
- Capability валидации проверяется и в контроллере, и в `TaskFSM`; он
  одноразовый и инвалидируется при pause/resume. Но capability не
  аутентифицирует валидатора: код, получивший запрос, может создать checks.
  Прямой `TaskFSM.apply_validation()` без capability отклоняется.
- `TaskFSM.from_dict()` проверяет инварианты, но не подлинность истории.
  Согласованная поддельная история принимается; поэтому restore не считает
  историю доверенной.
- Восстановленное `validation` нельзя подтвердить старым результатом:
  `validation_request`/`confirm_validation` заблокированы до
  `restart_current_step()` и нового исполнения.
- `ExecutionReceipt`, `executor_id`, `evidence`, `actor.source` — обычные
  данные. Проверки вокруг них не криптографические.
- `TaskController` создаётся только через `new_task()`/`restore()`;
  публичного builder/токена нет, внутренний фабричный класс-метод deepcopy-ит
  `TaskState`. Интроспекция приватного метода остаётся возможной и не
  считается границей безопасности.
- `TaskFSM.apply()` возвращает копию `Transition`; это защищает историю от
  случайного алиасинга, но private `_state`/`object.__setattr__` остаются
  доступны в процессе.
- `LocalCodeExecutor` безопасен только при доверенном фиксированном реестре
  runners; если реестр подменяем, граница рушится.
- `DiagnosticsSink` не меняет FSM. Исключение в диагностическом обработчике
  подавляется и не может ни откатить переход, ни превратить неуспех в успех;
  цена — отсутствие видимости диагностической ошибки.
- HMAC/KeyProvider и проверка целостности сохранённого конверта не
  реализованы; `task_store.py` остаётся read-only.
- Настоящая граница требует отдельного процесса/прав или внешнего подписанта,
  а не только разделения Python-интерфейсов.

## Запуск тестов

```bash
cd gemini-task13
../venv/bin/python -m pytest -q
../venv/bin/python -m pytest -q -W error
```

Текущий набор: 292 теста (77 исходных `test_task_state.py`, 116
`test_task_store.py`, 99 новых тестов контроллера/исполнителя/demo,
capability, attempt, restore validation и защиты от алиасинга).
