# Сводка эксперимента

Модель: `gemini-3.6-flash` (запасная; основная модель проекта — `gemini-3.8-flash`)  
Дата: 2026-11-04T22:29:42  
Порядок: варианты чередуются (v1, v2, v3, v1, …)  
Выполнено запусков: 9/9; статус: завершён

## Все запуски

Время — последней попытки; «Всего, с» — весь запуск, включая повторы и паузы.

| Вариант | Запуск | Попыток | finish_reason | Время, с | Всего, с | Символов | Вход. токены | Выход. токены | Thinking-токены | JSON | Pydantic | Книг | Итоговая ошибка API |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_unconstrained | 1 | 1 | STOP | 17.242 | 17.242 | 2001 | 37 | 567 | 1447 | json_error | skipped | — | — |
| v2_format_and_limits | 1 | 2 | MAX_TOKENS | 16.442 | 35.018 | 20 | 137 | 14 | 589 | json_error | skipped | — | — |
| v3_structured_output | 1 | 2 | — | 10.586 | 25.381 | — | — | — | — | — | — | — | 503: This model is currently experiencing high demand. Spikes in  |
| v1_unconstrained | 2 | 2 | — | 9.038 | 32.475 | — | — | — | — | — | — | — | 503: This model is currently experiencing high demand. Spikes in  |
| v2_format_and_limits | 2 | 1 | MAX_TOKENS | 78.618 | 78.618 | 38 | 137 | 12 | 384 | json_error | skipped | — | — |
| v3_structured_output | 2 | 2 | STOP | 9.871 | 24.511 | 589 | 37 | 158 | 668 | ok | ok | 3 | — |
| v1_unconstrained | 3 | 1 | STOP | 12.453 | 12.453 | 1879 | 37 | 536 | 1118 | json_error | skipped | — | — |
| v2_format_and_limits | 3 | 1 | MAX_TOKENS | 4.631 | 4.631 | 0 | 137 | — | 426 | empty | skipped | — | — |
| v3_structured_output | 3 | 1 | STOP | 5.396 | 5.396 | 533 | 37 | 144 | 654 | ok | ok | 3 | — |

## Итоги по вариантам

Успех/JSON/Pydantic считаются по запускам (итог после всех попыток). Средние — только по успешным запускам, время — успешной попытки.

| Вариант | Выполнено запусков | Успешных ответов API | JSON распарсен | Прошёл Pydantic | Попыток всего | Запусков с повтором | Ср. время, с | Ср. выход. токены | Ср. символов | Уникальных наборов книг |
|---|---|---|---|---|---|---|---|---|---|---|
| v1_unconstrained | 3/3 | 2/3 | 0/3 | 0/3 | 4 | 1/3 | 14.85 | 552 | 1940 | — |
| v2_format_and_limits | 3/3 | 3/3 | 0/3 | 0/3 | 4 | 1/3 | 33.23 | 13 | 19 | — |
| v3_structured_output | 3/3 | 2/3 | 2/3 | 2/3 | 5 | 2/3 | 7.63 | 151 | 561 | 2 |

## Ошибки парсинга JSON

- **v1_unconstrained / запуск 1** — json_error: Expecting value (строка 1, позиция 1)
- **v2_format_and_limits / запуск 1** — json_error: Unterminated string starting at (строка 1, позиция 1)
- **v2_format_and_limits / запуск 2** — json_error: Expecting property name enclosed in double quotes (строка 1, позиция 39)
- **v1_unconstrained / запуск 3** — json_error: Expecting value (строка 1, позиция 1)
- **v2_format_and_limits / запуск 3** — empty: Пустой ответ

## Ошибки валидации Pydantic

Нет.

## Повторные попытки (503/504)

- **v2_format_and_limits / запуск 1** — 2 попыток, итог: успех
  - попытка 1: 503: This model is currently experiencing high demand. Spikes in demand are usually t (5.573 с, пауза 13 с)
  - попытка 2: ok (16.442 с)
- **v3_structured_output / запуск 1** — 2 попыток, итог: неудача (503)
  - попытка 1: 503: This model is currently experiencing high demand. Spikes in demand are usually t (1.789 с, пауза 13 с)
  - попытка 2: 503: This model is currently experiencing high demand. Spikes in demand are usually t (10.586 с)
- **v1_unconstrained / запуск 2** — 2 попыток, итог: неудача (503)
  - попытка 1: 503: This model is currently experiencing high demand. Spikes in demand are usually t (10.434 с, пауза 13 с)
  - попытка 2: 503: This model is currently experiencing high demand. Spikes in demand are usually t (9.038 с)
- **v3_structured_output / запуск 2** — 2 попыток, итог: успех
  - попытка 1: 503: This model is currently experiencing high demand. Spikes in demand are usually t (1.634 с, пауза 13 с)
  - попытка 2: ok (9.871 с)

## Итоговые ошибки API (после всех попыток)

- **v3_structured_output / запуск 1** — APIError 503 (попыток: 2): This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.
- **v1_unconstrained / запуск 2** — APIError 503 (попыток: 2): This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.
