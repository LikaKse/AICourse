# Сводка эксперимента

Модель: `gemini-3.8-flash`  
Дата: 2026-11-04T22:45:41  
Порядок: варианты чередуются (v1, v2, v3, v1, …)  
Выполнено запусков: 3/9; статус: в процессе

## Все запуски

Время — последней попытки; «Всего, с» — весь запуск, включая повторы и паузы.

| Вариант | Запуск | Попыток | finish_reason | Время, с | Всего, с | Символов | Вход. токены | Выход. токены | Thinking-токены | JSON | Pydantic | Книг | Итоговая ошибка API |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_unconstrained | 1 | 2 | — | 9.438 | 54.424 | — | — | — | — | — | — | — | 503: This model is currently experiencing high demand. Spikes in  |
| v2_format_and_limits | 1 | 2 | — | 17.783 | 38.582 | — | — | — | — | — | — | — | 503: This model is currently experiencing high demand. Spikes in  |
| v3_structured_output | 1 | 2 | — | 118.897 | 133.162 | — | — | — | — | — | — | — | 504: Deadline expired before operation could complete. |

## Итоги по вариантам

Успех/JSON/Pydantic считаются по запускам (итог после всех попыток). Средние — только по успешным запускам, время — успешной попытки.

| Вариант | Выполнено запусков | Успешных ответов API | JSON распарсен | Прошёл Pydantic | Попыток всего | Запусков с повтором | Ср. время, с | Ср. выход. токены | Ср. символов | Уникальных наборов книг |
|---|---|---|---|---|---|---|---|---|---|---|
| v1_unconstrained | 1/3 | 0/1 | 0/1 | 0/1 | 2 | 1/1 | — | — | — | — |
| v2_format_and_limits | 1/3 | 0/1 | 0/1 | 0/1 | 2 | 1/1 | — | — | — | — |
| v3_structured_output | 1/3 | 0/1 | 0/1 | 0/1 | 2 | 1/1 | — | — | — | — |

## Ошибки парсинга JSON

Нет.

## Ошибки валидации Pydantic

Нет.

## Повторные попытки (503/504)

- **v1_unconstrained / запуск 1** — 2 попыток, итог: неудача (503)
  - попытка 1: 503: This model is currently experiencing high demand. Spikes in demand are usually t (31.985 с, пауза 13 с)
  - попытка 2: 503: This model is currently experiencing high demand. Spikes in demand are usually t (9.438 с)
- **v2_format_and_limits / запуск 1** — 2 попыток, итог: неудача (503)
  - попытка 1: 503: This model is currently experiencing high demand. Spikes in demand are usually t (7.797 с, пауза 13 с)
  - попытка 2: 503: This model is currently experiencing high demand. Spikes in demand are usually t (17.783 с)
- **v3_structured_output / запуск 1** — 2 попыток, итог: неудача (504)
  - попытка 1: 503: This model is currently experiencing high demand. Spikes in demand are usually t (1.26 с, пауза 13 с)
  - попытка 2: 504: Deadline expired before operation could complete. (118.897 с)

## Итоговые ошибки API (после всех попыток)

- **v1_unconstrained / запуск 1** — APIError 503 (попыток: 2): This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.
- **v2_format_and_limits / запуск 1** — APIError 503 (попыток: 2): This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.
- **v3_structured_output / запуск 1** — APIError 504 (попыток: 2): Deadline expired before operation could complete.
