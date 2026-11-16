# Сводка эксперимента

Модель: `gemini-3.8-flash`  
Дата: 2026-11-10T02:39:55  
Порядок: варианты чередуются (v1, v2, v3, v1, …)  
Выполнено запусков: 9/9; статус: завершён

## Все запуски

Время — последней попытки; «Всего, с» — весь запуск, включая повторы и паузы.

| Вариант | Запуск | Попыток | finish_reason | Время, с | Всего, с | Символов | Вход. токены | Выход. токены | Thinking-токены | JSON | Pydantic | Книг | Итоговая ошибка API |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_unconstrained | 1 | 1 | STOP | 14.243 | 14.243 | 1839 | 37 | 527 | 1028 | json_error | skipped | — | — |
| v2_format_and_limits | 1 | 1 | MAX_TOKENS | 4.431 | 4.431 | 33 | 137 | 14 | 382 | json_error | skipped | — | — |
| v3_structured_output | 1 | 1 | STOP | 6.684 | 6.684 | 799 | 37 | 213 | 400 | ok | ok | 3 | — |
| v1_unconstrained | 2 | 1 | STOP | 15.644 | 15.644 | 1840 | 37 | 528 | 1085 | json_error | skipped | — | — |
| v2_format_and_limits | 2 | 1 | MAX_TOKENS | 4.971 | 4.971 | 37 | 137 | 16 | 380 | json_error | skipped | — | — |
| v3_structured_output | 2 | 1 | STOP | 9.637 | 9.637 | 741 | 37 | 196 | 936 | ok | ok | 3 | — |
| v1_unconstrained | 3 | 1 | STOP | 16.527 | 16.527 | 1858 | 37 | 527 | 1242 | json_error | skipped | — | — |
| v2_format_and_limits | 3 | 1 | MAX_TOKENS | 5.639 | 5.639 | 28 | 137 | 13 | 383 | json_error | skipped | — | — |
| v3_structured_output | 3 | 1 | STOP | 8.185 | 8.185 | 574 | 37 | 156 | 715 | ok | ok | 3 | — |

## Итоги по вариантам

Успех/JSON/Pydantic считаются по запускам (итог после всех попыток). Средние — только по успешным запускам, время — успешной попытки.

| Вариант | Выполнено запусков | Успешных ответов API | JSON распарсен | Прошёл Pydantic | Попыток всего | Запусков с повтором | Ср. время, с | Ср. выход. токены | Ср. символов | Уникальных наборов книг |
|---|---|---|---|---|---|---|---|---|---|---|
| v1_unconstrained | 3/3 | 3/3 | 0/3 | 0/3 | 3 | 0/3 | 15.47 | 527 | 1846 | — |
| v2_format_and_limits | 3/3 | 3/3 | 0/3 | 0/3 | 3 | 0/3 | 5.01 | 14 | 33 | — |
| v3_structured_output | 3/3 | 3/3 | 3/3 | 3/3 | 3 | 0/3 | 8.17 | 188 | 705 | 3 |

## Ошибки парсинга JSON

- **v1_unconstrained / запуск 1** — json_error: Expecting value (строка 1, позиция 1)
- **v2_format_and_limits / запуск 1** — json_error: Unterminated string starting at (строка 4, позиция 7)
- **v1_unconstrained / запуск 2** — json_error: Expecting value (строка 1, позиция 1)
- **v2_format_and_limits / запуск 2** — json_error: Unterminated string starting at (строка 4, позиция 16)
- **v1_unconstrained / запуск 3** — json_error: Expecting value (строка 1, позиция 1)
- **v2_format_and_limits / запуск 3** — json_error: Unterminated string starting at (строка 4, позиция 7)

## Ошибки валидации Pydantic

Нет.

## Повторные попытки (503/504)

Не было.

## Итоговые ошибки API (после всех попыток)

Нет.
