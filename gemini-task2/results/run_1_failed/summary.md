# Сводка эксперимента

Модель: `gemini-3.8-flash`  
Дата: 2026-11-03T22:47:33

## Все запуски

| Вариант | Запуск | finish_reason | Время, с | Символов | Вход. токены | Выход. токены | Thinking-токены | JSON | Pydantic | Книг | Ошибка API |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_unconstrained | 1 | — | 119.028 | — | — | — | — | — | — | — | 504: Deadline expired before operation could complete. |
| v1_unconstrained | 2 | — | 118.856 | — | — | — | — | — | — | — | 504: Deadline expired before operation could complete. |
| v1_unconstrained | 3 | — | 3.237 | — | — | — | — | — | — | — | 503: This model is currently experiencing high demand. Spikes in  |
| v2_format_and_limits | 1 | — | 37.243 | — | — | — | — | — | — | — | 503: This model is currently experiencing high demand. Spikes in  |
| v2_format_and_limits | 2 | — | 1.728 | — | — | — | — | — | — | — | 503: This model is currently experiencing high demand. Spikes in  |
| v2_format_and_limits | 3 | — | 1.535 | — | — | — | — | — | — | — | 503: This model is currently experiencing high demand. Spikes in  |
| v3_structured_output | 1 | — | 0.069 | — | — | — | — | — | — | — | 400: Invalid JSON payload received. Unknown name "additional_prop |
| v3_structured_output | 2 | — | 0.08 | — | — | — | — | — | — | — | 400: Invalid JSON payload received. Unknown name "additional_prop |
| v3_structured_output | 3 | — | 0.085 | — | — | — | — | — | — | — | 400: Invalid JSON payload received. Unknown name "additional_prop |

## Итоги по вариантам

| Вариант | Успешных ответов API | JSON распарсен | Прошёл Pydantic | Ср. время, с | Ср. выход. токены | Ср. символов | Уникальных наборов книг |
|---|---|---|---|---|---|---|---|
| v1_unconstrained | 0/3 | 0/3 | 0/3 | — | — | — | — |
| v2_format_and_limits | 0/3 | 0/3 | 0/3 | — | — | — | — |
| v3_structured_output | 0/3 | 0/3 | 0/3 | — | — | — | — |

## Ошибки парсинга JSON

Нет.

## Ошибки валидации Pydantic

Нет.

## Ошибки API

- **v1_unconstrained / запуск 1** — APIError 504: Deadline expired before operation could complete.
- **v1_unconstrained / запуск 2** — APIError 504: Deadline expired before operation could complete.
- **v1_unconstrained / запуск 3** — APIError 503: This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.
- **v2_format_and_limits / запуск 1** — APIError 503: This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.
- **v2_format_and_limits / запуск 2** — APIError 503: This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.
- **v2_format_and_limits / запуск 3** — APIError 503: This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.
- **v3_structured_output / запуск 1** — APIError 400: Invalid JSON payload received. Unknown name "additional_properties" at 'generation_config.response_schema': Cannot find field.
Invalid JSON payload received. Unknown name "additional_properties" at 'generation_config.response_schema.properties[0].value.items': Cannot find field.
- **v3_structured_output / запуск 2** — APIError 400: Invalid JSON payload received. Unknown name "additional_properties" at 'generation_config.response_schema': Cannot find field.
Invalid JSON payload received. Unknown name "additional_properties" at 'generation_config.response_schema.properties[0].value.items': Cannot find field.
- **v3_structured_output / запуск 3** — APIError 400: Invalid JSON payload received. Unknown name "additional_properties" at 'generation_config.response_schema': Cannot find field.
Invalid JSON payload received. Unknown name "additional_properties" at 'generation_config.response_schema.properties[0].value.items': Cannot find field.
