# Сводка эксперимента

Модель: `gemini-3.8-flash`  
Дата: 2026-11-04T22:09:14  
Порядок: варианты чередуются (v1, v2, v3, v1, …)  
Выполнено запусков: 5/9; статус: **остановлен досрочно** — ошибка API 429 в v2_format_and_limits / запуск 2

## Все запуски

Время — последней попытки; «Всего, с» — весь запуск, включая повторы и паузы.

| Вариант | Запуск | Попыток | finish_reason | Время, с | Всего, с | Символов | Вход. токены | Выход. токены | Thinking-токены | JSON | Pydantic | Книг | Итоговая ошибка API |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_unconstrained | 1 | 2 | — | 1.313 | 15.085 | — | — | — | — | — | — | — | 503: This model is currently experiencing high demand. Spikes in  |
| v2_format_and_limits | 1 | 2 | — | 3.646 | 15.481 | — | — | — | — | — | — | — | 503: This model is currently experiencing high demand. Spikes in  |
| v3_structured_output | 1 | 2 | — | 0.558 | 17.105 | — | — | — | — | — | — | — | 503: This model is currently experiencing high demand. Spikes in  |
| v1_unconstrained | 2 | 2 | — | 1.033 | 11.816 | — | — | — | — | — | — | — | 503: This model is currently experiencing high demand. Spikes in  |
| v2_format_and_limits | 2 | 1 | — | 0.347 | 0.347 | — | — | — | — | — | — | — | 429: You exceeded your current quota, please check your plan and  |

## Итоги по вариантам

Успех/JSON/Pydantic считаются по запускам (итог после всех попыток). Средние — только по успешным запускам, время — успешной попытки.

| Вариант | Выполнено запусков | Успешных ответов API | JSON распарсен | Прошёл Pydantic | Попыток всего | Запусков с повтором | Ср. время, с | Ср. выход. токены | Ср. символов | Уникальных наборов книг |
|---|---|---|---|---|---|---|---|---|---|---|
| v1_unconstrained | 2/3 | 0/2 | 0/2 | 0/2 | 4 | 2/2 | — | — | — | — |
| v2_format_and_limits | 2/3 | 0/2 | 0/2 | 0/2 | 3 | 1/2 | — | — | — | — |
| v3_structured_output | 1/3 | 0/1 | 0/1 | 0/1 | 2 | 1/1 | — | — | — | — |

## Ошибки парсинга JSON

Нет.

## Ошибки валидации Pydantic

Нет.

## Досрочная остановка

Причина: ошибка API 429 в v2_format_and_limits / запуск 2

Не выполнены запуски:
- v3_structured_output / запуск 2
- v1_unconstrained / запуск 3
- v2_format_and_limits / запуск 3
- v3_structured_output / запуск 3

## Повторные попытки (503/504)

- **v1_unconstrained / запуск 1** — 2 попыток, итог: неудача (503)
  - попытка 1: 503: This model is currently experiencing high demand. Spikes in demand are usually t (3.768 с, пауза 10 с)
  - попытка 2: 503: This model is currently experiencing high demand. Spikes in demand are usually t (1.313 с)
- **v2_format_and_limits / запуск 1** — 2 попыток, итог: неудача (503)
  - попытка 1: 503: This model is currently experiencing high demand. Spikes in demand are usually t (1.83 с, пауза 10 с)
  - попытка 2: 503: This model is currently experiencing high demand. Spikes in demand are usually t (3.646 с)
- **v3_structured_output / запуск 1** — 2 попыток, итог: неудача (503)
  - попытка 1: 503: This model is currently experiencing high demand. Spikes in demand are usually t (6.543 с, пауза 10 с)
  - попытка 2: 503: This model is currently experiencing high demand. Spikes in demand are usually t (0.558 с)
- **v1_unconstrained / запуск 2** — 2 попыток, итог: неудача (503)
  - попытка 1: 503: This model is currently experiencing high demand. Spikes in demand are usually t (0.781 с, пауза 10 с)
  - попытка 2: 503: This model is currently experiencing high demand. Spikes in demand are usually t (1.033 с)

## Итоговые ошибки API (после всех попыток)

- **v1_unconstrained / запуск 1** — APIError 503 (попыток: 2): This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.
- **v2_format_and_limits / запуск 1** — APIError 503 (попыток: 2): This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.
- **v3_structured_output / запуск 1** — APIError 503 (попыток: 2): This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.
- **v1_unconstrained / запуск 2** — APIError 503 (попыток: 2): This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.
- **v2_format_and_limits / запуск 2** — APIError 429 (попыток: 1): You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. / * Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 5, model: gemini-3.8-flash, description: Generate content requests per minute limit exceeded on Free Tier. See https://ai.google.dev/gemini-api/docs/rate-limits. / Please retry in 45.033687206s.
