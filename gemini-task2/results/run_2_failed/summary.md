# Сводка эксперимента

Модель: `gemini-3.8-flash`  
Дата: 2026-11-04T00:50:59

## Все запуски

Время — последней попытки; «Всего, с» — весь запуск, включая повторы и паузы.

| Вариант | Запуск | Попыток | finish_reason | Время, с | Всего, с | Символов | Вход. токены | Выход. токены | Thinking-токены | JSON | Pydantic | Книг | Итоговая ошибка API |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_unconstrained | 1 | 4 | — | 68.014 | 194.34 | — | — | — | — | — | — | — | 503: This model is currently experiencing high demand. Spikes in  |
| v1_unconstrained | 2 | 4 | — | 6.557 | 127.788 | — | — | — | — | — | — | — | 503: This model is currently experiencing high demand. Spikes in  |
| v1_unconstrained | 3 | 4 | — | 68.097 | 291.995 | — | — | — | — | — | — | — | 503: This model is currently experiencing high demand. Spikes in  |
| v2_format_and_limits | 1 | 4 | — | 3.233 | 247.638 | — | — | — | — | — | — | — | 503: This model is currently experiencing high demand. Spikes in  |
| v2_format_and_limits | 2 | 1 | — | 0.237 | 0.237 | — | — | — | — | — | — | — | 429: You exceeded your current quota, please check your plan and  |
| v2_format_and_limits | 3 | 1 | — | 0.375 | 0.375 | — | — | — | — | — | — | — | 429: You exceeded your current quota, please check your plan and  |
| v3_structured_output | 1 | 1 | — | 0.342 | 0.342 | — | — | — | — | — | — | — | 429: You exceeded your current quota, please check your plan and  |
| v3_structured_output | 2 | 1 | — | 0.239 | 0.239 | — | — | — | — | — | — | — | 429: You exceeded your current quota, please check your plan and  |
| v3_structured_output | 3 | 2 | — | 0.325 | 19.798 | — | — | — | — | — | — | — | 429: You exceeded your current quota, please check your plan and  |

## Итоги по вариантам

Успех/JSON/Pydantic считаются по запускам (итог после всех попыток). Средние — только по успешным запускам, время — успешной попытки.

| Вариант | Успешных ответов API | JSON распарсен | Прошёл Pydantic | Попыток всего | Запусков с повтором | Ср. время, с | Ср. выход. токены | Ср. символов | Уникальных наборов книг |
|---|---|---|---|---|---|---|---|---|---|
| v1_unconstrained | 0/3 | 0/3 | 0/3 | 12 | 3/3 | — | — | — | — |
| v2_format_and_limits | 0/3 | 0/3 | 0/3 | 6 | 1/3 | — | — | — | — |
| v3_structured_output | 0/3 | 0/3 | 0/3 | 4 | 1/3 | — | — | — | — |

## Ошибки парсинга JSON

Нет.

## Ошибки валидации Pydantic

Нет.

## Повторные попытки (503/504)

- **v1_unconstrained / запуск 1** — 4 попыток, итог: неудача (503)
  - попытка 1: 503: This model is currently experiencing high demand. Spikes in demand are usually t (15.972 с, пауза 10 с)
  - попытка 2: 503: This model is currently experiencing high demand. Spikes in demand are usually t (9.494 с, пауза 30 с)
  - попытка 3: 503: This model is currently experiencing high demand. Spikes in demand are usually t (0.852 с, пауза 60 с)
  - попытка 4: 503: This model is currently experiencing high demand. Spikes in demand are usually t (68.014 с)
- **v1_unconstrained / запуск 2** — 4 попыток, итог: неудача (503)
  - попытка 1: 503: This model is currently experiencing high demand. Spikes in demand are usually t (1.31 с, пауза 10 с)
  - попытка 2: 503: This model is currently experiencing high demand. Spikes in demand are usually t (14.278 с, пауза 30 с)
  - попытка 3: 503: This model is currently experiencing high demand. Spikes in demand are usually t (5.631 с, пауза 60 с)
  - попытка 4: 503: This model is currently experiencing high demand. Spikes in demand are usually t (6.557 с)
- **v1_unconstrained / запуск 3** — 4 попыток, итог: неудача (503)
  - попытка 1: 504: Deadline expired before operation could complete. (118.832 с, пауза 10 с)
  - попытка 2: 503: This model is currently experiencing high demand. Spikes in demand are usually t (1.452 с, пауза 30 с)
  - попытка 3: 503: This model is currently experiencing high demand. Spikes in demand are usually t (3.603 с, пауза 60 с)
  - попытка 4: 503: This model is currently experiencing high demand. Spikes in demand are usually t (68.097 с)
- **v2_format_and_limits / запуск 1** — 4 попыток, итог: неудача (503)
  - попытка 1: 503: This model is currently experiencing high demand. Spikes in demand are usually t (11.476 с, пауза 10 с)
  - попытка 2: 504: Deadline expired before operation could complete. (118.956 с, пауза 30 с)
  - попытка 3: 503: This model is currently experiencing high demand. Spikes in demand are usually t (13.964 с, пауза 60 с)
  - попытка 4: 503: This model is currently experiencing high demand. Spikes in demand are usually t (3.233 с)
- **v3_structured_output / запуск 3** — 2 попыток, итог: неудача (429)
  - попытка 1: 503: This model is currently experiencing high demand. Spikes in demand are usually t (9.47 с, пауза 10 с)
  - попытка 2: 429: You exceeded your current quota, please check your plan and billing details. For (0.325 с)

## Итоговые ошибки API (после всех попыток)

- **v1_unconstrained / запуск 1** — APIError 503 (попыток: 4): This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.
- **v1_unconstrained / запуск 2** — APIError 503 (попыток: 4): This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.
- **v1_unconstrained / запуск 3** — APIError 503 (попыток: 4): This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.
- **v2_format_and_limits / запуск 1** — APIError 503 (попыток: 4): This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.
- **v2_format_and_limits / запуск 2** — APIError 429 (попыток: 1): You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. / * Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.8-flash, description: Generate content requests per day limit exceeded on Free Tier. See https://ai.google.dev/gemini-api/docs/rate-limits. / Please retry in 14h9m28.842268966s.
- **v2_format_and_limits / запуск 3** — APIError 429 (попыток: 1): You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. / * Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.8-flash, description: Generate content requests per day limit exceeded on Free Tier. See https://ai.google.dev/gemini-api/docs/rate-limits. / Please retry in 14h9m26.548927562s.
- **v3_structured_output / запуск 1** — APIError 429 (попыток: 1): You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. / * Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.8-flash, description: Generate content requests per day limit exceeded on Free Tier. See https://ai.google.dev/gemini-api/docs/rate-limits. / Please retry in 14h9m24.145269314s.
- **v3_structured_output / запуск 2** — APIError 429 (попыток: 1): You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. / * Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.8-flash, description: Generate content requests per day limit exceeded on Free Tier. See https://ai.google.dev/gemini-api/docs/rate-limits. / Please retry in 14h9m21.85894521s.
- **v3_structured_output / запуск 3** — APIError 429 (попыток: 2): You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. / * Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.8-flash, description: Generate content requests per day limit exceeded on Free Tier. See https://ai.google.dev/gemini-api/docs/rate-limits. / Please retry in 14h9m0.054248894s.
