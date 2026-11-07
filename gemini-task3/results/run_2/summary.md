# Сводка: разные способы рассуждения

Модель: `gemini-3.5-flash-lite`  
Режим: full; создан: 2026-11-05T00:05:10; обновлён: 2026-11-05T00:51:19  
Решений: 40/40; статус: завершён  
Финальные ответы всех способов — единая JSON-схема `{answer, explanation}`; правильность проверяется программно.

## Итоговая таблица

Точность — доля верных среди полученных ответов (в скобках — от всех решений группы, включая ошибки API). Токены — `total_token_count`; для Prompt generation — сумма двух вызовов. Среднее время — время успешных вызовов, без пауз и повторов.

| Способ | Reasoning | Верных ответов | Всего | Точность | Токены | Токенов на задачу | Среднее время, с | Ошибок API | Повторов |
|---|---|---|---|---|---|---|---|---|---|
| Direct | MINIMAL | 5 | 5 | 100% (100%) | 652 | 130 | 27.31 | 0 | 3 |
| Direct | HIGH | 5 | 5 | 100% (100%) | 3458 | 692 | 17.71 | 0 | 0 |
| Step-by-step | MINIMAL | 5 | 5 | 100% (100%) | 814 | 163 | 16.04 | 0 | 2 |
| Step-by-step | HIGH | 5 | 5 | 100% (100%) | 3794 | 759 | 13.02 | 0 | 2 |
| Prompt generation | MINIMAL | 5 | 5 | 100% (100%) | 2537 | 507 | 33.98 | 0 | 1 |
| Prompt generation | HIGH | 5 | 5 | 100% (100%) | 11018 | 2204 | 34.82 | 0 | 1 |
| Expert group | MINIMAL | 5 | 5 | 100% (100%) | 1369 | 274 | 15.34 | 0 | 0 |
| Expert group | HIGH | 5 | 5 | 100% (100%) | 4830 | 966 | 14.69 | 0 | 0 |

Всего: верных 40 из 40 полученных ответов (100.0%); решений 40; вызовов API (попыток) 59; ошибок API 0; повторных попыток 9.

## MINIMAL vs HIGH по каждому способу

| Способ | Точность MINIMAL | Точность HIGH | Δ точности | Токенов/задачу MINIMAL | Токенов/задачу HIGH | Thinking-токенов MINIMAL | Thinking-токенов HIGH | Время MINIMAL, с | Время HIGH, с |
|---|---|---|---|---|---|---|---|---|---|
| Direct | 100% | 100% | 0 п.п. | 130 | 692 | — | 575 | 27.31 | 17.71 |
| Step-by-step | 100% | 100% | 0 п.п. | 163 | 759 | — | 612 | 16.04 | 13.02 |
| Prompt generation | 100% | 100% | 0 п.п. | 507 | 2204 | — | 1571 | 33.98 | 34.82 |
| Expert group | 100% | 100% | 0 п.п. | 274 | 966 | — | 706 | 15.34 | 14.69 |

## Сравнение с Direct на том же уровне

| Способ | Reasoning | Δ точности к Direct | Токенов/задачу | × к Direct по токенам | Время, с | × к Direct по времени |
|---|---|---|---|---|---|---|
| Direct | MINIMAL | 0 п.п. | 130 | 1.0× | 27.31 | 1.0× |
| Step-by-step | MINIMAL | 0 п.п. | 163 | 1.25× | 16.04 | 0.59× |
| Prompt generation | MINIMAL | 0 п.п. | 507 | 3.89× | 33.98 | 1.24× |
| Expert group | MINIMAL | 0 п.п. | 274 | 2.1× | 15.34 | 0.56× |
| Direct | HIGH | 0 п.п. | 692 | 1.0× | 17.71 | 1.0× |
| Step-by-step | HIGH | 0 п.п. | 759 | 1.1× | 13.02 | 0.74× |
| Prompt generation | HIGH | 0 п.п. | 2204 | 3.19× | 34.82 | 1.97× |
| Expert group | HIGH | 0 п.п. | 966 | 1.4× | 14.69 | 0.83× |

## Выводы (вычислены автоматически)

- **Step-by-step** — MINIMAL: 100% против Direct 100% (0 п.п.) — без разницы; HIGH: 100% против Direct 100% (0 п.п.) — без разницы. Итог: при MINIMAL выигрыша нет.
- **Prompt generation** — MINIMAL: 100% против Direct 100% (0 п.п.) — без разницы; HIGH: 100% против Direct 100% (0 п.п.) — без разницы. Итог: при MINIMAL выигрыша нет.
- **Expert group** — MINIMAL: 100% против Direct 100% (0 п.п.) — без разницы; HIGH: 100% против Direct 100% (0 п.п.) — без разницы. Итог: при MINIMAL выигрыша нет.
- При MINIMAL Direct уже даёт 100%: улучшить точность на этом наборе нельзя (эффект потолка), сравнивать способы стоит по токенам и времени.
- При HIGH Direct уже даёт 100%: улучшить точность на этом наборе нельзя (эффект потолка), сравнивать способы стоит по токенам и времени.
- Выборка мала (5 задач на группу): разница в одну задачу = 20 п.п.; выводы предварительные.

## Результаты по задачам

✅ верно · ❌ неверно · ⚠️ ошибка API · · не выполнялось

| Задача | Direct MIN | Direct HIG | Step-by-step MIN | Step-by-step HIG | Prompt generation MIN | Prompt generation HIG | Expert group MIN | Expert group HIG |
|---|---|---|---|---|---|---|---|---|
| math_bat_ball | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| logic_syllogism | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| combinatorics_digits | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| algorithm_loop | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| order_logic | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

## Prompt generation: генерация промпта + решение

| Задача | Reasoning | Токены: генерация | Токены: решение | Токены: итого | Время: генерация, с | Время: решение, с | Время: итого, с | Ответ в промпте? |
|---|---|---|---|---|---|---|---|---|
| math_bat_ball | MINIMAL | 303 | 365 | 668 | 17.799 | 18.837 | 36.636 | нет |
| math_bat_ball | HIGH | 1320 | 996 | 2316 | 19.294 | 15.086 | 34.38 | нет |
| logic_syllogism | MINIMAL | 185 | 238 | 423 | 26.4 | 16.318 | 42.718 | — |
| logic_syllogism | HIGH | 1126 | 1166 | 2292 | 27.587 | 18.58 | 46.167 | — |
| combinatorics_digits | MINIMAL | 229 | 281 | 510 | 10.886 | 10.321 | 21.207 | нет |
| combinatorics_digits | HIGH | 1128 | 964 | 2092 | 16.518 | 8.354 | 24.872 | нет |
| algorithm_loop | MINIMAL | 201 | 283 | 484 | 21.724 | 9.255 | 30.979 | нет |
| algorithm_loop | HIGH | 1356 | 1250 | 2606 | 8.816 | 24.467 | 33.283 | нет |
| order_logic | MINIMAL | 201 | 251 | 452 | 19.852 | 18.522 | 38.374 | нет |
| order_logic | HIGH | 926 | 786 | 1712 | 25.092 | 10.298 | 35.39 | нет |

## Неверные ответы

Нет.

## Ошибки API

Нет.

## Повторные попытки

- math_bat_ball|direct|MINIMAL / solve / попытка 1: 503 This model is currently experiencing high demand. Spikes in demand are usually t, пауза 15 с
- math_bat_ball|direct|MINIMAL / solve / попытка 2: 503 This model is currently experiencing high demand. Spikes in demand are usually t, пауза 30 с
- math_bat_ball|step_by_step|MINIMAL / solve / попытка 1: 503 This model is currently experiencing high demand. Spikes in demand are usually t, пауза 15 с
- math_bat_ball|prompt_generation|MINIMAL / prompt_generation / попытка 1: 503 This model is currently experiencing high demand. Spikes in demand are usually t, пауза 15 с
- math_bat_ball|step_by_step|HIGH / solve / попытка 1: 503 This model is currently experiencing high demand. Spikes in demand are usually t, пауза 15 с
- logic_syllogism|step_by_step|MINIMAL / solve / попытка 1: 503 This model is currently experiencing high demand. Spikes in demand are usually t, пауза 15 с
- logic_syllogism|step_by_step|HIGH / solve / попытка 1: 503 This model is currently experiencing high demand. Spikes in demand are usually t, пауза 15 с
- logic_syllogism|prompt_generation|HIGH / solve / попытка 1: 504 Deadline expired before operation could complete., пауза 15 с
- order_logic|direct|MINIMAL / solve / попытка 1: 503 This model is currently experiencing high demand. Spikes in demand are usually t, пауза 15 с
