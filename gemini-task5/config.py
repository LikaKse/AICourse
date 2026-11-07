"""Настройки эксперимента «Версии моделей»: три модели Gemini на одних и тех же задачах.

Все условия генерации одинаковы для всех моделей. Отличается только идентификатор модели.
"""

# ---------- Модели ----------
# Идентификаторы проверены 06.11.2026: `python main.py --list-models` (client.models.list(),
# google-genai 2.34.0). Все три поддерживают generateContent. Описания взяты со страницы
# https://aistudio.google.com/docs/models

MODELS = [
    {
        "role": "small",
        "role_name": "Маленькая и дешёвая",
        "id": "gemini-3.5-flash-lite",
        "why": "Stable; «the fastest and most cost-effective 3.5 model». Та же модель, что в Днях 3–4.",
    },
    {
        "role": "medium",
        "role_name": "Средняя",
        "id": "gemini-3.8-flash",
        "why": "«The most intelligent Flash model»; на странице моделей отмечена как New (без метки Stable/Preview).",
    },
    {
        "role": "flagship",
        "role_name": "Флагманская",
        "id": "gemini-4-argon",
        "why": "«Our flagship reasoning model». Бесплатного уровня нет: нужен платный (billing) проект.",
    },
]

MODEL_IDS = [m["id"] for m in MODELS]
MODELS_BY_ROLE = {m["role"]: m for m in MODELS}

# ---------- Одинаковые условия генерации ----------
# thinking_level (проверено 06.11.2026, https://aistudio.google.com/docs/thinking):
#   gemini-3.5-flash-lite — minimal/low/medium/high, по умолчанию minimal;
#   gemini-3.8-flash      — low/medium/high, по умолчанию medium;
#   gemini-4-argon        — low/medium/high, по умолчанию medium.
# MINIMAL есть не у всех моделей, поэтому для всех задан LOW: это общий и самый дешёвый уровень.
THINKING_LEVEL = "LOW"

# temperature не задаём: Google рекомендует 1.0 (значение по умолчанию) для Gemini 3+.
# Это подтвердил День 4.

# ---------- Лимит выходных токенов ----------
# По странице thinking max_output_tokens ограничивает ответ ВМЕСТЕ с токенами рассуждений.
# Если модель упирается в лимит во время рассуждения, сгенерированные thinking-токены
# всё равно оплачиваются. Единый лимит делает максимальную стоимость конечной.
# Без лимита действует максимум модели (OUTPUT_TOKEN_LIMITS), и худший случай ≈ $4.4.
# 8192 — с большим запасом: в Дне 3 даже при HIGH ответ flash-lite занимал ≤ 2 204 токенов.
# Обрезанный ответ (finish_reason = MAX_TOKENS) отмечается отдельно (truncated) и считается неверным.
# None — не передавать лимит.
MAX_OUTPUT_TOKENS = 8192

# output_token_limit моделей из client.models.list() (google-genai 2.34.0, 06.11.2026).
# Нужен для расчёта худшего случая, если MAX_OUTPUT_TOKENS = None.
# Внимание: страница моделей пишет для gemini-4-argon «long decodes up to 1M tokens»,
# но API сообщает 32 768. В расчёте используется значение из API.
OUTPUT_TOKEN_LIMITS = {
    "gemini-3.5-flash-lite": 65536,
    "gemini-3.8-flash": 65536,
    "gemini-4-argon": 32768,
}

# Оценка СВЕРХУ входных токенов одного запроса для расчёта максимальной стоимости.
# Самый длинный промпт — 238 символов (≈ 100–150 токенов); 500 — с запасом на JSON-схему.
MAX_INPUT_TOKENS_ESTIMATE = 500

# Общий формат ответа (как в Дне 3) и JSON-схема ApiSolution из validators.py
ANSWER_FORMAT = (
    'Верни только JSON-объект вида {"answer": "короткий ответ", "explanation": "краткое объяснение"} '
    "без Markdown и без текста вне JSON."
)
PROMPT_TEMPLATE = "{question}\n\n{answer_format}"  # это Direct из Дня 3

# Сколько раз решать каждую пару «задача × модель». Можно переопределить флагом --repeats.
REPEATS = 1

# ---------- Темп и надёжность (как в Днях 3–4) ----------

MIN_INTERVAL_BETWEEN_REQUESTS_SEC = 13    # бесплатный тариф: 5 запросов в минуту
REQUEST_TIMEOUT_SEC = 180
RETRY_STATUS_CODES = (429, 500, 502, 503, 504)
RETRY_DELAYS_SEC = (15, 30)               # максимум 3 попытки на вызов
MAX_RETRY_AFTER_SEC = 60
DAILY_QUOTA_MARKERS = ("per day", "perday", "requests per day")
# Ошибки, после которых оставшиеся задачи ЭТОЙ модели не отправляются: модель недоступна,
# нет прав или нет billing. Повтор в таких случаях бесполезен.
MODEL_FATAL_STATUS_CODES = (403, 404)
# Неверный ключ: эксперимент останавливается целиком.
RUN_FATAL_STATUS_CODES = (401,)

PILOT_TASK_ID = "math_bat_ball"           # пилот: 1 задача × 3 модели = 3 вызова

# ---------- Тарифы ----------
# Standard (paid tier), USD за 1 000 000 токенов. На странице цен строка называется
# «Output price (including thinking tokens)»: thinking оплачивается по цене выходных токенов.
# Длина промпта не важна: у выбранных моделей нет порогов ≤200k / >200k.
# Источник: https://ai.google.dev/gemini-api/docs/pricing
#   (301 → https://aistudio.google.com/docs/pricing). Дата проверки: 2026-11-06
#   (все три модели перепроверены после замены средней модели на gemini-3.8-flash).
# У части моделей цена меняется с определённой даты, поэтому храним периоды
# [valid_from, valid_to] включительно (None — без ограничения). Цена выбирается по дате запроса.

PRICING_SOURCE_URL = "https://ai.google.dev/gemini-api/docs/pricing"
PRICING_SOURCE_URL_RESOLVED = "https://aistudio.google.com/docs/pricing"
MODELS_SOURCE_URL = "https://aistudio.google.com/docs/models"
THINKING_SOURCE_URL = "https://aistudio.google.com/docs/thinking"
PRICING_CHECKED_ON = "2026-11-06"
PRICING_TIER = "Standard (paid tier)"

PRICING = {
    "gemini-3.5-flash-lite": [
        {"valid_from": None, "valid_to": None,
         "input": 0.30, "output": 2.50, "cached_input": 0.03, "free_tier": True},
    ],
    "gemini-3.8-flash": [
        {"valid_from": None, "valid_to": "2026-12-31",
         "input": 0.75, "output": 3.75, "cached_input": 0.075, "free_tier": True},
        {"valid_from": "2027-01-01", "valid_to": None,
         "input": 1.50, "output": 7.50, "cached_input": 0.15, "free_tier": True},
    ],
    "gemini-4-argon": [
        {"valid_from": None, "valid_to": "2026-11-30",
         "input": 2.00, "output": 10.00, "cached_input": 0.10, "free_tier": False},
        {"valid_from": "2026-12-01", "valid_to": None,
         "input": 4.00, "output": 20.00, "cached_input": 0.20, "free_tier": False},
    ],
}
