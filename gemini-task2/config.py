"""Общие настройки эксперимента: модель, промпт, схема ответа и три варианта запроса."""

from pydantic import BaseModel, ConfigDict, Field

# Та же модель, что и в задаче №1
MODEL = "gemini-3.8-flash"

# Сколько раз запускать каждый вариант
RUNS_PER_VARIANT = 3

# Пауза между запусками (секунды). Бесплатный тариф: 5 запросов в минуту (третий прогон
# получил 429 «requests per minute, limit: 5»), поэтому между началами запросов >= 13 с.
PAUSE_BETWEEN_REQUESTS = 13

# Таймаут одного запроса (секунды): без него зависший запрос блокирует весь эксперимент
REQUEST_TIMEOUT_SEC = 120

# Базовый запрос — одинаковый для всех трёх вариантов
BASE_PROMPT = (
    "Порекомендуй 3 книги по программированию для начинающих. "
    "Для каждой укажи название, автора, год издания и коротко объясни, "
    "почему её стоит прочитать."
)

# Стоп-последовательность для варианта 2
STOP_MARKER = "<<END>>"

# Лимит выходных токенов для варианта 2
MAX_OUTPUT_TOKENS = 400

# Текстовое описание формата для варианта 2 (дописывается к базовому запросу)
FORMAT_INSTRUCTIONS = f"""

Формат ответа — строго JSON-объект без пояснений до или после него и без Markdown-разметки:
{{
  "books": [
    {{"title": "строка", "author": "строка", "year": целое число, "reason": "строка"}}
  ]
}}
В массиве "books" должно быть ровно 3 элемента.
Сразу после закрывающей фигурной скобки напиши {STOP_MARKER}"""


# ---------- Повторные попытки ----------

# Коды ошибок API, при которых запрос повторяется (перегрузка / таймаут на стороне сервера)
RETRY_STATUS_CODES = (503, 504)

# Паузы перед повторами (секунды): один повтор через 13 с, максимум 1 + 1 = 2 попытки на запуск.
# Бесплатный тариф: 20 запросов в сутки; 9 запусков × 2 попытки = максимум 18 запросов.
# 13 с — та же причина, что и у PAUSE_BETWEEN_REQUESTS: не больше 5 запросов в минуту.
RETRY_DELAYS_SEC = (13,)

# Коды ошибок API, при которых весь эксперимент немедленно останавливается
# (429 — исчерпана квота: дальнейшие запросы бессмысленны и только тратят лимит)
STOP_STATUS_CODES = (429,)


# ---------- Pydantic-схемы ----------
# 1) API-схема (ApiBook / ApiBookRecommendations) — передаётся в response_schema варианта 3.
#    Без extra="forbid": Gemini API не принимает поле additionalProperties
#    (первый прогон: 400 Unknown name "additional_properties").
# 2) Строгая схема (StrictBook / StrictBookRecommendations) — ей проверяются ответы
#    ВСЕХ трёх вариантов. Те же поля и ограничения + запрет лишних полей.

class ApiBook(BaseModel):
    """Одна рекомендованная книга (схема для API)."""

    title: str = Field(min_length=1, description="Название книги")
    author: str = Field(min_length=1, description="Автор или авторы")
    year: int = Field(description="Год издания")
    reason: str = Field(min_length=1, description="Почему стоит прочитать")


class ApiBookRecommendations(BaseModel):
    """Ответ модели: ровно 3 книги (схема для API)."""

    books: list[ApiBook] = Field(min_length=3, max_length=3, description="Ровно 3 книги")


class StrictBook(ApiBook):
    """Строгая проверка книги: те же поля и ограничения, лишние поля запрещены."""

    model_config = ConfigDict(extra="forbid")


class StrictBookRecommendations(ApiBookRecommendations):
    """Строгая проверка ответа: ровно 3 строгие книги, лишние поля запрещены."""

    model_config = ConfigDict(extra="forbid")

    books: list[StrictBook] = Field(min_length=3, max_length=3, description="Ровно 3 книги")


# ---------- Три варианта запроса ----------
# prompt — текст запроса, config — параметры для GenerateContentConfig (None = без конфига)

VARIANTS = [
    {
        "id": "v1_unconstrained",
        "name": "Без ограничений",
        "prompt": BASE_PROMPT,
        "config": None,
    },
    {
        "id": "v2_format_and_limits",
        "name": "Описание формата + max_output_tokens + stop_sequences",
        "prompt": BASE_PROMPT + FORMAT_INSTRUCTIONS,
        "config": {
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "stop_sequences": [STOP_MARKER],
        },
    },
    {
        "id": "v3_structured_output",
        "name": "Structured Output (JSON Schema)",
        "prompt": BASE_PROMPT,
        "config": {
            "response_mime_type": "application/json",
            "response_schema": ApiBookRecommendations,
        },
    },
]
