"""Эксперименты Дня 8 с токенами: короткий диалог, длинный диалог, переполнение контекста.

    ../venv/bin/python main.py --experiment short      # 3 коротких сообщения
    ../venv/bin/python main.py --experiment long       # 12 сообщений — рост истории
    ../venv/bin/python main.py --experiment overflow   # вход больше лимита контекста модели

Каждый эксперимент работает с ОТДЕЛЬНЫМ агентом в памяти: файл истории чата
(results/chat_history.json) не читается и не изменяется. Перед платными вызовами
печатается оценка и запрашивается подтверждение (--yes — без вопроса).
Протокол сохраняется в results/tokens_<эксперимент>_<дата>.json (без API-ключа и без
многомегабайтного текста переполнения — только его размеры).
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import config
import token_stats
from agent import Agent, AgentResult, redact_secrets
from google.genai import types

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"

SHORT_DIALOG = [
    "Привет! Меня зовут Аня.",
    "Сколько будет 2 + 2?",
    "Как меня зовут?",
]

LONG_DIALOG = [
    "Привет! Меня зовут Аня, я изучаю разработку AI-агентов.",
    "Расскажи подробно, что такое токены в языковых моделях.",
    "Чем контекстное окно отличается от лимита выходных токенов?",
    "Сколько будет 150 долларов в донгах?",
    "Добавь к этой сумме 10 процентов.",
    "Который сейчас час?",
    "Объясни, почему стоимость запроса растёт с длиной диалога.",
    "Приведи 5 способов сократить расход токенов в агенте.",
    "Сколько будет 1234 * 5678?",
    "Расскажи, что такое function calling, с коротким примером.",
    "Сделай краткое резюме нашего диалога.",
    "Как меня зовут и какую сумму в донгах ты получил в начале?",
]

# Наполнитель для переполнения. Номер строки делает строки разными. Сколько это токенов,
# измеряет count_tokens; длина в символах нужна только чтобы собрать текст нужного размера.
FILLER_LINE = "Строка {i}: это тестовый текст для проверки переполнения контекстного окна модели.\n"
FILLER_SAMPLE_LINES = 2_000
OVERFLOW_MARGIN = 0.03          # целимся на 3 % выше лимита


class _CountOnlyClient:
    """Обёртка клиента для шага «защита агента»: count_tokens разрешён, generate_content — нет.

    Гарантирует, что этот шаг бесплатный, даже если count_tokens вдруг не сработает
    и локальная проверка лимита не сможет остановить запрос.
    """

    class _Models:
        def __init__(self, models):
            self._models = models

        def count_tokens(self, **kwargs):
            return self._models.count_tokens(**kwargs)

        def generate_content(self, **kwargs):
            raise RuntimeError("generate_content заблокирован на шаге проверки защиты агента")

    def __init__(self, client):
        self.models = self._Models(client.models)


def _user(text: str) -> list[types.Content]:
    return [types.Content(role="user", parts=[types.Part.from_text(text=text)])]


def build_filler(lines: int) -> str:
    return "".join(FILLER_LINE.format(i=i) for i in range(lines))


def build_overflow_text(agent: Agent, limit: int, max_rounds: int = 4) -> tuple[str | None, int | None, list[dict]]:
    """Собирает текст, который по count_tokens больше limit. Возвращает (текст, токены, шаги)."""
    steps: list[dict] = []
    sample_tokens, err = agent.count_tokens(_user(build_filler(FILLER_SAMPLE_LINES)))
    steps.append({"step": "sample", "lines": FILLER_SAMPLE_LINES, "tokens": sample_tokens, "error": err})
    if not sample_tokens:
        return None, None, steps
    target = math.ceil(limit * (1 + OVERFLOW_MARGIN))
    lines = math.ceil(target / (sample_tokens / FILLER_SAMPLE_LINES))
    for _ in range(max_rounds):
        text = build_filler(lines)
        tokens, err = agent.count_tokens(_user(text))
        steps.append({"step": "full", "lines": lines, "chars": len(text), "tokens": tokens, "error": err})
        if tokens is None:
            return text, None, steps
        if tokens > limit:
            return text, tokens, steps
        lines = math.ceil(lines * target / tokens)
    return None, None, steps


def _turn_record(index: int, message: str, result: AgentResult) -> dict:
    return {
        "turn": index,
        "message": message if len(message) <= 300 else f"<{len(message)} символов>",
        "status": result.status,
        "answer": result.text,
        "tool_calls": [{"iteration": c.iteration, "name": c.name, "args": c.args} for c in result.tool_calls],
        "token_stats": result.stats.to_dict(),
    }


def _save(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"tokens_{name}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(redact_secrets(json.dumps(payload, ensure_ascii=False, indent=2, default=str)), encoding="utf-8")
    return out


def _print_growth(records: list[dict]) -> None:
    print("\nРост истории по ходам:")
    print(f"{'ход':>3} | {'запрос':>6} | {'история':>8} | {'вход за ход':>11} | {'выход':>6} | {'обращ.':>6} | статус")
    for r in records:
        st = r["token_stats"]
        def n(v):
            return "н/д" if v is None else str(v)
        print(f"{r['turn']:>3} | {n(st['request_tokens']):>6} | {n(st['history_tokens']):>8} | "
              f"{st['input_tokens']:>11} | {st['output_tokens']:>6} | {len(st['calls']):>6} | {r['status']}")


# ---------------------------------------------------------------------
# Диалоги (короткий / длинный)
# ---------------------------------------------------------------------

def run_dialog(name: str, messages: list[str], assume_yes: bool, *, client=None, confirm=None) -> int:
    import main  # отложенный импорт: main импортирует этот модуль только по флагу

    confirm = confirm or main.confirm
    worst = len(messages) * 3 * (20_000 * config.PRICE_INPUT_PER_1M + config.MAX_OUTPUT_TOKENS * config.PRICE_OUTPUT_PER_1M) / 1e6
    print(f"Эксперимент «{name}»: {len(messages)} сообщений к {config.MODEL}, отдельная история в памяти "
          "(файл чата не трогается).")
    print(f"Примерно {len(messages)}–{len(messages) * 2} вызовов generate_content + бесплатные count_tokens; "
          f"оценка сверху ≈ ${worst:.3f}, реально — центы или меньше.")
    print(token_stats.pricing_note())
    if not confirm("Запустить платные вызовы Gemini API?", assume_yes):
        print("Отменено.")
        return 1

    client = client if client is not None else main.make_client()
    limit, limit_error = main.fetch_input_token_limit(client)
    print(main.describe_limit(limit, limit_error))
    agent = Agent(client, input_token_limit=limit)
    records = []
    for i, message in enumerate(messages, 1):
        print(f"\n=== Ход {i}/{len(messages)} ===\nПользователь: {message}")
        result = agent.run(message)
        print(f"Агент: {result.text}\n")
        print(token_stats.format_turn(result.stats, agent.session_stats))
        records.append(_turn_record(i, message, result))

    _print_growth(records)
    out = _save(name, {
        "experiment": name, "model": config.MODEL, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_token_limit": limit, "input_token_limit_error": limit_error,
        "pricing": token_stats.model_pricing(), "turns": records,
        "session": {**vars(agent.session_stats), "cost_usd": agent.session_stats.cost_usd},
        "final_history_messages": len(agent.history),
        "final_history_tokens": agent.count_history_tokens(),
    })
    print(f"\n{token_stats.format_session(agent.session_stats)}\nПротокол: {out.relative_to(ROOT)}")
    return 0 if all(r["status"] == "ok" for r in records) else 1


# ---------------------------------------------------------------------
# Переполнение контекста
# ---------------------------------------------------------------------

def run_overflow(assume_yes: bool, *, client=None, confirm=None) -> int:
    import main

    confirm = confirm or main.confirm
    # 1. Лимит — только из метаданных модели (бесплатно). Не угадываем.
    client = client if client is not None else main.make_client(retry_attempts=1)
    limit, limit_error = main.fetch_input_token_limit(client)
    print(main.describe_limit(limit, limit_error))
    report: dict = {"experiment": "overflow", "model": config.MODEL,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "input_token_limit": limit, "input_token_limit_error": limit_error}
    if not limit:
        print("Точный лимит контекста неизвестен — эксперимент остановлен: без лимита нельзя "
              "собрать вход, который заведомо его превышает. Проверьте: main.py --check-model.")
        report["result"] = "limit_unknown"
        print(f"Протокол: {_save('overflow', report).relative_to(ROOT)}")
        return 1

    # 2. Собираем текст больше лимита, размер подтверждаем через count_tokens (бесплатно).
    measure = Agent(client, input_token_limit=limit)
    print(f"Собираю текст > {limit} токенов (размер измеряется через count_tokens)…")
    text, tokens, steps = build_overflow_text(measure, limit)
    report["build_steps"] = steps
    if text is None or tokens is None:
        print("Не удалось подтвердить размер через count_tokens: " + json.dumps(steps, ensure_ascii=False))
        report["result"] = "count_tokens_failed"
        print(f"Протокол: {_save('overflow', report).relative_to(ROOT)}")
        return 1
    report["overflow_message"] = {"chars": len(text), "bytes_utf8": len(text.encode()), "count_tokens": tokens}
    print(f"Текст: {len(text):,} символов, {len(text.encode()) / 1e6:.1f} МБ, "
          f"count_tokens = {tokens:,} > лимит {limit:,}.".replace(",", " "))

    # 3. Защита агента: запрос не уходит в модель (0 вызовов generate_content, бесплатно).
    guarded = Agent(_CountOnlyClient(client), input_token_limit=limit, enforce_context_limit=True)
    guarded_result = guarded.run(text)
    print(f"\n[Защита агента] статус: {guarded_result.status}\n{guarded_result.text}")
    report["guarded"] = {"status": guarded_result.status, "text": guarded_result.text,
                         "generate_content_calls": guarded_result.stats.model_calls,
                         "history_after": len(guarded.history)}

    # 4. Реальное поведение API: защиту выключаем и отправляем запрос.
    print("\nСледующий шаг отправит этот текст в generate_content без локальной защиты, чтобы "
          "зафиксировать реальный ответ API. Ожидается отказ (HTTP 400); отклонённый запрос не должен "
          f"тарифицироваться. Худший случай, если API всё же примет: ≈ ${token_stats.estimate_cost(tokens, 16) or 0:.2f}.")
    if not confirm("Отправить запрос, превышающий лимит, в Gemini API?", assume_yes):
        report["result"] = "guard_only"
        print(f"Отменено. Протокол: {_save('overflow', report).relative_to(ROOT)}")
        return 0
    raw = Agent(client, input_token_limit=limit, enforce_context_limit=False, max_output_tokens=16)
    raw_result = raw.run(text)
    api_error = raw_result.stats.calls[-1].error if raw_result.stats.calls else None
    print(f"\n[Реальный API] статус агента: {raw_result.status}")
    print(f"Ответ API: {api_error or raw_result.text}")
    report["api"] = {"status": raw_result.status, "error": api_error, "text": raw_result.text,
                     "token_stats": raw_result.stats.to_dict(), "history_after": len(raw.history)}

    # 5. После ошибки агент продолжает работать: история откатилась, следующий запрос — обычный.
    raw.max_output_tokens = config.MAX_OUTPUT_TOKENS
    after = raw.run("Ответь одним словом: работаешь?")
    print(f"\n[После ошибки] история перед запросом была пустой: {report['api']['history_after'] == 0}; "
          f"статус: {after.status}; ответ: {after.text}")
    report["after_error"] = {"status": after.status, "text": after.text, "token_stats": after.stats.to_dict()}
    report["result"] = raw_result.status
    print(f"\nПротокол: {_save('overflow', report).relative_to(ROOT)}")
    return 0


def run(name: str, assume_yes: bool) -> int:
    if name == "short":
        return run_dialog("short", SHORT_DIALOG, assume_yes)
    if name == "long":
        return run_dialog("long", LONG_DIALOG, assume_yes)
    if name == "overflow":
        return run_overflow(assume_yes)
    raise ValueError(f"неизвестный эксперимент: {name}")
