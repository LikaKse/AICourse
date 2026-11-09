"""CLI-чат с агентом на Gemini (День 6. Первый агент).

Запуск (из папки gemini-task6):
    ../venv/bin/python main.py                  # чат (платные вызовы на каждый запрос)
    ../venv/bin/python main.py --dry-run        # настройки и схемы инструментов, без API
    ../venv/bin/python main.py --check-model    # метаданные модели (бесплатно, без генерации)
    ../venv/bin/python main.py --scenarios      # 3 тестовых сценария (платно, с подтверждением)
    ../venv/bin/python main.py --scenarios 3    # только сценарий 3

Флаги: --verbose (подробный лог), --log-file (дублировать лог в logs/), --yes (без подтверждения).
Ключ берётся из переменной окружения GEMINI_API_KEY и нигде не печатается.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import config
import tools
from agent import Agent, AgentResult, redact_secrets

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
EXIT_COMMANDS = {"exit", "quit", "выход"}

# ---------- Тестовые сценарии ----------
# expected_tools — ожидаемая ПОСЛЕДОВАТЕЛЬНОСТЬ вызовов. Агенту она не передаётся:
# по ней скрипт только проверяет, что выбрала модель.
SCENARIOS = [
    {
        "id": 1,
        "title": "Без инструмента",
        "prompt": "Объясни простыми словами, что такое API",
        "expected_tools": [],
        "expected_number": None,
    },
    {
        "id": 2,
        "title": "Один инструмент",
        "prompt": "Сколько будет 150 долларов в донгах?",
        "expected_tools": ["convert_currency"],
        "expected_number": 3_750_000,
    },
    {
        "id": 3,
        "title": "Два инструмента последовательно",
        "prompt": (
            "Переведи 150 долларов в донги, а затем рассчитай, сколько получится, "
            "если добавить к этой сумме ещё 10 процентов"
        ),
        "expected_tools": ["convert_currency", "calculate"],
        "expected_number": 4_125_000,
    },
]


# ---------- Логирование ----------

class RedactSecretsFilter(logging.Filter):
    """Последний рубеж: вырезает значение API-ключа из любой строки лога."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        cleaned = redact_secrets(message)
        if cleaned != message:
            record.msg, record.args = cleaned, None
        return True


def setup_logging(verbose: bool, log_file: bool) -> Path | None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s [%(name)s] %(message)s", "%H:%M:%S")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    log_path = None
    if log_file:
        log_dir = ROOT / config.LOG_DIR
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"agent_{datetime.now():%Y%m%d_%H%M%S}.log"
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    for h in handlers:
        h.setFormatter(fmt)
        h.addFilter(RedactSecretsFilter())
    logging.basicConfig(level=logging.WARNING, handlers=handlers, force=True)
    logging.getLogger("agent").setLevel(logging.DEBUG if verbose else logging.INFO)
    # Сетевые библиотеки — только предупреждения.
    for name in ("httpx", "httpcore", "google_genai", "google.genai"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return log_path


# ---------- Клиент ----------

def make_client():
    """Создаёт клиента Gemini. Ключ читается из окружения и не выводится."""
    api_key = os.environ.get(config.API_KEY_ENV)
    if not api_key:
        sys.exit(f"Не задана переменная окружения {config.API_KEY_ENV}. "
                 f"Пример: export {config.API_KEY_ENV}=...  (ключ не сохраняйте в файлы проекта)")
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=config.REQUEST_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(
                attempts=config.RETRY_ATTEMPTS,
                http_status_codes=list(config.RETRY_STATUS_CODES),
            ),
        ),
    )


def confirm(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        return input(f"{question} [y/N]: ").strip().lower() in {"y", "yes", "д", "да"}
    except EOFError:
        return False


# ---------- Режимы ----------

def cmd_dry_run() -> None:
    agent = Agent(client=None)
    cfg = agent.build_config()
    print(f"Модель: {config.MODEL}")
    print(f"Макс. итераций: {config.MAX_ITERATIONS}; thinking_level: {config.THINKING_LEVEL}; "
          f"max_output_tokens: {config.MAX_OUTPUT_TOKENS}")
    print(f"Автоматическое выполнение функций SDK отключено: {cfg.automatic_function_calling.disable}")
    print("\nСистемная инструкция:\n" + config.SYSTEM_INSTRUCTION)
    print("\nСхемы инструментов (FunctionDeclaration):")
    for d in tools.FUNCTION_DECLARATIONS:
        print(json.dumps(d.model_dump(exclude_none=True), ensure_ascii=False, indent=2))
    print("\nТестовые сценарии:")
    for s in SCENARIOS:
        print(f"  {s['id']}. {s['title']}: «{s['prompt']}» → ожидаются {s['expected_tools'] or 'без инструментов'}")


def cmd_check_model() -> None:
    client = make_client()
    m = client.models.get(model=config.MODEL)
    print(f"{m.name}: version={m.version}, display_name={m.display_name}")
    print(f"supported_actions={m.supported_actions}")
    print(f"input_token_limit={m.input_token_limit}, output_token_limit={m.output_token_limit}")


def _digits(text: str) -> str:
    """'4 125 000 VND' / '4,125,000' / '4.125.000' → '4125000' внутри текста."""
    return re.sub(r"(?<=\d)[\s  .,](?=\d{3}\b)", "", text)


def check_scenario(scenario: dict, result: AgentResult) -> list[str]:
    """Автоматическая проверка: какие инструменты выбрала модель и есть ли нужное число в ответе."""
    problems = []
    if not result.ok:
        problems.append(f"статус {result.status}")
    names = [c.name for c in result.tool_calls if c.ok]
    if names != scenario["expected_tools"]:
        problems.append(f"инструменты {names}, ожидались {scenario['expected_tools']}")
    if scenario["id"] == 3 and len(result.tool_calls) >= 2:
        conv, calc = result.tool_calls[0], result.tool_calls[1]
        if calc.iteration <= conv.iteration:
            problems.append("calculate вызван в том же ходе, что convert_currency (не последовательно)")
        converted = conv.response.get("result", {}).get("converted_amount")
        if converted is not None and str(converted) not in _digits(calc.args.get("expression", "")):
            problems.append(f"calculate не использует результат конвертации {converted}")
    if scenario["expected_number"] is not None and str(scenario["expected_number"]) not in _digits(result.text):
        problems.append(f"в ответе нет числа {scenario['expected_number']}")
    if "convert_currency" in names and not re.search(r"фиксир|демонстрац", result.text, re.I):
        problems.append("в ответе не сказано, что курс фиксированный")
    return problems


def cmd_scenarios(ids: list[int], assume_yes: bool) -> int:
    selected = [s for s in SCENARIOS if not ids or s["id"] in ids]
    max_calls = len(selected) * config.MAX_ITERATIONS
    worst = max_calls * (3000 * config.PRICE_INPUT_PER_1M + config.MAX_OUTPUT_TOKENS * config.PRICE_OUTPUT_PER_1M) / 1e6
    print(f"Будут запущены сценарии {[s['id'] for s in selected]} на {config.MODEL}.")
    print(f"Ожидается ~{sum(len(s['expected_tools']) + 1 for s in selected)} вызовов generate_content "
          f"(максимум {max_calls}); оценка сверху ≈ ${worst:.3f}, реально — доли цента.")
    if not confirm("Запустить платные вызовы Gemini API?", assume_yes):
        print("Отменено.")
        return 1

    client = make_client()
    records, failed = [], 0
    for s in selected:
        print(f"\n=== Сценарий {s['id']}. {s['title']} ===\nПользователь: {s['prompt']}")
        agent = Agent(client)  # отдельный агент: сценарии не влияют друг на друга
        result = agent.run(s["prompt"])
        print(f"Агент: {result.text}")
        problems = check_scenario(s, result)
        verdict = "PASS" if not problems else "FAIL: " + "; ".join(problems)
        failed += bool(problems)
        print(f"[{verdict}] итераций: {result.iterations}, токены in/out: {result.input_tokens}/{result.output_tokens}")
        records.append({
            "id": s["id"], "title": s["title"], "prompt": s["prompt"],
            "expected_tools": s["expected_tools"], "status": result.status, "verdict": verdict,
            "iterations": result.iterations,
            "tool_calls": [
                {"iteration": c.iteration, "name": c.name, "args": c.args, "response": c.response}
                for c in result.tool_calls
            ],
            "answer": result.text,
            "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
        })

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"scenarios_{datetime.now():%Y%m%d_%H%M%S}.json"
    payload = {"model": config.MODEL, "timestamp": datetime.now().isoformat(timespec="seconds"),
               "scenarios": records}
    out.write_text(redact_secrets(json.dumps(payload, ensure_ascii=False, indent=2)), encoding="utf-8")
    print(f"\nИтог: {len(selected) - failed}/{len(selected)} PASS. Результаты: {out.relative_to(ROOT)}")
    return 0 if not failed else 1


def cmd_chat() -> None:
    agent = Agent(make_client())
    print(f"Привет! Я агент на {config.MODEL}. Умею: узнавать текущее время ({config.TIMEZONE}), "
          "считать и переводить USD ↔ VND (демо-курс).")
    print("Команды: exit — выход, reset — очистить историю.\n")
    while True:
        try:
            user_message = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            return
        if not user_message:
            continue
        if user_message.lower() in EXIT_COMMANDS:
            print("До свидания!")
            return
        if user_message.lower() == "reset":
            agent.reset()
            print("История очищена.\n")
            continue
        try:
            result = agent.run(user_message)
        except KeyboardInterrupt:
            print("\n(запрос прерван)\n")
            continue
        print(f"\nАгент: {result.text}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CLI-чат с агентом на Gemini (ручной Function Calling)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="показать настройки и схемы, без API")
    mode.add_argument("--check-model", action="store_true", help="метаданные модели (бесплатно)")
    mode.add_argument("--scenarios", nargs="*", type=int, metavar="N", help="тестовые сценарии (платно)")
    parser.add_argument("--yes", action="store_true", help="не спрашивать подтверждение платных вызовов")
    parser.add_argument("--verbose", action="store_true", help="подробный лог (DEBUG)")
    parser.add_argument("--log-file", action="store_true", help="дублировать лог в logs/")
    args = parser.parse_args(argv)

    log_path = setup_logging(args.verbose, args.log_file)
    if log_path:
        print(f"Лог пишется в {log_path.relative_to(ROOT)}", file=sys.stderr)

    if args.dry_run:
        cmd_dry_run()
        return 0
    if args.check_model:
        cmd_check_model()
        return 0
    if args.scenarios is not None:
        return cmd_scenarios(args.scenarios, args.yes)
    cmd_chat()
    return 0


if __name__ == "__main__":
    sys.exit(main())
