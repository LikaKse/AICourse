"""CLI «День 11. Модель памяти ассистента».

Запуск (из папки gemini-task11):
    ../venv/bin/python main.py --dry-run                 # настройки, правила, данные, прогноз вызовов и цены (без API)
    ../venv/bin/python main.py --offline-experiment      # эксперимент на FakeClient → results/offline/ (без API)
    ../venv/bin/python main.py --experiment              # эксперимент на Gemini → results/gemini/ (ПЛАТНО, с подтверждением)
    ../venv/bin/python main.py --report offline          # пересобрать report.md из experiment.json (без API)
    ../venv/bin/python main.py --chat                    # чат с трёхслойной памятью в state/ (платно)
    ../venv/bin/python main.py --chat --offline          # тот же чат на FakeClient (без API)

Ключ: переменная окружения GEMINI_API_KEY или файл .env (python-dotenv). Ключ не печатается.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import config

EXIT_COMMANDS = {"exit", "quit", "выход", "/exit"}
CHAT_HELP = (
    "Команды: /memory — содержимое слоёв; /profile — настройки профиля; /prompt — что уйдёт в модель; /rules — правила;\n"
    "         /clear short|working|long|all|profile — очистить; /task done — завершить задачу; /exit"
)


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for name in ("httpx", "httpcore", "google_genai", "google.genai"):
        logging.getLogger(name).setLevel(logging.WARNING)


class ConfigError(Exception):
    pass


def load_api_key(dotenv_path: Path = config.DOTENV_FILE) -> str:
    """Ключ из окружения; если его нет — из .env. Ничего не печатает."""
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise ConfigError("Не установлен python-dotenv: ../venv/bin/pip install -r requirements.txt") from exc
    load_dotenv(dotenv_path, override=False)
    key = os.environ.get(config.API_KEY_ENV, "").strip()
    if not key:
        raise ConfigError(
            f"Не задан {config.API_KEY_ENV}. Скопируйте .env.example в .env и впишите ключ "
            f"(https://aistudio.google.com/apikey) или выполните export {config.API_KEY_ENV}=..."
        )
    return key


def make_client():
    api_key = load_api_key()
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=config.REQUEST_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=config.RETRY_ATTEMPTS,
                                                 http_status_codes=list(config.RETRY_STATUS_CODES)),
        ),
    )


def make_offline_client():
    import experiments
    from fake_client import FakeClient, echo_responder, scripted_extractor

    return FakeClient(responder=echo_responder(scripted_extractor(experiments.load_offline_extraction())))


def confirm(question: str, assume_yes: bool, input_fn=input) -> bool:
    if assume_yes:
        return True
    try:
        return input_fn(f"{question} [y/N]: ").strip().lower() in {"y", "yes", "д", "да"}
    except EOFError:
        return False


# ---------- Прогноз ----------

def forecast(window: int) -> dict:
    """Офлайн-прогон во временной папке: число вызовов и ВЕРХНЯЯ граница цены реального прогона.

    Вход оценивается сверху числом символов запроса и system instruction (токен Gemini ≥ 1 символа),
    выход — max_output_tokens каждого вызова (включая thinking).
    """
    import experiments
    from fake_client import all_text
    from token_stats import estimate_cost

    client = make_offline_client()
    with tempfile.TemporaryDirectory() as tmp:
        data = experiments.run_experiment(client, Path(tmp), window=window)
    reqs = client.requests
    input_chars = sum(len(all_text(r["contents"])) + len(str(r["config"].system_instruction or "")) for r in reqs)
    output_max = sum(r["config"].max_output_tokens or 0 for r in reqs)
    memory_calls = len(client.memory_requests())
    return {
        "generate_content": len(reqs), "memory_update_calls": memory_calls,
        "answer_calls": len(reqs) - memory_calls, "count_tokens_calls": len(client.models.count_requests),
        "input_tokens_upper_bound": input_chars, "output_tokens_upper_bound": output_max,
        "cost_upper_bound_usd": estimate_cost(input_chars, output_max),
        "offline_matrix_ok": all(v["summary"]["matches_expectation"] for v in data["variants"].values()),
    }


def print_forecast(fc: dict) -> None:
    print(f"Вызовы generate_content: {fc['generate_content']} (менеджер памяти: {fc['memory_update_calls']}, "
          f"ответы: {fc['answer_calls']}); count_tokens (бесплатно): {fc['count_tokens_calls']}")
    print(f"Верхняя граница: вход ≤ {fc['input_tokens_upper_bound']} ток., выход ≤ {fc['output_tokens_upper_bound']} ток. "
          f"→ ≤ ${fc['cost_upper_bound_usd']} по тарифу {config.MODEL}")


# ---------- Режимы ----------

def cmd_dry_run(window: int) -> int:
    import experiments
    from memory import LAYER_TITLES, READ_RULES, WRITE_RULES

    print(f"Модель: {config.MODEL}; окно краткосрочной памяти: {window}; цитаты в LT: {config.RENDER_EVIDENCE}")
    print(f"Файлы слоёв: {config.SHORT_TERM_FILE}, {config.WORKING_FILE}, {config.LONG_TERM_FILE} (в {config.STATE_DIR})")
    for title, rules in (("Правила записи", WRITE_RULES), ("Правила чтения", READ_RULES)):
        print(f"\n{title}:")
        for layer, items in rules.items():
            print(f"  {LAYER_TITLES[layer]}:")
            for item in items:
                print(f"    • {item}")
    pairs, questions, expected = (experiments.load_scenario(), experiments.load_questions(),
                                  experiments.load_expected())
    experiments.validate_data(pairs, questions, expected)
    print(f"\nДанные: {len(pairs)} реплик ({2 * len(pairs)} сообщений), {len(questions)} вопросов — эталон корректен.")
    print("\nПрогноз реального эксперимента (по офлайн-прогону на FakeClient):")
    fc = forecast(window)
    print_forecast(fc)
    print(f"Офлайн-матрица совпала с ожиданиями: {'да' if fc['offline_matrix_ok'] else 'НЕТ'}")
    print("\nПлатных запросов не было.")
    return 0


def run_and_save(client, results_dir: Path, window: int, mode: str, overwrite: bool) -> dict:
    import experiments

    data = experiments.run_experiment(client, Path(results_dir) / experiments.STATE_SUBDIR, window=window, mode=mode)
    for p in experiments.save_results(results_dir, data, overwrite=overwrite):
        print(f"Сохранено: {p}")
    for v in data["variants"].values():
        s = v["summary"]
        print(f"  {v['name']:<14} верно {s['correct']}/{s['total']}; потеряны: {s['lost_vs_full'] or '—'}; "
              f"промпт ≈ {s['tokens']['total_avg']} ток.; как ожидалось: {'да' if s['matches_expectation'] else 'НЕТ'}")
    return data


def cmd_offline_experiment(results_dir: Path, window: int, overwrite: bool) -> int:
    import experiments

    experiments.ensure_can_write(results_dir, overwrite)
    print("Офлайн-эксперимент на FakeClient (API не вызывается).")
    run_and_save(make_offline_client(), results_dir, window, "offline", overwrite)
    return 0


def cmd_experiment(results_dir: Path, window: int, overwrite: bool, assume_yes: bool, input_fn=input) -> int:
    import experiments

    experiments.ensure_can_write(results_dir, overwrite)
    fc = forecast(window)
    print("Эксперимент на реальном Gemini (ПЛАТНО).")
    print_forecast(fc)
    if not confirm("Запустить платные запросы?", assume_yes, input_fn):
        print("Отменено, запросов не было.")
        return 0
    run_and_save(make_client(), results_dir, window, "gemini", overwrite)
    return 0


def cmd_report(results_dir: Path) -> int:
    import experiments
    from utils import atomic_write_text

    path = Path(results_dir) / experiments.RESULT_JSON
    if not path.is_file():
        print(f"Нет {path}: сначала запустите эксперимент.")
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    report = experiments.markdown_report(data)
    atomic_write_text(Path(results_dir) / experiments.RESULT_REPORT, report)
    print(report)
    return 0


def _print_memory(memory) -> None:
    from memory import LAYER_TITLES

    for name, layer in memory.layers.items():
        print(f"== {LAYER_TITLES[name]} ({layer.path.name if layer.path else 'в памяти'}): "
              f"{json.dumps(layer.stats(), ensure_ascii=False)}")
        print(json.dumps(layer.dump(), ensure_ascii=False, indent=1))


def cmd_chat(state_dir: Path, window: int, client=None, input_fn=input) -> int:
    from agent import Agent
    from memory import LAYER_TITLES, READ_RULES, WRITE_RULES, MemoryLoadError, MemoryManager

    try:
        memory = MemoryManager(state_dir, window=window)
    except MemoryLoadError as exc:
        print(f"Не удалось загрузить память из {state_dir}: {exc}\nПереименуйте или удалите повреждённый файл слоя.")
        return 1
    agent = Agent(client or make_client(), memory)
    s = memory.stats()
    print(f"Чат ({config.MODEL}). Загружено: ST {s['short_term']['stored_messages']} сообщ., "
          f"WM {'задача #' + str(s['working']['task_id']) if s['working']['has_task'] else 'пусто'}, "
          f"LT {s['long_term']['facts']} фактов.")
    print(CHAT_HELP)
    while True:
        try:
            line = input_fn("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.lower() in EXIT_COMMANDS:
            return 0
        if line.startswith("/"):
            cmd, *args = line.split()
            if cmd == "/memory":
                _print_memory(memory)
            elif cmd == "/profile":
                agent.profile.load()
                print(agent.profile.profile.render() or "Профиль пока пуст.")
            elif cmd == "/prompt":
                agent.profile.load()
                profile_text = agent.profile.profile.render()
                print("\n".join(x for x in (profile_text, memory.context_text()) if x) or "(промпт пуст — только вопрос)")
            elif cmd == "/rules":
                for title, rules in (("Запись", WRITE_RULES), ("Чтение", READ_RULES)):
                    for layer, items in rules.items():
                        print(f"[{title}] {LAYER_TITLES[layer]}:\n  • " + "\n  • ".join(items))
            elif cmd == "/clear" and len(args) == 1:
                try:
                    if args[0] == "profile":
                        agent.profile.clear()
                        print("Очищен профиль пользователя.")
                        continue
                    cleared = memory.clear(args[0])
                except ValueError as exc:
                    print(exc)
                else:
                    print("Очищено: " + ", ".join(LAYER_TITLES[n] for n in cleared))
            elif cmd == "/task" and args == ["done"]:
                print("Задача завершена (в долговременную память ничего не перенесено)."
                      if memory.working.finish_task() else "Нет активной задачи.")
            else:
                print(CHAT_HELP)
            continue
        res = agent.ask(line)
        print(res.text)
        for c in res.service_calls:
            d = c.meta.get("decisions") or {}
            written = {k: len(d.get(k, {}).get("applied", [])) for k in ("working", "long_term")}
            rejected = sum(len(d.get(k, {}).get("rejected", [])) for k in ("working", "long_term")) + len(d.get("unrouted", []))
            status = f"WM +{written['working']}, LT +{written['long_term']}, отклонено {rejected}" if c.ok else f"ошибка: {c.error}"
            print(f"[память: {status}]")
        t = res.tokens
        print(f"[токены (count_tokens): LT {t['layers']['long_term']}, WM {t['layers']['working']}, "
              f"ST {t['layers']['short_term']}, вопрос {t['question']}, всего {t['total']}; "
              f"usage: вход {res.usage.input_tokens}, выход {res.usage.output_tokens}]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Агент Gemini с трёхслойной памятью (День 11)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="настройки, правила, прогноз вызовов (без API)")
    mode.add_argument("--offline-experiment", action="store_true", help="эксперимент на FakeClient (без API)")
    mode.add_argument("--experiment", action="store_true", help="эксперимент на Gemini (платно, с подтверждением)")
    mode.add_argument("--report", choices=("offline", "gemini"), help="пересобрать report.md (без API)")
    mode.add_argument("--chat", action="store_true", help="интерактивный чат (платно; с --offline — FakeClient)")
    parser.add_argument("--offline", action="store_true", help="для --chat: FakeClient вместо Gemini")
    parser.add_argument("--window", type=int, default=config.SHORT_TERM_WINDOW,
                        help=f"окно краткосрочной памяти (по умолчанию {config.SHORT_TERM_WINDOW})")
    parser.add_argument("--yes", action="store_true", help="не спрашивать подтверждение платных вызовов")
    parser.add_argument("--overwrite", action="store_true", help="разрешить перезапись результатов")
    parser.add_argument("--results-dir", type=Path, default=None, help="папка результатов")
    parser.add_argument("--state-dir", type=Path, default=config.STATE_DIR, help="папка памяти чата")
    parser.add_argument("--verbose", action="store_true", help="подробный лог")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    if args.window < 0:
        parser.error("--window должен быть ≥ 0")

    import experiments

    try:
        if args.dry_run:
            return cmd_dry_run(args.window)
        if args.offline_experiment:
            return cmd_offline_experiment(args.results_dir or config.RESULTS_DIR / "offline", args.window, args.overwrite)
        if args.experiment:
            return cmd_experiment(args.results_dir or config.RESULTS_DIR / "gemini", args.window, args.overwrite, args.yes)
        if args.report:
            return cmd_report(args.results_dir or config.RESULTS_DIR / args.report)
        return cmd_chat(args.state_dir, args.window, make_offline_client() if args.offline else None)
    except (ConfigError, experiments.ResultsExistError, experiments.DataError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # понятное сообщение вместо трейсбэка; ключ маскируется
        from utils import redact_secrets
        print(f"Ошибка: {redact_secrets(f'{type(exc).__name__}: {exc}')}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
