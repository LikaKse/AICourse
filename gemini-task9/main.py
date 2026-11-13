"""CLI «День 9. Context Compression».

Запуск (из папки gemini-task9):
    ../venv/bin/python main.py --dry-run              # настройки и данные, без API
    ../venv/bin/python main.py --chat                 # чат со сжатием (платно; состояние в state/)
    ../venv/bin/python main.py --experiment all       # эксперименты A и B + сравнение (платно, с подтверждением)
    ../venv/bin/python main.py --experiment A         # только A (без сжатия)
    ../venv/bin/python main.py --experiment B         # только B (со сжатием)
    ../venv/bin/python main.py --compare              # comparison.json и summary.json из сохранённых A и B (без API)
    ../venv/bin/python main.py --report               # Markdown-таблицы для README из сохранённых JSON (без API)
    ../venv/bin/python main.py --audit                # офлайн-аудит summary: что сохранилось/потерялось (без API)

Ключ: переменная окружения GEMINI_API_KEY или файл .env (python-dotenv). Ключ не печатается.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import config

EXIT_COMMANDS = {"exit", "quit", "выход"}


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
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


def confirm(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        return input(f"{question} [y/N]: ").strip().lower() in {"y", "yes", "д", "да"}
    except EOFError:
        return False


# ---------- Режимы ----------

def cmd_dry_run() -> int:
    import experiments

    dialogue = experiments.load_dialogue()
    questions = experiments.load_questions()
    facts = experiments.load_facts()
    experiments.validate_data(dialogue, questions, facts)
    import compression_audit
    catalog = compression_audit.load_catalog()
    compression_audit.validate_catalog(catalog, dialogue)
    print(f"Модель: {config.MODEL}")
    print(f"SUMMARY_EVERY={config.SUMMARY_EVERY}, KEEP_LAST={config.KEEP_LAST} "
          f"(сжатие при {config.SUMMARY_EVERY + config.KEEP_LAST} активных сообщениях)")
    print(f"Thinking: ответы {config.ANSWER_THINKING_LEVEL}, summary {config.SUMMARY_THINKING_LEVEL}")
    print(f"Сценарий: {len(dialogue)} сообщений; вопросов: {len(questions)}; фактов в эталоне: {len(facts)}")
    n_summaries = sum(1 for total in range(2, len(dialogue) + 1, 2)
                      if total >= config.KEEP_LAST + config.SUMMARY_EVERY
                      and (total - config.KEEP_LAST - config.SUMMARY_EVERY) % config.SUMMARY_EVERY == 0)
    print(f"Ожидаемые платные вызовы: A — {len(questions)}, B — {n_summaries} (summary) + {len(questions)}")
    print(f"Каталог аудита: {len(catalog['details'])} деталей, {len(catalog.get('attributions', []))} проверок авторства")
    print("Данные корректны; каждый эталонный ответ и каждая деталь подтверждены цитатой из диалога. "
          "API не вызывался.")
    return 0


def cmd_experiment(which: str, assume_yes: bool, results_dir: Path = config.RESULTS_DIR) -> int:
    import experiments

    names = ["A", "B"] if which == "all" else [which]
    if not confirm(f"Запустить эксперимент(ы) {', '.join(names)}? Это платные вызовы {config.MODEL}.", assume_yes):
        print("Отменено.")
        return 1
    client = make_client()
    for name in names:
        state_dir = results_dir / f"state_{name}"
        print(f"\n=== Эксперимент {name} ===")
        data = experiments.run_experiment(client, compression=(name == "B"), state_dir=state_dir)
        path = experiments.save_results(results_dir, name, data)
        print(f"Факты: {data['facts_correct']}/{data['facts_total']}; токены (usage_metadata): "
              f"{data['tokens']['total']['total_tokens']}; файл: {path}")
    return cmd_compare(results_dir)


def cmd_compare(results_dir: Path = config.RESULTS_DIR) -> int:
    import experiments

    out = experiments.compare_saved(results_dir)
    if out is None:
        print("Для сравнения нужны оба файла: результаты экспериментов A и B.")
        return 0
    _, summary = out
    print("\nСравнение сохранено: comparison.json, summary.json")
    print(f"Факты: A {summary['facts_retained_percent']['A_without_compression']}%, "
          f"B {summary['facts_retained_percent']['B_with_compression']}%")
    print(f"Экономия на контрольных запросах (вход): {summary['savings_control_requests_input_percent']}%")
    print(f"Общая экономия с учётом суммаризации: {summary['savings_overall_with_summarization_percent']}%")
    return 0


def cmd_report(results_dir: Path = config.RESULTS_DIR) -> int:
    import experiments

    a, b = experiments.load_result(results_dir, "A"), experiments.load_result(results_dir, "B")
    comparison = experiments.load_result(results_dir, "comparison")
    if not (a and b and comparison):
        print("Нет сохранённых результатов. Сначала: python main.py --experiment all")
        return 1
    print(experiments.markdown_report(a, b, comparison))
    return 0


def cmd_audit(results_dir: Path = config.RESULTS_DIR, output: Path | None = None) -> int:
    """Офлайн-аудит summary из сохранённого прогона. Только чтение results/; API не вызывается."""
    import compression_audit
    import experiments

    try:
        audit = compression_audit.audit_saved_run(results_dir, experiments.load_dialogue())
    except compression_audit.AuditError as exc:
        print(f"Аудит невозможен: {exc}")
        return 1
    report = compression_audit.markdown_report(audit)
    if output:
        if Path(output).resolve().is_relative_to(Path(results_dir).resolve()):
            print("Отчёт аудита не записывается в папку результатов, чтобы не смешивать его с данными прогона.")
            return 1
        from utils import atomic_write_text
        atomic_write_text(output, report + "\n")
        print(f"Отчёт сохранён: {output}")
    else:
        print(report)
    return 0


def cmd_chat(state_dir: Path = config.STATE_DIR, client=None, input_fn=input) -> int:
    from agent import Agent
    from context_manager import ContextManager, StateError
    from history_store import HistoryLoadError

    try:
        context = ContextManager.from_dir(state_dir)
    except (HistoryLoadError, StateError) as exc:
        print(f"Не удалось загрузить состояние из {state_dir}: {exc}\n"
              "Переименуйте/удалите повреждённый файл или папку состояния и запустите снова.")
        return 1
    agent = Agent(client or make_client(), context)
    print(f"Чат со сжатием контекста ({config.MODEL}). Восстановлено: {len(context.full_history)} сообщ., "
          f"в summary {context.summary.summarized_upto}, активных {len(context.active)}.")
    print("Команды: summary, stats, reset, exit.")
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
        if line == "summary":
            print(context.summary.text or "(summary пока нет)")
            continue
        if line == "stats":
            print(f"Полная история: {len(context.full_history)}, сжато: {context.summary.summarized_upto} "
                  f"(версия summary {context.summary.version}), активных: {len(context.active)}")
            continue
        if line == "reset":
            context.reset()
            print("История и summary очищены.")
            continue
        res = agent.ask(line)
        print(res.text)
        for ev in res.compression:
            print(f"[сжатие сообщений {ev.start}–{ev.end - 1}: {'ок, summary v' + str(ev.version) if ev.ok else 'ошибка: ' + str(ev.error)}]")
        u = res.usage
        print(f"[токены: запрос {res.request_tokens} (count_tokens), вход {u.input_tokens}, выход {u.output_tokens}"
              f"{'' if u.present else ' — usage_metadata нет'}]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Агент Gemini со сжатием истории диалога (День 9)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="настройки и проверка данных, без API")
    mode.add_argument("--chat", action="store_true", help="интерактивный чат со сжатием (платно)")
    mode.add_argument("--experiment", choices=("A", "B", "all"), help="эксперименты (платно)")
    mode.add_argument("--compare", action="store_true", help="сравнение сохранённых A и B (без API)")
    mode.add_argument("--report", action="store_true", help="Markdown-отчёт по сохранённым JSON (без API)")
    mode.add_argument("--audit", action="store_true", help="офлайн-аудит summary сохранённого прогона (без API)")
    parser.add_argument("--yes", action="store_true", help="не спрашивать подтверждение платных вызовов")
    parser.add_argument("--verbose", action="store_true", help="подробный лог")
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR, help="папка результатов (для --audit)")
    parser.add_argument("--audit-output", type=Path, metavar="PATH", help="сохранить отчёт аудита в файл (вне results/)")
    parser.add_argument("--state-dir", type=Path, default=config.STATE_DIR, help="папка состояния чата")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    try:
        if args.dry_run:
            return cmd_dry_run()
        if args.experiment:
            return cmd_experiment(args.experiment, args.yes)
        if args.compare:
            return cmd_compare()
        if args.report:
            return cmd_report()
        if args.audit:
            return cmd_audit(args.results_dir, args.audit_output)
        return cmd_chat(args.state_dir)
    except ConfigError as exc:
        print(f"Ошибка настройки: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # понятное сообщение вместо трейсбэка; ключ маскируется
        from utils import redact_secrets
        print(f"Ошибка: {redact_secrets(f'{type(exc).__name__}: {exc}')}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
