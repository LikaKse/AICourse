"""CLI «День 10. Управление контекстом: сравнение стратегий».

Запуск (из папки gemini-task10):
    ../venv/bin/python main.py --dry-run                     # настройки, данные, прогноз вызовов + офлайн-прогон на fake-клиенте
    ../venv/bin/python main.py --chat --strategy facts       # чат (платно; состояние и ветки в state/<strategy>/)
    ../venv/bin/python main.py --experiment all              # три стратегии + сравнение (платно, с подтверждением)
    ../venv/bin/python main.py --experiment sliding          # одна стратегия (sliding | facts | summary)
    ../venv/bin/python main.py --compare                     # comparison.json из сохранённых результатов (без API)
    ../venv/bin/python main.py --report                      # Markdown-таблицы для README (без API)
    ../venv/bin/python main.py --audit facts                 # аудит 36 деталей сохранённого прогона (без API)
    ../venv/bin/python main.py --branch-demo --strategy facts  # checkpoint и ветки A/B на fake-клиенте (без API)

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

def expected_calls(n_messages: int, n_questions: int) -> dict:
    """Сколько generate_content сделает каждая стратегия на сценарии (по правилам, без API)."""
    n_user = n_messages // 2
    n_summaries = sum(1 for total in range(2, n_messages + 1, 2)
                      if total >= config.KEEP_LAST + config.SUMMARY_EVERY
                      and (total - config.KEEP_LAST - config.SUMMARY_EVERY) % config.SUMMARY_EVERY == 0)
    return {"sliding": {"processing": 0, "questions": n_questions},
            "facts": {"processing": n_user, "questions": n_questions},
            "summary": {"processing": n_summaries, "questions": n_questions}}


def offline_run(strategies: list[str]) -> dict:
    """Полный прогон эксперимента на fake-клиенте во временной папке (results/ не трогается)."""
    import tempfile

    import experiments
    from fake_client import FakeClient, echo_responder

    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in strategies:
            client = FakeClient(responder=echo_responder())
            res = experiments.run_strategy_experiment(client, name, state_dir=Path(tmp) / name)
            out[name] = {"generate_content": len(client.requests), "count_tokens": len(client.models.count_requests),
                         "processing_calls": res["tokens"]["processing"]["calls"],
                         "question_calls": res["tokens"]["control_questions"]["calls"]}
    return out


def cmd_dry_run(strategy: str | None = None) -> int:
    import branching
    import compression_audit
    import experiments

    dialogue = experiments.load_dialogue()
    questions = experiments.load_questions()
    facts = experiments.load_facts()
    experiments.validate_data(dialogue, questions, facts)
    catalog = compression_audit.load_catalog()
    compression_audit.validate_catalog(catalog, dialogue)
    scenario = branching.load_scenario()
    names = [strategy] if strategy else list(config.STRATEGIES)

    print(f"Модель: {config.MODEL}")
    print(f"Стратегия по умолчанию: {config.STRATEGY}; WINDOW_SIZE (N) = {config.WINDOW_SIZE}")
    print(f"Summary: SUMMARY_EVERY={config.SUMMARY_EVERY}, KEEP_LAST={config.KEEP_LAST} "
          f"(сжатие при {config.SUMMARY_EVERY + config.KEEP_LAST} активных сообщениях)")
    print(f"Thinking: ответы {config.ANSWER_THINKING_LEVEL}, summary {config.SUMMARY_THINKING_LEVEL}, "
          f"facts {config.FACTS_THINKING_LEVEL}; max_output: ответы {config.MAX_OUTPUT_TOKENS}, "
          f"summary {config.SUMMARY_MAX_OUTPUT_TOKENS}, facts {config.FACTS_MAX_OUTPUT_TOKENS}")
    print(f"Сценарий: {len(dialogue)} сообщений; вопросов: {len(questions)}; фактов в эталоне: {len(facts)}; "
          f"каталог аудита: {len(catalog['details'])} деталей, {len(catalog.get('attributions', []))} проверок авторства")
    print(f"Ветвление: checkpoint после сообщения {scenario['checkpoint_after']}, ветки "
          f"{', '.join(b['name'] for b in scenario['branches'])}, вопросов {len(scenario['questions'])}")

    exp = expected_calls(len(dialogue), len(questions))
    price = config.PRICING[config.MODEL]
    max_out = {"sliding": 0, "facts": config.FACTS_MAX_OUTPUT_TOKENS, "summary": config.SUMMARY_MAX_OUTPUT_TOKENS}
    print("\nОжидаемые платные вызовы generate_content (count_tokens бесплатный, не считается):")
    for n in names:
        e = exp[n]
        worst_out = e["processing"] * max_out[n] + e["questions"] * config.MAX_OUTPUT_TOKENS
        print(f"  {n:8s}: обработка диалога {e['processing']:2d} + вопросы {e['questions']} = "
              f"{e['processing'] + e['questions']}; верхняя граница выхода {worst_out} ток. "
              f"(≤ ${worst_out / 1e6 * price['output_per_1m']:.4f} за выход)")
    print("  Входные токены до запуска неизвестны (зависят от ответов модели) — будут измерены usage_metadata.")

    print("\nОфлайн-прогон всей цепочки на fake-клиенте (Gemini не вызывается):")
    for n, r in offline_run(names).items():
        ok = (r["processing_calls"], r["question_calls"]) == (exp[n]["processing"], exp[n]["questions"])
        print(f"  {n:8s}: generate_content {r['generate_content']} (обработка {r['processing_calls']}, "
              f"вопросы {r['question_calls']}), count_tokens {r['count_tokens']} — "
              f"{'совпадает с прогнозом' if ok else 'НЕ совпадает с прогнозом'}")
    print("\nДанные корректны; каждый эталонный ответ и каждая деталь подтверждены цитатой из диалога. "
          "API не вызывался.")
    return 0


def cmd_experiment(which: str, assume_yes: bool, results_dir: Path = config.RESULTS_DIR) -> int:
    import experiments

    names = list(config.STRATEGIES) if which == "all" else [which]
    exp = expected_calls(len(experiments.load_dialogue()), len(experiments.load_questions()))
    total_calls = sum(exp[n]["processing"] + exp[n]["questions"] for n in names)
    if not confirm(f"Запустить стратегии {', '.join(names)}? Это ≈{total_calls} платных вызовов {config.MODEL}.",
                   assume_yes):
        print("Отменено.")
        return 1
    client = make_client()
    for name in names:
        print(f"\n=== Стратегия {name} ===")
        data = experiments.run_strategy_experiment(client, name, state_dir=results_dir / f"state_{name}")
        path = experiments.save_results(results_dir, name, data)
        t = data["tokens"]
        print(f"Ответы: {data['facts_correct']}/{data['facts_total']}; деталей: "
              f"{(data['detail_audit'] or {}).get('totals', {}).get('preserved')}/36; вызовы: "
              f"{t['processing']['calls']} + {t['control_questions']['calls']}; токены: {t['total']['total_tokens']}; "
              f"файл: {path}")
    return cmd_compare(results_dir)


def cmd_compare(results_dir: Path = config.RESULTS_DIR) -> int:
    import experiments

    cmp = experiments.compare_saved(results_dir)
    if cmp is None:
        print("Нет сохранённых результатов. Сначала: python main.py --experiment all")
        return 0
    print(f"\nСравнение сохранено: {results_dir / experiments.RESULT_FILES['comparison']}")
    from utils import atomic_write_text
    atomic_write_text(results_dir / "report.md", experiments.markdown_report(cmp) + "\n")
    for r in cmp["table"]:
        print(f"  {r['strategy']:8s}: деталей {r['details_preserved']}/36, ответов {r['answers_correct']}/"
              f"{r['answers_total']}, вызовов {r['total']['generate_content_calls']}, "
              f"токенов {r['total']['total_tokens']}, ${r['total']['estimated_cost_usd']}")
    return 0


def cmd_report(results_dir: Path = config.RESULTS_DIR) -> int:
    import experiments

    cmp = experiments.load_result(results_dir, "comparison")
    if not cmp:
        print("Нет сохранённых результатов. Сначала: python main.py --experiment all")
        return 1
    print(experiments.markdown_report(cmp))
    return 0


def cmd_audit(strategy: str, results_dir: Path = config.RESULTS_DIR) -> int:
    """Отчёт аудита 36 деталей из сохранённого прогона. Только чтение results/; API не вызывается."""
    import compression_audit
    import experiments

    res = experiments.load_result(results_dir, strategy)
    if not res or not res.get("detail_audit"):
        print(f"Нет сохранённого прогона стратегии {strategy}. Сначала: python main.py --experiment {strategy}")
        return 1
    print(compression_audit.markdown_report(res["detail_audit"]))
    return 0


def cmd_branch_demo(strategy: str, output: Path | None = None) -> int:
    """Сценарий ветвления на fake-клиенте (эхо-модель отвечает текстом контекста). Без API."""
    import tempfile

    import branching
    import experiments
    from agent import Agent
    from fake_client import FakeClient, echo_responder

    scenario = branching.load_scenario()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(output) if output else Path(tmp) / "session"
        agent = Agent(FakeClient(responder=echo_responder()))
        out = branching.run_scenario(root, strategy, experiments.load_dialogue(), agent, scenario)
        print(out["manager"].describe())
        for name, br in out["branches"].items():
            c = br["continue"]
            print(f"\nВетка {name}: добавлено {c['added']} унаследованных сообщ., пропущено {c['skipped'] or '—'}, "
                  f"текстовых конфликтов {len(c['text_conflicts'])}, конфликтов facts {len(c['facts_conflicts'])}")
            for a in br["answers"]:
                print(f"  {'✅' if a['correct'] else '❌'} {a['question']} (ожидается /{a['expected']}/)")
        print("\nFake-клиент: ответы — эхо контекста; проверяется, какие данные ветки попали в её запрос. API не вызывался.")
    return 0


CHAT_HELP = ("Команды: /stats, /memory, /checkpoint ID, /branch NAME CHECKPOINT, /switch NAME, /branches, "
             "/restore CHECKPOINT, /reset, exit.")


def cmd_chat(strategy: str = config.STRATEGY, state_dir: Path = config.STATE_DIR, client=None, input_fn=input) -> int:
    from agent import Agent
    from branching import BranchError, BranchManager
    from context_manager import StateError
    from history_store import HistoryLoadError

    root = Path(state_dir) / strategy
    try:
        agent = Agent(client or make_client())
        bm = BranchManager(root, strategy=strategy, agent=agent)
    except (HistoryLoadError, StateError, BranchError) as exc:
        print(f"Не удалось загрузить состояние из {root}: {exc}\n"
              "Переименуйте/удалите повреждённый файл или папку состояния и запустите снова.")
        return 1
    print(f"Чат ({config.MODEL}), стратегия {strategy}. Ветка {bm.current}. Восстановлено: {bm.strategy.describe()}")
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
            try:
                if cmd == "/stats":
                    print(bm.strategy.describe())
                elif cmd == "/memory":
                    print(bm.strategy.memory_text() or "(контекст пуст)")
                elif cmd == "/checkpoint" and len(args) == 1:
                    cp = bm.checkpoint(args[0])
                    print(f"Checkpoint {cp['id']}: {cp['messages']} сообщ. ветки {cp['branch']}")
                elif cmd == "/branch" and len(args) == 2:
                    bm.create_branch(args[0], args[1])
                    print(f"Создана ветка {args[0]} от {args[1]}; текущая — {bm.current}")
                elif cmd == "/switch" and len(args) == 1:
                    bm.switch(args[0])
                    print(f"Текущая ветка: {bm.current}. {bm.strategy.describe()}")
                elif cmd == "/branches":
                    print(bm.describe())
                elif cmd == "/restore" and len(args) == 1:
                    bm.restore(args[0])
                    print(f"Ветка {bm.current} возвращена к checkpoint {args[0]}. {bm.strategy.describe()}")
                elif cmd == "/reset":
                    bm.strategy.reset()
                    print(f"Состояние ветки {bm.current} очищено.")
                else:
                    print(CHAT_HELP)
            except BranchError as exc:
                print(f"Ошибка: {exc}")
            continue
        res = agent.ask(line)
        print(res.text)
        for c in res.service_calls:
            status = "ок" if c.ok else f"ошибка: {c.error}"
            print(f"[{c.kind}: {status}; вход {c.usage.input_tokens}, выход {c.usage.output_tokens}]")
        u = res.usage
        print(f"[ветка {bm.current}; токены ответа: запрос {res.request_tokens} (count_tokens), вход {u.input_tokens}, "
              f"выход {u.output_tokens}{'' if u.present else ' — usage_metadata нет'}]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Агент Gemini: сравнение стратегий управления контекстом (День 10)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="настройки, данные, прогноз вызовов; офлайн-прогон, без API")
    mode.add_argument("--chat", action="store_true", help="интерактивный чат с ветками (платно)")
    mode.add_argument("--experiment", choices=(*config.STRATEGIES, "all"), help="эксперимент (платно)")
    mode.add_argument("--compare", action="store_true", help="сравнение сохранённых результатов (без API)")
    mode.add_argument("--report", action="store_true", help="Markdown-отчёт по сохранённым JSON (без API)")
    mode.add_argument("--audit", choices=config.STRATEGIES, help="аудит 36 деталей сохранённого прогона (без API)")
    mode.add_argument("--branch-demo", action="store_true", help="checkpoint и ветки на fake-клиенте (без API)")
    parser.add_argument("--strategy", choices=config.STRATEGIES, default=None,
                        help=f"стратегия для --chat / --branch-demo / --dry-run (по умолчанию {config.STRATEGY})")
    parser.add_argument("--yes", action="store_true", help="не спрашивать подтверждение платных вызовов")
    parser.add_argument("--verbose", action="store_true", help="подробный лог")
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR, help="папка результатов")
    parser.add_argument("--state-dir", type=Path, default=config.STATE_DIR, help="папка состояния чата")
    parser.add_argument("--branch-dir", type=Path, default=None, help="сохранить состояние --branch-demo в папку")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    try:
        if args.dry_run:
            return cmd_dry_run(args.strategy)
        if args.experiment:
            return cmd_experiment(args.experiment, args.yes, args.results_dir)
        if args.compare:
            return cmd_compare(args.results_dir)
        if args.report:
            return cmd_report(args.results_dir)
        if args.audit:
            return cmd_audit(args.audit, args.results_dir)
        if args.branch_demo:
            return cmd_branch_demo(args.strategy or "facts", args.branch_dir)
        return cmd_chat(args.strategy or config.STRATEGY, args.state_dir)
    except ConfigError as exc:
        print(f"Ошибка настройки: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # понятное сообщение вместо трейсбэка; ключ маскируется
        from utils import redact_secrets
        print(f"Ошибка: {redact_secrets(f'{type(exc).__name__}: {exc}')}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
