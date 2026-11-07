"""Демонстрация результатов для видео — читает сохранённые результаты, БЕЗ вызовов API.

Запуск (из папки gemini-task4):
    ../venv/bin/python demo.py                 # последний запуск results/run_N
    ../venv/bin/python demo.py --run run_1     # конкретный запуск
    ../venv/bin/python demo.py --pause         # пауза (Enter) между разделами — удобно для записи экрана
    ../venv/bin/python demo.py --no-color      # без цветов

google-genai не импортируется: демонстрация физически не может отправить запрос.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import config
import summary

RESULTS_DIR = Path(__file__).parent / "results"


class Style:
    def __init__(self, enabled):
        self.enabled = enabled

    def _wrap(self, code, text):
        return f"\033[{code}m{text}\033[0m" if self.enabled else str(text)

    def bold(self, t): return self._wrap("1", t)
    def dim(self, t): return self._wrap("2", t)
    def green(self, t): return self._wrap("32", t)
    def red(self, t): return self._wrap("31", t)
    def yellow(self, t): return self._wrap("33", t)
    def cyan(self, t): return self._wrap("36", t)


def latest_run(results_dir):
    runs = [(int(m.group(1)), p) for p in results_dir.glob("run_*") if (m := re.fullmatch(r"run_(\d+)", p.name))]
    return max(runs)[1] if runs else None


def load_run(run_dir):
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    raw_path = run_dir / "raw_results.json"
    records = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else []
    return meta, records


def section(title, s, pause):
    if pause:
        input(s.dim("\n[Enter — дальше] "))
    print("\n" + s.bold(s.cyan("═" * 78)))
    print(s.bold(s.cyan(f" {title}")))
    print(s.bold(s.cyan("═" * 78)))


def single_cell(r, s):
    if r is None:
        return s.dim("·")
    if r["api_error"] is not None:
        return s.yellow(f"API {r['api_error']['code']}")
    e = r["evaluation"]
    value = e["answer_key"] or e["format_status"]
    text = {True: s.green(f"{value} ✓"), False: s.red(f"{value} ✗"), None: s.yellow(f"{value} ?")}[e["is_correct"]]
    if not e["format_ok"]:
        text += s.red(" [формат]")
    if e["truncated"]:
        text += s.red(" [обрезан]")
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description="Демонстрация результатов эксперимента с temperature (без API)")
    parser.add_argument("--run", help="папка запуска, например run_1 (по умолчанию — последняя)")
    parser.add_argument("--pause", action="store_true", help="пауза между разделами")
    parser.add_argument("--no-color", action="store_true", help="без цветов")
    args = parser.parse_args(argv)
    s = Style(enabled=not args.no_color and sys.stdout.isatty())

    if args.run and not re.fullmatch(r"run_\d+", args.run):
        print(f"Ошибка: ожидается имя вида run_N, получено {args.run!r}")
        return 2
    run_dir = RESULTS_DIR / args.run if args.run else latest_run(RESULTS_DIR)
    if run_dir is None or not (run_dir / "run_meta.json").exists():
        print("Нет сохранённых результатов. Сначала запустите эксперимент (experiment.py --full).")
        return 1
    meta, records = load_run(run_dir)
    temps = meta["temperatures"]
    tasks = meta["tasks"]
    runs = range(1, meta["runs_per_combination"] + 1)

    section(f"Эксперимент «Температура» — {run_dir.name}", s, False)
    print(f"Модель: {s.bold(meta['model'])}   thinking_level: {meta['thinking_level']}   "
          f"max_output_tokens: {meta['max_output_tokens']}")
    print(f"Меняется только temperature: {', '.join(summary.temp_label(t) for t in temps)}   "
          f"(Google рекомендует 1.0 для Gemini 3)")
    print(f"Повторов на комбинацию: {meta['runs_per_combination']}   ответов получено: "
          f"{sum(1 for r in records if r['api_error'] is None)}/{meta['planned_calls']}")

    for task_id, tmeta in tasks.items():
        section(tmeta["title"], s, args.pause)
        print(s.dim("Промпт: " + summary.one_line(tmeta["prompt"])))
        if tmeta["kind"] == "single_answer":
            print(f"Эталон: {s.bold(tmeta['expected'])}\n")
            for t in temps:
                by_run = {r["run"]: r for r in summary.combo_records(records, task_id, t)}
                cells = "  ".join(f"{single_cell(by_run.get(i), s):<14}" for i in runs)
                print(f"  t = {summary.temp_label(t):<4} │ {cells}")
        else:
            for t in temps:
                rs = {r["run"]: r for r in summary.combo_records(records, task_id, t)}
                m = summary.combo_metrics(list(rs.values()))
                repeated = set(m.get("repeated_across_runs", {}))
                stats = "уникальных {}/{}, Jaccard {}".format(
                    m.get("unique_names", 0), m.get("total_names", 0), summary.fmt(m.get("mean_jaccard"), 2))
                print(f"\n  {s.bold('t = ' + summary.temp_label(t))}  {s.dim(stats)}")
                for i in runs:
                    r = rs.get(i)
                    if r is None:
                        line = s.dim("·")
                    elif r["api_error"] is not None:
                        line = s.yellow(f"ошибка API {r['api_error']['code']}")
                    elif not r["evaluation"]["names"]:
                        line = s.red(f"формат сломан: {r['evaluation']['format_status']}")
                    else:
                        e = r["evaluation"]
                        line = " · ".join(s.yellow(n) if nn in repeated else n
                                          for n, nn in zip(e["names"], e["names_normalized"]))
                        if not e["format_ok"]:
                            line += s.red("  [формат]")
                    print(f"    {i}. {line}")
            print(s.dim("\n  Жёлтым — названия, повторившиеся в разных прогонах при одной температуре."))

    if 0.0 in temps:
        section("Даёт ли temperature = 0 одинаковый ответ каждый раз?", s, args.pause)
        for task_id in tasks:
            m = summary.combo_metrics(summary.combo_records(records, task_id, 0.0))
            if not m["received"]:
                print(f"  {task_id}: нет ответов")
                continue
            same = m["identical_texts_max"] == m["received"]
            verdict = s.green("ДА — тексты совпали побайтно") if same else s.red("НЕТ — тексты различаются")
            print(f"  {task_id:11} одинаковых текстов {m['identical_texts_max']}/{m['received']}, "
                  f"различных текстов {m['distinct_texts']}  →  {verdict}")

    section("Сводка по метрикам", s, args.pause)
    print(f"  {'задача':11} {'t':>4}  {'формат ok':>9}  {'нарушен':>7}  {'обрезан':>7}  "
          f"{'качество':>16}  {'схожесть':>8}  {'ошибок API':>10}")
    for task_id, tmeta in tasks.items():
        for t in temps:
            m = summary.combo_metrics(summary.combo_records(records, task_id, t))
            quality = (f"верно {m.get('correct', 0)}/{m.get('evaluated', 0)}" if tmeta["kind"] == "single_answer"
                       else f"уник. {m.get('unique_names', 0)}/{m.get('total_names', 0)}")
            print(f"  {task_id:11} {summary.temp_label(t):>4}  {m['format_ok']:>9}  {m['format_broken']:>7}  "
                  f"{m['truncated']:>7}  {quality:>16}  {summary.fmt(m['mean_similarity'], 2):>8}  {m['api_errors']:>10}")
    print(s.dim(f"\nПолный отчёт: {run_dir / 'summary.md'}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
