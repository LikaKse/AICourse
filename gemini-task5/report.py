"""Итоговая таблица и сравнение моделей. Работает только с сохранёнными записями, API не нужен.

Правила подсчёта:
- «Всего» — число запланированных вызовов модели (задачи × повторы). Ошибка API, пропуск или
  ещё не выполненный вызов считаются неверным ответом и показываются отдельными колонками.
- Время — latency успешной попытки, без пауз темпа и без повторов.
- Входные токены — prompt_token_count. Выходные — candidates + thoughts (так их тарифицирует Google).
  Всего — total_token_count из API.
- Стоимость — сумма по успешным вызовам. Если хотя бы у одного успешного вызова стоимость
  неизвестна, сумма помечается как нижняя оценка (cost_complete = False).
"""

import csv
import json
import os
from statistics import mean, median

SUMMARY_MD = "summary.md"
SUMMARY_CSV = "summary.csv"
SUMMARY_JSON = "summary.json"

SUMMARY_FIELDS = [
    "role", "role_name", "model", "correct", "total", "accuracy_pct", "api_errors", "skipped", "not_run",
    "truncated",
    "avg_latency_sec", "median_latency_sec", "total_latency_sec",
    "input_tokens", "output_tokens", "thoughts_tokens", "total_tokens", "avg_total_tokens",
    "cost_usd", "cost_complete", "avg_cost_usd", "cost_per_correct_usd",
]


def _sum(values):
    known = [v for v in values if v is not None]
    return sum(known) if known else None


def build_summary(records, meta):
    planned = meta["planned_record_ids"]
    by_model = {m["id"]: [] for m in meta["models"]}
    for r in records:
        by_model.setdefault(r["model"], []).append(r)

    rows = []
    for m in meta["models"]:
        recs = by_model.get(m["id"], [])
        ok = [r for r in recs if r["status"] == "ok"]
        total = sum(1 for rid in planned if rid.split("|")[1] == m["id"])
        correct = sum(1 for r in recs if r["is_correct"] is True)
        latencies = [r["latency_sec"] for r in ok if r["latency_sec"] is not None]
        costs = [r["cost_usd"] for r in ok]
        cost = _sum(costs)
        total_tokens = _sum(r["total_tokens"] for r in ok)
        rows.append({
            "role": m["role"],
            "role_name": m["role_name"],
            "model": m["id"],
            "correct": correct,
            "total": total,
            "accuracy_pct": round(100 * correct / total, 1) if total else None,
            "api_errors": sum(1 for r in recs if r["status"] == "api_error"),
            "skipped": sum(1 for r in recs if r["status"] == "skipped"),
            "not_run": total - len(recs),
            "truncated": sum(1 for r in ok if r.get("truncated")),
            "avg_latency_sec": round(mean(latencies), 2) if latencies else None,
            "median_latency_sec": round(median(latencies), 2) if latencies else None,
            "total_latency_sec": round(sum(latencies), 2) if latencies else None,
            "input_tokens": _sum(r["input_tokens"] for r in ok),
            "output_tokens": _sum(r["output_tokens"] for r in ok),
            "thoughts_tokens": _sum(r["thoughts_tokens"] for r in ok),
            "total_tokens": total_tokens,
            "avg_total_tokens": round(total_tokens / len(ok)) if ok and total_tokens is not None else None,
            "cost_usd": cost,
            "cost_complete": bool(ok) and all(c is not None for c in costs),
            "avg_cost_usd": cost / len(ok) if ok and cost is not None else None,
            "cost_per_correct_usd": cost / correct if correct and cost is not None else None,
        })

    by_role = {r["role"]: r for r in rows}
    return {"run_id": meta["run_id"], "rows": rows, "comparison": compare(by_role), "pricing": meta["pricing"],
            "per_task": per_task_matrix(records, meta)}


def compare(by_role):
    small, medium, flagship = by_role.get("small"), by_role.get("medium"), by_role.get("flagship")
    out = {}

    def ratio(a, b, key):
        if not a or not b:
            return None, "нет данных по одной из моделей"
        if a[key] is None or b[key] is None:
            return None, "стоимость неизвестна (нет usage_metadata)"
        if b[key] == 0:
            return None, "у маленькой модели стоимость 0 — делить нельзя"
        note = None if (a.get("cost_complete", True) and b.get("cost_complete", True)) else "по неполным данным"
        return a[key] / b[key], note

    out["flagship_vs_small_cost_ratio"], out["flagship_vs_small_cost_note"] = ratio(flagship, small, "cost_usd")
    out["medium_vs_small_cost_ratio"], out["medium_vs_small_cost_note"] = ratio(medium, small, "cost_usd")

    def diff(a, b):
        if a and b and a["accuracy_pct"] is not None and b["accuracy_pct"] is not None:
            return round(a["accuracy_pct"] - b["accuracy_pct"], 1)
        return None

    out["flagship_minus_small_accuracy_pp"] = diff(flagship, small)
    out["medium_minus_small_accuracy_pp"] = diff(medium, small)
    out["flagship_minus_medium_accuracy_pp"] = diff(flagship, medium)
    out["avg_latency_sec"] = {r["model"]: r["avg_latency_sec"] for r in by_role.values()}
    if flagship and small and flagship["avg_latency_sec"] and small["avg_latency_sec"]:
        out["flagship_vs_small_latency_ratio"] = flagship["avg_latency_sec"] / small["avg_latency_sec"]
    else:
        out["flagship_vs_small_latency_ratio"] = None
    return out


def per_task_matrix(records, meta):
    """task_id → model → '✓' / '✗' / 'ошибка' / 'пропуск' (при повторах — «верных/всего»)."""
    matrix = {}
    for r in records:
        cell = matrix.setdefault(r["task_id"], {}).setdefault(r["model"], [])
        cell.append(r)
    out = {}
    for task_id in meta["task_ids"]:
        out[task_id] = {}
        for m in meta["models"]:
            recs = matrix.get(task_id, {}).get(m["id"], [])
            if not recs:
                out[task_id][m["id"]] = "—"
            elif len(recs) == 1:
                r = recs[0]
                if r["status"] == "ok":
                    mark = "✓" if r["is_correct"] else "✗"
                    out[task_id][m["id"]] = mark + (" (обрезан)" if r.get("truncated") else "")
                else:
                    out[task_id][m["id"]] = "ошибка API" if r["status"] == "api_error" else "пропуск"
            else:
                out[task_id][m["id"]] = f"{sum(1 for r in recs if r['is_correct'])}/{len(recs)}"
    return out


# ---------- Форматирование ----------

def fmt_int(v):
    return "—" if v is None else f"{v:,}".replace(",", " ")


def fmt_sec(v):
    return "—" if v is None else f"{v:.2f}"


def fmt_usd(v, complete=True):
    if v is None:
        return "—"
    return f"${v:.6f}" + ("" if complete else "*")


def fmt_ratio(v):
    return "—" if v is None else f"{v:.1f}×"


def fmt_pp(v):
    return "—" if v is None else f"{v:+.1f} п.п."


def table_rows(summary):
    header = ["Модель", "Роль", "Верных / всего", "Точность", "Ошибок API", "Обрезано", "Ср. время, с",
              "Сумм. время, с", "Входные", "Выходные (в т.ч. thinking)", "Всего токенов", "Стоимость"]
    body = []
    for r in summary["rows"]:
        out_tokens = fmt_int(r["output_tokens"])
        if r["thoughts_tokens"]:
            out_tokens += f" ({fmt_int(r['thoughts_tokens'])})"
        body.append([
            r["model"], r["role_name"], f"{r['correct']} / {r['total']}",
            "—" if r["accuracy_pct"] is None else f"{r['accuracy_pct']:.1f}%",
            str(r["api_errors"] + r["skipped"]), str(r["truncated"]),
            fmt_sec(r["avg_latency_sec"]), fmt_sec(r["total_latency_sec"]),
            fmt_int(r["input_tokens"]), out_tokens, fmt_int(r["total_tokens"]),
            fmt_usd(r["cost_usd"], r["cost_complete"]),
        ])
    return header, body


def comparison_lines(summary):
    c = summary["comparison"]
    rows = {r["role"]: r for r in summary["rows"]}
    small, medium, flagship = rows.get("small"), rows.get("medium"), rows.get("flagship")
    lines = []
    for r in summary["rows"]:
        accuracy = "—" if r["accuracy_pct"] is None else f"{r['accuracy_pct']:.1f}%"
        lines.append(f"Точность {r['model']}: {accuracy} ({r['correct']} из {r['total']}); "
                     f"среднее время ответа {fmt_sec(r['avg_latency_sec'])} с")
    note = f" ({c['flagship_vs_small_cost_note']})" if c["flagship_vs_small_cost_note"] else ""
    if flagship and small:
        lines.append(f"Флагман дороже маленькой модели в {fmt_ratio(c['flagship_vs_small_cost_ratio'])}{note}")
        lines.append(f"Разница в точности (флагман − маленькая): {fmt_pp(c['flagship_minus_small_accuracy_pp'])}")
    if medium and small:
        note_m = f" ({c['medium_vs_small_cost_note']})" if c["medium_vs_small_cost_note"] else ""
        lines.append(f"Средняя дороже маленькой в {fmt_ratio(c['medium_vs_small_cost_ratio'])}{note_m}; "
                     f"разница в точности (средняя − маленькая): {fmt_pp(c['medium_minus_small_accuracy_pp'])}")
    if c["flagship_vs_small_latency_ratio"]:
        lines.append(f"Флагман в среднем медленнее маленькой в {fmt_ratio(c['flagship_vs_small_latency_ratio'])}")
    return lines


def render_markdown(summary):
    header, body = table_rows(summary)
    p = summary["pricing"]
    out = [f"# Итоги {summary['run_id']}", "", "## Итоговая таблица", "",
           "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(row) + " |" for row in body]
    out += ["", "«Ошибок API» включает пропущенные вызовы. «Обрезано» — ответ упёрся в max_output_tokens "
            "(finish_reason = MAX_TOKENS). «*» — стоимость по неполным данным.",
            "Выходные токены = candidates + thoughts: thinking оплачивается по цене выходных токенов.",
            "Время — среднее и сумма latency успешных вызовов, без пауз и повторов.", "",
            "## Сравнение", ""]
    out += [f"- {line}" for line in comparison_lines(summary)]
    out += ["", "## Ответы по задачам", "",
            "| Задача | " + " | ".join(r["model"] for r in summary["rows"]) + " |",
            "|" + "---|" * (1 + len(summary["rows"]))]
    for task_id, cells in summary["per_task"].items():
        out.append(f"| {task_id} | " + " | ".join(cells[r["model"]] for r in summary["rows"]) + " |")
    out += ["", "## Тарифы", "", f"{p['tier']}, USD за 1M токенов, проверено {p['checked_on']}: "
            f"{p['source']} (→ {p['source_resolved']}).", "",
            "| Модель | Период | Вход | Выход (вкл. thinking) |", "|---|---|---|---|"]
    for model, periods in p["prices"].items():
        for per in periods:
            period = f"{per['valid_from'] or '…'} — {per['valid_to'] or '…'}"
            out.append(f"| {model} | {period} | ${per['input']:.2f} | ${per['output']:.2f} |")
    return "\n".join(out) + "\n"


def render_console(summary):
    header, body = table_rows(summary)
    widths = [max(len(str(x)) for x in col) for col in zip(header, *body)]
    line = lambda cells: "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))
    out = ["", line(header), "  ".join("-" * w for w in widths)] + [line(r) for r in body] + [""]
    out += comparison_lines(summary)
    return "\n".join(out)


def write_reports(run_dir, records, meta):
    summary = build_summary(records, meta)
    tmp = run_dir / (SUMMARY_JSON + ".tmp")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, run_dir / SUMMARY_JSON)
    tmp = run_dir / (SUMMARY_MD + ".tmp")
    tmp.write_text(render_markdown(summary), encoding="utf-8")
    os.replace(tmp, run_dir / SUMMARY_MD)
    tmp = run_dir / (SUMMARY_CSV + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in summary["rows"]:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in SUMMARY_FIELDS})
    os.replace(tmp, run_dir / SUMMARY_CSV)
    return summary
