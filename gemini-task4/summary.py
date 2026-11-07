"""Метрики вариативности и Markdown-отчёт. Не импортирует google-genai (используется и в demo.py).

Отсутствующие значения (None) не превращаются в нули: средние — только по известным значениям.
"""

import difflib
import statistics
from collections import Counter
from datetime import datetime
from itertools import combinations

import config


# ---------- Вспомогательное ----------

def one_line(text, limit=None):
    s = " / ".join(part.strip() for part in str(text).splitlines() if part.strip())
    return s if limit is None or len(s) <= limit else s[:limit - 1] + "…"


def fmt(value, digits=None, suffix=""):
    if value is None:
        return "—"
    if digits is not None:
        value = round(value, digits)
        if digits == 0:
            value = int(value)
    return f"{value}{suffix}"


def md_cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def text_similarity(a, b):
    """Схожесть двух текстов 0..1 (difflib)."""
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


def mean_pairwise(items, fn):
    pairs = list(combinations(items, 2))
    return statistics.mean(fn(x, y) for x, y in pairs) if pairs else None


def mean_known(values):
    known = [v for v in values if v is not None]
    return statistics.mean(known) if known else None


def median_known(values):
    known = [v for v in values if v is not None]
    return statistics.median(known) if known else None


def temp_label(t):
    return f"{t:g}" if t != int(t) else f"{t:.1f}"


# ---------- Метрики одной комбинации «задача × температура» ----------

def combo_records(records, task_id, temperature):
    return sorted((r for r in records if r["task_id"] == task_id and r["temperature"] == temperature),
                  key=lambda r: r["run"])


def combo_metrics(rs):
    ok = [r for r in rs if r["api_error"] is None]
    ev = [r["evaluation"] for r in ok]
    texts = [r["raw_text"] or "" for r in ok]
    text_counts = Counter(texts)
    broken = [e for e in ev if not e["format_ok"]]
    m = {
        "planned": len(rs),
        "received": len(ok),
        "api_errors": sum(1 for r in rs if r["api_error"] is not None),
        "retries": sum(r["retries"] for r in rs),
        "attempts": sum(r["attempts"] for r in rs),
        "format_ok": len(ev) - len(broken),
        "format_broken": len(broken),
        "format_breakdown": dict(Counter(e["format_status"] if e["format_status"] != "ok" else "schema"
                                         for e in broken)),
        "truncated": sum(1 for e in ev if e["truncated"]),
        "identical_texts_max": max(text_counts.values()) if ok else None,
        "distinct_texts": len(text_counts) if ok else None,
        "mean_similarity": mean_pairwise(texts, text_similarity),
        "avg_total_tokens": mean_known([r["total_token_count"] for r in ok]),
        "avg_output_tokens": mean_known([r["candidates_token_count"] for r in ok]),
        "median_latency": median_known([r["latency_sec"] for r in ok]),
        "mean_latency": mean_known([r["latency_sec"] for r in ok]),
    }
    kind = rs[0]["kind"] if rs else None
    if kind == "single_answer":
        evaluated = [e for e in ev if e["is_correct"] is not None]
        answers = Counter(e["answer_key"] for e in ev if e["answer_key"] is not None)
        m.update(
            evaluated=len(evaluated),
            correct=sum(1 for e in evaluated if e["is_correct"]),
            wrong=sum(1 for e in evaluated if not e["is_correct"]),
            not_evaluated=len(ev) - len(evaluated),
            answers=dict(answers),
            distinct_answers=len(answers),
            identical_answers_max=max(answers.values()) if answers else None,
        )
    elif kind == "generation":
        with_names = [e for e in ev if e["names_normalized"]]
        name_sets = [frozenset(e["names_normalized"]) for e in with_names]
        all_names = [n for e in with_names for n in e["names_normalized"]]
        runs_per_name = Counter(n for s in name_sets for n in s)
        m.update(
            lists_evaluated=sum(1 for e in ev if e["structure_ok"] is not None),
            lists_valid=sum(1 for e in ev if e["structure_ok"]),
            lists_with_inner_duplicates=sum(1 for e in ev if e["duplicates_within"]),
            total_names=len(all_names),
            unique_names=len(set(all_names)),
            repeated_across_runs={n: c for n, c in runs_per_name.most_common() if c >= 2},
            mean_jaccard=mean_pairwise(name_sets, jaccard),
            identical_sets_max=max(Counter(name_sets).values()) if name_sets else None,
        )
    return m


def all_metrics(records):
    tasks = list(dict.fromkeys(r["task_id"] for r in records))
    temps = sorted({r["temperature"] for r in records})
    return {(task, t): combo_metrics(combo_records(records, task, t)) for task in tasks for t in temps}


# ---------- Строка для терминала ----------

def console_line(r):
    if r["api_error"] is not None:
        e = r["api_error"]
        return f"ошибка API {e['code']} ({e['type']}): {one_line(e['message'], 100)}"
    e = r["evaluation"]
    fmt_note = "формат ok" if e["format_ok"] else f"ФОРМАТ НАРУШЕН ({e['format_status']})"
    if e["truncated"]:
        fmt_note += ", ОБРЕЗАН"
    tail = f"{fmt_note} | {r['latency_sec']} с | токенов {fmt(r['total_token_count'])}"
    if r["kind"] == "single_answer":
        mark = {True: "верно", False: "НЕВЕРНО", None: "не оценено"}[e["is_correct"]]
        return f"ответ {e['answer']!r} — {mark} | {tail}"
    names = "; ".join(e["names"] or []) or "—"
    return f"{len(e['names'] or [])} назв.: {one_line(names, 90)} | {tail}"


# ---------- Markdown-отчёт ----------

def _mark_single(r):
    if r is None:
        return "·"
    if r["api_error"] is not None:
        return "⚠️ API"
    e = r["evaluation"]
    mark = {True: "✅", False: "❌", None: "❔"}[e["is_correct"]]
    value = e["answer"] if e["answer"] is not None else e["format_status"]
    flags = ("" if e["format_ok"] else " ⛔формат") + (" ✂️обрезан" if e["truncated"] else "")
    return f"{mark} {md_cell(one_line(value, 40))}{flags}"


def build_observations(metrics, temps, tasks_meta):
    lines = []
    single = next((k for k, v in tasks_meta.items() if v["kind"] == "single_answer"), None)
    gen = next((k for k, v in tasks_meta.items() if v["kind"] == "generation"), None)
    if single:
        parts = [f"t={temp_label(t)}: верно {metrics[(single, t)].get('correct', 0)}/"
                 f"{metrics[(single, t)].get('evaluated', 0)}, "
                 f"различных ответов {metrics[(single, t)].get('distinct_answers', 0)}" for t in temps]
        lines.append("- **Один правильный ответ** — " + "; ".join(parts) + ".")
    if gen:
        parts = [f"t={temp_label(t)}: уникальных {metrics[(gen, t)].get('unique_names', 0)}/"
                 f"{metrics[(gen, t)].get('total_names', 0)}, "
                 f"сходство списков (Jaccard) {fmt(metrics[(gen, t)].get('mean_jaccard'), 2)}" for t in temps]
        lines.append("- **Генерация** — " + "; ".join(parts) + ". Чем ниже Jaccard, тем сильнее списки "
                     "отличаются между прогонами. Разнообразие здесь измеряется, а оригинальность и "
                     "качество названий автоматически не оцениваются.")
    fmt_parts = []
    for t in temps:
        broken = sum(metrics[(task, t)]["format_broken"] for task in tasks_meta)
        trunc = sum(metrics[(task, t)]["truncated"] for task in tasks_meta)
        received = sum(metrics[(task, t)]["received"] for task in tasks_meta)
        fmt_parts.append(f"t={temp_label(t)}: {broken}/{received}" + (f" (обрезано {trunc})" if trunc else ""))
    lines.append("- **Нарушения формата** — " + "; ".join(fmt_parts) + ".")
    if 0.0 in temps:
        for task in tasks_meta:
            m = metrics[(task, 0.0)]
            if m["received"]:
                same = m["identical_texts_max"] == m["received"]
                lines.append(f"- **temperature = 0, {task}:** одинаковых текстов {m['identical_texts_max']}/{m['received']} — "
                             + ("все ответы побайтно совпали." if same else "ответы НЕ совпадают полностью."))
    lines.append(f"- Выборка: {config.RUNS_PER_COMBINATION} прогонов на комбинацию — выводы предварительные.")
    return lines


def build_summary(records, meta, status):
    temps = list(meta.get("temperatures", sorted({r["temperature"] for r in records})))
    tasks_meta = meta.get("tasks", {})
    metrics = {(task, t): combo_metrics(combo_records(records, task, t)) for task in tasks_meta for t in temps}
    planned = meta.get("planned_calls", len(records))
    received = sum(1 for r in records if r["api_error"] is None)
    lines = [
        "# Сводка: влияние temperature",
        "",
        f"Модель: `{meta.get('model')}`; thinking_level: `{meta.get('thinking_level')}`; "
        f"max_output_tokens: {meta.get('max_output_tokens')}; Structured Output: нет (JSON только по инструкции)  ",
        f"Создан: {meta.get('created_at')}; обновлён: {datetime.now().isoformat(timespec='seconds')}  ",
        f"Записей: {len(records)}/{planned}; успешных ответов API: {received}/{planned}; статус: {status}",
        "",
        "Меняется только temperature; промпты одинаковы во всех вызовах.",
        "",
        "## Сводная таблица",
        "",
        "«Одинаковых текстов» — сколько прогонов из полученных дали побайтно один и тот же текст (максимальная группа). "
        "Схожесть — средняя попарная схожесть текстов (difflib, 0–1). Время — медиана успешных ответов.",
        "",
        "| Задача | t | Получено | Формат OK | Формат нарушен | Обрезано | Верно / оценено | Различных ответов "
        "| Одинаковых текстов | Схожесть текстов | Токенов (ср.) | Время, с | Ошибок API | Повторов |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for task, tmeta in tasks_meta.items():
        for t in temps:
            m = metrics[(task, t)]
            if tmeta["kind"] == "single_answer":
                quality = f"{m.get('correct', 0)} / {m.get('evaluated', 0)}"
                distinct = fmt(m.get("distinct_answers"))
            else:
                quality = f"валидных списков {m.get('lists_valid', 0)} / {m.get('lists_evaluated', 0)}"
                distinct = f"уник. названий {m.get('unique_names', 0)}/{m.get('total_names', 0)}"
            breakdown = ", ".join(f"{k}: {v}" for k, v in m["format_breakdown"].items())
            lines.append(
                f"| {task} | {temp_label(t)} | {m['received']}/{m['planned']} | {m['format_ok']} "
                f"| {m['format_broken']}{f' ({breakdown})' if breakdown else ''} | {m['truncated']} | {quality} "
                f"| {distinct} | {fmt(m['identical_texts_max'])}/{m['received']} | {fmt(m['mean_similarity'], 2)} "
                f"| {fmt(m['avg_total_tokens'], 0)} | {fmt(m['median_latency'], 1)} | {m['api_errors']} | {m['retries']} |"
            )

    # temperature = 0
    if 0.0 in temps:
        lines += ["", "## Даёт ли temperature = 0 одинаковый ответ каждый раз?", ""]
        for task, tmeta in tasks_meta.items():
            m = metrics[(task, 0.0)]
            if not m["received"]:
                lines.append(f"- **{task}:** нет полученных ответов.")
                continue
            text_same = m["identical_texts_max"] == m["received"]
            if tmeta["kind"] == "single_answer":
                content = f"одинаковых ответов {fmt(m['identical_answers_max'])}/{m['received']} ({m['answers']})"
            else:
                content = f"одинаковых наборов названий {fmt(m['identical_sets_max'])}/{m['received']}"
            lines.append(f"- **{task}:** одинаковых текстов {m['identical_texts_max']}/{m['received']}; {content}. "
                         + ("**Да — все тексты совпали побайтно.**" if text_same else "**Нет — тексты различаются.**"))

    # Примеры: один правильный ответ
    for task, tmeta in tasks_meta.items():
        if tmeta["kind"] != "single_answer":
            continue
        lines += ["", f"## Примеры ответов: {tmeta['title']}", "",
                  f"Эталон: **{tmeta['expected']}**. ✅ верно · ❌ неверно · ❔ не оценено · ⛔ формат нарушен · ✂️ обрезан · ⚠️ ошибка API",
                  "", "| t | " + " | ".join(f"Прогон {i}" for i in range(1, config.RUNS_PER_COMBINATION + 1)) + " |",
                  "|---|" + "---|" * config.RUNS_PER_COMBINATION]
        for t in temps:
            by_run = {r["run"]: r for r in combo_records(records, task, t)}
            lines.append(f"| {temp_label(t)} | " + " | ".join(_mark_single(by_run.get(i)) for i in
                                                            range(1, config.RUNS_PER_COMBINATION + 1)) + " |")

    # Примеры: генерация — все названия рядом
    for task, tmeta in tasks_meta.items():
        if tmeta["kind"] != "generation":
            continue
        lines += ["", f"## Все варианты названий: {tmeta['title']}", "",
                  "**Жирным** — название, которое встретилось в нескольких прогонах при той же температуре.", ""]
        for t in temps:
            m = metrics[(task, t)]
            rs = {r["run"]: r for r in combo_records(records, task, t)}
            repeated = set(m.get("repeated_across_runs", {}))
            lines += [f"### t = {temp_label(t)}", "",
                      "| № | " + " | ".join(f"Прогон {i}" for i in range(1, config.RUNS_PER_COMBINATION + 1)) + " |",
                      "|---|" + "---|" * config.RUNS_PER_COMBINATION]
            columns = []
            for i in range(1, config.RUNS_PER_COMBINATION + 1):
                r = rs.get(i)
                if r is None:
                    columns.append(["·"])
                elif r["api_error"] is not None:
                    columns.append(["⚠️ ошибка API"])
                elif not r["evaluation"]["names"]:
                    columns.append([f"⛔ {r['evaluation']['format_status']}"])
                else:
                    e = r["evaluation"]
                    columns.append([f"**{md_cell(n)}**" if nn in repeated else md_cell(n)
                                    for n, nn in zip(e["names"], e["names_normalized"])])
            for row in range(max(len(c) for c in columns)):
                lines.append(f"| {row + 1} | " + " | ".join(c[row] if row < len(c) else "" for c in columns) + " |")
            rep = ", ".join(f"{n} ×{c}" for n, c in m.get("repeated_across_runs", {}).items()) or "нет"
            lines += ["", f"Уникальных названий: {m.get('unique_names', 0)} из {m.get('total_names', 0)}; "
                          f"сходство списков между прогонами (Jaccard): {fmt(m.get('mean_jaccard'), 2)}; "
                          f"списков с повторами внутри: {m.get('lists_with_inner_duplicates', 0)}; "
                          f"повторы между прогонами: {rep}.", ""]

    lines += ["## Наблюдения (вычислены автоматически)", ""] + build_observations(metrics, temps, tasks_meta)

    # Нарушения формата, неверные ответы, ошибки
    broken = [r for r in records if r["api_error"] is None and not r["evaluation"]["format_ok"]]
    lines += ["", "## Нарушения формата", ""]
    if broken:
        for r in broken:
            e = r["evaluation"]
            reason = e["format_error"] or ("схема: " + "; ".join(f"{x['loc']} {x['type']}" for x in e["schema_errors"] or []))
            lines.append(f"- **{r['solution_id']}** — {e['format_status']}: {reason}; finish_reason={r['finish_reason']}; "
                         f"начало ответа: `{md_cell(one_line(r['raw_text'] or '', 120)).replace('`', 'ʼ')}`")
    else:
        lines.append("Нет.")
    wrong = [r for r in records if r["api_error"] is None and r["kind"] == "single_answer"
             and r["evaluation"]["is_correct"] is False]
    lines += ["", "## Неверные ответы", ""]
    lines += ([f"- **{r['solution_id']}** — ответ `{md_cell(r['evaluation']['answer'])}`, "
               f"эталон `{r['expected_answer']}`: {r['evaluation']['check_reason']}" for r in wrong] or ["Нет."])
    errs = [r for r in records if r["api_error"] is not None]
    lines += ["", "## Ошибки API (отдельно от ответов модели)", ""]
    lines += ([f"- **{r['solution_id']}** (попыток {r['attempts']}) — {r['api_error']['type']} "
               f"{fmt(r['api_error']['code'])}: {one_line(r['api_error']['message'])}" for r in errs] or ["Нет."])
    retried = [(r, a) for r in records for a in r["attempt_log"] if a["error"] and r["retries"]]
    lines += ["", "## Повторные попытки", ""]
    lines += ([f"- {r['solution_id']} / попытка {a['attempt']}: {fmt(a['error']['code'])} "
               f"{one_line(a['error']['message'], 80)}" + (f", пауза {a['retry_after_sec']} с" if a["retry_after_sec"] else "")
               for r, a in retried] or ["Не было."])
    return "\n".join(lines) + "\n"
