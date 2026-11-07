"""Метрики и Markdown-отчёт по результатам эксперимента.

Отсутствующие значения (None) не превращаются в нули: средние считаются только по известным
значениям, а число учтённых значений показывается рядом, если известны не все.
"""

import statistics
from datetime import datetime

import config


def one_line(text):
    return " / ".join(part.strip() for part in str(text).splitlines() if part.strip())


def fmt(value, digits=None, suffix=""):
    if value is None:
        return "—"
    if digits is not None:
        value = round(value, digits)
        if digits == 0:
            value = int(value)
    return f"{value}{suffix}"


def mean_known(values):
    known = [v for v in values if v is not None]
    return (statistics.mean(known) if known else None), len(known)


def group_metrics(records):
    """Метрики группы решений (например, один способ при одном уровне рассуждения)."""
    total = len(records)
    answered = [r for r in records if r["is_correct"] is not None]
    correct = sum(1 for r in answered if r["is_correct"])
    tokens = [r["total_token_count"] for r in answered]
    tokens_known = [t for t in tokens if t is not None]
    avg_tokens, tokens_n = mean_known(tokens)
    avg_latency, latency_n = mean_known([r["latency_sec"] for r in answered])
    avg_thoughts, thoughts_n = mean_known([r["thoughts_token_count"] for r in answered])
    return {
        "total": total,
        "answered": len(answered),
        "correct": correct,
        "accuracy": correct / len(answered) * 100 if answered else None,       # от полученных ответов
        "accuracy_of_all": correct / total * 100 if total else None,           # от всех решений
        "tokens_sum": sum(tokens_known) if tokens_known else None,
        "tokens_known": tokens_n,
        "avg_tokens": avg_tokens,
        "avg_thoughts": avg_thoughts,
        "thoughts_known": thoughts_n,
        "avg_latency": avg_latency,
        "latency_known": latency_n,
        "api_errors": sum(1 for r in records if r["api_error"] is not None),
        "retries": sum(r["retries"] for r in records),
        "attempts": sum(r["attempts"] for r in records),
        "tokens_incomplete": any(r["tokens_incomplete"] for r in records),
    }


def metrics_by_group(records):
    groups = {}
    for strategy in config.STRATEGIES:
        for level in config.THINKING_LEVELS:
            rs = [r for r in records if r["strategy"] == strategy and r["thinking_level"] == level]
            groups[(strategy, level)] = group_metrics(rs)
    return groups


def _count_note(value_n, of_n):
    """Пометка «(n из m)», если значение известно не для всех ответов."""
    return "" if value_n == of_n else f" ({value_n} из {of_n})"


def _delta(a, b, digits=0, suffix=" п.п."):
    if a is None or b is None:
        return "—"
    d = round(a - b, digits)
    if digits == 0:
        d = int(d)
    return f"{'+' if d > 0 else ''}{d}{suffix}"


def _verdict(acc, base):
    if acc is None or base is None:
        return None
    if acc > base:
        return "помогает"
    if acc < base:
        return "хуже"
    return "без разницы"


def build_conclusions(groups):
    """Механические выводы по точности (без интерпретации модели)."""
    lines = []
    for strategy in config.STRATEGIES:
        if strategy == "direct":
            continue
        name = config.STRATEGY_NAMES[strategy]
        parts = []
        verdicts = {}
        for level in config.THINKING_LEVELS:
            acc = groups[(strategy, level)]["accuracy"]
            base = groups[("direct", level)]["accuracy"]
            verdicts[level] = _verdict(acc, base)
            parts.append(f"{level}: {fmt(acc, 0, '%')} против Direct {fmt(base, 0, '%')} "
                         f"({_delta(acc, base)}) — {verdicts[level] or 'нет данных'}")
        v_min, v_high = verdicts.get("MINIMAL"), verdicts.get("HIGH")
        if v_min == "помогает":
            tail = ("преимущество сохраняется при HIGH" if v_high == "помогает"
                    else "при HIGH преимущество исчезает" if v_high else "данных по HIGH нет")
        elif v_min is None:
            tail = "данных по MINIMAL нет"
        else:
            tail = "при MINIMAL выигрыша нет"
        lines.append(f"- **{name}** — " + "; ".join(parts) + f". Итог: {tail}.")
    for level in config.THINKING_LEVELS:
        base = groups[("direct", level)]["accuracy"]
        if base == 100:
            lines.append(f"- При {level} Direct уже даёт 100%: улучшить точность на этом наборе нельзя "
                         f"(эффект потолка), сравнивать способы стоит по токенам и времени.")
    lines.append("- Выборка мала (5 задач на группу): разница в одну задачу = 20 п.п.; выводы предварительные.")
    return lines


def build_summary(records, meta, status):
    groups = metrics_by_group(records)
    planned = meta.get("planned_solutions", len(records))
    lines = [
        "# Сводка: разные способы рассуждения",
        "",
        f"Модель: `{meta.get('model')}`  ",
        f"Режим: {meta.get('mode')}; создан: {meta.get('created_at')}; обновлён: "
        f"{datetime.now().isoformat(timespec='seconds')}  ",
        f"Решений: {len(records)}/{planned}; статус: {status}  ",
        "Финальные ответы всех способов — единая JSON-схема `{answer, explanation}`; "
        "правильность проверяется программно.",
        "",
        "## Итоговая таблица",
        "",
        "Точность — доля верных среди полученных ответов (в скобках — от всех решений группы, "
        "включая ошибки API). Токены — `total_token_count`; для Prompt generation — сумма двух вызовов. "
        "Среднее время — время успешных вызовов, без пауз и повторов.",
        "",
        "| Способ | Reasoning | Верных ответов | Всего | Точность | Токены | Токенов на задачу "
        "| Среднее время, с | Ошибок API | Повторов |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for strategy in config.STRATEGIES:
        for level in config.THINKING_LEVELS:
            m = groups[(strategy, level)]
            if m["total"] == 0:
                continue
            star = "*" if m["tokens_incomplete"] else ""
            lines.append(
                f"| {config.STRATEGY_NAMES[strategy]} | {level} | {m['correct']} | {m['total']} "
                f"| {fmt(m['accuracy'], 0, '%')} ({fmt(m['accuracy_of_all'], 0, '%')}) "
                f"| {fmt(m['tokens_sum'])}{star}{_count_note(m['tokens_known'], m['answered'])} "
                f"| {fmt(m['avg_tokens'], 0)} | {fmt(m['avg_latency'], 2)}{_count_note(m['latency_known'], m['answered'])} "
                f"| {m['api_errors']} | {m['retries']} |"
            )
    if any(m["tokens_incomplete"] for m in groups.values()):
        lines += ["", "\\* часть полей usage_metadata не вернулась — сумма является нижней оценкой."]

    total_m = group_metrics(records)
    lines += [
        "",
        f"Всего: верных {total_m['correct']} из {total_m['answered']} полученных ответов "
        f"({fmt(total_m['accuracy'], 1, '%')}); решений {total_m['total']}; вызовов API (попыток) "
        f"{total_m['attempts']}; ошибок API {total_m['api_errors']}; повторных попыток {total_m['retries']}.",
        "",
        "## MINIMAL vs HIGH по каждому способу",
        "",
        "| Способ | Точность MINIMAL | Точность HIGH | Δ точности | Токенов/задачу MINIMAL | Токенов/задачу HIGH "
        "| Thinking-токенов MINIMAL | Thinking-токенов HIGH | Время MINIMAL, с | Время HIGH, с |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for strategy in config.STRATEGIES:
        lo, hi = groups[(strategy, "MINIMAL")], groups[(strategy, "HIGH")]
        if lo["total"] == 0 and hi["total"] == 0:
            continue
        lines.append(
            f"| {config.STRATEGY_NAMES[strategy]} | {fmt(lo['accuracy'], 0, '%')} | {fmt(hi['accuracy'], 0, '%')} "
            f"| {_delta(hi['accuracy'], lo['accuracy'])} | {fmt(lo['avg_tokens'], 0)} | {fmt(hi['avg_tokens'], 0)} "
            f"| {fmt(lo['avg_thoughts'], 0)} | {fmt(hi['avg_thoughts'], 0)} "
            f"| {fmt(lo['avg_latency'], 2)} | {fmt(hi['avg_latency'], 2)} |"
        )

    lines += [
        "",
        "## Сравнение с Direct на том же уровне",
        "",
        "| Способ | Reasoning | Δ точности к Direct | Токенов/задачу | × к Direct по токенам | Время, с | × к Direct по времени |",
        "|---|---|---|---|---|---|---|",
    ]
    for level in config.THINKING_LEVELS:
        base = groups[("direct", level)]
        for strategy in config.STRATEGIES:
            m = groups[(strategy, level)]
            if m["total"] == 0:
                continue
            ratio_t = (m["avg_tokens"] / base["avg_tokens"]) if m["avg_tokens"] and base["avg_tokens"] else None
            ratio_l = (m["avg_latency"] / base["avg_latency"]) if m["avg_latency"] and base["avg_latency"] else None
            lines.append(
                f"| {config.STRATEGY_NAMES[strategy]} | {level} | {_delta(m['accuracy'], base['accuracy'])} "
                f"| {fmt(m['avg_tokens'], 0)} | {fmt(ratio_t, 2, '×')} | {fmt(m['avg_latency'], 2)} | {fmt(ratio_l, 2, '×')} |"
            )

    lines += ["", "## Выводы (вычислены автоматически)", ""] + build_conclusions(groups)

    # Матрица по задачам
    task_ids = meta.get("task_ids") or sorted({r["task_id"] for r in records})
    columns = [(s, l) for s in config.STRATEGIES for l in config.THINKING_LEVELS]
    lines += [
        "",
        "## Результаты по задачам",
        "",
        "✅ верно · ❌ неверно · ⚠️ ошибка API · · не выполнялось",
        "",
        "| Задача | " + " | ".join(f"{config.STRATEGY_NAMES[s]} {l[:3]}" for s, l in columns) + " |",
        "|---|" + "---|" * len(columns),
    ]
    by_id = {r["solution_id"]: r for r in records}
    for task_id in task_ids:
        cells = []
        for s, l in columns:
            r = by_id.get(f"{task_id}|{s}|{l}")
            cells.append("·" if r is None else "⚠️" if r["is_correct"] is None else "✅" if r["is_correct"] else "❌")
        lines.append(f"| {task_id} | " + " | ".join(cells) + " |")

    # Prompt generation: из чего складываются токены и время
    pg = [r for r in records if r["strategy"] == "prompt_generation"]
    if pg:
        lines += [
            "",
            "## Prompt generation: генерация промпта + решение",
            "",
            "| Задача | Reasoning | Токены: генерация | Токены: решение | Токены: итого | Время: генерация, с "
            "| Время: решение, с | Время: итого, с | Ответ в промпте? |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in pg:
            stages = {c["stage"]: c for c in r["calls"]}
            gen, sol = stages.get("prompt_generation"), stages.get("solve")
            leak = {True: "⚠️ да", False: "нет", None: "—"}[r["generated_prompt_leaks_answer"]]
            lines.append(
                f"| {r['task_id']} | {r['thinking_level']} "
                f"| {fmt(gen and gen['total_token_count'])} | {fmt(sol and sol['total_token_count'])} "
                f"| {fmt(r['total_token_count'])}{'*' if r['tokens_incomplete'] else ''} "
                f"| {fmt(gen and gen['latency_sec'])} | {fmt(sol and sol['latency_sec'])} | {fmt(r['latency_sec'])} "
                f"| {leak} |"
            )

    # Неверные ответы, ошибки разбора и API
    wrong = [r for r in records if r["is_correct"] is False]
    lines += ["", "## Неверные ответы", ""]
    if wrong:
        for r in wrong:
            lines.append(f"- **{r['solution_id']}** — ответ: `{one_line(r['answer']) if r['answer'] else '—'}`, "
                         f"эталон: `{r['expected_answer']}`; причина: {r['check_reason']}")
    else:
        lines.append("Нет.")

    lines += ["", "## Ошибки API", ""]
    errs = [r for r in records if r["api_error"] is not None]
    if errs:
        for r in errs:
            e = r["api_error"]
            lines.append(f"- **{r['solution_id']}** ({e['stage']}, попыток: {r['attempts']}) — "
                         f"{e['type']} {fmt(e['code'])}: {one_line(e['message'])}")
    else:
        lines.append("Нет.")

    retried = [r for r in records if r["retries"] > 0]
    lines += ["", "## Повторные попытки", ""]
    if retried:
        for r in retried:
            for c in r["calls"]:
                for a in c["attempts"]:
                    if a["error"]:
                        pause = f", пауза {a['retry_after_sec']} с" if a["retry_after_sec"] else ""
                        lines.append(f"- {r['solution_id']} / {c['stage']} / попытка {a['attempt']}: "
                                     f"{fmt(a['error']['code'])} {one_line(a['error']['message'])[:80]}{pause}")
    else:
        lines.append("Не было.")

    return "\n".join(lines) + "\n"
