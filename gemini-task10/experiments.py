"""Сравнительный эксперимент Дня 10: Sliding Window vs Sticky Facts vs Summary.

Для КАЖДОЙ стратегии (с чистого состояния):
  1. проигрывается один и тот же сценарий data/dialogue.json (50 сообщений: 25 пользователя + 25 агента)
     — в порядке реального диалога: before_user_message (facts: вызов Gemini) → add_exchange
     (summary: сжатие по SUMMARY_EVERY/KEEP_LAST);
  2. задаются 15 контрольных вопросов data/control_questions.json, каждый НЕЗАВИСИМО (remember=False:
     ни история, ни память не меняются);
  3. ответы проверяются по эталону data/facts.json (правила Дня 9, эталон модели не передаётся);
  4. офлайн-аудит 36 деталей data/summary_details.json — по ТЕКСТУ КОНТЕКСТА, который модель получает
     перед вопросом (memory_text: summary/facts + последние сообщения). Для summary дополнительно —
     аудит в формате Дня 9 (только текст summary).

Токены и вызовы раздельно: processing (служебные вызовы при обработке диалога: summary или
обновление facts) и control_questions (ответы на 15 вопросов). Источник — usage_metadata;
count_tokens (бесплатно) — отдельной оценкой. Стоимость — расчёт по config.PRICING.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from google.genai import types

import config
import compression_audit
from agent import Agent
from context_manager import content_text, text_content
from strategies import CallRecord, make_strategy
from token_stats import (
    Usage, compare_count_with_usage, estimate_cost, savings_percent, sum_known, sum_usage,
)
from utils import normalize_text, write_json

logger = logging.getLogger("experiments")

STRATEGY_TITLES = {
    "sliding": "Sliding Window (последние N сообщений)",
    "facts": "Sticky Facts (key-value память + последние N)",
    "summary": "Summary (День 9: summary + последние KEEP_LAST)",
}
RESULT_FILES = {
    "sliding": "experiment_sliding.json",
    "facts": "experiment_facts.json",
    "summary": "experiment_summary.json",
    "comparison": "comparison.json",
}


class DataError(Exception):
    """Файлы сценария, вопросов или эталона некорректны."""


# ---------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------

def _load(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataError(f"Файл не найден: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataError(f"{path}: некорректный JSON (строка {exc.lineno}): {exc.msg}") from exc


def load_dialogue(path: Path = config.DIALOGUE_FILE) -> list[types.Content]:
    items = _load(path).get("messages")
    if not isinstance(items, list) or not items:
        raise DataError(f"{path}: нет списка 'messages'")
    if len(items) % 2:
        raise DataError(f"{path}: нечётное число сообщений ({len(items)}) — ожидаются пары вопрос/ответ")
    dialogue = []
    for i, item in enumerate(items):
        expected_role = "user" if i % 2 == 0 else "model"
        if item.get("role") != expected_role or not str(item.get("text", "")).strip():
            raise DataError(f"{path}: сообщение #{i} должно иметь роль {expected_role} и непустой текст")
        dialogue.append(text_content(item["role"], item["text"]))
    return dialogue


def load_questions(path: Path = config.QUESTIONS_FILE) -> list[dict]:
    questions = _load(path).get("questions")
    if not isinstance(questions, list) or not questions:
        raise DataError(f"{path}: нет списка 'questions'")
    for q in questions:
        if not isinstance(q.get("id"), int) or not str(q.get("question", "")).strip():
            raise DataError(f"{path}: у каждого вопроса должны быть id и question")
    return questions


def load_facts(path: Path = config.FACTS_FILE) -> list[dict]:
    facts = _load(path).get("facts")
    if not isinstance(facts, list) or not facts:
        raise DataError(f"{path}: нет списка 'facts'")
    for f in facts:
        if not f.get("accept_patterns"):
            raise DataError(f"{path}: у факта {f.get('id')} нет accept_patterns")
        for p in f["accept_patterns"] + f.get("reject_patterns", []):
            try:
                re.compile(p)
            except re.error as exc:
                raise DataError(f"{path}: факт {f.get('id')}: неверное регулярное выражение {p!r}: {exc}") from exc
    return facts


def validate_data(dialogue: list[types.Content], questions: list[dict], facts: list[dict]) -> None:
    """Эталон должен подтверждаться диалогом: цитата из указанного сообщения указанной роли.

    Это защищает от «придуманных» эталонных ответов: каждый факт привязан к конкретной реплике.
    """
    q_ids = [q["id"] for q in questions]
    f_ids = [f["id"] for f in facts]
    if sorted(q_ids) != sorted(f_ids) or len(set(q_ids)) != len(q_ids):
        raise DataError(f"id вопросов {q_ids} и фактов {f_ids} не совпадают")
    texts = [(c.role, "\n".join(p.text or "" for p in c.parts)) for c in dialogue]
    for f in facts:
        if not f.get("evidence"):
            raise DataError(f"факт {f['id']}: нет evidence (цитаты из диалога)")
        for ev in f["evidence"]:
            n = ev["message"]
            if not 1 <= n <= len(texts):
                raise DataError(f"факт {f['id']}: сообщения #{n} нет в диалоге")
            role, text = texts[n - 1]
            if role != ev["role"] or ev["quote"] not in text:
                raise DataError(f"факт {f['id']}: цитата «{ev['quote']}» не найдена в сообщении #{n} ({ev['role']})")
        for rule in f.get("absent_in", []):
            hits = [i + 1 for i, (role, text) in enumerate(texts)
                    if role == rule["role"] and re.search(rule["pattern"], normalize_answer(text))]
            if hits:
                raise DataError(f"факт {f['id']}: «{rule['pattern']}» встречается в сообщениях {rule['role']} {hits}")
        # Сам эталонный ответ должен проходить собственную проверку.
        if not check_answer(f["expected"], f):
            raise DataError(f"факт {f['id']}: эталон «{f['expected']}» не проходит собственные accept_patterns")


# ---------------------------------------------------------------------
# Проверка ответа
# ---------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """Нижний регистр, ё→е, разделители тысяч убраны: «18 500 000» → «18500000»."""
    return normalize_text(text)


def check_answer(answer: str, fact: dict) -> bool:
    """Верно, если совпал хотя бы один accept_pattern и ни один reject_pattern."""
    normalized = normalize_answer(answer or "")
    if any(re.search(p, normalized) for p in fact.get("reject_patterns", [])):
        return False
    return any(re.search(p, normalized) for p in fact["accept_patterns"])


# ---------------------------------------------------------------------
# Запуск эксперимента для одной стратегии
# ---------------------------------------------------------------------

def _prepare_state_dir(state_dir: Path) -> None:
    """Каждый прогон начинается с чистого состояния — стратегии независимы."""
    if state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True)


def _usage_of(d: dict) -> Usage:
    return Usage(**{k: v for k, v in (d or {}).items() if k in Usage.__dataclass_fields__})


def cost_block(usages: list[Usage], request_tokens: list[int | None]) -> dict:
    block = sum_usage(usages)
    block["request_tokens_count_tokens"] = sum_known(request_tokens) if request_tokens else 0
    block["estimated_cost_usd"] = estimate_cost(block["input_tokens"], block["output_tokens"])
    return block


def processing_costs(calls: list[CallRecord]) -> dict:
    """Служебные вызовы при обработке диалога — всего и по видам (summary / facts_update)."""
    out = cost_block([c.usage for c in calls], [c.request_tokens for c in calls])
    out["failed_calls"] = sum(1 for c in calls if not c.ok)
    out["by_kind"] = {kind: {**cost_block([c.usage for c in calls if c.kind == kind],
                                          [c.request_tokens for c in calls if c.kind == kind]),
                             "failed_calls": sum(1 for c in calls if c.kind == kind and not c.ok)}
                      for kind in sorted({c.kind for c in calls})}
    return out


def run_strategy_experiment(
    client: Any,
    strategy_name: str,
    *,
    state_dir: Path,
    dialogue: list[types.Content] | None = None,
    questions: list[dict] | None = None,
    facts: list[dict] | None = None,
    agent_kwargs: dict | None = None,
    strategy_kwargs: dict | None = None,
) -> dict:
    dialogue = dialogue if dialogue is not None else load_dialogue()
    questions = questions if questions is not None else load_questions()
    facts = facts if facts is not None else load_facts()
    validate_data(dialogue, questions, facts)
    facts_by_id = {f["id"]: f for f in facts}
    started = datetime.now()

    _prepare_state_dir(state_dir)
    strategy = make_strategy(strategy_name, state_dir, **(strategy_kwargs or {}))
    agent = Agent(client, strategy, **(agent_kwargs or {}))
    counter = agent.counter

    # 1. Обработка диалога: те же 50 сообщений, в порядке реального диалога.
    logger.info("Стратегия %s: проигрываю %d сообщений", strategy_name, len(dialogue))
    processing: list[CallRecord] = []
    for i in range(0, len(dialogue), 2):
        processing += strategy.before_user_message(dialogue[i])
        processing += strategy.add_exchange(dialogue[i], dialogue[i + 1])

    # 2. Состав контекста перед контрольными вопросами (count_tokens, бесплатно).
    memory = strategy.memory_contents()
    recent = strategy.window() if hasattr(strategy, "window") else list(strategy.context.active)
    extra = memory[: len(memory) - len(recent)]           # summary-пара или facts-пара
    full_t, full_e = counter.count(strategy.full_history)
    extra_t, extra_e = counter.count(extra)
    recent_t, recent_e = counter.count(recent)
    context_t, context_e = counter.count(memory)
    context_breakdown = {
        "full_history_messages": len(strategy.full_history),
        "full_history_tokens": full_t,
        "memory_block_tokens": extra_t,          # summary или facts (пара сообщений)
        "recent_messages": len(recent),
        "recent_tokens": recent_t,
        "request_context_tokens": context_t,     # memory_block + recent (без вопроса)
        "count_errors": [e for e in (full_e, extra_e, recent_e, context_e) if e],
    }

    # 3. Контрольные вопросы — независимо (remember=False).
    answers = []
    question_service: list[CallRecord] = []
    for q in questions:
        fact = facts_by_id[q["id"]]
        res = agent.ask(q["question"], remember=False)
        question_service += res.service_calls        # summary: отложенное сжатие (обычно пусто)
        correct = res.ok and check_answer(res.text, fact)
        answers.append({
            "id": q["id"], "fact": fact["fact"], "category": fact.get("category", "core"),
            "kind": fact.get("kind", "fact"), "question": q["question"], "expected": fact["expected"],
            "answer": res.text, "status": res.status, "correct": correct,
            "request_messages": res.request_messages,
            "request_tokens_count_tokens": res.request_tokens,
            "usage": res.usage.to_dict(),
            "count_vs_usage": compare_count_with_usage(res.request_tokens, res.usage),
            "error": res.error,
        })
        logger.info("Вопрос %d: %s → %s", q["id"], res.status, "верно" if correct else "НЕВЕРНО")

    # 4. Затраты: обработка диалога и ответы — раздельно, затем вместе.
    processing += question_service
    proc = processing_costs(processing)
    control = cost_block([_usage_of(a["usage"]) for a in answers], [a["request_tokens_count_tokens"] for a in answers])
    total = {
        "generate_content_calls": proc["calls"] + control["calls"],
        "input_tokens": proc["input_tokens"] + control["input_tokens"],
        "output_tokens": proc["output_tokens"] + control["output_tokens"],
        "total_tokens": proc["total_tokens"] + control["total_tokens"],
        "complete": proc["complete"] and control["complete"],
    }
    total["estimated_cost_usd"] = estimate_cost(total["input_tokens"], total["output_tokens"])

    # 5. Офлайн-аудит 36 деталей по тексту контекста (без API).
    audit, summary_audit = None, None
    memory_text = strategy.memory_text()
    try:
        catalog = compression_audit.load_catalog()
        compression_audit.validate_catalog(catalog, dialogue)
        source = {"file": "(этот прогон)", "strategy": strategy_name, "started_at": started.isoformat(timespec="seconds"),
                  "model": agent.model}
        audit = compression_audit.audit_context(catalog, memory_text, len(dialogue), source=source)
        if strategy_name == "summary":
            s = strategy.context.summary
            versions = [{"version": u["version"], "end": u["end"], "text": u.get("text", "")} for u in s.updates]
            summary_audit = compression_audit.audit_summary(catalog, s.text, summarized_upto=s.summarized_upto,
                                                            versions=versions, source=source)
    except compression_audit.AuditError as exc:
        logger.warning("Аудит не выполнен: %s", exc)

    correct = sum(a["correct"] for a in answers)
    extra_state: dict = {}
    if strategy_name == "summary":
        s = strategy.context.summary
        extra_state = {"final_summary": s.text, "summary_version": s.version, "summarized_messages": s.summarized_upto,
                       "summary_versions": [{"version": u["version"], "start": u["start"], "end": u["end"],
                                             "text": u.get("text", "")} for u in s.updates]}
    elif strategy_name == "facts":
        f = strategy.facts
        journal = f.journal
        extra_state = {
            "final_facts": {k: v.to_dict() for k, v in f.facts.items()},
            "facts_by_kind": {k: sum(1 for v in f.facts.values() if v.kind == k)
                              for k in ("user", "accepted_proposal", "assistant_proposal")},
            "facts_version": f.version,
            "operations": {key: sum(len(e.get(key, [])) for e in journal)
                           for key in ("applied", "rejected", "conflicts", "unchanged")},
            "rejected_operations": [r for e in journal for r in e.get("rejected", [])],
        }

    return {
        "strategy": strategy_name,
        "title": STRATEGY_TITLES.get(strategy_name, strategy_name),
        "model": agent.model,
        "settings": {**strategy.settings(),
                     "answer_thinking_level": agent.answer_thinking_level,
                     "summary_thinking_level": agent.summary_thinking_level,
                     "facts_thinking_level": agent.facts_thinking_level,
                     "max_output_tokens": agent.max_output_tokens,
                     "answer_system_instruction_chars": len(agent.system_instruction)},
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "state_dir": str(state_dir),
        "dialogue_messages": len(dialogue),
        "dialogue_fingerprint": compression_audit.dialogue_fingerprint(dialogue),
        "context": context_breakdown,
        "memory_text": memory_text,
        **extra_state,
        "service_calls": [c.to_dict() for c in processing],
        "detail_audit": audit,
        "summary_only_audit": summary_audit,
        "answers": answers,
        "facts_correct": correct,
        "facts_total": len(answers),
        "facts_retained_percent": round(correct / len(answers) * 100, 1) if answers else None,
        "facts_by_category": facts_by_category(answers),
        "tokens": {
            "processing": proc,
            "control_questions": control,
            "total": total,
            "source": "usage_metadata (фактически); request_tokens_count_tokens — count_tokens до отправки",
            "count_tokens_calls": counter.calls,
        },
    }


def facts_by_category(answers: list[dict]) -> dict:
    """Процент верных ответов отдельно для основных фактов (core) и дополнительных деталей (detail)."""
    out = {}
    for cat in sorted({a.get("category", "core") for a in answers}):
        items = [a for a in answers if a.get("category", "core") == cat]
        ok = sum(a["correct"] for a in items)
        out[cat] = {"correct": ok, "total": len(items), "percent": round(ok / len(items) * 100, 1)}
    return out


# ---------------------------------------------------------------------
# Сравнение стратегий
# ---------------------------------------------------------------------

def compare_strategies(results: dict[str, dict], complexity: dict | None = None) -> dict:
    names = [n for n in config.STRATEGIES if n in results]
    if not names:
        raise DataError("Нет результатов для сравнения")
    base = results[names[0]]
    for n in names[1:]:
        r = results[n]
        if (r["dialogue_fingerprint"] != base["dialogue_fingerprint"]
                or [a["question"] for a in r["answers"]] != [a["question"] for a in base["answers"]]):
            raise DataError(f"Стратегия {n} прогнана на другом сценарии или вопросах — сравнение некорректно")
        if r["model"] != base["model"]:
            raise DataError(f"Стратегия {n}: другая модель ({r['model']} ≠ {base['model']})")

    rows = []
    for n in names:
        r = results[n]
        audit = r.get("detail_audit") or {}
        totals = audit.get("totals") or {}
        details = audit.get("details") or []
        t = r["tokens"]
        rows.append({
            "strategy": n,
            "title": r["title"],
            "details_preserved": totals.get("preserved"),
            "details_lost": totals.get("lost"),
            "details_distorted": totals.get("distorted"),
            "details_checked": totals.get("checked"),
            "lost_ids": [d["id"] for d in details if d["status"] == "lost"],
            "distorted_ids": [d["id"] for d in details if d["status"] == "distorted"],
            "attribution_issues": audit.get("attribution_issues"),
            "answers_correct": r["facts_correct"],
            "answers_total": r["facts_total"],
            "answers_by_category": r["facts_by_category"],
            "context_tokens_before_questions": r["context"]["request_context_tokens"],
            "processing": {k: t["processing"][k] for k in ("calls", "input_tokens", "output_tokens", "total_tokens",
                                                           "estimated_cost_usd", "failed_calls")},
            "processing_by_kind": t["processing"]["by_kind"],
            "control": {k: t["control_questions"][k] for k in ("calls", "input_tokens", "output_tokens",
                                                               "total_tokens", "estimated_cost_usd")},
            "total": t["total"],
            "count_tokens_calls": t["count_tokens_calls"],
            "usage_complete": t["total"]["complete"],
            "complexity": (complexity or {}).get(n),
        })
    per_question = [{
        "id": a["id"], "fact": a["fact"], "category": a["category"], "expected": a["expected"],
        **{f"{n}_correct": results[n]["answers"][i]["correct"] for n in names},
        **{f"{n}_answer": results[n]["answers"][i]["answer"] for n in names},
    } for i, a in enumerate(base["answers"])]
    per_detail = []
    audits = {n: {d["id"]: d for d in (results[n].get("detail_audit") or {}).get("details", [])} for n in names}
    for d in (results[names[0]].get("detail_audit") or {}).get("details", []):
        per_detail.append({"id": d["id"], "detail": d["detail"], "importance": d["importance"], "kind": d["kind"],
                           **{n: audits[n].get(d["id"], {}).get("status") for n in names}})
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": base["model"],
        "strategies": names,
        "table": rows,
        "per_question": per_question,
        "per_detail": per_detail,
        "notes": [
            "Деталей сохранено — офлайн-аудит 36 деталей по тексту контекста перед вопросом "
            "(блок памяти + последние сообщения), регулярные выражения data/summary_details.json.",
            "Токены — usage_metadata (фактически). Стоимость — расчёт по config.PRICING, не счёт Google.",
            "Вызовы модели — generate_content; count_tokens (бесплатный) указан отдельно.",
        ],
    }


def save_results(results_dir: Path, name: str, data: dict) -> Path:
    path = Path(results_dir) / RESULT_FILES[name]
    write_json(path, data)
    return path


def load_result(results_dir: Path, name: str) -> dict | None:
    path = Path(results_dir) / RESULT_FILES[name]
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compare_saved(results_dir: Path) -> dict | None:
    """comparison.json из сохранённых результатов стратегий (без API)."""
    import complexity
    results = {n: r for n in config.STRATEGIES if (r := load_result(results_dir, n)) is not None}
    if not results:
        return None
    comparison = compare_strategies(results, complexity.measure_all())
    save_results(results_dir, "comparison", comparison)
    return comparison


# ---------------------------------------------------------------------
# Отчёт в Markdown (из сохранённых JSON)
# ---------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None:
        return "н/д"
    if isinstance(v, float):
        return f"{v:.2f}"
    if isinstance(v, int):
        return f"{v:,}".replace(",", " ")
    return str(v)


def _usd(v) -> str:
    return "н/д" if v is None else f"${v:.5f}"


def markdown_report(comparison: dict) -> str:
    rows = comparison["table"]
    names = comparison["strategies"]
    lines = ["### Итоговая таблица", "",
             "| Стратегия | Деталей сохранено (из 36) | Ответов верно (из 15) | Токены: обработка диалога | "
             "Токены: контрольные вопросы | Токены всего | Вызовы модели (обработка + вопросы) | "
             "Стоимость (расчёт) | Сложность |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c = r.get("complexity") or {}
        cx = (f"{c.get('loc', 'н/д')} строк, {c.get('components', 'н/д')} компон., "
              f"{c.get('extra_model_call_types', 'н/д')} тип(а) служебн. вызовов") if c else "н/д"
        lines.append(
            f"| {r['strategy']} | {_fmt(r['details_preserved'])} | {r['answers_correct']}/{r['answers_total']} | "
            f"{_fmt(r['processing']['total_tokens'])} | {_fmt(r['control']['total_tokens'])} | "
            f"{_fmt(r['total']['total_tokens'])} | {r['processing']['calls']} + {r['control']['calls']} = "
            f"{r['total']['generate_content_calls']} | {_usd(r['total']['estimated_cost_usd'])} | {cx} |")

    lines += ["", "### Затраты подробно (usage_metadata)", "",
              "| Стратегия | Статья | Вызовы | Вход | Выход (вкл. thinking) | Всего | Стоимость |",
              "|---|---|---|---|---|---|---|"]
    kind_ru = {"summary": "суммаризация", "facts_update": "обновление facts"}
    for r in rows:
        for kind, b in r["processing_by_kind"].items():
            lines.append(f"| {r['strategy']} | обработка: {kind_ru.get(kind, kind)} | {b['calls']} "
                         f"(ошибок {b['failed_calls']}) | {_fmt(b['input_tokens'])} | {_fmt(b['output_tokens'])} | "
                         f"{_fmt(b['total_tokens'])} | {_usd(b['estimated_cost_usd'])} |")
        if not r["processing_by_kind"]:
            lines.append(f"| {r['strategy']} | обработка: служебных вызовов нет | 0 | 0 | 0 | 0 | $0.00000 |")
        c = r["control"]
        lines.append(f"| {r['strategy']} | контрольные вопросы | {c['calls']} | {_fmt(c['input_tokens'])} | "
                     f"{_fmt(c['output_tokens'])} | {_fmt(c['total_tokens'])} | {_usd(c['estimated_cost_usd'])} |")
        t = r["total"]
        lines.append(f"| **{r['strategy']}** | **итого** | **{t['generate_content_calls']}** | "
                     f"**{_fmt(t['input_tokens'])}** | **{_fmt(t['output_tokens'])}** | **{_fmt(t['total_tokens'])}** | "
                     f"**{_usd(t['estimated_cost_usd'])}** |")
    lines += ["", "Контекст перед вопросом (count_tokens, без вопроса): " +
              ", ".join(f"{r['strategy']} — {_fmt(r['context_tokens_before_questions'])}" for r in rows) + "."]

    lines += ["", "### Контрольные вопросы", "", "| # | Факт | Эталон | " + " | ".join(names) + " |",
              "|---|---|---|" + "---|" * len(names)]
    for q in comparison["per_question"]:
        lines.append(f"| {q['id']} | {q['fact']} | {q['expected']} | " +
                     " | ".join("✅" if q[f"{n}_correct"] else "❌" for n in names) + " |")

    mark = {"preserved": "✅", "lost": "❌", "distorted": "⚠️", None: "·"}
    lines += ["", "### Детали диалога (36)", "", "| ID | Деталь | Важность | " + " | ".join(names) + " |",
              "|---|---|---|" + "---|" * len(names)]
    for d in comparison["per_detail"]:
        lines.append(f"| {d['id']} | {d['detail']} | {d['importance']} | " +
                     " | ".join(mark.get(d[n], d[n]) for n in names) + " |")
    lines += ["", "Потерянные / искажённые детали:"]
    for r in rows:
        lines.append(f"- **{r['strategy']}**: потеряны {', '.join(r['lost_ids']) or '—'}; "
                     f"искажены {', '.join(r['distorted_ids']) or '—'}; "
                     f"проблемы авторства {', '.join(r['attribution_issues'] or []) or '—'}")
    return "\n".join(lines)
