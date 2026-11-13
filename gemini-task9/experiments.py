"""Эксперименты A (без сжатия) и B (со сжатием): сохранение фактов и расход токенов.

Оба эксперимента:
  1. берут ОДИН И ТОТ ЖЕ сценарий из data/dialogue.json (50 сообщений: 25 пользователя + 25 агента);
  2. «проигрывают» его через ContextManager (A — compression_enabled=False, B — True);
     в B при этом вызывается Gemini для создания/обновления summary — эти вызовы учитываются;
  3. задают одни и те же 10 вопросов из data/control_questions.json, каждый НЕЗАВИСИМО
     (remember=False: ответ на вопрос не попадает в контекст следующего);
  4. проверяют ответы по эталону data/facts.json. Эталон агенту и суммаризатору не передаётся.

Токены: count_tokens (до отправки) и usage_metadata (фактически, после ответа) — раздельно.
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
from context_manager import ContextManager, text_content
from token_stats import (
    Usage, compare_count_with_usage, estimate_cost, savings_percent, sum_known, sum_usage,
)
from utils import normalize_text, write_json

logger = logging.getLogger("experiments")

RESULT_FILES = {
    "A": "experiment_without_compression.json",
    "B": "experiment_with_compression.json",
    "comparison": "comparison.json",
    "summary": "summary.json",
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
# Запуск эксперимента
# ---------------------------------------------------------------------

def _prepare_state_dir(state_dir: Path) -> None:
    """Каждый эксперимент начинается с чистого состояния — эксперименты независимы."""
    if state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True)


def run_experiment(
    client: Any,
    *,
    compression: bool,
    state_dir: Path,
    dialogue: list[types.Content] | None = None,
    questions: list[dict] | None = None,
    facts: list[dict] | None = None,
    agent_kwargs: dict | None = None,
) -> dict:
    dialogue = dialogue if dialogue is not None else load_dialogue()
    questions = questions if questions is not None else load_questions()
    facts = facts if facts is not None else load_facts()
    validate_data(dialogue, questions, facts)
    facts_by_id = {f["id"]: f for f in facts}
    name = "B" if compression else "A"
    started = datetime.now()

    _prepare_state_dir(state_dir)
    context = ContextManager.from_dir(state_dir, compression_enabled=compression)
    agent = Agent(client, context, **(agent_kwargs or {}))
    counter = agent.counter

    # 1. Проигрываем сценарий: те же 50 сообщений; в B по пути создаётся summary.
    logger.info("Эксперимент %s: проигрываю %d сообщений", name, len(dialogue))
    compression_events = []
    for i in range(0, len(dialogue), 2):
        compression_events.extend(context.add_exchange(dialogue[i], dialogue[i + 1]))

    # 2. Состав контекста перед контрольными вопросами (count_tokens).
    def count(contents):
        return counter.count(contents)

    full_t, full_e = count(context.full_history)
    summary_t, summary_e = count(context.summary_contents())
    recent_t, recent_e = count(context.active)
    context_t, context_e = count(context.build_contents())
    context_breakdown = {
        "full_history_messages": len(context.full_history),
        "full_history_tokens": full_t,
        "summarized_messages": context.summary.summarized_upto,
        "summary_version": context.summary.version,
        "summary_tokens": summary_t,               # пара сообщений summary + подтверждение
        "recent_messages": len(context.active),
        "recent_tokens": recent_t,
        "request_context_tokens": context_t,       # summary + recent (без вопроса)
        "count_errors": [e for e in (full_e, summary_e, recent_e, context_e) if e],
    }

    # 3. Контрольные вопросы — независимо, история не меняется (remember=False).
    answers = []
    for q in questions:
        fact = facts_by_id[q["id"]]
        res = agent.ask(q["question"], remember=False)
        correct = res.ok and check_answer(res.text, fact)
        answers.append({
            "id": q["id"],
            "fact": fact["fact"],
            "category": fact.get("category", "core"),
            "kind": fact.get("kind", "fact"),
            "question": q["question"],
            "expected": fact["expected"],
            "answer": res.text,
            "status": res.status,
            "correct": correct,
            "request_messages": res.request_messages,
            "request_tokens_count_tokens": res.request_tokens,
            "usage": res.usage.to_dict(),
            "count_vs_usage": compare_count_with_usage(res.request_tokens, res.usage),
            "error": res.error,
        })
        logger.info("Вопрос %d: %s → %s", q["id"], res.status, "верно" if correct else "НЕВЕРНО")

    # 4. Токены: контрольные запросы и суммаризация — раздельно и вместе.
    control_usage = [Usage(**{k: v for k, v in a["usage"].items() if k in Usage.__dataclass_fields__})
                     for a in answers]
    summary_calls = [e for e in compression_events if e.meta.get("usage")]
    summary_usage = [Usage(**{k: v for k, v in e.meta["usage"].items() if k in Usage.__dataclass_fields__})
                     for e in summary_calls]
    control = sum_usage(control_usage)
    control["request_tokens_count_tokens"] = sum_known(a["request_tokens_count_tokens"] for a in answers)
    summarization = sum_usage(summary_usage)
    summarization["request_tokens_count_tokens"] = sum_known(e.meta.get("request_tokens") for e in summary_calls)
    summarization["failed_attempts"] = sum(1 for e in compression_events if not e.ok)
    total = {
        "input_tokens": control["input_tokens"] + summarization["input_tokens"],
        "output_tokens": control["output_tokens"] + summarization["output_tokens"],
        "total_tokens": control["total_tokens"] + summarization["total_tokens"],
        "complete": control["complete"] and summarization["complete"],
    }
    for block in (control, summarization, total):
        block["estimated_cost_usd"] = estimate_cost(block["input_tokens"], block["output_tokens"])

    correct = sum(a["correct"] for a in answers)

    # 5. Версии summary целиком (v1, v2, …) и офлайн-аудит итогового summary (без API).
    summary_versions = [{"version": u["version"], "start": u["start"], "end": u["end"], "text": u.get("text", "")}
                        for u in context.summary.updates]
    audit = None
    if compression:
        try:
            catalog = compression_audit.load_catalog()
            compression_audit.validate_catalog(catalog, dialogue)
            audit = compression_audit.audit_summary(
                catalog, context.summary.text, summarized_upto=context.summary.summarized_upto,
                versions=summary_versions, source={"file": "(этот прогон)", "started_at": started.isoformat(timespec="seconds"),
                                                   "model": agent.model})
        except compression_audit.AuditError as exc:
            logger.warning("Аудит summary не выполнен: %s", exc)

    return {
        "experiment": name,
        "title": "Со сжатием (summary + последние сообщения)" if compression else "Без сжатия (полная история)",
        "model": agent.model,
        "settings": {
            "compression": compression,
            "summary_every": context.summary_every,
            "keep_last": context.keep_last,
            "answer_thinking_level": agent.answer_thinking_level,
            "summary_thinking_level": agent.summary_thinking_level,
        },
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "state_dir": str(state_dir),
        "dialogue_messages": len(dialogue),
        "dialogue_fingerprint": compression_audit.dialogue_fingerprint(dialogue),
        "context": context_breakdown,
        "compression_events": [e.to_dict() for e in compression_events],
        "final_summary": context.summary.text,
        "summary_versions": summary_versions,
        "compression_audit": audit,
        "answers": answers,
        "facts_correct": correct,
        "facts_total": len(answers),
        "facts_retained_percent": round(correct / len(answers) * 100, 1) if answers else None,
        "facts_by_category": facts_by_category(answers),
        "tokens": {
            "control_questions": control,
            "summarization": summarization,
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
# Сравнение
# ---------------------------------------------------------------------

def compare(a: dict, b: dict) -> dict:
    if a["dialogue_messages"] != b["dialogue_messages"] or [x["question"] for x in a["answers"]] != [
            x["question"] for x in b["answers"]]:
        raise DataError("Эксперименты A и B выполнены на разных сценариях или вопросах — сравнение некорректно")
    ta, tb = a["tokens"], b["tokens"]
    ca, cb = ta["control_questions"], tb["control_questions"]
    n = len(a["answers"])

    per_question_saving = (ca["input_tokens"] - cb["input_tokens"]) / n if n else None
    summary_cost = tb["summarization"]["total_tokens"]
    total_saving_per_q = ((ca["total_tokens"] - cb["total_tokens"]) / n) if n else None
    break_even = (round(summary_cost / total_saving_per_q, 1)
                  if total_saving_per_q and total_saving_per_q > 0 else None)

    facts = [{
        "id": x["id"], "fact": x["fact"], "category": x.get("category", "core"), "expected": x["expected"],
        "A_correct": x["correct"], "A_answer": x["answer"],
        "B_correct": y["correct"], "B_answer": y["answer"],
    } for x, y in zip(a["answers"], b["answers"])]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": a["model"],
        "usage_metadata_complete": ta["total"]["complete"] and tb["total"]["complete"],
        "facts": facts,
        "facts_retained": {
            "A": {"correct": a["facts_correct"], "total": a["facts_total"], "percent": a["facts_retained_percent"]},
            "B": {"correct": b["facts_correct"], "total": b["facts_total"], "percent": b["facts_retained_percent"]},
            "lost_in_B": [f["id"] for f in facts if f["A_correct"] and not f["B_correct"]],
            "gained_in_B": [f["id"] for f in facts if not f["A_correct"] and f["B_correct"]],
            "by_category": {"A": facts_by_category(a["answers"]), "B": facts_by_category(b["answers"])},
        },
        "context": {
            "A": a["context"], "B": b["context"],
            "request_context_savings_percent_count_tokens": savings_percent(
                a["context"]["request_context_tokens"], b["context"]["request_context_tokens"]),
        },
        # 1) Только контрольные запросы (без затрат на создание summary).
        "control_requests": {
            "input_tokens": {"A": ca["input_tokens"], "B": cb["input_tokens"],
                             "savings_percent": savings_percent(ca["input_tokens"], cb["input_tokens"]),
                             "source": "usage_metadata.prompt_token_count (фактически)"},
            "input_tokens_count_tokens": {
                "A": ca["request_tokens_count_tokens"], "B": cb["request_tokens_count_tokens"],
                "savings_percent": savings_percent(ca["request_tokens_count_tokens"], cb["request_tokens_count_tokens"]),
                "source": "count_tokens до отправки (оценка, без system instruction)"},
            "output_tokens": {"A": ca["output_tokens"], "B": cb["output_tokens"]},
            "total_tokens": {"A": ca["total_tokens"], "B": cb["total_tokens"],
                             "savings_percent": savings_percent(ca["total_tokens"], cb["total_tokens"]),
                             "source": "usage_metadata (фактически)"},
            "input_saving_per_question": per_question_saving,
        },
        # 2) Весь эксперимент: в B добавлены все вызовы суммаризации.
        "overall_with_summarization": {
            "A_total_tokens": ta["total"]["total_tokens"],
            "B_control_tokens": cb["total_tokens"],
            "B_summarization_tokens": summary_cost,
            "B_summarization_calls": tb["summarization"]["calls"],
            "B_total_tokens": tb["total"]["total_tokens"],
            "savings_percent": savings_percent(ta["total"]["total_tokens"], tb["total"]["total_tokens"]),
            "estimated_cost_usd": {"A": ta["total"]["estimated_cost_usd"], "B": tb["total"]["estimated_cost_usd"]},
            "source": "usage_metadata (фактически); стоимость — расчёт по тарифу config.PRICING",
        },
        "break_even_questions": {
            "value": break_even,
            "note": "РАСЧЁТ: сколько запросов к сжатому контексту окупают затраты на создание summary "
                    "(затраты на summary / экономия токенов на одном запросе). Не измерение.",
        },
    }


def build_summary(a: dict, b: dict, comparison: dict) -> dict:
    """Краткая сводка для README и быстрого просмотра."""
    cr = comparison["control_requests"]
    ov = comparison["overall_with_summarization"]
    return {
        "generated_at": comparison["generated_at"],
        "model": comparison["model"],
        "settings": b["settings"],
        "facts_retained_percent": {"A_without_compression": a["facts_retained_percent"],
                                   "B_with_compression": b["facts_retained_percent"]},
        "facts_lost_in_B": comparison["facts_retained"]["lost_in_B"],
        "facts_by_category": comparison["facts_retained"]["by_category"],
        "summary_audit": (b.get("compression_audit") or {}).get("totals"),
        "context_tokens_before_questions": {
            "A_full_history": a["context"]["request_context_tokens"],
            "B_summary_plus_recent": b["context"]["request_context_tokens"],
            "B_summary": b["context"]["summary_tokens"],
            "B_recent": b["context"]["recent_tokens"],
        },
        "savings_control_requests_input_percent": cr["input_tokens"]["savings_percent"],
        "savings_control_requests_total_percent": cr["total_tokens"]["savings_percent"],
        "savings_overall_with_summarization_percent": ov["savings_percent"],
        "summarization_tokens": ov["B_summarization_tokens"],
        "break_even_questions_estimate": comparison["break_even_questions"]["value"],
        "usage_metadata_complete": comparison["usage_metadata_complete"],
        "final_summary": b["final_summary"],
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


def compare_saved(results_dir: Path) -> tuple[dict, dict] | None:
    """Построить comparison.json и summary.json из сохранённых результатов A и B (без API)."""
    a, b = load_result(results_dir, "A"), load_result(results_dir, "B")
    if a is None or b is None:
        return None
    comparison = compare(a, b)
    summary = build_summary(a, b, comparison)
    save_results(results_dir, "comparison", comparison)
    save_results(results_dir, "summary", summary)
    return comparison, summary


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


def markdown_report(a: dict, b: dict, comparison: dict) -> str:
    def mark(ok):
        return "✅" if ok else "❌"

    def short(text, limit=70):
        text = " ".join((text or "").split())
        return (text[:limit] + "…") if len(text) > limit else text

    cat_ru = {"core": "основной", "detail": "деталь"}
    lines = [f"### Проверка {len(comparison['facts'])} контрольных вопросов", "",
             "| # | Факт | Тип | Эталон | A: без сжатия | B: со сжатием | Ответ B |",
             "|---|---|---|---|---|---|---|"]
    for f in comparison["facts"]:
        lines.append(f"| {f['id']} | {f['fact']} | {cat_ru.get(f.get('category', 'core'), f.get('category'))} | "
                     f"{f['expected']} | {mark(f['A_correct'])} | "
                     f"{mark(f['B_correct'])} | {short(f['B_answer']).replace('|', '/')} |")
    fr = comparison["facts_retained"]
    lines += ["", f"Сохранено фактов: **A — {fr['A']['correct']}/{fr['A']['total']} ({_fmt(fr['A']['percent'])}%)**, "
              f"**B — {fr['B']['correct']}/{fr['B']['total']} ({_fmt(fr['B']['percent'])}%)**."]
    for cat, label in (("core", "основные факты"), ("detail", "дополнительные детали")):
        ca_, cb_ = fr.get("by_category", {}).get("A", {}).get(cat), fr.get("by_category", {}).get("B", {}).get(cat)
        if ca_ and cb_:
            lines.append(f"- {label}: A {ca_['correct']}/{ca_['total']}, B {cb_['correct']}/{cb_['total']}")
    lines.append("")

    ca, cb = a["context"], b["context"]
    lines += ["### Состав контекста перед вопросами (count_tokens)", "",
              "| Показатель | A | B |", "|---|---|---|",
              f"| Сообщений в полной истории | {ca['full_history_messages']} | {cb['full_history_messages']} |",
              f"| Токены полной истории | {_fmt(ca['full_history_tokens'])} | {_fmt(cb['full_history_tokens'])} |",
              f"| Сжато сообщений / версия summary | {ca['summarized_messages']} / {ca['summary_version']} | "
              f"{cb['summarized_messages']} / {cb['summary_version']} |",
              f"| Токены summary | {_fmt(ca['summary_tokens'])} | {_fmt(cb['summary_tokens'])} |",
              f"| Последних сообщений / токены | {ca['recent_messages']} / {_fmt(ca['recent_tokens'])} | "
              f"{cb['recent_messages']} / {_fmt(cb['recent_tokens'])} |",
              f"| Контекст запроса (summary + последние) | {_fmt(ca['request_context_tokens'])} | "
              f"{_fmt(cb['request_context_tokens'])} |", ""]

    ta, tb = a["tokens"], b["tokens"]
    cr, ov = comparison["control_requests"], comparison["overall_with_summarization"]
    n = len(a["answers"])

    def usd(v):
        return "н/д" if v is None else f"${v:.5f}"

    lines += ["### Расход токенов (usage_metadata — фактически)", "",
              "| Статья | A | B | Экономия B |", "|---|---|---|---|",
              f"| Вход {n} контрольных запросов (usage_metadata) | {_fmt(cr['input_tokens']['A'])} | "
              f"{_fmt(cr['input_tokens']['B'])} | {_fmt(cr['input_tokens']['savings_percent'])}% |",
              f"| Вход {n} контрольных запросов (count_tokens, оценка) | {_fmt(cr['input_tokens_count_tokens']['A'])} | "
              f"{_fmt(cr['input_tokens_count_tokens']['B'])} | {_fmt(cr['input_tokens_count_tokens']['savings_percent'])}% |",
              f"| Выход (ответы, вкл. thinking) | {_fmt(cr['output_tokens']['A'])} | {_fmt(cr['output_tokens']['B'])} | — |",
              f"| **Итого контрольные запросы** | {_fmt(cr['total_tokens']['A'])} | {_fmt(cr['total_tokens']['B'])} | "
              f"**{_fmt(cr['total_tokens']['savings_percent'])}%** |",
              f"| Суммаризация: вход / выход ({tb['summarization']['calls']} вызова) | 0 | "
              f"{_fmt(tb['summarization']['input_tokens'])} / {_fmt(tb['summarization']['output_tokens'])} | — |",
              f"| **Итого с учётом суммаризации** | {_fmt(ov['A_total_tokens'])} | {_fmt(ov['B_total_tokens'])} | "
              f"**{_fmt(ov['savings_percent'])}%** |",
              f"| Расчётная стоимость (оценка по тарифу, не счёт Google) | {usd(ov['estimated_cost_usd']['A'])} | "
              f"{usd(ov['estimated_cost_usd']['B'])} | — |", "",
              f"Точка окупаемости summary (расчёт): {_fmt(comparison['break_even_questions']['value'])} запросов."]
    return "\n".join(lines)
