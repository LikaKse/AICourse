"""Сравнение профилей и проверка устойчивости; запуск с настроенным GEMINI_API_KEY.

Каждый прогон использует СВЕЖЕЕ временное состояние: старые results/personalization/state_*
не читаются. Ошибки Gemini API помечаются как failed, не оцениваются и не записываются
в short-term. Отчёт строится из тех же оценок, что сохранены в JSON.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import config
from agent import Agent
from main import confirm, make_client
from memory import MemoryManager
from memory.storage import text_content
from utils import atomic_write_text, redact_secrets, write_json

DEFAULT_OUTPUT = Path("results/personalization")
RESULT_JSON = "personalization_experiment.json"
RESULT_REPORT = "report.md"
ASSESSMENT_VERSION = "2026-11-18"
ASSESSMENT_CRITERIA = (
    "длина: «кратко» — не более 60 слов, «подробно» — не менее 100 слов "
    "(считается проза без fenced code); код: fenced block или строки синтаксиса; "
    "стиль: явные маркеры, нейтральный ответ не оценён (None)"
)

QUESTIONS = [
    "Что такое рекурсия? Объясни и покажи короткий пример на Python.",
    "Как отсортировать список чисел в Python? Покажи код.",
    "Чем список отличается от кортежа в Python?",
    "Как обработать исключение в Python? Покажи пример.",
    "Что такое HTTP API? Объясни кратко.",
]
STABILITY_QUESTIONS = [f"Вопрос {i}: объясни понятие функции в Python и приведи пример." for i in range(1, 11)]
PROFILES = {
    "A": {"length": "кратко", "code": "показывать", "style": "неформальный"},
    "B": {"length": "подробно", "code": "не показывать", "style": "формальный"},
    "C": {"length": "кратко", "code": "не показывать", "style": "формальный"},
}
REQUIRED_PREFS = ("length", "code", "style")


class ResultsExistError(Exception):
    """Результаты уже есть — без --overwrite они не перезаписываются."""


def assess(text: str, prefs: dict) -> dict:
    prose = re.sub(r"```.*?```", " ", text, flags=re.S)
    words = re.findall(r"\w+", prose, re.UNICODE)
    has_code = bool(re.search(r"```.*?```", text, re.S) or re.search(
        r"(?m)^\s*(?:def\s+\w+\s*\(|return\b|print\s*\(|import\s+\w+|from\s+\w+\s+import|if\s+.+:)", text))
    low = prose.lower()
    formal_markers = re.findall(r"\b(вы|вам|ваш|рекомендуется|следует|представляет собой|является|таким образом|в соответствии|необходимо|определяется|предусмотрен)\b", low)
    informal_markers = re.findall(r"\b(ты|тебе|твой|давай|смотри|круто|понятно|держи|проще говоря|используй|попробуй|если хочешь)\b", low)
    formal, informal = bool(formal_markers), bool(informal_markers)
    if prefs["style"] == "неформальный":
        style_ok = True if informal and not formal else False if formal and not informal else None
    else:
        style_ok = True if formal and not informal else False if informal and not formal else None
    checks = {
        "length": len(words) <= 60 if prefs["length"] == "кратко" else len(words) >= 100,
        "code": has_code if prefs["code"] == "показывать" else not has_code,
        "style": style_ok,
    }
    return {"checks": checks, "satisfied": sum(value is True for value in checks.values()), "total": len(checks),
            "unverified": [key for key, value in checks.items() if value is None], "word_count": len(words),
            "has_code": has_code, "style_markers": {"formal": formal_markers, "informal": informal_markers}}


def seed(agent: Agent, prefs: dict) -> None:
    for key, value in prefs.items():
        agent.profile.update(key, value, source="experiment_seed", confidence=1.0,
                             evidence="Заданный экспериментальный профиль")


# ---------------------------------------------------------------------
# Вспомогательные функции эксперимента
# ---------------------------------------------------------------------

def planned_requests() -> int:
    return len(PROFILES) * len(QUESTIONS) + len(STABILITY_QUESTIONS)


def _ensure_can_write(output: Path, overwrite: bool) -> None:
    output = Path(output)
    existing = [p for p in (output / RESULT_JSON, output / RESULT_REPORT) if p.exists()]
    if existing and not overwrite:
        raise ResultsExistError(
            f"В {output} уже есть результаты ({', '.join(p.name for p in existing)}). "
            "Укажите --overwrite, если хотите заменить их."
        )


def _failure_row(question_no: int, question: str, answer) -> dict:
    return {
        "question_no": question_no,
        "question": question,
        "status": "failed",
        "answer": None,
        "assessment": None,
        "tokens": answer.tokens,
        "error_status": answer.status,
        "error": answer.error or answer.text or "неизвестная ошибка",
    }


def _ok_row(question_no: int, question: str, answer, assessment: dict) -> dict:
    return {
        "question_no": question_no,
        "question": question,
        "status": "ok",
        "answer": answer.text,
        "assessment": assessment,
        "tokens": answer.tokens,
    }


def _violation_reason(assessment: dict) -> str | None:
    failed = [key for key, value in assessment["checks"].items() if value is False]
    return ", ".join(failed) or None


def _summarize_profile(name: str, prefs: dict, rows: list[dict], planned: int) -> dict:
    ok = [row for row in rows if row["status"] == "ok"]
    satisfied = sum(row["assessment"]["satisfied"] for row in ok)
    total = sum(row["assessment"]["total"] for row in ok)
    return {
        "preferences": prefs,
        "planned": planned,
        "answered": len(ok),
        "failed": len(rows) - len(ok),
        "complete": len(ok) == planned,
        "questions": rows,
        "satisfied": satisfied,
        "total": total,
        "percent": round(satisfied / total * 100, 2) if total else None,
    }


def _summarize_stability(profile: str, rows: list[dict], planned: int) -> dict:
    ok = [row for row in rows if row["status"] == "ok"]
    violations = [row["question_no"] for row in ok if row["violation_reason"]]
    failed = len(rows) - len(ok)
    return {
        "profile": profile,
        "planned": planned,
        "answered": len(ok),
        "failed": failed,
        "complete": failed == 0,
        "turns": rows,
        "first_violation_question": violations[0] if violations else None,
        "zero_violations": not violations and failed == 0,
    }


# ---------------------------------------------------------------------
# Запуск эксперимента
# ---------------------------------------------------------------------

def run(output: Path = DEFAULT_OUTPUT, client=None, *, overwrite: bool = False,
        assume_yes: bool = False, input_fn=input) -> dict | None:
    """Провести эксперимент. client=None — реальный Gemini после подтверждения.

    Возвращает сохранённый результат или None, если платный запуск отменён.
    """
    output = Path(output)
    _ensure_can_write(output, overwrite)
    if client is None:
        if not confirm(f"Запустить платные запросы к Gemini ({planned_requests()} ответов)?",
                       assume_yes, input_fn):
            print("Отменено, запросов не было.")
            return None
        client = make_client()

    output.mkdir(parents=True, exist_ok=True)
    checked_at = datetime.now().astimezone().isoformat(timespec="seconds")
    result = {
        "generated_at": checked_at,
        "model": config.MODEL,
        "settings": {"thinking": config.ANSWER_THINKING_LEVEL,
                     "max_output_tokens": config.MAX_OUTPUT_TOKENS},
        "assessment": {"version": ASSESSMENT_VERSION, "criteria": ASSESSMENT_CRITERIA,
                       "source": "run"},
        "state": {"isolation": "fresh temporary directory per run", "reused_previous": False},
        "requests": {"planned": planned_requests(), "attempted": 0, "successful": 0, "failed": 0,
                     "generate_content": 0},
        "errors": [],
        "profiles": {},
    }
    attempted = successful = failed = 0
    errors: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="day12-personalization-") as tmp:
        state_root = Path(tmp)

        for name, prefs in PROFILES.items():
            agent = Agent(client, MemoryManager(state_root / f"state_{name}"))
            seed(agent, prefs)
            rows = []
            for question_no, question in enumerate(QUESTIONS, 1):
                answer = agent.ask(question, remember=False)
                attempted += 1
                if not answer.ok:
                    failed += 1
                    error = answer.error or answer.text or "неизвестная ошибка"
                    errors.append({"profile": name, "question_no": question_no, "question": question,
                                   "status": answer.status, "error": error})
                    rows.append(_failure_row(question_no, question, answer))
                    continue
                successful += 1
                assessment = assess(answer.text, prefs)
                rows.append(_ok_row(question_no, question, answer, assessment))
                agent.memory.record_exchange(text_content("user", question),
                                             text_content("model", answer.text))
            result["profiles"][name] = _summarize_profile(name, prefs, rows, len(QUESTIONS))

        # Один разговор: один Agent и неизменный профиль на все 10 вопросов.
        prefs = PROFILES["C"]
        agent = Agent(client, MemoryManager(state_root / "state_stability"))
        seed(agent, prefs)
        rows = []
        for question_no, question in enumerate(STABILITY_QUESTIONS, 1):
            answer = agent.ask(question, remember=False)
            attempted += 1
            if not answer.ok:
                failed += 1
                error = answer.error or answer.text or "неизвестная ошибка"
                errors.append({"profile": "stability", "question_no": question_no,
                               "question": question, "status": answer.status, "error": error})
                row = _failure_row(question_no, question, answer)
                row["violation_reason"] = None
                rows.append(row)
                continue
            successful += 1
            assessment = assess(answer.text, prefs)
            row = _ok_row(question_no, question, answer, assessment)
            row["violation_reason"] = _violation_reason(assessment)
            rows.append(row)
            agent.memory.record_exchange(text_content("user", question),
                                         text_content("model", answer.text))
        result["stability"] = _summarize_stability("C", rows, len(STABILITY_QUESTIONS))

    result["requests"].update(attempted=attempted, successful=successful, failed=failed,
                              generate_content=attempted)
    result["errors"] = errors

    write_json(output / RESULT_JSON, result)
    atomic_write_text(output / RESULT_REPORT, redact_secrets(_markdown_report(result)))
    return result


# ---------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------

def _fmt_percent(value) -> str:
    return "—" if value is None else f"{value:.2f}%"


def _fmt_prefs(prefs: dict) -> str:
    return ", ".join(f"{key}: {value}" for key, value in prefs.items()) or "—"


def _mark(value) -> str:
    return "✓" if value is True else "✗" if value is False else "?"


def _checks_line(assessment: dict) -> str:
    checks = assessment.get("checks", {})
    failed = [key for key, value in checks.items() if value is False]
    unverified = assessment.get("unverified", [])
    return (
        f"Проверки: длина {_mark(checks.get('length'))}, код {_mark(checks.get('code'))}, "
        f"стиль {_mark(checks.get('style'))}. "
        f"Нарушения: {', '.join(failed) if failed else 'нет'}; "
        f"не оценено: {', '.join(unverified) if unverified else 'нет'}."
    )


def _markdown_report(data: dict) -> str:
    requests = data.get("requests") or {}
    criteria = data.get("assessment") or {}
    state = data.get("state") or {}
    lines = [
        "# День 12 — эксперимент персонализации",
        "",
        f"Сгенерировано: `{data.get('generated_at', '—')}`.",
        f"Модель: `{data.get('model', '—')}`; thinking: "
        f"`{(data.get('settings') or {}).get('thinking', '—')}`; "
        f"max_output_tokens: `{(data.get('settings') or {}).get('max_output_tokens', '—')}`.",
        f"Критерии оценки: версия `{criteria.get('version', '—')}`; "
        f"{criteria.get('criteria', 'описание отсутствует')}.",
        f"Состояние: {state.get('isolation', 'не указано')}; "
        f"старые state_* не используются: {'да' if state.get('reused_previous') is False else 'не указано'}.",
        f"Запросов: планировалось {requests.get('planned', '—')}, выполнено "
        f"{requests.get('attempted', '—')}, успешных {requests.get('successful', '—')}, "
        f"ошибок {requests.get('failed', '—')}.",
    ]
    if data.get("assessment_note"):
        lines.extend(["", f"> ⚠ {data['assessment_note']}"])
    if requests.get("failed"):
        lines.extend(["", "Ошибки API не оценивались и не записывались в short-term. "
                          "Проценты рассчитаны только по успешным ответам."])

    # Сначала вся таблица, потом разделы профилей — иначе B и C выпадают из таблицы.
    lines.extend([
        "",
        "| Профиль | Предпочтения | Ответов | Соблюдено | Процент |",
        "|---|---|---:|---:|---:|",
    ])
    for name, item in (data.get("profiles") or {}).items():
        planned = item.get("planned", len(item.get("questions", [])))
        answered = item.get("answered", len([r for r in item.get("questions", []) if r.get("status") == "ok"]))
        lines.append(
            f"| {name} | {_fmt_prefs(item.get('preferences', {}))} | {answered}/{planned} | "
            f"{item.get('satisfied', 0)}/{item.get('total', 0)} | {_fmt_percent(item.get('percent'))} |"
        )

    for name, item in (data.get("profiles") or {}).items():
        lines.extend(["", f"## Профиль {name}", "", f"Предпочтения: {_fmt_prefs(item.get('preferences', {}))}.", ""])
        for index, row in enumerate(item.get("questions", []), 1):
            question_no = row.get("question_no", index)
            lines.extend([f"### Вопрос {question_no}: {row.get('question', '—')}", ""])
            if row.get("status") == "failed" or row.get("assessment") is None or row.get("answer") is None:
                lines.extend([f"**Ошибка:** {row.get('error', '—')}", ""])
            else:
                lines.extend([row.get("answer", ""), "", _checks_line(row["assessment"]), ""])

    stability = data.get("stability") or {}
    lines.extend(["", f"## Устойчивость профиля {stability.get('profile', 'C')}", ""])
    if stability.get("failed"):
        lines.append(f"Ошибок: {stability['failed']} из {stability.get('planned', '—')}; "
                     "вывод о нарушениях неполный.")
        lines.append("")
    first = stability.get("first_violation_question")
    if first:
        first_text = f"Вопрос {first}"
    elif stability.get("failed"):
        first_text = "не определено (есть ошибки)"
    else:
        first_text = "нарушений за 10 вопросов не было"
    lines.extend([f"Первое нарушение: {first_text}.", ""])
    for index, row in enumerate(stability.get("turns", []), 1):
        question_no = row.get("question_no", index)
        lines.extend([f"### Вопрос {question_no}", ""])
        if row.get("status") == "failed" or row.get("assessment") is None or row.get("answer") is None:
            lines.extend([f"**Ошибка:** {row.get('error', '—')}", ""])
        else:
            assessment = row["assessment"]
            lines.extend([
                row.get("answer", ""), "",
                f"Соблюдено {assessment.get('satisfied', 0)}/{assessment.get('total', 0)}; "
                f"нарушение: {row.get('violation_reason') or 'нет'}; "
                f"не оценено: {', '.join(assessment.get('unverified', [])) or 'нет'}.", "",
            ])
    return "\n".join(lines)


def _has_prefs(prefs: dict) -> bool:
    return all(key in prefs for key in REQUIRED_PREFS)


def _recalculate_profile(item: dict) -> None:
    rows = item.get("questions", [])
    ok = [row for row in rows if row.get("status") == "ok"]
    item["planned"] = item.get("planned", len(rows))
    item["answered"] = len(ok)
    item["failed"] = len(rows) - len(ok)
    item["complete"] = len(ok) == item["planned"]
    item["satisfied"] = sum(row["assessment"]["satisfied"] for row in ok if row.get("assessment"))
    item["total"] = sum(row["assessment"]["total"] for row in ok if row.get("assessment"))
    item["percent"] = round(item["satisfied"] / item["total"] * 100, 2) if item["total"] else None


def _recalculate_stability(stability: dict) -> None:
    rows = stability.get("turns", [])
    ok = [row for row in rows if row.get("status") == "ok"]
    violations = [row["question_no"] for row in ok if row.get("violation_reason")]
    stability["planned"] = stability.get("planned", len(rows))
    stability["answered"] = len(ok)
    stability["failed"] = len(rows) - len(ok)
    stability["complete"] = stability["failed"] == 0
    stability["first_violation_question"] = violations[0] if violations else None
    stability["zero_violations"] = not violations and stability["failed"] == 0


def _reassess_if_stale(data: dict) -> dict:
    """Для старого снимка пересчитать оценки в отчёте текущим assess(), JSON не меняя."""
    version = (data.get("assessment") or {}).get("version")
    if version == ASSESSMENT_VERSION:
        return data
    result = copy.deepcopy(data)
    result["assessment"] = {"version": ASSESSMENT_VERSION, "criteria": ASSESSMENT_CRITERIA,
                            "source": "report_only"}
    result["assessment_note"] = (
        f"Снимок JSON создан по версии критериев {version or 'без версии'}; "
        f"отчёт пересчитан текущим assess() ({ASSESSMENT_VERSION}). JSON не изменялся."
    )
    for name, item in (result.get("profiles") or {}).items():
        prefs = {**PROFILES.get(name, {}), **item.get("preferences", {})}
        if not _has_prefs(prefs):
            continue
        for row in item.get("questions", []):
            if row.get("status") == "failed" or not isinstance(row.get("answer"), str):
                continue
            row["status"] = "ok"
            row["assessment"] = assess(row["answer"], prefs)
        _recalculate_profile(item)
    stability = result.get("stability") or {}
    if stability:
        profile_name = stability.get("profile", "C")
        prefs = {**PROFILES.get(profile_name, PROFILES["C"]), **stability.get("preferences", {})}
        if _has_prefs(prefs):
            for row in stability.get("turns", []):
                if row.get("status") == "failed" or not isinstance(row.get("answer"), str):
                    continue
                row["status"] = "ok"
                row["assessment"] = assess(row["answer"], prefs)
                row["violation_reason"] = _violation_reason(row["assessment"])
            _recalculate_stability(stability)
    return result


def report_only(output: Path = DEFAULT_OUTPUT) -> Path:
    """Пересобрать report.md без API. JSON не перезаписывается."""
    output = Path(output)
    source = output / RESULT_JSON
    if not source.is_file():
        raise FileNotFoundError(f"Нет {source}: сначала запустите эксперимент.")
    data = json.loads(source.read_text(encoding="utf-8"))
    report_data = _reassess_if_stale(data)
    target = output / RESULT_REPORT
    atomic_write_text(target, redact_secrets(_markdown_report(report_data)))
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="День 12: эксперимент персонализации")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"папка результатов (по умолчанию {DEFAULT_OUTPUT})")
    parser.add_argument("--overwrite", action="store_true",
                        help="разрешить замену существующих результатов")
    parser.add_argument("--yes", action="store_true",
                        help="не спрашивать подтверждение платных запросов")
    parser.add_argument("--report-only", action="store_true",
                        help="только пересобрать report.md без API")
    args = parser.parse_args(argv)
    try:
        if args.report_only:
            path = report_only(args.output)
            print(f"Сохранено: {path}")
            return 0
        result = run(args.output, overwrite=args.overwrite, assume_yes=args.yes)
        if result is None:
            return 0
        print(f"Сохранено: {args.output / RESULT_JSON}, {args.output / RESULT_REPORT}")
        print(f"Запросов: {result['requests']['attempted']}, успешных: "
              f"{result['requests']['successful']}, ошибок: {result['requests']['failed']}")
        return 0
    except ResultsExistError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # понятное сообщение вместо трейсбэка; ключ маскируется
        print(f"Ошибка: {redact_secrets(f'{type(exc).__name__}: {exc}')}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
