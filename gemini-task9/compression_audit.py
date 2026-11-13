"""Офлайн-аудит сжатия: что из исходного диалога сохранилось в summary, что потерялось или исказилось.

Работает только с файлами, Gemini API НЕ вызывает:

* каталог деталей ``data/summary_details.json`` — детали диалога (включая промежуточные значения,
  причины, альтернативы, советы ассистента) с цитатами-доказательствами и шаблонами;
* текст summary (итоговый и, если сохранены, все версии v1, v2, …) из результатов эксперимента B.

Статусы детали:
    preserved      — признаки детали найдены в summary;
    lost           — признаков нет;
    distorted      — в summary значение, противоречащее диалогу (правило distort: например, прежнее
                     время 08:00 без пометки «заменено», а итогового 09:30 нигде нет);
    not_compressed — деталь из сообщений, которые ещё не сжимались (они в активном окне);
    not_comparable — summary создано на другой версии этих сообщений диалога.

Проверка авторства: предложения ассистента (venv, launchd, структура папок, …) не должны
появляться в summary как решения пользователя. Оценивается каждое предложение summary,
где упомянут совет ассистента (а не вся строка, куда summary мог склеить несколько тем).

Метод эвристический (регулярные выражения): отчёт — повод посмотреть строку summary глазами,
а не окончательный вердикт.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import config
from utils import normalize_text

DETAILS_FILE = config.DATA_DIR / "summary_details.json"

ASSISTANT_MARKERS = r"ассистент|предлож|посовет|совет|рекоменд|можно|вариант"
DECISION_MARKERS = (r"использу|выбран|решен|решил|будем|принят|утвержд|ограничени|хран(ят|им|ится)|"
                    r"запланир|зафиксир|пользовател\w*\s+(выбрал|решил|хочет)")

STATUS_RU = {
    "preserved": "✅ сохранена", "lost": "❌ потеряна", "distorted": "⚠️ искажена",
    "not_compressed": "— не сжималась", "not_comparable": "≠ несравнимо",
}
ATTR_RU = {
    "as_decision": "❌ записано как решение", "unattributed": "⚠️ без указания автора",
    "ok_assistant": "✅ приписано ассистенту", "ok_user_accepted": "✅ принято пользователем",
    "absent": "— нет в summary", "not_comparable": "≠ несравнимо",
}
IMPORTANCE_RU = {"high": "высокая", "medium": "средняя", "low": "низкая"}
KIND_RU = {"fact": "факт", "decision": "решение", "intermediate": "промежуточное значение", "reason": "причина",
           "preference": "предпочтение", "alternative": "альтернатива", "constraint": "ограничение",
           "advice": "совет ассистента"}


class AuditError(Exception):
    """Каталог деталей не согласован с диалогом или нет данных для аудита."""


# ---------------------------------------------------------------------
# Каталог и диалог
# ---------------------------------------------------------------------

def dialogue_texts(dialogue) -> list[tuple[str, str]]:
    """[(role, text)] из types.Content или из словарей {"role", "text"}."""
    out = []
    for item in dialogue:
        if isinstance(item, dict):
            out.append((item["role"], item["text"]))
        else:
            out.append((item.role, "\n".join(p.text or "" for p in item.parts or [])))
    return out


def dialogue_fingerprint(dialogue) -> str:
    data = json.dumps(dialogue_texts(dialogue), ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def load_catalog(path: Path = DETAILS_FILE) -> dict:
    try:
        catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AuditError(f"Каталог деталей не найден: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AuditError(f"{path}: некорректный JSON: {exc.msg}") from exc
    if not catalog.get("details"):
        raise AuditError(f"{path}: нет списка 'details'")
    return catalog


def validate_catalog(catalog: dict, dialogue) -> None:
    """Каждая деталь подтверждается цитатой из диалога; предложения ассистента — только его сообщениями."""
    msgs = dialogue_texts(dialogue)

    def message(n: int, where: str) -> tuple[str, str]:
        if not 1 <= n <= len(msgs):
            raise AuditError(f"{where}: сообщения #{n} нет в диалоге ({len(msgs)} сообщений)")
        return msgs[n - 1]

    for d in catalog["details"]:
        if not d.get("evidence") or not d.get("keep"):
            raise AuditError(f"{d.get('id')}: нужны evidence и keep")
        for ev in d["evidence"]:
            role, text = message(ev["message"], d["id"])
            if role != ev["role"] or ev["quote"] not in text:
                raise AuditError(f"{d['id']}: цитата «{ev['quote']}» не найдена в сообщении #{ev['message']} ({ev['role']})")
        for rule in d["keep"]:
            _compile_rule(rule, d["id"])
        for rule in d.get("distort", []):
            for key in ("line_has", "line_lacks", "text_lacks"):
                if rule.get(key):
                    re.compile(rule[key])
    user_text = normalize_text("\n".join(t for r, t in msgs if r == "user"))
    for a in catalog.get("attributions", []):
        for n in a["assistant_messages"]:
            role, text = message(n, a["id"])
            if role != "model" or not re.search(a["pattern"], normalize_text(text)):
                raise AuditError(f"{a['id']}: в сообщении ассистента #{n} нет «{a['item']}»")
        if a.get("accepted_by_user"):
            if not any(r == "user" and a["user_acceptance_quote"] in t for r, t in msgs):
                raise AuditError(f"{a['id']}: нет реплики пользователя, принимающей предложение")
        elif re.search(a["pattern"], user_text):
            raise AuditError(f"{a['id']}: «{a['item']}» упоминает пользователь — это не только предложение ассистента")


def _compile_rule(rule, where: str) -> None:
    patterns = [rule] if isinstance(rule, str) else rule.get("line_all") or rule.get("text_all")
    if not patterns:
        raise AuditError(f"{where}: неизвестный формат правила keep: {rule!r}")
    for p in patterns:
        try:
            re.compile(p)
        except re.error as exc:
            raise AuditError(f"{where}: неверное регулярное выражение {p!r}: {exc}") from exc


# ---------------------------------------------------------------------
# Проверка одного summary
# ---------------------------------------------------------------------

def summary_lines(text: str) -> list[str]:
    return [line for line in (normalize_text(l) for l in (text or "").splitlines()) if line]


def _keep_match(rule, lines: list[str], full: str) -> str | None:
    """Строка summary, в которой найдена деталь (или None)."""
    if isinstance(rule, str):
        return next((l for l in lines if re.search(rule, l)), None) or (full if re.search(rule, full) else None)
    if "line_all" in rule:
        return next((l for l in lines if all(re.search(p, l) for p in rule["line_all"])), None)
    if all(re.search(p, full) for p in rule["text_all"]):
        return next((l for l in lines if re.search(rule["text_all"][0], l)), full)
    return None


def check_detail(detail: dict, text: str) -> dict:
    lines = summary_lines(text)
    full = "\n".join(lines)
    for rule in detail.get("distort", []):
        # Итоговое значение есть где-то в summary → прежнее значение рядом с ним не искажение.
        if rule.get("text_lacks") and re.search(rule["text_lacks"], full):
            continue
        for line in lines:
            if re.search(rule["line_has"], line) and not re.search(rule["line_lacks"], line):
                return {"status": "distorted", "line": line, "note": rule.get("note")}
    for rule in detail["keep"]:
        line = _keep_match(rule, lines, full)
        if line:
            return {"status": "preserved", "line": line, "note": None}
    return {"status": "lost", "line": None, "note": None}


def summary_sentences(text: str) -> list[str]:
    """Предложения summary: строки делятся по . ! ? ; с последующим пробелом.

    Пункт summary часто склеивает несколько тем в одну строку («ОС macOS (zsh, Homebrew).
    … выбрана библиотека httpx. API-ключи хранятся в .env»), поэтому авторство оценивается
    по предложению, где упомянуто предложение ассистента, а не по всей строке.
    """
    out = []
    for line in summary_lines(text):
        out += [s.strip() for s in re.split(r"(?<=[.!?;])\s+", line) if s.strip()]
    return out


def check_attribution(item: dict, text: str) -> dict:
    """Эвристика: как в summary подано предложение ассистента (по предложениям, не по строкам)."""
    sentences = [s for s in summary_sentences(text) if re.search(item["pattern"], s)]
    if not sentences:
        return {"status": "absent", "lines": []}
    if item.get("accepted_by_user"):
        return {"status": "ok_user_accepted", "lines": sentences}
    statuses = []
    for sentence in sentences:
        if re.search(ASSISTANT_MARKERS, sentence):
            statuses.append("ok_assistant")
        elif re.search(DECISION_MARKERS, sentence):
            statuses.append("as_decision")
        else:
            statuses.append("unattributed")
    worst = next(s for s in ("as_decision", "unattributed", "ok_assistant") if s in statuses)
    return {"status": worst, "lines": sentences}


# ---------------------------------------------------------------------
# Аудит прогона
# ---------------------------------------------------------------------

def audit_summary(
    catalog: dict,
    final_summary: str,
    *,
    summarized_upto: int,
    versions: list[dict] | None = None,
    mismatched_messages: set[int] | None = None,
    stored_messages: list[tuple[str, str]] | None = None,
    source: dict | None = None,
    warnings: list[str] | None = None,
) -> dict:
    """summarized_upto — сколько первых сообщений диалога вошло в итоговое summary.

    versions — [{"version", "end", "text"}] (end — сколько сообщений покрывает версия);
    mismatched_messages — номера сообщений (с 1), которые отличаются от текущего диалога;
    stored_messages — сообщения того прогона: если цитата детали в изменённом сообщении не тронута,
    деталь всё равно сравнима (например, сообщение 1 отличается только именем).
    """
    mismatched = mismatched_messages or set()

    def incomparable(evidence: list[dict]) -> list[int]:
        bad = []
        for ev in evidence:
            n = ev["message"]
            if n not in mismatched:
                continue
            if stored_messages and n <= len(stored_messages) and ev["quote"] in stored_messages[n - 1][1]:
                continue
            bad.append(n)
        return sorted(bad)
    versions = [v for v in (versions or []) if v.get("text")]
    rows = []
    for d in catalog["details"]:
        msgs = [ev["message"] for ev in d["evidence"]]
        row = {"id": d["id"], "detail": d["detail"], "importance": d["importance"], "kind": d["kind"],
               "messages": msgs}
        bad = incomparable(d["evidence"])
        if bad:
            row.update(status="not_comparable", line=None,
                       note=f"в сообщениях {bad} прогона эта деталь отличается от текущего диалога")
        elif max(msgs) > summarized_upto:
            row.update(status="not_compressed", line=None, note="в активном окне, передаётся без сжатия")
        else:
            row.update(check_detail(d, final_summary))
        # Отслеживание по версиям: где деталь появилась и где пропала.
        history = []
        for v in versions:
            if max(msgs) > v["end"]:
                history.append({"version": v["version"], "status": None})
            elif bad:
                history.append({"version": v["version"], "status": "not_comparable"})
            else:
                history.append({"version": v["version"], "status": check_detail(d, v["text"])["status"]})
        row["versions"] = history
        row["first_problem_version"] = _first_problem(history)
        rows.append(row)

    attributions = []
    for a in catalog.get("attributions", []):
        rec = {"id": a["id"], "item": a["item"], "assistant_messages": a["assistant_messages"],
               "accepted_by_user": a.get("accepted_by_user", False)}
        stored_ok = stored_messages and all(
            n <= len(stored_messages) and re.search(a["pattern"], normalize_text(stored_messages[n - 1][1]))
            for n in a["assistant_messages"])
        if mismatched & set(a["assistant_messages"]) and not stored_ok:
            rec.update(status="not_comparable", lines=[])
        elif min(a["assistant_messages"]) > summarized_upto:
            rec.update(status="absent", lines=[])
        else:
            rec.update(check_attribution(a, final_summary))
        attributions.append(rec)

    counts = Counter(r["status"] for r in rows)
    by_importance = {imp: dict(Counter(r["status"] for r in rows if r["importance"] == imp))
                     for imp in ("high", "medium", "low")}
    checked = [r for r in rows if r["status"] in ("preserved", "lost", "distorted")]
    return {
        "source": source or {},
        "warnings": list(warnings or []),
        "summarized_messages": summarized_upto,
        "versions_available": [v["version"] for v in versions],
        "totals": {**{k: counts.get(k, 0) for k in STATUS_RU}, "checked": len(checked)},
        "preserved_percent": round(counts.get("preserved", 0) / len(checked) * 100, 1) if checked else None,
        "by_importance": by_importance,
        "details": rows,
        "attributions": attributions,
        "attribution_issues": [a["id"] for a in attributions if a["status"] in ("as_decision", "unattributed")],
    }


def _first_problem(history: list[dict]) -> dict | None:
    """Первая версия, где деталь потеряна/искажена: после того как была сохранена — или сразу."""
    seen = False
    for h in history:
        if h["status"] == "preserved":
            seen = True
        elif h["status"] in ("lost", "distorted"):
            return {"version": h["version"], "status": h["status"], "after_preserved": seen}
    return None


# ---------------------------------------------------------------------
# Загрузка сохранённого прогона (только чтение)
# ---------------------------------------------------------------------

def mismatched_messages(stored_history: list[tuple[str, str]], dialogue) -> set[int]:
    current = dialogue_texts(dialogue)
    diff = {i + 1 for i, (a, b) in enumerate(zip(stored_history, current)) if a != b}
    if len(stored_history) != len(current):
        diff |= set(range(min(len(stored_history), len(current)) + 1, max(len(stored_history), len(current)) + 1))
    return diff


def load_run(results_dir: Path, dialogue) -> dict:
    """Собрать из results/ данные для аудита: summary, версии, сравнение с текущим диалогом."""
    results_dir = Path(results_dir)
    exp_path = results_dir / "experiment_with_compression.json"
    if not exp_path.is_file():
        raise AuditError(f"Нет результатов эксперимента B: {exp_path}. Сначала: python main.py --experiment B")
    exp = json.loads(exp_path.read_text(encoding="utf-8"))
    warnings = []

    versions = exp.get("summary_versions") or []
    state_summary = results_dir / "state_B" / config.SUMMARY_FILE
    if not versions and state_summary.is_file():
        updates = json.loads(state_summary.read_text(encoding="utf-8")).get("updates", [])
        versions = [{"version": u["version"], "end": u["end"], "text": u["text"]} for u in updates if u.get("text")]
    if not any(v.get("text") for v in versions):
        warnings.append("Тексты промежуточных версий summary не сохранены (прогон до доработки) — "
                        "аудит только по итоговому summary, отслеживание по версиям недоступно.")

    mismatched: set[int] = set()
    stored = None
    history_path = results_dir / "state_B" / config.FULL_HISTORY_FILE
    if history_path.is_file():
        payload = json.loads(history_path.read_text(encoding="utf-8"))
        stored = [(c["role"], "\n".join(p.get("text", "") for p in c.get("parts", []))) for c in payload["history"]]
        mismatched = mismatched_messages(stored, dialogue)
    elif exp.get("dialogue_fingerprint") and exp["dialogue_fingerprint"] != dialogue_fingerprint(dialogue):
        mismatched = set(range(1, len(dialogue_texts(dialogue)) + 1))
        warnings.append("Отпечаток диалога не совпадает, а полной истории прогона нет — сравнение невозможно.")
    elif not exp.get("dialogue_fingerprint"):
        warnings.append("Не удалось проверить, на какой версии диалога создано summary.")
    if mismatched:
        warnings.append(f"Summary создано на другой версии диалога: отличаются сообщения {sorted(mismatched)}. "
                        "Детали, которые в этих сообщениях отличаются, помечены «несравнимо».")

    return {
        "final_summary": exp.get("final_summary", ""),
        "summarized_upto": exp.get("context", {}).get("summarized_messages", 0),
        "versions": versions,
        "mismatched": mismatched,
        "stored": stored,
        "warnings": warnings,
        "source": {"file": str(exp_path), "started_at": exp.get("started_at"), "model": exp.get("model")},
    }


def audit_saved_run(results_dir: Path, dialogue, catalog: dict | None = None) -> dict:
    catalog = catalog or load_catalog()
    validate_catalog(catalog, dialogue)
    run = load_run(results_dir, dialogue)
    return audit_summary(catalog, run["final_summary"], summarized_upto=run["summarized_upto"],
                         versions=run["versions"], mismatched_messages=run["mismatched"],
                         stored_messages=run["stored"],
                         source=run["source"], warnings=run["warnings"])


# ---------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------

def _cut(text: str | None, limit: int = 80) -> str:
    if not text:
        return ""
    text = " ".join(text.split()).replace("|", "/")
    return text if len(text) <= limit else text[:limit] + "…"


def markdown_report(audit: dict) -> str:
    src = audit["source"]
    t = audit["totals"]
    lines = ["# Офлайн-аудит сжатия контекста", "",
             f"Источник: `{src.get('file', '—')}` (прогон {src.get('started_at', '—')}, модель {src.get('model', '—')}). "
             f"Сжато сообщений: {audit['summarized_messages']}. Gemini API не вызывался.", ""]
    for w in audit["warnings"]:
        lines.append(f"> ⚠️ {w}")
    if audit["warnings"]:
        lines.append("")
    lines += ["## Итог", "",
              f"Проверено деталей: **{t['checked']}** — сохранено {t['preserved']}, потеряно {t['lost']}, "
              f"искажено {t['distorted']} (**{audit['preserved_percent']}%** сохранено). "
              f"Несравнимо: {t['not_comparable']}, не сжимались: {t['not_compressed']}.", "",
              "| Важность | Сохранено | Потеряно | Искажено | Несравнимо |", "|---|---|---|---|---|"]
    for imp, c in audit["by_importance"].items():
        lines.append(f"| {IMPORTANCE_RU[imp]} | {c.get('preserved', 0)} | {c.get('lost', 0)} | "
                     f"{c.get('distorted', 0)} | {c.get('not_comparable', 0)} |")

    versions = audit["versions_available"]
    head = "| ID | Деталь | Тип | Важность | Сообщ. | Статус | Строка summary / примечание |"
    sep = "|---|---|---|---|---|---|---|"
    if versions:
        head += "".join(f" v{v} |" for v in versions) + " Проблема с |"
        sep += "---|" * (len(versions) + 1)
    lines += ["", "## Детали диалога", "", head, sep]
    short = {"preserved": "✅", "lost": "❌", "distorted": "⚠️", "not_comparable": "≠", None: "·"}
    for r in audit["details"]:
        row = (f"| {r['id']} | {r['detail']} | {KIND_RU.get(r['kind'], r['kind'])} | {IMPORTANCE_RU[r['importance']]} | "
               f"{', '.join(map(str, r['messages']))} | {STATUS_RU[r['status']]} | {_cut(r.get('note') or r.get('line'))} |")
        if versions:
            fp = r["first_problem_version"]
            row += "".join(f" {short.get(h['status'], '?')} |" for h in r["versions"])
            row += f" {'v' + str(fp['version']) if fp else '—'} |"
        lines.append(row)

    lines += ["", "## Авторство: предложения ассистента", "",
              "| ID | Предложение ассистента | Сообщ. | Статус | Предложения summary |", "|---|---|---|---|---|"]
    for a in audit["attributions"]:
        lines.append(f"| {a['id']} | {a['item']} | {', '.join(map(str, a['assistant_messages']))} | "
                     f"{ATTR_RU[a['status']]} | {_cut(' // '.join(a['lines']), 110)} |")
    lines += ["", "Статусы (по отдельным предложениям summary): «записано как решение» — предложение подаёт совет "
              "ассистента как принятое решение "
              "(есть «используется», «хранятся», «ограничения» и т. п.); «без указания автора» — упомянуто как факт "
              "проекта без пометки, что это предложение ассистента.", "",
              "_Метод эвристический (регулярные выражения по нормализованному тексту summary); спорные строки стоит "
              "проверить вручную._"]
    return "\n".join(lines)
