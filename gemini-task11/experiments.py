"""Эксперимент Дня 11: что хранит каждый слой памяти, что попадает в промпт и что теряется при очистке.

1. Сценарий data/scenario.json проигрывается ОДИН раз с чистого состояния (папка base/):
   на каждую реплику — 1 вызов менеджера памяти, затем пара записывается в краткосрочную память.
2. Проверка перезапуска: новый MemoryManager из тех же файлов → содержимое и промпт совпадают.
3. Для каждого варианта состояние base/ копируется в свою папку, загружается заново (= перезапуск агента),
   очищается НЕ БОЛЕЕ ОДНОГО слоя, и задаются одни и те же 15 вопросов (remember=False):
       full           — все слои;
       no_short_term  — очищена только краткосрочная память;
       no_working     — очищена только рабочая память;
       no_long_term   — очищена только долговременная память.
4. Ответы проверяются по эталону data/expected.json (агенту не передаётся). У каждого вопроса есть
   ожидаемый слой: ответ должен пропасть ровно тогда, когда очищен его слой.

Токены: count_tokens каждого блока промпта отдельно (LT, WM, ST, вопрос) и всего промпта; usage_metadata
по вызовам. В офлайн-режиме (FakeClient) числа — детерминированный фейковый счёт, а не токены Gemini.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import config
from agent import Agent, CallRecord
from memory import LAYER_TITLES, LAYERS, PROMPT_ORDER, READ_RULES, WRITE_RULES, MemoryManager
from memory.manager import FILE_NAMES
from token_stats import Usage, estimate_cost, sum_known, sum_usage
from utils import atomic_write_text, normalize_text, write_json

logger = logging.getLogger("experiments")

VARIANTS = (
    {"name": "full", "clear": None, "title": "Все три слоя"},
    {"name": "no_short_term", "clear": "short_term", "title": "Очищена только краткосрочная память"},
    {"name": "no_working", "clear": "working", "title": "Очищена только рабочая память"},
    {"name": "no_long_term", "clear": "long_term", "title": "Очищена только долговременная память"},
)
RESULT_JSON = "experiment.json"
RESULT_REPORT = "report.md"
STATE_SUBDIR = "state"


class DataError(Exception):
    """Файлы сценария, вопросов или эталона некорректны."""


class ResultsExistError(Exception):
    """Результаты уже есть — без --overwrite они не перезаписываются."""


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


def load_scenario(path: Path = config.SCENARIO_FILE) -> list[tuple[str, str]]:
    items = _load(path).get("exchanges")
    if not isinstance(items, list) or not items:
        raise DataError(f"{path}: нет списка 'exchanges'")
    pairs = []
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict) or not str(item.get("user", "")).strip() or not str(item.get("model", "")).strip():
            raise DataError(f"{path}: реплика #{i} должна содержать непустые 'user' и 'model'")
        pairs.append((item["user"].strip(), item["model"].strip()))
    return pairs


def load_questions(path: Path = config.QUESTIONS_FILE) -> list[dict]:
    items = _load(path).get("questions")
    if not isinstance(items, list) or not items:
        raise DataError(f"{path}: нет списка 'questions'")
    return items


def load_expected(path: Path = config.EXPECTED_FILE) -> dict[int, dict]:
    items = _load(path).get("answers")
    if not isinstance(items, list) or not items:
        raise DataError(f"{path}: нет списка 'answers'")
    return {a["id"]: a for a in items}


def load_offline_extraction(path: Path = config.OFFLINE_EXTRACTION_FILE) -> dict[int, list[dict]]:
    data = _load(path).get("by_user_message")
    if not isinstance(data, dict):
        raise DataError(f"{path}: нет объекта 'by_user_message'")
    return {int(k): v for k, v in data.items()}


def check_answer(answer: str, exp: dict) -> bool:
    """Верно, если совпали ВСЕ accept_all и ни один reject (по нормализованному тексту)."""
    text = normalize_text(answer or "")
    if any(re.search(p, text) for p in exp.get("reject", [])):
        return False
    return all(re.search(p, text) for p in exp["accept_all"])


def validate_data(pairs: list[tuple[str, str]], questions: list[dict], expected: dict[int, dict]) -> None:
    ids = [q.get("id") for q in questions]
    if len(set(ids)) != len(ids):
        raise DataError("повторяющиеся id вопросов")
    if set(ids) != set(expected):
        raise DataError(f"id вопросов и эталона не совпадают: {sorted(set(ids) ^ set(expected))}")
    for q in questions:
        if not str(q.get("question", "")).strip():
            raise DataError(f"вопрос {q.get('id')}: пустой текст")
    n_messages = 2 * len(pairs)
    for i, e in expected.items():
        if e.get("layer") not in LAYERS:
            raise DataError(f"эталон {i}: неизвестный слой {e.get('layer')!r}")
        if not e.get("accept_all"):
            raise DataError(f"эталон {i}: пустой accept_all")
        for p in e["accept_all"] + e.get("reject", []):
            try:
                re.compile(p)
            except re.error as exc:
                raise DataError(f"эталон {i}: неверный шаблон {p!r}: {exc}") from exc
        if not check_answer(e["expected"], e):
            raise DataError(f"эталон {i}: «{e['expected']}» не проходит собственные шаблоны")
        if not 1 <= int(e.get("evidence_message", 0)) <= n_messages:
            raise DataError(f"эталон {i}: evidence_message вне сценария")


# ---------------------------------------------------------------------
# Вспомогательное
# ---------------------------------------------------------------------

def file_digests(state_dir: Path) -> dict[str, str | None]:
    out = {}
    for layer, name in FILE_NAMES.items():
        p = Path(state_dir) / name
        out[layer] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    return out


def cost_block(usages: list[Usage], request_tokens: list[int | None]) -> dict:
    block = sum_usage(usages)
    block["request_tokens_count_tokens"] = sum_known(request_tokens) if request_tokens else 0
    block["estimated_cost_usd"] = estimate_cost(block["input_tokens"], block["output_tokens"])
    return block


def write_log(calls: list[CallRecord]) -> dict:
    """Что модель предложила и что код записал/отклонил — по слоям, со ссылкой на сообщение."""
    out = {"working": {"applied": [], "rejected": [], "unchanged": []},
           "long_term": {"applied": [], "rejected": [], "unchanged": []},
           "unrouted": [], "failed_calls": []}
    for c in calls:
        msg = c.meta.get("message")
        if not c.ok:
            out["failed_calls"].append({"message": msg, "error": c.error})
            continue
        d = c.meta.get("decisions") or {}
        for layer in ("working", "long_term"):
            for status in ("applied", "rejected", "unchanged"):
                for rec in (d.get(layer) or {}).get(status, []):
                    item = {"message": msg, "key": rec.get("key"), "value": rec.get("value"), "kind": rec.get("kind")}
                    if status == "rejected":
                        item["rejection"] = rec.get("rejection")
                    out[layer][status].append(item)
        for rec in d.get("unrouted", []):
            out["unrouted"].append({"message": msg, "layer": rec.get("layer"), "key": rec.get("key"),
                                    "rejection": rec.get("rejection")})
    return out


# ---------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------

def play_scenario(agent: Agent, pairs: list[tuple[str, str]]) -> list[CallRecord]:
    calls = []
    for i, (user, model) in enumerate(pairs, 1):
        call = agent.play_exchange(user, model)
        logger.info("Реплика %d/%d: память %s", i, len(pairs), "обновлена" if call.ok else f"НЕ обновлена ({call.error})")
        calls.append(call)
    return calls


def run_variant(client: Any, base_dir: Path, work_dir: Path, variant: dict, questions: list[dict],
                expected: dict[int, dict], window: int) -> dict:
    vdir = Path(work_dir) / variant["name"]
    if vdir.exists():
        shutil.rmtree(vdir)
    shutil.copytree(base_dir, vdir)
    before = file_digests(vdir)
    memory = MemoryManager(vdir, window=window)          # загрузка из файлов = перезапуск агента
    if variant["clear"]:
        memory.clear(variant["clear"])
    after_clear = file_digests(vdir)
    agent = Agent(client, memory)
    blocks = memory.prompt_blocks()

    answers = []
    for q in questions:
        exp = expected[q["id"]]
        res = agent.ask(q["question"], remember=False)
        correct = res.ok and check_answer(res.text, exp)
        expected_correct = exp["layer"] != variant["clear"]
        answers.append({
            "id": q["id"], "question": q["question"], "layer": exp["layer"], "expected": exp["expected"],
            "answer": res.text, "status": res.status, "correct": correct,
            "expected_correct": expected_correct, "matches_expectation": correct == expected_correct,
            "tokens": res.tokens, "usage": res.usage.to_dict(),
        })
    unchanged_by_questions = file_digests(vdir) == after_clear

    by_layer = {layer: {"correct": sum(a["correct"] for a in answers if a["layer"] == layer),
                        "total": sum(1 for a in answers if a["layer"] == layer)} for layer in LAYERS}
    totals = [a["tokens"].get("total") for a in answers]
    known = [t for t in totals if t is not None]
    usages = [Usage(**{k: v for k, v in a["usage"].items() if k in Usage.__dataclass_fields__}) for a in answers]
    return {
        "name": variant["name"], "title": variant["title"], "cleared_layer": variant["clear"],
        "changed_files": sorted(layer for layer in LAYERS if before[layer] != after_clear[layer]),
        "memory_unchanged_by_questions": unchanged_by_questions,
        "available_data": memory.stats(),
        "prompt_layers": [b.layer for b in blocks],
        "prompt_context_text": memory.context_text(),
        "answers": answers,
        "summary": {
            "correct": sum(a["correct"] for a in answers), "total": len(answers),
            "correct_ids": [a["id"] for a in answers if a["correct"]],
            "wrong_ids": [a["id"] for a in answers if not a["correct"]],
            "by_layer": by_layer,
            "matches_expectation": all(a["matches_expectation"] for a in answers),
            "tokens": {
                "layers": answers[0]["tokens"]["layers"] if answers else {},   # одинаковы для всех вопросов
                "question_avg": round(sum(a["tokens"]["question"] or 0 for a in answers) / len(answers), 1),
                "total_avg": round(sum(known) / len(known), 1) if known else None,
                "total_sum": sum_known(totals),
            },
            "usage": cost_block(usages, totals),
        },
    }


def run_experiment(client: Any, work_dir: Path, *, window: int = config.SHORT_TERM_WINDOW,
                   pairs: list[tuple[str, str]] | None = None, questions: list[dict] | None = None,
                   expected: dict[int, dict] | None = None, mode: str = "offline") -> dict:
    pairs = pairs if pairs is not None else load_scenario()
    questions = questions if questions is not None else load_questions()
    expected = expected if expected is not None else load_expected()
    validate_data(pairs, questions, expected)

    work_dir = Path(work_dir)
    base = work_dir / "base"
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)

    memory = MemoryManager(base, window=window)
    agent = Agent(client, memory)
    calls = play_scenario(agent, pairs)

    restarted = MemoryManager(base, window=window)
    restart_check = {"same_content": restarted.dump() == memory.dump(),
                     "same_prompt": restarted.context_text() == memory.context_text(),
                     "files": {layer: name for layer, name in FILE_NAMES.items() if (base / name).is_file()}}
    restart_check["ok"] = restart_check["same_content"] and restart_check["same_prompt"]

    variants = {v["name"]: run_variant(client, base, work_dir, v, questions, expected, window) for v in VARIANTS}
    full_ok = set(variants["full"]["summary"]["correct_ids"])
    for v in variants.values():
        ok = set(v["summary"]["correct_ids"])
        v["summary"]["still_correct_vs_full"] = sorted(full_ok & ok)
        v["summary"]["lost_vs_full"] = sorted(full_ok - ok)

    matrix = [{"id": q["id"], "question": q["question"], "layer": expected[q["id"]]["layer"],
               **{name: next(a["correct"] for a in v["answers"] if a["id"] == q["id"]) for name, v in variants.items()}}
              for q in questions]
    processing = cost_block([c.usage for c in calls], [c.request_tokens for c in calls])
    processing["failed_calls"] = sum(1 for c in calls if not c.ok)

    return {
        "meta": {
            "mode": mode, "model": config.MODEL, "created_at": datetime.now().isoformat(timespec="seconds"),
            # По хэшу видно, с какой версией инструкций сделан прогон (v1 и v2 различаются инструкцией памяти).
            "memory_instruction_sha256": hashlib.sha256(config.MEMORY_SYSTEM_INSTRUCTION.encode()).hexdigest()[:12],
            "answer_instruction_sha256": hashlib.sha256(config.SYSTEM_INSTRUCTION.encode()).hexdigest()[:12],
            "short_term_window": window, "render_evidence": config.RENDER_EVIDENCE,
            "exchanges": len(pairs), "messages": 2 * len(pairs), "questions": len(questions),
            "token_source": ("FakeClient: детерминированный фейковый счёт (3 на часть + 1 на слово), НЕ токены Gemini"
                             if mode == "offline" else "Gemini: count_tokens (contents, без system instruction) "
                             "и usage_metadata"),
            "note": ("Офлайн: ответ = эхо контекста, поэтому «верно» означает «нужный факт попал в промпт»."
                     if mode == "offline" else "Реальные ответы Gemini."),
        },
        "rules": {"write": WRITE_RULES, "read": READ_RULES},
        "processing": processing,
        "write_log": write_log(calls),
        "layers_after_scenario": {name: {"stats": memory.stats()[name], "content": memory.dump()[name]}
                                  for name in LAYERS},
        "restart_check": restart_check,
        "variants": variants,
        "matrix": matrix,
    }


# ---------------------------------------------------------------------
# Сохранение
# ---------------------------------------------------------------------

def ensure_can_write(results_dir: Path, overwrite: bool = False) -> None:
    """Проверить ДО запуска (чтобы не тратить платные вызовы): старые результаты не перезаписываются."""
    results_dir = Path(results_dir)
    existing = [p for p in (results_dir / RESULT_JSON, results_dir / RESULT_REPORT, results_dir / STATE_SUBDIR)
                if p.exists()]
    if existing and not overwrite:
        raise ResultsExistError(f"В {results_dir} уже есть результаты ({', '.join(p.name for p in existing)}). "
                                "Укажите другую папку (--results-dir) или --overwrite.")


def save_results(results_dir: Path, data: dict, overwrite: bool = False) -> list[Path]:
    results_dir = Path(results_dir)
    paths = [results_dir / RESULT_JSON, results_dir / RESULT_REPORT]
    if not overwrite and any(p.exists() for p in paths):
        raise ResultsExistError(f"В {results_dir} уже есть результаты; используйте --overwrite")
    write_json(paths[0], data)
    atomic_write_text(paths[1], markdown_report(data))
    return paths


# ---------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------

SHORT = {"short_term": "ST", "working": "WM", "long_term": "LT"}


def _fmt(v) -> str:
    return "—" if v is None else str(v)


def markdown_report(data: dict) -> str:
    meta, variants = data["meta"], data["variants"]
    lines = [f"# День 11 — результаты эксперимента ({meta['mode']})", "",
             f"Модель: `{meta['model']}`; окно краткосрочной памяти: {meta['short_term_window']}; "
             f"сценарий: {meta['exchanges']} реплик ({meta['messages']} сообщений); вопросов: {meta['questions']}.",
             f"Источник токенов: {meta['token_source']}.", f"{meta['note']}", ""]

    lines += ["## 1. Что сохранено в каждом слое после сценария", "",
              "| Слой | Файл | Содержимое |", "|---|---|---|"]
    la = data["layers_after_scenario"]
    st, wm, lt = la["short_term"]["stats"], la["working"]["stats"], la["long_term"]["stats"]
    lines.append(f"| {LAYER_TITLES['short_term']} | `{config.SHORT_TERM_FILE}` | {st['stored_messages']} сообщений, "
                 f"в промпт — последние {st['in_prompt_messages']} |")
    lines.append(f"| {LAYER_TITLES['working']} | `{config.WORKING_FILE}` | задача #{wm['task_id']}, статус {wm['status']}; "
                 f"поля: {', '.join(wm['filled_fields'])}; ограничений {wm['constraints']}, результатов {wm['results']} |")
    kinds = ", ".join(f"{k}: {n}" for k, n in lt["by_kind"].items())
    lines.append(f"| {LAYER_TITLES['long_term']} | `{config.LONG_TERM_FILE}` | {lt['facts']} фактов ({kinds}) |")
    lines.append("")
    lines += ["Долговременная память:", ""]
    for k, f in la["long_term"]["content"]["facts"].items():
        lines.append(f"- `{k}` = {f['value']} — *{f['kind']}*, {f['category']}")
    lines += ["", "Рабочая память:", "", "```json",
              json.dumps(la["working"]["content"]["task"], ensure_ascii=False, indent=1), "```", ""]

    wl = data["write_log"]
    lines += ["## 2. Правила записи в действии: отклонённые операции", "",
              "| Сообщение | Слой | Ключ | Причина |", "|---|---|---|---|"]
    for layer in ("working", "long_term"):
        for r in wl[layer]["rejected"]:
            lines.append(f"| {r['message']} | {SHORT[layer]} | `{r['key']}` | {r['rejection']} |")
    for r in wl["unrouted"]:
        lines.append(f"| {r['message']} | {r['layer']} | `{r['key']}` | {r['rejection']} |")
    lines += ["", f"Записано: WM {len(wl['working']['applied'])}, LT {len(wl['long_term']['applied'])}; "
              f"неудачных вызовов менеджера памяти: {len(wl['failed_calls'])}.", ""]

    full = variants["full"]["summary"]["tokens"]
    lines += ["## 3. Токены по слоям (вариант full, count_tokens)", "",
              "| Блок промпта | Токены |", "|---|---|"]
    for layer in PROMPT_ORDER:
        lines.append(f"| {LAYER_TITLES[layer]} | {_fmt(full['layers'].get(layer))} |")
    lines += [f"| Вопрос (в среднем) | {full['question_avg']} |",
              f"| **Весь промпт (в среднем)** | **{_fmt(full['total_avg'])}** |", "",
              "System instruction в count_tokens не входит (см. День 8); она есть в usage_metadata.prompt_token_count.", ""]

    lines += ["## 4. Варианты очистки", "",
              "| Вариант | Верно | ST | WM | LT | Токены LT / WM / ST | Промпт (ср.) | Изменённые файлы | Как ожидалось |",
              "|---|---|---|---|---|---|---|---|---|"]
    for v in variants.values():
        s = v["summary"]
        bl, t = s["by_layer"], s["tokens"]["layers"]
        lines.append(
            f"| {v['title']} | {s['correct']}/{s['total']} | {bl['short_term']['correct']}/{bl['short_term']['total']} | "
            f"{bl['working']['correct']}/{bl['working']['total']} | {bl['long_term']['correct']}/{bl['long_term']['total']} | "
            f"{_fmt(t.get('long_term'))} / {_fmt(t.get('working'))} / {_fmt(t.get('short_term'))} | "
            f"{_fmt(s['tokens']['total_avg'])} | {', '.join(v['changed_files']) or '—'} | "
            f"{'да' if s['matches_expectation'] else 'НЕТ'} |")
    lines += ["", "| Вариант | Потеряны (id) | Что осталось доступно |", "|---|---|---|"]
    for v in variants.values():
        a = v["available_data"]
        avail = (f"ST: {a['short_term']['stored_messages']} сообщ.; "
                 f"WM: {'задача #' + str(a['working']['task_id']) if a['working']['has_task'] else 'пусто'}; "
                 f"LT: {a['long_term']['facts']} фактов")
        lines.append(f"| {v['name']} | {', '.join(map(str, v['summary']['lost_vs_full'])) or '—'} | {avail} |")
    lines += ["", "## 5. Матрица ответов", "",
              "| # | Вопрос | Слой | " + " | ".join(variants) + " |",
              "|---|---|---|" + "---|" * len(variants)]
    for row in data["matrix"]:
        marks = " | ".join("✓" if row[name] else "✗" for name in variants)
        lines.append(f"| {row['id']} | {row['question']} | {SHORT[row['layer']]} | {marks} |")

    p = data["processing"]
    lines += ["", "## 6. Стоимость", ""]
    if meta["mode"] == "offline":
        lines += ["*Офлайн: токены фейковые, поэтому суммы ниже — не реальная цена (см. `--dry-run`).*", ""]
    lines += [f"Обновление памяти: {p['calls']} вызовов, вход {p['input_tokens']}, выход {p['output_tokens']} токенов.",
              ""]
    for v in variants.values():
        u = v["summary"]["usage"]
        lines.append(f"- {v['name']}: {u['calls']} ответов, вход {u['input_tokens']}, выход {u['output_tokens']}, "
                     f"оценка ${_fmt(u['estimated_cost_usd'])}")
    r = data["restart_check"]
    lines += ["", f"Перезапуск (новый MemoryManager из тех же файлов): содержимое {'совпало' if r['same_content'] else 'НЕ совпало'}, "
              f"промпт {'совпал' if r['same_prompt'] else 'НЕ совпал'}.", ""]
    return "\n".join(lines)
