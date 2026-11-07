"""Эксперимент «Температура»: 2 задачи × 4 температуры × 5 повторов = 40 вызовов API.

Запуск (из папки gemini-task4):
    ../venv/bin/python experiment.py --dry-run          # план и параметры, без запросов к API
    ../venv/bin/python experiment.py --full             # 40 вызовов
    ../venv/bin/python experiment.py --resume run_N     # дописать незавершённый запуск

Без --full / --resume / --dry-run скрипт ничего не отправляет. При импорте ничего не выполняется.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import errors, types

import config
import summary
from tasks import TASKS, TASKS_BY_ID

RESULTS_DIR = Path(__file__).parent / "results"
RAW_RESULTS_NAME = "raw_results.json"
SUMMARY_NAME = "summary.md"
META_NAME = "run_meta.json"
SUPERSEDED_NAME = "superseded.json"

USAGE_FIELDS = ("prompt_token_count", "candidates_token_count", "thoughts_token_count", "total_token_count")

# Подменяются в офлайн-проверках
sleep = time.sleep
monotonic = time.monotonic


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def log(message="", end="\n"):
    print(message, end=end, flush=True)


class Pacer:
    """Минимальный интервал между началами любых двух запросов (включая повторы)."""

    def __init__(self, min_interval_sec):
        self.min_interval_sec = min_interval_sec
        self.last_start = None

    def wait(self):
        if self.last_start is not None:
            remaining = self.min_interval_sec - (monotonic() - self.last_start)
            if remaining > 0:
                sleep(remaining)
        self.last_start = monotonic()


# ---------- Конфигурация: меняется только temperature ----------

def build_config(temperature):
    return types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=config.MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel[config.THINKING_LEVEL]),
    )


def describe_config(temperature):
    return build_config(temperature).model_dump(exclude_none=True, mode="json")


# ---------- Ответ API ----------

def extract_usage(response):
    usage = getattr(response, "usage_metadata", None)
    return {f: (getattr(usage, f, None) if usage is not None else None) for f in USAGE_FIELDS}


def extract_finish_reason(response):
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    reason = getattr(candidates[0], "finish_reason", None)
    return None if reason is None else getattr(reason, "name", str(reason))


def extract_text(response):
    try:
        return response.text
    except Exception:
        return None


def is_daily_quota_error(error):
    if not error or error.get("code") != 429:
        return False
    compact = (error.get("message") or "").lower().replace(" ", "")
    return any(m.replace(" ", "") in compact for m in config.DAILY_QUOTA_MARKERS)


def retry_hint_sec(message):
    match = re.search(r"retry in ([\d.]+)s\b", message or "")
    return float(match.group(1)) if match else None


def call_model(client, prompt, temperature, pacer):
    """Один вызов с ограниченными повторами при 503/504/429. Возвращает dict с попытками и ответом."""
    call = {"attempts": [], "api_error": None, "daily_quota_exceeded": False, "latency_sec": None,
            "total_elapsed_sec": None, "finish_reason": None, "model_version": None,
            **{f: None for f in USAGE_FIELDS}, "raw_text": None}
    max_attempts = 1 + len(config.RETRY_DELAYS_SEC)
    call_start = monotonic()
    response = None
    for attempt in range(1, max_attempts + 1):
        pacer.wait()
        started_at, attempt_start = now_iso(), monotonic()
        response, error = None, None
        try:
            response = client.models.generate_content(model=config.MODEL, contents=prompt,
                                                      config=build_config(temperature))
        except errors.APIError as e:
            error = {"type": "APIError", "code": e.code, "status": e.status, "message": e.message}
        except Exception as e:  # таймаут клиента, сеть — записываем, не скрываем
            error = {"type": type(e).__name__, "code": None, "status": None, "message": str(e)}
        latency = round(monotonic() - attempt_start, 3)
        daily = is_daily_quota_error(error)
        will_retry = (error is not None and not daily and error["code"] in config.RETRY_STATUS_CODES
                      and attempt < max_attempts)
        delay = None
        if will_retry:
            delay = config.RETRY_DELAYS_SEC[attempt - 1]
            if error["code"] == 429 and (hint := retry_hint_sec(error["message"])) is not None:
                delay = min(max(delay, hint), config.MAX_RETRY_AFTER_SEC)
        call["attempts"].append({"attempt": attempt, "started_at": started_at, "latency_sec": latency,
                                 "status": "ok" if error is None else "api_error", "error": error,
                                 "retry_after_sec": delay})
        call.update(latency_sec=latency, api_error=error, daily_quota_exceeded=daily)
        if not will_retry:
            break
        log(f"\n    попытка {attempt} — ошибка {error['code']}, повтор через {delay} с...", end=" ")
        sleep(delay)
    call["total_elapsed_sec"] = round(monotonic() - call_start, 3)
    if call["api_error"] is None and response is not None:
        call["raw_text"] = extract_text(response)
        call["finish_reason"] = extract_finish_reason(response)
        call["model_version"] = getattr(response, "model_version", None)
        call.update(extract_usage(response))
    else:
        call["latency_sec"] = None   # время считаем только для успешного ответа
    return call


# ---------- Один прогон ----------

def solution_id(task_id, temperature, run):
    return f"{task_id}|t={temperature}|run{run}"


def run_one(client, task, temperature, run, pacer):
    call = call_model(client, task["prompt"], temperature, pacer)
    record = {
        "solution_id": solution_id(task["id"], temperature, run),
        "task_id": task["id"],
        "kind": task["kind"],
        "temperature": temperature,
        "run": run,
        "model": config.MODEL,
        "thinking_level": config.THINKING_LEVEL,
        "max_output_tokens": config.MAX_OUTPUT_TOKENS,
        "timestamp": now_iso(),
        "prompt": task["prompt"],
        "request_config": describe_config(temperature),
        "expected_answer": task["expected"],
        "raw_text": call["raw_text"],                 # полный исходный текст, даже если JSON не разобран
        "finish_reason": call["finish_reason"],
        "model_version": call["model_version"],
        **{f: call[f] for f in USAGE_FIELDS},
        "latency_sec": call["latency_sec"],
        "total_elapsed_sec": call["total_elapsed_sec"],
        "attempts": len(call["attempts"]),
        "retries": len(call["attempts"]) - 1,
        "attempt_log": call["attempts"],
        "api_error": call["api_error"],
        "daily_quota_exceeded": call["daily_quota_exceeded"],
        "evaluation": None,
    }
    if call["api_error"] is None:
        record["evaluation"] = task["evaluate"](call["raw_text"], call["finish_reason"])
    return record


# ---------- План, папки, сохранение ----------

def build_plan():
    """Порядок: повтор → задача → температура (сбои сервера распределяются по всем комбинациям)."""
    return [(task["id"], t, run)
            for run in range(1, config.RUNS_PER_COMBINATION + 1)
            for task in TASKS
            for t in config.TEMPERATURES]


def next_run_dir(results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)
    numbers = [int(m.group(1)) for p in results_dir.iterdir() if (m := re.fullmatch(r"run_(\d+)", p.name))]
    run_dir = results_dir / f"run_{max(numbers, default=0) + 1}"
    run_dir.mkdir(exist_ok=False)
    return run_dir


def atomic_write_text(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path, data):
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def save_progress(run_dir, records, meta, status):
    order = {solution_id(*item): i for i, item in enumerate(meta["plan"])}
    records.sort(key=lambda r: order.get(r["solution_id"], len(order)))
    atomic_write_json(run_dir / RAW_RESULTS_NAME, records)
    atomic_write_text(run_dir / SUMMARY_NAME, summary.build_summary(records, meta, status))


def make_meta(plan):
    return {
        "model": config.MODEL,
        "created_at": now_iso(),
        "thinking_level": config.THINKING_LEVEL,
        "max_output_tokens": config.MAX_OUTPUT_TOKENS,
        "temperatures": list(config.TEMPERATURES),
        "runs_per_combination": config.RUNS_PER_COMBINATION,
        "tasks": {t["id"]: {"kind": t["kind"], "title": t["title"], "prompt": t["prompt"],
                            "expected": t["expected"]} for t in TASKS},
        "plan": [list(item) for item in plan],
        "planned_calls": len(plan),
        "settings": {
            "min_interval_between_requests_sec": config.MIN_INTERVAL_BETWEEN_REQUESTS_SEC,
            "request_timeout_sec": config.REQUEST_TIMEOUT_SEC,
            "retry_status_codes": list(config.RETRY_STATUS_CODES),
            "retry_delays_sec": list(config.RETRY_DELAYS_SEC),
            "structured_output": False,
        },
        "sdk": {"google-genai": getattr(genai, "__version__", None)},
    }


# ---------- Точка входа ----------

def dry_run(plan):
    log(f"Модель: {config.MODEL}; thinking_level: {config.THINKING_LEVEL}; max_output_tokens: {config.MAX_OUTPUT_TOKENS}")
    log(f"Температуры: {config.TEMPERATURES}; повторов: {config.RUNS_PER_COMBINATION}; "
        f"вызовов API (без повторов): {len(plan)}")
    log(f"Интервал между запросами ≥ {config.MIN_INTERVAL_BETWEEN_REQUESTS_SEC} с; повторы при "
        f"{config.RETRY_STATUS_CODES}: паузы {config.RETRY_DELAYS_SEC} с; дневная квота 429 — остановка")
    log("Конфиги (отличаются только temperature):")
    for t in config.TEMPERATURES:
        log(f"  {json.dumps(describe_config(t), ensure_ascii=False)}")
    log("Порядок вызовов:")
    for i, (task_id, t, run) in enumerate(plan, 1):
        log(f"  {i:2}. {task_id:11} t={t:<4} прогон {run}")
    for task in TASKS:
        log(f"\n=== Промпт {task['id']} (одинаковый во всех вызовах):\n{task['prompt']}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Влияние temperature на ответы Gemini")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="показать план, без запросов к API")
    mode.add_argument("--full", action="store_true", help="полный эксперимент: 40 вызовов")
    mode.add_argument("--resume", metavar="RUN_DIR", help="дописать незавершённый запуск, например run_1")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        dry_run(build_plan())
        return 0
    if not (args.full or args.resume):
        log("Режим не указан. Используйте --dry-run, --full или --resume RUN_DIR.")
        return 2
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        log("Ошибка: не задан API-ключ. Установите переменную окружения GEMINI_API_KEY.")
        return 1

    if args.resume:
        if not re.fullmatch(r"run_\d+", args.resume):
            log(f"Ошибка: ожидается имя папки вида run_N, получено {args.resume!r}")
            return 2
        run_dir = RESULTS_DIR / args.resume
        meta_path = run_dir / META_NAME
        if not meta_path.exists():
            log(f"Ошибка: {run_dir} не найден или в нём нет {META_NAME}")
            return 1
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        current = {"model": config.MODEL, "thinking_level": config.THINKING_LEVEL,
                   "max_output_tokens": config.MAX_OUTPUT_TOKENS,
                   "prompts": {t["id"]: t["prompt"] for t in TASKS}}
        saved = {"model": meta["model"], "thinking_level": meta["thinking_level"],
                 "max_output_tokens": meta["max_output_tokens"],
                 "prompts": {k: v["prompt"] for k, v in meta["tasks"].items()}}
        if current != saved:
            log("Ошибка: параметры или промпты в config.py отличаются от сохранённых в run_meta.json — "
                "продолжение исказило бы эксперимент.")
            return 1
        plan = [tuple(item) for item in meta["plan"]]
        raw_path = run_dir / RAW_RESULTS_NAME
        previous = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else []
        records = [r for r in previous if r["api_error"] is None]
        failed = [r for r in previous if r["api_error"] is not None]
        if failed:
            sup_path = run_dir / SUPERSEDED_NAME
            sup = json.loads(sup_path.read_text(encoding="utf-8")) if sup_path.exists() else []
            atomic_write_json(sup_path, sup + failed)
        done = {r["solution_id"] for r in records}
        todo = [item for item in plan if solution_id(*item) not in done]
        log(f"Продолжение {run_dir.name}: готово {len(done)}/{len(plan)}, к выполнению {len(todo)}"
            + (f" (неудачных ранее: {len(failed)} — сохранены в {SUPERSEDED_NAME})" if failed else ""))
    else:
        plan = build_plan()
        run_dir = next_run_dir(RESULTS_DIR)
        meta = make_meta(plan)
        atomic_write_json(run_dir / META_NAME, meta)
        records, todo = [], plan
        log(f"Новый запуск {run_dir.name}: {len(plan)} вызовов API")

    client = genai.Client(api_key=api_key,
                          http_options=types.HttpOptions(timeout=config.REQUEST_TIMEOUT_SEC * 1000))
    pacer = Pacer(config.MIN_INTERVAL_BETWEEN_REQUESTS_SEC)
    save_progress(run_dir, records, meta, "в процессе")
    status, exit_code = "завершён", 0
    try:
        for index, (task_id, t, run) in enumerate(todo, 1):
            log(f"[{index}/{len(todo)}] {task_id} | t={t} | прогон {run}...", end=" ")
            record = run_one(client, TASKS_BY_ID[task_id], t, run, pacer)
            records.append(record)
            log(summary.console_line(record))
            if record["daily_quota_exceeded"]:
                status, exit_code = "остановлен: исчерпана дневная квота (429)", 3
                save_progress(run_dir, records, meta, status)
                log(f"\nДневная квота исчерпана. Продолжить позже: "
                    f"../venv/bin/python experiment.py --resume {run_dir.name}")
                break
            save_progress(run_dir, records, meta, "в процессе")
    except KeyboardInterrupt:
        status, exit_code = "прерван пользователем (Ctrl+C)", 130
        log(f"\nПрервано. Сохранено: {len(records)}. Продолжить: "
            f"../venv/bin/python experiment.py --resume {run_dir.name}")
    if exit_code == 0 and any(r["api_error"] for r in records):
        status = "завершён с ошибками API — продолжить через --resume"
    save_progress(run_dir, records, meta, status)
    log(f"\nРезультаты: {run_dir / RAW_RESULTS_NAME}\nСводка: {run_dir / SUMMARY_NAME}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
