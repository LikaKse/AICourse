"""Эксперимент «Разные способы рассуждения»: 5 задач × 4 способа × 2 уровня thinking_level.

Запуск (из папки gemini-task3):
    ../venv/bin/python experiment.py --dry-run            # показать план, без запросов к API
    ../venv/bin/python experiment.py --pilot              # пилот: 1 задача, 8 решений, 10 вызовов
    ../venv/bin/python experiment.py --full               # полный эксперимент: 40 решений, 50 вызовов
    ../venv/bin/python experiment.py --resume run_3       # дописать незавершённый запуск

Без --pilot / --full / --resume / --dry-run скрипт ничего не отправляет: полный эксперимент
запускается только явной командой. При импорте модуля ничего не выполняется.
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
from validators import ApiSolution, contains_any, parse_solution

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


# ---------- Темп запросов ----------

class Pacer:
    """Гарантирует минимальный интервал между началами любых двух запросов (включая повторы)."""

    def __init__(self, min_interval_sec):
        self.min_interval_sec = min_interval_sec
        self.last_start = None

    def wait(self):
        if self.last_start is not None:
            remaining = self.min_interval_sec - (monotonic() - self.last_start)
            if remaining > 0:
                sleep(remaining)
        self.last_start = monotonic()


# ---------- Конфигурация запроса ----------

def build_config(thinking_level, structured):
    """GenerateContentConfig: только thinking_level (без thinking_budget) и, для финальных
    ответов, единая JSON-схема {answer, explanation}."""
    kwargs = {
        "thinking_config": types.ThinkingConfig(thinking_level=types.ThinkingLevel[thinking_level]),
    }
    if structured:
        kwargs["response_mime_type"] = "application/json"
        kwargs["response_schema"] = ApiSolution
    return types.GenerateContentConfig(**kwargs)


def describe_config(thinking_level, structured):
    """Описание конфигурации для записи в результаты (JSON-совместимое)."""
    described = {"thinking_config": {"thinking_level": thinking_level}}
    if structured:
        described["response_mime_type"] = "application/json"
        described["response_schema"] = ApiSolution.model_json_schema()
    return described


# ---------- Разбор ответа API ----------

def extract_usage(response):
    """Поля usage_metadata; отсутствующие значения остаются None (не 0)."""
    usage = getattr(response, "usage_metadata", None)
    return {field: (getattr(usage, field, None) if usage is not None else None) for field in USAGE_FIELDS}


def extract_finish_reason(response):
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    reason = getattr(candidates[0], "finish_reason", None)
    if reason is None:
        return None
    return getattr(reason, "name", str(reason))


def extract_text(response):
    try:
        return response.text
    except Exception:
        return None


def is_daily_quota_error(error):
    """429 из-за ДНЕВНОЙ квоты (повторять бессмысленно до сброса)."""
    if not error or error.get("code") != 429:
        return False
    message = (error.get("message") or "").lower()
    compact = message.replace(" ", "")
    return any(marker.replace(" ", "") in compact for marker in config.DAILY_QUOTA_MARKERS)


def retry_hint_sec(message):
    """Подсказка сервера «Please retry in 45.03s» → 45.03 (или None)."""
    match = re.search(r"retry in ([\d.]+)s\b", message or "")
    return float(match.group(1)) if match else None


# ---------- Один вызов модели с повторами ----------

def call_model(client, model, stage, prompt, thinking_level, structured, pacer):
    """Вызов generate_content с ограниченными повторами при 503/504/429.

    Возвращает запись вызова: все попытки, итоговую ошибку, текст, finish_reason, usage.
    Дневная квота (429 «per day») не повторяется: daily_quota_exceeded=True.
    """
    call = {
        "stage": stage,
        "prompt": prompt,
        "request_config": describe_config(thinking_level, structured),
        "attempts": [],
        "attempts_count": 0,
        "retries_count": 0,
        "api_error": None,
        "daily_quota_exceeded": False,
        "latency_sec": None,
        "total_elapsed_sec": None,
        "finish_reason": None,
        "model_version": None,
        **{field: None for field in USAGE_FIELDS},
        "raw_text": None,
    }
    max_attempts = 1 + len(config.RETRY_DELAYS_SEC)
    call_start = monotonic()
    response = None
    for attempt in range(1, max_attempts + 1):
        pacer.wait()
        started_at = now_iso()
        attempt_start = monotonic()
        response, error = None, None
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=build_config(thinking_level, structured),
            )
        except errors.APIError as e:
            error = {"type": "APIError", "code": e.code, "status": e.status, "message": e.message}
        except Exception as e:  # сеть, таймаут клиента и т.п. — не скрываем, записываем
            error = {"type": type(e).__name__, "code": None, "status": None, "message": str(e)}
        latency = round(monotonic() - attempt_start, 3)

        daily = is_daily_quota_error(error)
        will_retry = (error is not None and not daily
                      and error["code"] in config.RETRY_STATUS_CODES and attempt < max_attempts)
        delay = None
        if will_retry:
            delay = config.RETRY_DELAYS_SEC[attempt - 1]
            if error["code"] == 429:
                hint = retry_hint_sec(error["message"])
                if hint is not None:
                    delay = min(max(delay, hint), config.MAX_RETRY_AFTER_SEC)

        call["attempts"].append({
            "attempt": attempt,
            "started_at": started_at,
            "latency_sec": latency,
            "status": "ok" if error is None else "api_error",
            "error": error,
            "retry_after_sec": delay,
        })
        call["latency_sec"] = latency
        call["api_error"] = error
        call["daily_quota_exceeded"] = daily
        if not will_retry:
            break
        log(f"\n    {stage}: попытка {attempt} — ошибка {error['code']}, повтор через {delay} с...", end=" ")
        sleep(delay)

    call["attempts_count"] = len(call["attempts"])
    call["retries_count"] = call["attempts_count"] - 1
    call["total_elapsed_sec"] = round(monotonic() - call_start, 3)
    if call["api_error"] is None and response is not None:
        call["raw_text"] = extract_text(response)
        call["finish_reason"] = extract_finish_reason(response)
        call["model_version"] = getattr(response, "model_version", None)
        call.update(extract_usage(response))
    return call


# ---------- Решение одной задачи одним способом ----------

def sum_optional(values):
    """Сумма известных значений. (None, False) — если неизвестно ни одно;
    флаг incomplete=True — если часть значений отсутствует (сумма тогда — нижняя оценка)."""
    known = [v for v in values if v is not None]
    if not known:
        return None, False
    total = sum(known)
    if isinstance(total, float):
        total = round(total, 3)
    return total, len(known) < len(values)


def build_final_prompt(strategy, task, generated_prompt=None):
    question, answer_format = task["question"], config.ANSWER_FORMAT
    if strategy == "direct":
        return config.DIRECT_TEMPLATE.format(question=question, answer_format=answer_format)
    if strategy == "step_by_step":
        return config.STEP_BY_STEP_TEMPLATE.format(question=question, answer_format=answer_format)
    if strategy == "prompt_generation":
        return config.PROMPT_GENERATION_SOLVE_TEMPLATE.format(
            generated_prompt=generated_prompt, question=question, answer_format=answer_format)
    if strategy == "expert_group":
        return config.EXPERT_GROUP_TEMPLATE.format(question=question, answer_format=answer_format)
    raise ValueError(f"Неизвестный способ: {strategy}")


def solution_id(task_id, strategy, thinking_level):
    return f"{task_id}|{strategy}|{thinking_level}"


def solve(client, model, task, strategy, thinking_level, pacer):
    """Одно решение: 1 вызов (или 2 для prompt_generation). Возвращает запись решения."""
    calls = []
    generated_prompt = None
    stage_error = None

    if strategy == "prompt_generation":
        generation_prompt = config.PROMPT_GENERATION_TEMPLATE.format(question=task["question"])
        generation = call_model(client, model, "prompt_generation", generation_prompt,
                                thinking_level, structured=False, pacer=pacer)
        calls.append(generation)
        if generation["api_error"] is None:
            generated_prompt = (generation["raw_text"] or "").strip()
            if not generated_prompt:
                stage_error = "модель вернула пустой промпт на этапе генерации"

    final_prompt = None
    if not calls or (calls[-1]["api_error"] is None and stage_error is None):
        final_prompt = build_final_prompt(strategy, task, generated_prompt)
        calls.append(call_model(client, model, "solve", final_prompt, thinking_level,
                                structured=True, pacer=pacer))

    successful = [c for c in calls if c["api_error"] is None]
    solve_call = calls[-1] if calls[-1]["stage"] == "solve" else None
    first_error = next(({"stage": c["stage"], **c["api_error"]} for c in calls if c["api_error"]), None)

    record = {
        "solution_id": solution_id(task["id"], strategy, thinking_level),
        "task_id": task["id"],
        "category": task["category"],
        "strategy": strategy,
        "thinking_level": thinking_level,
        "model": model,
        "timestamp": now_iso(),
        "prompt": final_prompt,
        "generated_prompt": generated_prompt,
        "generated_prompt_leaks_answer": None,
        "answer": None,
        "explanation": None,
        "expected_answer": task["expected"],
        "is_correct": None,
        "check_reason": None,
        "parse_status": None,
        "parse_error": None,
        "validation_status": None,
        "validation_errors": None,
        "finish_reason": solve_call["finish_reason"] if solve_call else None,
        "attempts": sum(c["attempts_count"] for c in calls),
        "retries": sum(c["retries_count"] for c in calls),
        "api_error": first_error,
        "daily_quota_exceeded": any(c["daily_quota_exceeded"] for c in calls),
        "stage_error": stage_error,
        "tokens_incomplete": False,
        "raw_text": solve_call["raw_text"] if solve_call else None,
        "calls": calls,
    }

    # Метрики: сумма по успешным вызовам (для prompt_generation — генерация + решение)
    record["latency_sec"], _ = sum_optional([c["latency_sec"] for c in successful])
    record["total_elapsed_sec"], _ = sum_optional([c["total_elapsed_sec"] for c in calls])
    incomplete = False
    for field in USAGE_FIELDS:
        value, partial = sum_optional([c[field] for c in successful])
        record[field] = value
        if field != "thoughts_token_count":  # thoughts может законно отсутствовать
            incomplete = incomplete or partial
    record["tokens_incomplete"] = incomplete

    if generated_prompt is not None and task["leak_patterns"]:
        record["generated_prompt_leaks_answer"] = contains_any(generated_prompt, task["leak_patterns"])

    # Правильность: только программная проверка
    if first_error is not None:
        return record                      # ответа нет — is_correct остаётся None
    if stage_error is not None:
        record["is_correct"] = False
        record["check_reason"] = stage_error
        return record
    parsed = parse_solution(solve_call["raw_text"])
    record.update({k: parsed[k] for k in ("parse_status", "parse_error", "validation_status",
                                          "validation_errors", "answer", "explanation")})
    if parsed["validation_status"] != "ok":
        record["is_correct"] = False
        record["check_reason"] = (f"ответ не разобран: {parsed['parse_status']}"
                                  if parsed["parse_status"] != "ok" else "ответ не прошёл схему")
        return record
    record["is_correct"], record["check_reason"] = task["check"](parsed["answer"])
    return record


# ---------- План, папки, сохранение ----------

def build_plan(task_ids, strategies, levels):
    """Порядок: задача → уровень → способ (способы чередуются, сбои API распределяются равномерно)."""
    return [(task_id, strategy, level)
            for task_id in task_ids
            for level in levels
            for strategy in strategies]


def expected_api_calls(plan):
    return sum(2 if strategy == "prompt_generation" else 1 for _, strategy, _ in plan)


def next_run_dir(results_dir):
    """Новая папка results/run_N (N — следующий свободный номер). Существующие не переиспользуются."""
    results_dir.mkdir(parents=True, exist_ok=True)
    numbers = [int(m.group(1)) for p in results_dir.iterdir()
               if (m := re.fullmatch(r"run_(\d+)", p.name))]
    run_dir = results_dir / f"run_{max(numbers, default=0) + 1}"
    run_dir.mkdir(exist_ok=False)
    return run_dir


def atomic_write_text(path, text):
    """Запись через временный файл + os.replace: прерывание не оставит полузаписанный файл."""
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


def make_meta(model, plan, mode):
    return {
        "model": model,
        "mode": mode,
        "created_at": now_iso(),
        "thinking_levels": list(config.THINKING_LEVELS),
        "strategies": list(config.STRATEGIES),
        "task_ids": sorted({t for t, _, _ in plan}, key=lambda t: [x["id"] for x in TASKS].index(t)),
        "plan": [list(item) for item in plan],
        "planned_solutions": len(plan),
        "expected_api_calls": expected_api_calls(plan),
        "settings": {
            "min_interval_between_requests_sec": config.MIN_INTERVAL_BETWEEN_REQUESTS_SEC,
            "request_timeout_sec": config.REQUEST_TIMEOUT_SEC,
            "retry_status_codes": list(config.RETRY_STATUS_CODES),
            "retry_delays_sec": list(config.RETRY_DELAYS_SEC),
            "max_retry_after_sec": config.MAX_RETRY_AFTER_SEC,
            "structured_final_answers": True,
            "max_output_tokens": None,
        },
        "sdk": {"google-genai": getattr(genai, "__version__", None)},
    }


# ---------- Точка входа ----------

def dry_run(plan, model):
    log(f"Модель: {model}")
    log(f"Решений: {len(plan)}, ожидаемых вызовов API (без повторов): {expected_api_calls(plan)}")
    log(f"Интервал между запросами: ≥ {config.MIN_INTERVAL_BETWEEN_REQUESTS_SEC} с; повторы при "
        f"{config.RETRY_STATUS_CODES}: паузы {config.RETRY_DELAYS_SEC} с "
        f"(до {1 + len(config.RETRY_DELAYS_SEC)} попыток на вызов); дневная квота 429 — остановка")
    log("Порядок решений:")
    for i, (task_id, strategy, level) in enumerate(plan, 1):
        calls = 2 if strategy == "prompt_generation" else 1
        log(f"  {i:2}. {task_id:22} {strategy:18} {level:8} ({calls} вызов{'а' if calls > 1 else ''})")
    task = TASKS_BY_ID[plan[0][0]]
    log(f"\nПромпты для задачи {task['id']}:")
    for strategy in config.STRATEGIES:
        log(f"\n=== {config.STRATEGY_NAMES[strategy]}")
        if strategy == "prompt_generation":
            log("[вызов 1, без JSON-схемы]\n" + config.PROMPT_GENERATION_TEMPLATE.format(question=task["question"]))
            log("\n[вызов 2, с JSON-схемой]\n" + build_final_prompt(strategy, task, "<сгенерированный промпт>"))
        else:
            log(build_final_prompt(strategy, task))
    for level in config.THINKING_LEVELS:
        for structured in (False, True):
            cfg = build_config(level, structured)
            log(f"\nКонфиг {level}, structured={structured}: "
                + json.dumps(cfg.model_dump(exclude_none=True, mode="json", exclude={"response_schema"}),
                             ensure_ascii=False)
                + (" + response_schema=ApiSolution" if structured else ""))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Сравнение способов рассуждения в Gemini API")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="показать план и промпты, без запросов к API")
    mode.add_argument("--pilot", action="store_true",
                      help=f"пилот: задача {config.PILOT_TASK_ID}, 4 способа × 2 уровня (10 вызовов)")
    mode.add_argument("--full", action="store_true", help="полный эксперимент: 5 задач (50 вызовов)")
    mode.add_argument("--resume", metavar="RUN_DIR", help="дописать незавершённый запуск, например run_3")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model = config.MODEL

    if args.dry_run:
        dry_run(build_plan([t["id"] for t in TASKS], config.STRATEGIES, config.THINKING_LEVELS), model)
        return 0
    if not (args.pilot or args.full or args.resume):
        log("Режим не указан. Используйте --dry-run, --pilot, --full или --resume RUN_DIR.")
        return 2

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        log("Ошибка: не задан API-ключ. Установите переменную окружения GEMINI_API_KEY.")
        return 1

    # Папка запуска: новая run_N или существующая при --resume
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
        if meta["model"] != model:
            log(f"Ошибка: запуск сделан на {meta['model']}, а в config.MODEL — {model}")
            return 1
        plan = [tuple(item) for item in meta["plan"]]
        raw_path = run_dir / RAW_RESULTS_NAME
        previous = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else []
        records = [r for r in previous if r["api_error"] is None]
        failed = [r for r in previous if r["api_error"] is not None]
        if failed:
            # Неудачные решения не удаляются: переносятся в superseded.json, затем переделываются
            superseded_path = run_dir / SUPERSEDED_NAME
            superseded = (json.loads(superseded_path.read_text(encoding="utf-8"))
                          if superseded_path.exists() else [])
            atomic_write_json(superseded_path, superseded + failed)
        done = {r["solution_id"] for r in records}
        todo = [item for item in plan if solution_id(*item) not in done]
        log(f"Продолжение {run_dir.name}: готово {len(done)}/{len(plan)}, к выполнению {len(todo)}"
            + (f" (ранее неудачных: {len(failed)} — сохранены в {SUPERSEDED_NAME})" if failed else ""))
    else:
        task_ids = [config.PILOT_TASK_ID] if args.pilot else [t["id"] for t in TASKS]
        plan = build_plan(task_ids, config.STRATEGIES, config.THINKING_LEVELS)
        run_dir = next_run_dir(RESULTS_DIR)
        meta = make_meta(model, plan, "pilot" if args.pilot else "full")
        atomic_write_json(run_dir / META_NAME, meta)
        records, todo = [], plan
        log(f"Новый запуск {run_dir.name} ({meta['mode']}): {len(plan)} решений, "
            f"~{meta['expected_api_calls']} вызовов API")

    client = genai.Client(api_key=api_key,
                          http_options=types.HttpOptions(timeout=config.REQUEST_TIMEOUT_SEC * 1000))
    pacer = Pacer(config.MIN_INTERVAL_BETWEEN_REQUESTS_SEC)
    save_progress(run_dir, records, meta, "в процессе")

    status = "завершён"
    exit_code = 0
    try:
        for index, (task_id, strategy, level) in enumerate(todo, 1):
            log(f"[{index}/{len(todo)}] {task_id} | {strategy} | {level}...", end=" ")
            record = solve(client, model, TASKS_BY_ID[task_id], strategy, level, pacer)
            records.append(record)
            if record["api_error"]:
                e = record["api_error"]
                log(f"ошибка API {e['code']} ({e['stage']}): {summary.one_line(e['message'])[:100]}")
            else:
                mark = {True: "верно", False: "НЕВЕРНО"}[record["is_correct"]]
                log(f"{mark}: {summary.one_line(record['answer'] or record['check_reason'] or '')[:60]} "
                    f"| {record['latency_sec']} с | токенов {record['total_token_count']}")
            if record["daily_quota_exceeded"]:
                status = "остановлен: исчерпана дневная квота (429)"
                save_progress(run_dir, records, meta, status)
                log(f"\nДневная квота исчерпана. Продолжить позже: "
                    f"../venv/bin/python experiment.py --resume {run_dir.name}")
                exit_code = 3
                break
            save_progress(run_dir, records, meta, "в процессе")
    except KeyboardInterrupt:
        status = "прерван пользователем (Ctrl+C)"
        exit_code = 130
        log(f"\nПрервано. Сохранено решений: {len(records)}. Продолжить: "
            f"../venv/bin/python experiment.py --resume {run_dir.name}")

    save_progress(run_dir, records, meta, status)
    log(f"\nРезультаты: {run_dir / RAW_RESULTS_NAME}\nСводка: {run_dir / SUMMARY_NAME}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
