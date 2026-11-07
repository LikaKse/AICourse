"""Эксперимент «Версии моделей»: одни и те же задачи на трёх моделях Gemini.

Запуск (из папки gemini-task5):
    ../venv/bin/python main.py --list-models       # какие модели доступны (бесплатно, без генерации)
    ../venv/bin/python main.py --dry-run           # план, промпты, тарифы, оценка цены; без API
    ../venv/bin/python main.py --pilot             # 1 задача × 3 модели = 3 платных вызова
    ../venv/bin/python main.py --full              # 6 задач × 3 модели = 18 платных вызовов
    ../venv/bin/python main.py --resume run_N      # повторить неудачные вызовы запуска run_N
    ../venv/bin/python main.py --report run_N      # пересобрать итоговую таблицу без API

Перед платными запросами скрипт показывает план и спрашивает подтверждение (--yes отключает вопрос).
При импорте модуля ничего не выполняется.
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import config
import pricing
import report
from tasks import TASKS, TASKS_BY_ID

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
RESULTS_JSON = "results.json"
RESULTS_CSV = "results.csv"
META_JSON = "run_meta.json"
SUPERSEDED_JSON = "superseded.json"
LOG_NAME = "run.log"

API_KEY_ENV = "GEMINI_API_KEY"

# Порядок и состав колонок results.csv (в JSON — те же поля плюс attempts_log)
CSV_FIELDS = [
    "record_id", "timestamp", "model", "model_role", "model_version", "task_id", "category", "repeat",
    "status", "expected_answer", "answer", "is_correct", "check_reason", "parse_status",
    "latency_sec", "total_elapsed_sec", "attempts", "retries",
    "input_tokens", "output_tokens", "candidates_tokens", "thoughts_tokens", "cached_tokens", "total_tokens",
    "price_input_per_1m", "price_output_per_1m", "input_cost_usd", "output_cost_usd", "cost_usd", "cost_note",
    "finish_reason", "truncated", "error_code", "error_status", "error_message", "explanation",
]

# Подменяются в офлайн-проверках
sleep = time.sleep
monotonic = time.monotonic

log = logging.getLogger("task5")


# ---------- Логирование без утечки ключа ----------

class RedactSecretsFilter(logging.Filter):
    """Заменяет значение API-ключа на *** в любом сообщении лога (на случай, если оно попадёт в текст ошибки)."""

    def __init__(self, secrets):
        super().__init__()
        self.secrets = [s for s in secrets if s]

    def filter(self, record):
        if self.secrets:
            message = record.getMessage()
            for secret in self.secrets:
                message = message.replace(secret, "***")
            record.msg, record.args = message, None
        return True


def redact(text):
    """Убирает API-ключ из строки, которая попадёт в результаты."""
    key = os.environ.get(API_KEY_ENV)
    if text and key:
        return str(text).replace(key, "***")
    return text


def setup_logging(log_file=None):
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    redactor = RedactSecretsFilter([os.environ.get(API_KEY_ENV)])
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    console.addFilter(redactor)
    log.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        file_handler.addFilter(redactor)
        log.addHandler(file_handler)


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


# ---------- Темп запросов ----------

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


# ---------- Запрос ----------

def build_prompt(task):
    return config.PROMPT_TEMPLATE.format(question=task["question"], answer_format=config.ANSWER_FORMAT)


def build_config():
    """Одинаковая конфигурация для всех моделей."""
    from google.genai import types
    from validators import ApiSolution
    kwargs = {
        "thinking_config": types.ThinkingConfig(thinking_level=types.ThinkingLevel[config.THINKING_LEVEL]),
        "response_mime_type": "application/json",
        "response_schema": ApiSolution,
    }
    if config.MAX_OUTPUT_TOKENS is not None:
        kwargs["max_output_tokens"] = config.MAX_OUTPUT_TOKENS
    return types.GenerateContentConfig(**kwargs)


def describe_config():
    from validators import ApiSolution
    return {
        "thinking_config": {"thinking_level": config.THINKING_LEVEL},
        "response_mime_type": "application/json",
        "response_schema": ApiSolution.model_json_schema(),
        "temperature": "не задана (по умолчанию модели)",
        "max_output_tokens": (config.MAX_OUTPUT_TOKENS if config.MAX_OUTPUT_TOKENS is not None
                              else "не задан (максимум модели)"),
    }


def make_client():
    from google import genai
    from google.genai import types
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise SystemExit(f"Переменная окружения {API_KEY_ENV} не задана. "
                         f"Выполните: export {API_KEY_ENV}=\"ваш_ключ\"")
    return genai.Client(api_key=api_key,
                        http_options=types.HttpOptions(timeout=config.REQUEST_TIMEOUT_SEC * 1000))


# ---------- Разбор ответа API ----------

def extract_usage(response):
    """Токены из usage_metadata. Отсутствующие значения остаются None, а не 0."""
    usage = getattr(response, "usage_metadata", None)

    def get(name):
        return getattr(usage, name, None) if usage is not None else None

    return {
        "input_tokens": get("prompt_token_count"),
        "candidates_tokens": get("candidates_token_count"),
        "thoughts_tokens": get("thoughts_token_count"),
        "cached_tokens": get("cached_content_token_count"),
        "total_tokens": get("total_token_count"),
    }


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


def classify_exception(exc):
    """Исключение → dict ошибки: type, code, status, message, retryable."""
    from google.genai import errors
    if isinstance(exc, errors.APIError):
        code = exc.code
        return {"type": type(exc).__name__, "code": code, "status": exc.status,
                "message": redact(exc.message), "retryable": code in config.RETRY_STATUS_CODES}
    # Сетевые ошибки и таймауты клиента (httpx) считаем временными
    name = type(exc).__name__
    transient = any(marker in name for marker in ("Timeout", "Connect", "RemoteProtocol", "Network", "ReadError"))
    return {"type": name, "code": None, "status": None, "message": redact(str(exc)), "retryable": transient}


def is_daily_quota_error(error):
    if not error or error.get("code") != 429:
        return False
    compact = (error.get("message") or "").lower().replace(" ", "")
    return any(marker.replace(" ", "") in compact for marker in config.DAILY_QUOTA_MARKERS)


def retry_hint_sec(message):
    """Подсказка сервера «Please retry in 45.03s» → 45.03 (или None)."""
    match = re.search(r"retry in ([\d.]+)s\b", message or "")
    return float(match.group(1)) if match else None


def call_with_retries(client, model, prompt, pacer):
    """generate_content с ограниченными повторами при временных ошибках.

    Возвращает (response | None, итоговая ошибка | None, журнал попыток, latency успешной попытки,
    общее время с паузами).
    """
    attempts = []
    max_attempts = 1 + len(config.RETRY_DELAYS_SEC)
    call_start = monotonic()
    response, error, latency = None, None, None
    for attempt in range(1, max_attempts + 1):
        pacer.wait()
        started_at = now_iso()
        t0 = monotonic()
        response, error = None, None
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=build_config())
        except Exception as exc:  # ошибки не скрываем: всё записывается в attempts
            error = classify_exception(exc)
        latency = round(monotonic() - t0, 3)

        daily = is_daily_quota_error(error)
        will_retry = error is not None and error["retryable"] and not daily and attempt < max_attempts
        delay = None
        if will_retry:
            delay = config.RETRY_DELAYS_SEC[attempt - 1]
            if error["code"] == 429:
                hint = retry_hint_sec(error["message"])
                if hint is not None:
                    delay = min(max(delay, hint), config.MAX_RETRY_AFTER_SEC)
        attempts.append({"attempt": attempt, "started_at": started_at, "latency_sec": latency,
                         "status": "ok" if error is None else "error", "error": error,
                         "daily_quota": daily, "retry_after_sec": delay})
        if error is None:
            log.info(f"    попытка {attempt}: ок, {latency:.2f} с")
            break
        log.warning(f"    попытка {attempt}: ошибка {error['code'] or error['type']} "
                    f"({error['status'] or ''}) {error['message'][:200]}")
        if not will_retry:
            break
        log.info(f"    повтор через {delay} с")
        sleep(delay)
    total_elapsed = round(monotonic() - call_start, 3)
    return response, error, attempts, (latency if error is None else None), total_elapsed


# ---------- Одно решение ----------

def record_id(task_id, model, repeat):
    return f"{task_id}|{model}|{repeat}"


def empty_record(task, model_info, repeat):
    return {
        "record_id": record_id(task["id"], model_info["id"], repeat),
        "timestamp": now_iso(),
        "model": model_info["id"],
        "model_role": model_info["role"],
        "model_version": None,
        "task_id": task["id"],
        "category": task["category"],
        "repeat": repeat,
        "status": None,                # ok | api_error | skipped
        "expected_answer": task["expected"],
        "answer": None,
        "explanation": None,
        "is_correct": None,
        "check_reason": None,
        "parse_status": None,
        "latency_sec": None,
        "total_elapsed_sec": None,
        "attempts": 0,
        "retries": 0,
        "input_tokens": None,
        "output_tokens": None,         # оплачиваемые: candidates + thoughts
        "candidates_tokens": None,
        "thoughts_tokens": None,
        "cached_tokens": None,
        "total_tokens": None,
        "price_input_per_1m": None,
        "price_output_per_1m": None,
        "input_cost_usd": None,
        "output_cost_usd": None,
        "cost_usd": None,
        "cost_note": None,
        "finish_reason": None,
        "truncated": None,             # True, если ответ упёрся в max_output_tokens
        "error_code": None,
        "error_status": None,
        "error_message": None,
        "daily_quota": False,
        "prompt": build_prompt(task),
        "raw_text": None,
        "attempts_log": [],
    }


def skipped_record(task, model_info, repeat, reason):
    record = empty_record(task, model_info, repeat)
    record.update(status="skipped", is_correct=False, check_reason=f"не отправлялся: {reason}",
                  cost_note="запрос не отправлялся")
    return record


def grade(record, task):
    """Программная проверка ответа (модели не доверяем)."""
    from validators import parse_solution
    parsed = parse_solution(record["raw_text"])
    record["parse_status"] = parsed["parse_status"]
    if parsed["validation_status"] != "ok":
        record["is_correct"] = False
        record["check_reason"] = (f"ответ не разобран: {parsed['parse_status']} {parsed['parse_error'] or ''}".strip()
                                  if parsed["parse_status"] != "ok" else "ответ не прошёл схему")
        return
    record["answer"] = parsed["answer"]
    record["explanation"] = parsed["explanation"]
    record["is_correct"], record["check_reason"] = task["check"](parsed["answer"])


def solve(client, task, model_info, repeat, pacer):
    record = empty_record(task, model_info, repeat)
    response, error, attempts, latency, total_elapsed = call_with_retries(
        client, model_info["id"], record["prompt"], pacer)
    record.update(attempts_log=attempts, attempts=len(attempts), retries=len(attempts) - 1,
                  latency_sec=latency, total_elapsed_sec=total_elapsed, timestamp=now_iso())
    if error is not None:
        record.update(status="api_error", is_correct=False, check_reason="ошибка API — ответа нет",
                      error_code=error["code"], error_status=error["status"],
                      error_message=error["message"], daily_quota=attempts[-1]["daily_quota"],
                      cost_note="запрос завершился ошибкой, usage_metadata нет")
        return record

    record["status"] = "ok"
    record["raw_text"] = extract_text(response)
    record["finish_reason"] = extract_finish_reason(response)
    record["truncated"] = record["finish_reason"] == "MAX_TOKENS"
    record["model_version"] = getattr(response, "model_version", None)
    record.update(extract_usage(response))
    record["output_tokens"] = pricing.billable_output_tokens(record["candidates_tokens"], record["thoughts_tokens"])
    record.update(pricing.compute_cost(model_info["id"], record["timestamp"], record["input_tokens"],
                                       record["candidates_tokens"], record["thoughts_tokens"],
                                       record["cached_tokens"]))
    grade(record, task)
    if record["truncated"]:
        record["check_reason"] = f"ответ обрезан по max_output_tokens ({config.MAX_OUTPUT_TOKENS}); {record['check_reason']}"
        log.warning(f"    ответ обрезан: finish_reason=MAX_TOKENS (лимит {config.MAX_OUTPUT_TOKENS})")
    return record


# ---------- Сохранение ----------

def atomic_write_text(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path, data):
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def write_csv(path, rows, fields):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})
    os.replace(tmp, path)


def save_all(run_dir, records, meta):
    order = {rid: i for i, rid in enumerate(meta["planned_record_ids"])}
    records = sorted(records, key=lambda r: order.get(r["record_id"], 10**9))
    write_json(run_dir / RESULTS_JSON, records)
    write_csv(run_dir / RESULTS_CSV, records, CSV_FIELDS)
    report.write_reports(run_dir, records, meta)


def new_run_dir():
    RESULTS_DIR.mkdir(exist_ok=True)
    numbers = [int(m.group(1)) for p in RESULTS_DIR.iterdir() if (m := re.fullmatch(r"run_(\d+)", p.name))]
    run_dir = RESULTS_DIR / f"run_{max(numbers, default=0) + 1}"
    run_dir.mkdir()
    return run_dir


# ---------- План ----------

def plan(task_ids, repeats):
    """Порядок вызовов: повтор → задача → модель (модели чередуются, чтобы нагрузка на сервер
    в разное время влияла на все модели примерно одинаково)."""
    items = []
    for repeat in range(1, repeats + 1):
        for task_id in task_ids:
            for model_info in config.MODELS:
                items.append((TASKS_BY_ID[task_id], model_info, repeat))
    return items


TYPICAL_INPUT_TOKENS = 150     # типичная оценка (не предел): по Дню 3 ответы были 130–2 200 токенов
TYPICAL_OUTPUT_TOKENS = 1000


def print_plan(items, repeats):
    n_tasks = len({t["id"] for t, _, _ in items})
    model_ids = sorted({m["id"] for _, m, _ in items}, key=config.MODEL_IDS.index)
    today = datetime.now().date()
    limit = (f"max_output_tokens={config.MAX_OUTPUT_TOKENS} (включая thinking)"
             if config.MAX_OUTPUT_TOKENS is not None else "max_output_tokens не задан (максимум модели)")
    log.info(f"Модели ({len(model_ids)}), thinking_level={config.THINKING_LEVEL}, {limit}, "
             f"JSON-схема {{answer, explanation}}:")
    for m in config.MODELS:
        if m["id"] not in model_ids:
            continue
        p = pricing.price_for(m["id"], today)
        log.info(f"  {m['role_name']:<20} {m['id']:<24} ${p['input']:.2f} / ${p['output']:.2f} за 1M "
                 f"(вход / выход вкл. thinking), до {pricing.max_output_tokens_for(m['id'])} вых. токенов"
                 f"{'' if p['free_tier'] else '  [бесплатного уровня нет]'}")
    log.info(f"Задач: {n_tasks}, повторов: {repeats}, вызовов: {len(items)}; "
             f"интервал между запросами ≥ {config.MIN_INTERVAL_BETWEEN_REQUESTS_SEC} с "
             f"(≈ {len(items) * config.MIN_INTERVAL_BETWEEN_REQUESTS_SEC / 60:.1f} мин минимум)")

    # Стоимость считается по вызовам в плане: модель → число вызовов
    calls = {mid: sum(1 for _, m, _ in items if m["id"] == mid) for mid in model_ids}
    typical = sum(c * pricing.max_call_cost(mid, today, TYPICAL_INPUT_TOKENS, TYPICAL_OUTPUT_TOKENS)
                  for mid, c in calls.items())
    worst = {mid: c * pricing.max_call_cost(mid, today) for mid, c in calls.items()}
    worst_total = sum(worst.values())
    attempts = 1 + len(config.RETRY_DELAYS_SEC)
    log.info(f"Типичная стоимость ({TYPICAL_INPUT_TOKENS} вход + {TYPICAL_OUTPUT_TOKENS} выход на вызов, "
             f"не предел): ≈ ${typical:.4f}")
    log.info(f"МАКСИМАЛЬНАЯ стоимость (≤{config.MAX_INPUT_TOKENS_ESTIMATE} вход + весь лимит выхода, "
             f"1 оплаченная попытка на вызов): "
             + ", ".join(f"{mid} ≤ ${c:.4f}" for mid, c in worst.items()) + f"; всего ≤ ${worst_total:.4f}")
    log.info(f"Абсолютный худший случай (все {attempts} попытки оплачены, напр. таймауты после генерации): "
             f"≤ ${worst_total * attempts:.4f}")


def confirm(items, assume_yes):
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        log.error("Нет терминала для подтверждения. Добавьте --yes, если согласны на платные запросы.")
        return False
    answer = input(f"Отправить {len(items)} платных запросов к Gemini API? [y/N] ").strip().lower()
    return answer in ("y", "yes", "д", "да")


# ---------- Запуск ----------

def run(items, run_dir, meta, records, client):
    pacer = Pacer(config.MIN_INTERVAL_BETWEEN_REQUESTS_SEC)
    blocked = {}   # model -> причина: дальше эту модель не вызываем
    done = {r["record_id"] for r in records}
    try:
        for i, (task, model_info, repeat) in enumerate(items, 1):
            rid = record_id(task["id"], model_info["id"], repeat)
            if rid in done:
                continue
            if model_info["id"] in blocked:
                records.append(skipped_record(task, model_info, repeat, blocked[model_info["id"]]))
                log.info(f"[{i}/{len(items)}] {rid}: пропуск — {blocked[model_info['id']]}")
                save_all(run_dir, records, meta)
                continue
            log.info(f"[{i}/{len(items)}] {rid}")
            record = solve(client, task, model_info, repeat, pacer)
            records.append(record)
            if record["status"] == "ok":
                cost = "—" if record["cost_usd"] is None else f"${record['cost_usd']:.6f}"
                if record["thoughts_tokens"]:
                    cost += f" (thinking {record['thoughts_tokens']} ток. учтены как выходные)"
                log.info(f"    ответ: {record['answer']!r} → {'ВЕРНО' if record['is_correct'] else 'НЕВЕРНО'} "
                         f"({record['check_reason']}); токены {record['input_tokens']}/"
                         f"{record['output_tokens']}/{record['total_tokens']}; {cost}")
            else:
                code = record["error_code"]
                if code in config.RUN_FATAL_STATUS_CODES:
                    save_all(run_dir, records, meta)
                    raise SystemExit(f"Ошибка {code}: проверьте {API_KEY_ENV}. Эксперимент остановлен.")
                if code in config.MODEL_FATAL_STATUS_CODES:
                    blocked[model_info["id"]] = f"ошибка {code} {record['error_status']} (модель недоступна для ключа)"
                elif record["daily_quota"]:
                    blocked[model_info["id"]] = "исчерпана дневная квота"
            save_all(run_dir, records, meta)
    except KeyboardInterrupt:
        log.warning("Прервано (Ctrl+C). Сохранено всё полученное; продолжить: "
                    f"python main.py --resume {run_dir.name}")
    finally:
        meta["finished_at"] = now_iso()
        write_json(run_dir / META_JSON, meta)
        save_all(run_dir, records, meta)
    return records


def make_meta(run_dir, mode, task_ids, repeats, items):
    import importlib.metadata
    return {
        "run_id": run_dir.name,
        "mode": mode,
        "started_at": now_iso(),
        "finished_at": None,
        "sdk": f"google-genai {importlib.metadata.version('google-genai')}",
        "python": sys.version.split()[0],
        "models": config.MODELS,
        "task_ids": task_ids,
        "repeats": repeats,
        "request_config": describe_config(),
        "prompt_template": config.PROMPT_TEMPLATE,
        "answer_format": config.ANSWER_FORMAT,
        "pricing": {
            "tier": config.PRICING_TIER,
            "checked_on": config.PRICING_CHECKED_ON,
            "source": config.PRICING_SOURCE_URL,
            "source_resolved": config.PRICING_SOURCE_URL_RESOLVED,
            "models_source": config.MODELS_SOURCE_URL,
            "thinking_source": config.THINKING_SOURCE_URL,
            "prices": config.PRICING,
            "thinking_billing": "thoughts_token_count оплачивается по цене выходных токенов",
        },
        "max_output_tokens": config.MAX_OUTPUT_TOKENS,
        "model_output_token_limits": config.OUTPUT_TOKEN_LIMITS,
        "planned_record_ids": [record_id(t["id"], m["id"], r) for t, m, r in items],
    }


def cmd_list_models():
    """Бесплатный запрос метаданных: какие модели видит ключ и поддерживают ли они generateContent."""
    client = make_client()
    available = {}
    for m in client.models.list():
        available[m.name.removeprefix("models/")] = m
    log.info("Модели с generateContent:")
    for name, m in sorted(available.items()):
        if "generateContent" in (m.supported_actions or []):
            log.info(f"  {name:<45} {m.display_name}")
    log.info("\nВыбранные для эксперимента:")
    ok = True
    for mi in config.MODELS:
        m = available.get(mi["id"])
        supported = m is not None and "generateContent" in (m.supported_actions or [])
        ok &= supported
        log.info(f"  {mi['role_name']:<20} {mi['id']:<24} "
                 f"{'доступна, generateContent ✓' if supported else 'НЕ НАЙДЕНА или нет generateContent'}")
    return 0 if ok else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Сравнение трёх моделей Gemini на одинаковых задачах")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list-models", action="store_true", help="список моделей (без генерации)")
    mode.add_argument("--dry-run", action="store_true", help="план и промпты, без API")
    mode.add_argument("--pilot", action="store_true", help=f"1 задача ({config.PILOT_TASK_ID}) × 3 модели")
    mode.add_argument("--full", action="store_true", help="все задачи × 3 модели")
    mode.add_argument("--resume", metavar="RUN", help="повторить неудачные/недостающие вызовы run_N")
    mode.add_argument("--report", metavar="RUN", help="пересобрать отчёт run_N без API")
    parser.add_argument("--repeats", type=int, default=config.REPEATS, help="повторов на пару задача×модель")
    parser.add_argument("--yes", action="store_true", help="не спрашивать подтверждение платных запросов")
    args = parser.parse_args(argv)

    setup_logging()

    if args.list_models:
        return cmd_list_models()

    if args.report:
        run_dir = RESULTS_DIR / args.report
        meta = json.loads((run_dir / META_JSON).read_text(encoding="utf-8"))
        records = json.loads((run_dir / RESULTS_JSON).read_text(encoding="utf-8"))
        save_all(run_dir, records, meta)
        print(report.render_console(report.build_summary(records, meta)))
        return 0

    if args.resume:
        run_dir = RESULTS_DIR / args.resume
        meta = json.loads((run_dir / META_JSON).read_text(encoding="utf-8"))
        old = json.loads((run_dir / RESULTS_JSON).read_text(encoding="utf-8"))
        records = [r for r in old if r["status"] == "ok"]
        failed = [r for r in old if r["status"] != "ok"]
        items = plan(meta["task_ids"], meta["repeats"])
        todo = [it for it in items if record_id(it[0]["id"], it[1]["id"], it[2]) not in
                {r["record_id"] for r in records}]
        setup_logging(run_dir / LOG_NAME)
        log.info(f"Продолжение {run_dir.name}: готово {len(records)}, осталось {len(todo)} вызовов")
        if not todo:
            save_all(run_dir, records, meta)
            print(report.render_console(report.build_summary(records, meta)))
            return 0
        print_plan(todo, meta["repeats"])
        if not confirm(todo, args.yes):
            log.info("Отменено, запросы не отправлялись.")
            return 1
        if failed:
            superseded_path = run_dir / SUPERSEDED_JSON
            prev = json.loads(superseded_path.read_text(encoding="utf-8")) if superseded_path.exists() else []
            write_json(superseded_path, prev + failed)
        meta.setdefault("resumed_at", []).append(now_iso())
        records = run(items, run_dir, meta, records, make_client())
        print(report.render_console(report.build_summary(records, meta)))
        return 0

    task_ids = [config.PILOT_TASK_ID] if args.pilot else [t["id"] for t in TASKS]
    items = plan(task_ids, args.repeats)

    if args.dry_run:
        print_plan(items, args.repeats)
        log.info("\nКонфигурация запроса: " + json.dumps(
            {k: v for k, v in describe_config().items() if k != "response_schema"}, ensure_ascii=False))
        log.info("\nЗадачи и промпты:")
        for t in TASKS:
            log.info(f"--- {t['id']} [{t['category']}] эталон: {t['expected']} (проверка {t['checker']})")
            log.info(build_prompt(t))
        log.info("\nDry-run: запросы к API не отправлялись.")
        return 0

    print_plan(items, args.repeats)
    if not confirm(items, args.yes):
        log.info("Отменено, запросы не отправлялись.")
        return 1
    client = make_client()
    run_dir = new_run_dir()
    setup_logging(run_dir / LOG_NAME)
    meta = make_meta(run_dir, "pilot" if args.pilot else "full", task_ids, args.repeats, items)
    write_json(run_dir / META_JSON, meta)
    log.info(f"Запуск {run_dir.name}: {len(items)} вызовов")
    records = run(items, run_dir, meta, [], client)
    print(report.render_console(report.build_summary(records, meta)))
    log.info(f"\nРезультаты: {run_dir.relative_to(ROOT)}/ ({RESULTS_JSON}, {RESULTS_CSV}, "
             f"{report.SUMMARY_MD}, {report.SUMMARY_CSV}, {report.SUMMARY_JSON}, {LOG_NAME})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
