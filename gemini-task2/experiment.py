"""Эксперимент: сравнение трёх способов отправки одного запроса в Gemini API.

Запуск:
    ../venv/bin/python experiment.py            # модель из config.MODEL, результаты в results/
    ../venv/bin/python experiment.py --dry-run  # только показывает, что будет отправлено
    ../venv/bin/python experiment.py --model gemini-3.6-flash   # другая модель,
                                                # результаты в results/model_gemini-3.6-flash/
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from config import (
    MODEL,
    PAUSE_BETWEEN_REQUESTS,
    REQUEST_TIMEOUT_SEC,
    RETRY_DELAYS_SEC,
    RETRY_STATUS_CODES,
    RUNS_PER_VARIANT,
    STOP_STATUS_CODES,
    VARIANTS,
    StrictBookRecommendations,
)

RESULTS_DIR = Path(__file__).parent / "results"
RAW_RESULTS_NAME = "raw_results.json"
SUMMARY_NAME = "summary.md"

# Допустимое имя модели: буквы, цифры, точка, дефис, подчёркивание (имя идёт в путь к папке)
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def results_dir_for(model):
    """Папка результатов: модель по умолчанию — results/, любая другая — results/model_<имя>/."""
    if model == MODEL:
        return RESULTS_DIR
    return RESULTS_DIR / f"model_{model}"


# ---------- Проверки ответа (одинаковые для всех вариантов) ----------

def check_response_text(text):
    """Проверяет текст ответа в два этапа: JSON-парсинг, затем валидация Pydantic.

    Парсинг строгий: json.loads по тексту без обрамляющих пробелов.
    Markdown-обёртку ```json ... ``` не снимаем — она считается ошибкой парсинга,
    но отдельно отмечается флагом has_code_fence.
    """
    result = {
        "has_code_fence": bool(text) and "```" in text,
        "parse_status": None,       # ok | empty | json_error
        "parse_error": None,
        "validation_status": None,  # ok | validation_error | skipped
        "validation_errors": None,
        "books_count": None,
        "titles": None,
    }

    # Этап 1: JSON-парсинг
    if not text or not text.strip():
        result["parse_status"] = "empty"
        result["parse_error"] = "Пустой ответ"
        result["validation_status"] = "skipped"
        return result

    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError as e:
        result["parse_status"] = "json_error"
        result["parse_error"] = f"{e.msg} (строка {e.lineno}, позиция {e.colno})"
        result["validation_status"] = "skipped"
        return result

    result["parse_status"] = "ok"
    if isinstance(data, dict) and isinstance(data.get("books"), list):
        result["books_count"] = len(data["books"])

    # Этап 2: валидация структуры через строгую Pydantic-модель (одна для всех вариантов)
    try:
        recs = StrictBookRecommendations.model_validate(data)
    except ValidationError as e:
        result["validation_status"] = "validation_error"
        result["validation_errors"] = [
            {"loc": ".".join(str(p) for p in err["loc"]), "type": err["type"], "msg": err["msg"]}
            for err in e.errors()
        ]
        return result

    result["validation_status"] = "ok"
    result["titles"] = [book.title for book in recs.books]
    return result


# ---------- Отправка запроса ----------

def build_config(variant):
    """Собирает GenerateContentConfig для варианта (или None для варианта без ограничений)."""
    if variant["config"] is None:
        return None
    return types.GenerateContentConfig(**variant["config"])


def extract_finish_reason(response):
    """Возвращает finish_reason первого кандидата строкой."""
    if not response.candidates:
        return None
    reason = response.candidates[0].finish_reason
    if reason is None:
        return None
    return getattr(reason, "name", str(reason))


def send_request(client, variant, model):
    """Одна попытка запроса. Возвращает (response, error, latency_sec)."""
    start = time.perf_counter()
    try:
        response = client.models.generate_content(
            model=model,
            contents=variant["prompt"],
            config=build_config(variant),
        )
        return response, None, round(time.perf_counter() - start, 3)
    except errors.APIError as e:
        error = {"type": "APIError", "code": e.code, "message": e.message}
    except Exception as e:
        error = {"type": type(e).__name__, "code": None, "message": str(e)}
    return None, error, round(time.perf_counter() - start, 3)


def run_once(client, variant, run_number, model=MODEL):
    """Выполняет один запуск варианта (с повторами при 503/504) и возвращает запись с метриками.

    Повтор делается только при ошибке API с кодом из RETRY_STATUS_CODES.
    Каждая попытка (включая неудачные) записывается в record["attempts"].
    latency_sec — время последней попытки; total_elapsed_sec — всё время запуска с паузами.
    """
    record = {
        "variant_id": variant["id"],
        "variant_name": variant["name"],
        "run": run_number,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "prompt": variant["prompt"],
        "request_config": describe_config(variant),
        "attempts": [],
        "attempts_count": 0,
        "retries_count": 0,
        "api_error": None,
        "latency_sec": None,
        "total_elapsed_sec": None,
        "finish_reason": None,
        "usage_metadata": None,
        "raw_text": None,
        "text_length_chars": None,
        "checks": None,
    }

    run_start = time.perf_counter()
    max_attempts = 1 + len(RETRY_DELAYS_SEC)
    response = None
    for attempt in range(1, max_attempts + 1):
        started_at = datetime.now().isoformat(timespec="seconds")
        response, error, latency = send_request(client, variant, model)
        will_retry = (
            error is not None
            and error["code"] in RETRY_STATUS_CODES
            and attempt < max_attempts
        )
        pause = RETRY_DELAYS_SEC[attempt - 1] if will_retry else None
        record["attempts"].append({
            "attempt": attempt,
            "started_at": started_at,
            "latency_sec": latency,
            "status": "ok" if error is None else "api_error",
            "error": error,
            "retry_after_sec": pause,
        })
        record["latency_sec"] = latency
        record["api_error"] = error
        if not will_retry:
            break
        print(f"\n    попытка {attempt}: ошибка {error['code']}, повтор через {pause} с...", end=" ", flush=True)
        time.sleep(pause)

    record["attempts_count"] = len(record["attempts"])
    record["retries_count"] = record["attempts_count"] - 1
    record["total_elapsed_sec"] = round(time.perf_counter() - run_start, 3)
    if response is None:
        return record

    # response.text может выбросить исключение или вернуть None, если кандидатов нет
    try:
        text = response.text
    except Exception:
        text = None

    record["finish_reason"] = extract_finish_reason(response)
    if response.usage_metadata is not None:
        record["usage_metadata"] = response.usage_metadata.model_dump(mode="json", exclude_none=True)
    record["raw_text"] = text
    record["text_length_chars"] = len(text) if text else 0
    record["checks"] = check_response_text(text)
    return record


def describe_config(variant):
    """Описание конфигурации в виде, пригодном для JSON (схема — как JSON Schema)."""
    if variant["config"] is None:
        return None
    described = {}
    for key, value in variant["config"].items():
        if key == "response_schema":
            described[key] = value.model_json_schema()
        else:
            described[key] = value
    return described


# ---------- Сводка ----------

def usage_value(record, key):
    usage = record.get("usage_metadata") or {}
    return usage.get(key)


def fmt(value):
    return "—" if value is None else str(value)


def one_line(text):
    """Сообщение об ошибке в одну строку (многострочные ломают Markdown-таблицы и списки)."""
    return " / ".join(part.strip() for part in str(text).splitlines() if part.strip())


def planned_runs():
    """Порядок запусков с чередованием вариантов: v1, v2, v3, v1, v2, v3, ..."""
    return [(variant, run_number)
            for run_number in range(1, RUNS_PER_VARIANT + 1)
            for variant in VARIANTS]


def build_summary(records, stop_info=None, model=MODEL):
    """Формирует Markdown-сводку по всем запускам.

    stop_info — None, если эксперимент не останавливался досрочно, иначе
    {"reason": ..., "not_run": [(variant_id, run), ...]}.
    """
    total_planned = len(planned_runs())
    if stop_info:
        status = f"**остановлен досрочно** — {stop_info['reason']}"
    elif len(records) < total_planned:
        status = "в процессе"
    else:
        status = "завершён"
    lines = [
        "# Сводка эксперимента",
        "",
        f"Модель: `{model}`" + ("  " if model == MODEL else f" (запасная; основная модель проекта — `{MODEL}`)  "),
        f"Дата: {datetime.now().isoformat(timespec='seconds')}  ",
        f"Порядок: варианты чередуются (v1, v2, v3, v1, …)  ",
        f"Выполнено запусков: {len(records)}/{total_planned}; статус: {status}",
        "",
        "## Все запуски",
        "",
        "Время — последней попытки; «Всего, с» — весь запуск, включая повторы и паузы.",
        "",
        "| Вариант | Запуск | Попыток | finish_reason | Время, с | Всего, с | Символов | Вход. токены "
        "| Выход. токены | Thinking-токены | JSON | Pydantic | Книг | Итоговая ошибка API |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        checks = r["checks"] or {}
        api_err = (f"{r['api_error']['code']}: {one_line(r['api_error']['message'])[:60]}"
                   if r["api_error"] else "—")
        lines.append(
            f"| {r['variant_id']} | {r['run']} | {r['attempts_count']} | {fmt(r['finish_reason'])} "
            f"| {fmt(r['latency_sec'])} | {fmt(r['total_elapsed_sec'])} "
            f"| {fmt(r['text_length_chars'])} | {fmt(usage_value(r, 'prompt_token_count'))} "
            f"| {fmt(usage_value(r, 'candidates_token_count'))} "
            f"| {fmt(usage_value(r, 'thoughts_token_count'))} "
            f"| {fmt(checks.get('parse_status'))} | {fmt(checks.get('validation_status'))} "
            f"| {fmt(checks.get('books_count'))} | {api_err} |"
        )

    lines += [
        "",
        "## Итоги по вариантам",
        "",
        "Успех/JSON/Pydantic считаются по запускам (итог после всех попыток). "
        "Средние — только по успешным запускам, время — успешной попытки.",
        "",
        "| Вариант | Выполнено запусков | Успешных ответов API | JSON распарсен | Прошёл Pydantic | Попыток всего "
        "| Запусков с повтором | Ср. время, с | Ср. выход. токены | Ср. символов | Уникальных наборов книг |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for variant in VARIANTS:
        rs = [r for r in records if r["variant_id"] == variant["id"]]
        ok = [r for r in rs if not r["api_error"]]
        parsed = [r for r in ok if r["checks"]["parse_status"] == "ok"]
        valid = [r for r in ok if r["checks"]["validation_status"] == "ok"]
        latencies = [r["latency_sec"] for r in ok]
        out_tokens = [usage_value(r, "candidates_token_count") for r in ok]
        out_tokens = [t for t in out_tokens if t is not None]
        lengths = [r["text_length_chars"] for r in ok]
        unique_sets = len({tuple(sorted(r["checks"]["titles"])) for r in valid}) if valid else None
        lines.append(
            f"| {variant['id']} | {len(rs)}/{RUNS_PER_VARIANT} "
            f"| {len(ok)}/{len(rs)} | {len(parsed)}/{len(rs)} | {len(valid)}/{len(rs)} "
            f"| {sum(r['attempts_count'] for r in rs)} | {sum(1 for r in rs if r['retries_count'] > 0)}/{len(rs)} "
            f"| {fmt(round(statistics.mean(latencies), 2) if latencies else None)} "
            f"| {fmt(round(statistics.mean(out_tokens)) if out_tokens else None)} "
            f"| {fmt(round(statistics.mean(lengths)) if lengths else None)} "
            f"| {fmt(unique_sets)} |"
        )

    # Подробности ошибок — отдельно парсинг и отдельно валидация
    lines += ["", "## Ошибки парсинга JSON", ""]
    parse_errs = [r for r in records if r["checks"] and r["checks"]["parse_status"] != "ok"]
    if parse_errs:
        for r in parse_errs:
            c = r["checks"]
            fence = " (есть Markdown-обёртка ```)" if c["has_code_fence"] else ""
            lines.append(f"- **{r['variant_id']} / запуск {r['run']}** — {c['parse_status']}: {c['parse_error']}{fence}")
    else:
        lines.append("Нет.")

    lines += ["", "## Ошибки валидации Pydantic", ""]
    val_errs = [r for r in records if r["checks"] and r["checks"]["validation_status"] == "validation_error"]
    if val_errs:
        for r in val_errs:
            lines.append(f"- **{r['variant_id']} / запуск {r['run']}**:")
            for err in r["checks"]["validation_errors"]:
                lines.append(f"  - `{err['loc']}` — {err['type']}: {err['msg']}")
    else:
        lines.append("Нет.")

    if stop_info:
        lines += ["", "## Досрочная остановка", "", f"Причина: {stop_info['reason']}", "",
                  "Не выполнены запуски:"]
        lines += [f"- {vid} / запуск {run}" for vid, run in stop_info["not_run"]]

    lines += ["", "## Повторные попытки (503/504)", ""]
    retried = [r for r in records if r["retries_count"] > 0]
    if retried:
        for r in retried:
            outcome = "успех" if not r["api_error"] else f"неудача ({r['api_error']['code']})"
            lines.append(f"- **{r['variant_id']} / запуск {r['run']}** — {r['attempts_count']} попыток, итог: {outcome}")
            for a in r["attempts"]:
                if a["error"]:
                    desc = f"{fmt(a['error']['code'])}: {one_line(a['error']['message'])[:80]}"
                else:
                    desc = "ok"
                pause = f", пауза {a['retry_after_sec']} с" if a["retry_after_sec"] else ""
                lines.append(f"  - попытка {a['attempt']}: {desc} ({a['latency_sec']} с{pause})")
    else:
        lines.append("Не было.")

    lines += ["", "## Итоговые ошибки API (после всех попыток)", ""]
    api_errs = [r for r in records if r["api_error"]]
    if api_errs:
        for r in api_errs:
            e = r["api_error"]
            lines.append(f"- **{r['variant_id']} / запуск {r['run']}** — {e['type']} {fmt(e['code'])} "
                         f"(попыток: {r['attempts_count']}): {one_line(e['message'])}")
    else:
        lines.append("Нет.")

    return "\n".join(lines) + "\n"


# ---------- Точка входа ----------

def dry_run(model=MODEL):
    """Печатает, какие запросы будут отправлены, без обращения к API."""
    print(f"Модель: {model}" + ("" if model == MODEL else f" (не по умолчанию; в config.MODEL — {MODEL})"))
    print(f"Папка результатов: {results_dir_for(model)}")
    print(f"Запусков на вариант: {RUNS_PER_VARIANT}, всего запусков: {RUNS_PER_VARIANT * len(VARIANTS)}")
    print(f"Таймаут запроса: {REQUEST_TIMEOUT_SEC} с; повтор при кодах {RETRY_STATUS_CODES}, "
          f"паузы {RETRY_DELAYS_SEC} с (до {1 + len(RETRY_DELAYS_SEC)} попыток на запуск); "
          f"остановка эксперимента при кодах {STOP_STATUS_CODES}")
    print("Порядок запусков: " + ", ".join(f"{v['id'][:2]}#{n}" for v, n in planned_runs()))
    print(f"Максимум запросов к API: {len(planned_runs()) * (1 + len(RETRY_DELAYS_SEC))}\n")
    for variant in VARIANTS:
        build_config(variant)  # проверяем, что конфиг собирается без ошибок
        print(f"=== {variant['id']} — {variant['name']}")
        print("Промпт:")
        print(variant["prompt"])
        print("Конфиг:")
        print(json.dumps(describe_config(variant), ensure_ascii=False, indent=2))
        print()


def main():
    parser = argparse.ArgumentParser(description="Сравнение трёх способов запроса к Gemini API")
    parser.add_argument("--dry-run", action="store_true", help="не отправлять запросы, только показать их")
    parser.add_argument("--model", default=MODEL,
                        help=f"модель Gemini (по умолчанию {MODEL}); для другой модели результаты "
                             f"пишутся в results/model_<имя>/")
    args = parser.parse_args()

    model = args.model.strip()
    if not MODEL_NAME_RE.match(model):
        print(f"Ошибка: недопустимое имя модели: {args.model!r}")
        sys.exit(2)

    results_dir = results_dir_for(model)
    raw_results_file = results_dir / RAW_RESULTS_NAME
    summary_file = results_dir / SUMMARY_NAME

    if args.dry_run:
        dry_run(model)
        return

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Ошибка: не задан API-ключ. Установите переменную окружения GEMINI_API_KEY.")
        sys.exit(1)

    # Защита от перезаписи результатов предыдущего прогона
    existing = [p for p in (raw_results_file, summary_file) if p.exists()]
    if existing:
        print(f"Ошибка: в {results_dir} уже есть результаты прогона: " + ", ".join(p.name for p in existing))
        print("Перенесите их в отдельную папку (например, run_N/) и запустите снова.")
        sys.exit(1)

    # Таймаут в HttpOptions задаётся в миллисекундах
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_SEC * 1000),
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Модель: {model}; результаты: {results_dir}")

    records = []
    stop_info = None
    plan = planned_runs()
    for index, (variant, run_number) in enumerate(plan):
        print(f"[{index + 1}/{len(plan)}] {variant['id']}, запуск {run_number}...", end=" ", flush=True)
        record = run_once(client, variant, run_number, model)
        records.append(record)

        if record["api_error"]:
            print(f"ошибка API {record['api_error']['code']}: {one_line(record['api_error']['message'])[:120]}")
        else:
            c = record["checks"]
            print(f"{record['latency_sec']} с, попыток={record['attempts_count']}, "
                  f"finish_reason={record['finish_reason']}, "
                  f"JSON={c['parse_status']}, Pydantic={c['validation_status']}")

        error_code = record["api_error"]["code"] if record["api_error"] else None
        if error_code in STOP_STATUS_CODES:
            stop_info = {
                "reason": f"ошибка API {error_code} в {record['variant_id']} / запуск {record['run']}",
                "not_run": [(v["id"], n) for v, n in plan[index + 1:]],
            }

        # Сохраняем сырые результаты и сводку после каждого запуска (включая неудачные попытки)
        raw_results_file.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_file.write_text(build_summary(records, stop_info, model), encoding="utf-8")

        if stop_info:
            print(f"\nЭксперимент остановлен: {stop_info['reason']}. "
                  f"Не выполнено запусков: {len(stop_info['not_run'])}.")
            break

        if index + 1 < len(plan):
            time.sleep(PAUSE_BETWEEN_REQUESTS)

    print(f"\nСырые результаты: {raw_results_file}")
    print(f"Сводка: {summary_file}")


if __name__ == "__main__":
    main()
