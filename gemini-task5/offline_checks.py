"""Проверки без сети и без API-ключа: python offline_checks.py

Проверяется: задачи и эталоны, функции проверки, расчёт стоимости по датам тарифов,
повторы при временных ошибках, остановка модели при 404, отсутствие ключа в логах и
результатах, формирование CSV/JSON/MD и итоговых метрик. Все вызовы API поддельные.
"""

import csv
import json
import os
import shutil
import tempfile
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from google.genai import errors

import config
import main
import pricing
import report
from tasks import TASKS, TASKS_BY_ID

FAILED = []


def check(name, condition, details=""):
    status = "OK  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {details}" if details and not condition else ""))
    if not condition:
        FAILED.append(name)


# ---------- Задачи и проверки ----------

def test_tasks():
    check("6 задач с эталонами", len(TASKS) == 6 and all(t["expected"] for t in TASKS))
    expected = {"math_bat_ball": "0.05 доллара", "logic_syllogism": "да", "combinatorics_digits": "24",
                "algorithm_loop": "14", "order_logic": "D, A, B, C", "days_count": "107"}
    check("эталоны совпадают с Днями 3–4", {t["id"]: t["expected"] for t in TASKS} == expected)
    check("пилотная задача существует", config.PILOT_TASK_ID in TASKS_BY_ID)
    check("эталон 107 = datetime", (date(2025, 6, 17) - date(2025, 3, 3)).days + 1 == 107)


def test_checkers():
    cases = {
        "math_bat_ball": [("$0.05", True), ("5 центов", True), ("0,05 доллара", True), ("$0.10", False),
                          ("0.05 или 0.10", False)],
        "logic_syllogism": [("Да", True), ("да, следует", True), ("Нет", False), ("да и нет", False)],
        "combinatorics_digits": [("24", True), ("24 числа", True), ("64", False), ("24 или 64", False)],
        "algorithm_loop": [("14", True), ("total = 14", True), ("6", False)],
        "order_logic": [("D, A, B, C", True), ("D → A → B → C", True), ("DABC", True), ("A, B, C, D", False)],
        "days_count": [("107", True), ("107 дней", True), ("106", False), ("106 или 107", False)],
    }
    for task_id, items in cases.items():
        fn = TASKS_BY_ID[task_id]["check"]
        bad = [(a, exp) for a, exp in items if fn(a)[0] != exp]
        check(f"проверка {task_id}", not bad, f"ошибки на {bad}")


# ---------- Стоимость ----------

def test_pricing():
    p = pricing.price_for("gemini-4-argon", "2026-11-06")
    check("Argon 06.11.2026: $2/$10", (p["input"], p["output"]) == (2.00, 10.00))
    p = pricing.price_for("gemini-4-argon", "2026-12-01")
    check("Argon с 01.12.2026: $4/$20", (p["input"], p["output"]) == (4.00, 20.00))
    p = pricing.price_for("gemini-3.8-flash", "2027-01-01")
    check("3.8 Flash с 01.01.2027: $1.50/$7.50", (p["input"], p["output"]) == (1.50, 7.50))
    c = pricing.compute_cost("gemini-3.5-flash-lite", "2026-11-06", 1_000_000, 500_000, 500_000)
    check("формула: 1M вход + (0.5M+0.5M) выход = 0.30 + 2.50", abs(c["cost_usd"] - 2.80) < 1e-12)
    c = pricing.compute_cost("gemini-4-argon", "2026-11-06", 100, 50, 150)
    check("Argon 100/50+150: 0.0002 + 0.002", abs(c["cost_usd"] - 0.0022) < 1e-12)
    c = pricing.compute_cost("gemini-3.8-flash", "2026-11-06", 1_000_000, 0, 0, cached_tokens=400_000)
    check("кэш по цене cached_input", abs(c["cost_usd"] - (0.6 * 0.75 + 0.4 * 0.075)) < 1e-12)
    c = pricing.compute_cost("gemini-3.8-flash", "2026-11-06", None, None, None)
    check("нет usage → стоимость None с причиной", c["cost_usd"] is None and c["cost_note"])
    c = pricing.compute_cost("unknown-model", "2026-11-06", 1, 1, 1)
    check("нет тарифа → стоимость None с причиной", c["cost_usd"] is None and "нет тарифа" in c["cost_note"])
    check("у всех моделей есть тариф", all(pricing.price_for(m["id"], date.today()) for m in config.MODELS))
    check("модели: 3.5-flash-lite / 3.8-flash / 4-argon",
          config.MODEL_IDS == ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-4-argon"])
    check("thinking оплачивается как выход: 0 ответа + 1M thinking = цена выхода",
          abs(pricing.compute_cost("gemini-3.8-flash", "2026-11-06", 0, 0, 1_000_000)["cost_usd"] - 3.75) < 1e-12)

    # Максимальная стоимость: лимит выхода (вкл. thinking) × цена выхода + вход по оценке сверху
    check("лимит выходных токенов задан", config.MAX_OUTPUT_TOKENS is not None)
    cap, inp = config.MAX_OUTPUT_TOKENS, config.MAX_INPUT_TOKENS_ESTIMATE
    expected = 6 * sum((inp * pi + cap * po) / 1e6 for pi, po in ((0.30, 2.50), (0.75, 3.75), (2.00, 10.00)))
    _, total = pricing.max_run_cost(config.MODEL_IDS, 6, 1, "2026-11-06")
    check(f"макс. стоимость 18 вызовов = {expected:.4f}", abs(total - expected) < 1e-12)
    _, total3 = pricing.max_run_cost(config.MODEL_IDS, 6, 1, "2026-11-06", attempts_per_call=3)
    check("3 оплаченные попытки = ×3", abs(total3 - 3 * expected) < 1e-12)
    old = config.MAX_OUTPUT_TOKENS
    try:
        config.MAX_OUTPUT_TOKENS = None
        check("без лимита берётся output_token_limit модели",
              pricing.max_output_tokens_for("gemini-4-argon") == config.OUTPUT_TOKEN_LIMITS["gemini-4-argon"])
    finally:
        config.MAX_OUTPUT_TOKENS = old


# ---------- Поддельный клиент ----------

CORRECT = {"math_bat_ball": "0.05 доллара", "logic_syllogism": "да", "combinatorics_digits": "24",
           "algorithm_loop": "14", "order_logic": "D, A, B, C", "days_count": "107"}


def fake_response(answer, model, thoughts=30, text=None, finish="STOP"):
    usage = SimpleNamespace(prompt_token_count=100, candidates_token_count=20, thoughts_token_count=thoughts,
                            cached_content_token_count=None, total_token_count=120 + (thoughts or 0))
    candidate = SimpleNamespace(finish_reason=SimpleNamespace(name=finish))
    if text is None:
        text = json.dumps({"answer": answer, "explanation": "..."}, ensure_ascii=False)
    return SimpleNamespace(text=text, usage_metadata=usage, candidates=[candidate], model_version=model)


class FakeModels:
    """small — всегда верно, но первый вызов 503; medium — ошибается в days_count и обрезан
    по лимиту в algorithm_loop; flagship — 404 (модель недоступна)."""

    def __init__(self, secret):
        self.calls = []
        self.configs = []
        self.secret = secret

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        self.configs.append(config)
        task_id = next(t["id"] for t in TASKS if t["question"] in contents)
        if model == "gemini-4-argon":
            raise errors.ClientError(404, {"error": {"code": 404, "status": "NOT_FOUND",
                                                     "message": f"model not found key={self.secret}"}})
        if model == "gemini-3.5-flash-lite" and self.calls.count(model) == 1:
            raise errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE",
                                                     "message": "The model is overloaded"}})
        if model == "gemini-3.8-flash" and task_id == "algorithm_loop":
            # упёрлись в лимит: JSON оборван, thinking израсходовал почти весь лимит
            return fake_response(None, model, thoughts=8000, text='{"answer": "1', finish="MAX_TOKENS")
        answer = "106" if (model == "gemini-3.8-flash" and task_id == "days_count") else CORRECT[task_id]
        return fake_response(answer, model)


def test_full_fake_run():
    secret = "AIzaFAKE-secret-key-123"
    old_key = os.environ.get(main.API_KEY_ENV)
    os.environ[main.API_KEY_ENV] = secret
    tmp = Path(tempfile.mkdtemp(prefix="task5_offline_"))
    old_results, old_sleep = main.RESULTS_DIR, main.sleep
    main.RESULTS_DIR = tmp
    main.sleep = lambda s: None
    try:
        task_ids = [t["id"] for t in TASKS]
        items = main.plan(task_ids, 1)
        run_dir = main.new_run_dir()
        main.setup_logging(run_dir / main.LOG_NAME)
        meta = main.make_meta(run_dir, "full", task_ids, 1, items)
        fake = FakeModels(secret)
        client = SimpleNamespace(models=fake)
        records = main.run(items, run_dir, meta, [], client)

        check("18 записей", len(records) == 18)
        check("argon вызван 1 раз, остальные пропущены", fake.calls.count("gemini-4-argon") == 1)
        argon = [r for r in records if r["model"] == "gemini-4-argon"]
        check("argon: 1 ошибка API + 5 пропусков",
              sum(r["status"] == "api_error" for r in argon) == 1 and sum(r["status"] == "skipped" for r in argon) == 5)
        first_small = next(r for r in records if r["model"] == "gemini-3.5-flash-lite")
        check("503 → повтор и успех", first_small["status"] == "ok" and first_small["retries"] == 1)
        check("выходные = candidates + thoughts", first_small["output_tokens"] == 50)
        check("стоимость small: 100×0.30 + 50×2.50 за 1M",
              abs(first_small["cost_usd"] - (100 * 0.30 + 50 * 2.50) / 1e6) < 1e-15)

        summary = json.loads((run_dir / report.SUMMARY_JSON).read_text(encoding="utf-8"))
        rows = {r["role"]: r for r in summary["rows"]}
        check("small 6/6 = 100%", (rows["small"]["correct"], rows["small"]["accuracy_pct"]) == (6, 100.0))
        check("medium 4/6 = 66.7%", (rows["medium"]["correct"], rows["medium"]["accuracy_pct"]) == (4, 66.7))
        check("flagship 0/6, стоимость неизвестна", rows["flagship"]["correct"] == 0
              and rows["flagship"]["cost_usd"] is None)

        # Лимит и обрезка
        sent = fake.configs[0]
        check(f"в запрос передан max_output_tokens={config.MAX_OUTPUT_TOKENS} и thinking LOW",
              sent.max_output_tokens == config.MAX_OUTPUT_TOKENS
              and sent.thinking_config.thinking_level.name == config.THINKING_LEVEL)
        check("одинаковая конфигурация во всех запросах",
              all(c.model_dump() == sent.model_dump() for c in fake.configs))
        cut = next(r for r in records if r["record_id"] == "algorithm_loop|gemini-3.8-flash|1")
        check("обрезанный ответ: truncated, неверно, причина указана",
              cut["truncated"] is True and cut["is_correct"] is False and "обрезан" in cut["check_reason"])
        check("обрезанный ответ: thinking оплачен как выход (20 + 8000)",
              cut["output_tokens"] == 8020
              and abs(cut["cost_usd"] - (100 * 0.75 + 8020 * 3.75) / 1e6) < 1e-15)
        check("в сводке 1 обрезанный ответ у medium", rows["medium"]["truncated"] == 1)
        check("матрица: ✗ (обрезан)", summary["per_task"]["algorithm_loop"]["gemini-3.8-flash"] == "✗ (обрезан)")

        comp = summary["comparison"]
        check("ratio флагман/малая = None с причиной", comp["flagship_vs_small_cost_ratio"] is None
              and comp["flagship_vs_small_cost_note"])
        medium_cost = (5 * (100 * 0.75 + 50 * 3.75) + (100 * 0.75 + 8020 * 3.75)) / 1e6
        small_cost = 6 * (100 * 0.30 + 50 * 2.50) / 1e6
        check("medium/small cost ratio посчитан",
              abs(comp["medium_vs_small_cost_ratio"] - medium_cost / small_cost) < 1e-9)
        check("разница точности medium − small = −33.3 п.п.", comp["medium_minus_small_accuracy_pp"] == -33.3)

        for name in (main.RESULTS_JSON, main.RESULTS_CSV, main.META_JSON, main.LOG_NAME,
                     report.SUMMARY_MD, report.SUMMARY_CSV, report.SUMMARY_JSON):
            check(f"файл {name} создан", (run_dir / name).exists())
        with open(run_dir / main.RESULTS_CSV, encoding="utf-8") as f:
            check("CSV читается, 18 строк", len(list(csv.DictReader(f))) == 18)
        leaked = [p.name for p in run_dir.iterdir() if secret in p.read_text(encoding="utf-8")]
        check("ключ не попал ни в один файл", not leaked, f"найден в {leaked}")
        md = (run_dir / report.SUMMARY_MD).read_text(encoding="utf-8")
        check("в summary.md есть источник тарифов", config.PRICING_SOURCE_URL in md)
        print("\nПример итоговой таблицы (поддельные данные):")
        print(report.render_console(report.build_summary(records, meta)))
        print()
    finally:
        main.RESULTS_DIR, main.sleep = old_results, old_sleep
        main.setup_logging()
        shutil.rmtree(tmp, ignore_errors=True)
        if old_key is None:
            os.environ.pop(main.API_KEY_ENV, None)
        else:
            os.environ[main.API_KEY_ENV] = old_key


def test_misc():
    check("одинаковый промпт-шаблон для всех моделей", "{question}" in config.PROMPT_TEMPLATE)
    check("3 модели с ролями small/medium/flagship",
          [m["role"] for m in config.MODELS] == ["small", "medium", "flagship"])
    err = {"code": 429, "message": "Quota exceeded ... requests per day"}
    check("дневная квота распознаётся", main.is_daily_quota_error(err))
    check("подсказка retry in Ns", main.retry_hint_sec("Please retry in 45.03s.") == 45.03)
    check("порядок: модели чередуются", [m["id"] for _, m, _ in main.plan(["math_bat_ball"], 1)] == config.MODEL_IDS)


if __name__ == "__main__":
    main.setup_logging()
    test_tasks()
    test_checkers()
    test_pricing()
    test_misc()
    test_full_fake_run()
    print(f"Итого: {'все проверки пройдены' if not FAILED else f'провалено {len(FAILED)}: {FAILED}'}")
    raise SystemExit(1 if FAILED else 0)
