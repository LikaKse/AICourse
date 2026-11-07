"""Офлайн-проверки эксперимента — без единого запроса к API.

Запуск:
    ../venv/bin/python offline_checks.py

Что гарантирует отсутствие сети:
  * genai.Client подменяется заглушкой (ответы и ошибки задаются сценарием);
  * socket.socket.connect и socket.create_connection подменяются функцией,
    которая падает — любая попытка выйти в сеть провалит проверку;
  * результаты пишутся во временную папку, настоящая results/ не затрагивается.
"""

import sys

sys.dont_write_bytecode = True  # не создавать __pycache__ в проекте

import io
import json
import os
import socket
import tempfile
import types as pytypes
from contextlib import redirect_stdout
from pathlib import Path

import config
import experiment as ex
from google import genai
from google.genai import _transformers as transformers
from google.genai import errors


# ---------- Блокировка сети ----------

def _no_network(*args, **kwargs):
    raise RuntimeError("Сетевое соединение запрещено в офлайн-проверках")


socket.socket.connect = _no_network
socket.create_connection = _no_network
ex.time.sleep = lambda seconds: None  # паузы не нужны

# ---------- Мини-фреймворк проверок ----------

PASSED, FAILED = [], []


def check(name, condition, details=""):
    (PASSED if condition else FAILED).append(name)
    mark = "✅" if condition else "❌"
    print(f"  {mark} {name}" + (f" — {details}" if details and not condition else ""))


GOOD = {"books": [{"title": f"T{i}", "author": "A", "year": 2000 + i, "reason": "r"} for i in range(3)]}
GOOD_TEXT = json.dumps(GOOD)


def api_error(code):
    return errors.APIError(code, {"error": {"code": code, "message": f"fake {code}", "status": "FAKE"}})


class FakeResponse:
    text = GOOD_TEXT
    usage_metadata = None
    candidates = [pytypes.SimpleNamespace(finish_reason=pytypes.SimpleNamespace(name="STOP"))]


def variant_of(config_obj):
    """Определяет вариант по конфигу запроса (промпты v1 и v3 совпадают, поэтому по конфигу)."""
    if config_obj is None:
        return "v1"
    if config_obj.response_schema is not None:
        return "v3"
    return "v2"


def run_main(argv, script, results_root):
    """Запускает experiment.main() с заглушкой клиента. Возвращает (exit_code, calls, stdout)."""
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = self

        def generate_content(self, model, contents, config):
            raw_files = list(results_root.rglob(ex.RAW_RESULTS_NAME))
            saved = {str(p.relative_to(results_root)): len(json.loads(p.read_text())) for p in raw_files}
            calls.append({"model": model, "variant": variant_of(config), "saved": saved})
            item = script.pop(0) if script else FakeResponse()
            if isinstance(item, Exception):
                raise item
            return item

    ex.genai.Client = FakeClient
    ex.RESULTS_DIR = results_root
    sys.argv = ["experiment.py", *argv]
    os.environ["GEMINI_API_KEY"] = "offline-fake-key"
    out = io.StringIO()
    code = 0
    with redirect_stdout(out):
        try:
            ex.main()
        except SystemExit as e:
            code = e.code
    return code, calls, out.getvalue()


def load_records(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


# ---------- Проверки ----------

def checks_config():
    print("\n1. Конфигурация")
    check("MODEL в config.py не изменён (gemini-3.8-flash)", config.MODEL == "gemini-3.8-flash")
    check("3 варианта, 3 запуска на вариант", len(config.VARIANTS) == 3 and config.RUNS_PER_VARIANT == 3)
    check("v2: max_output_tokens=400, stop_sequences=['<<END>>']",
          config.VARIANTS[1]["config"] == {"max_output_tokens": 400, "stop_sequences": ["<<END>>"]})
    check("v3: response_schema = ApiBookRecommendations",
          config.VARIANTS[2]["config"]["response_schema"] is config.ApiBookRecommendations)
    check("максимум 2 попытки на запуск", 1 + len(config.RETRY_DELAYS_SEC) == 2)
    check("повтор при 503/504, остановка при 429",
          config.RETRY_STATUS_CODES == (503, 504) and config.STOP_STATUS_CODES == (429,))
    check("паузы между запусками и перед повтором >= 13 с (не больше 5 запросов в минуту)",
          config.PAUSE_BETWEEN_REQUESTS >= 13 and all(d >= 13 for d in config.RETRY_DELAYS_SEC))


def checks_schema_and_validation():
    print("\n2. API-схема и строгая валидация")
    schema = transformers.t_schema(genai.Client(api_key="offline"), config.ApiBookRecommendations)
    dumped = schema.model_dump(exclude_none=True, mode="json")
    books = dumped["properties"]["books"]
    check("в API-схеме нет additional_properties", "additional_properties" not in json.dumps(dumped))
    check("books: ARRAY, min_items=3, max_items=3",
          books["type"] == "ARRAY" and books["min_items"] == 3 and books["max_items"] == 3)

    first = GOOD["books"][0]
    cases = [
        ("правильный ответ", GOOD_TEXT, "ok", None),
        ("2 книги", json.dumps({"books": GOOD["books"][:2]}), "validation_error", "too_short"),
        ("4 книги", json.dumps({"books": GOOD["books"] + GOOD["books"][:1]}), "validation_error", "too_long"),
        ("лишнее поле в корне", json.dumps({**GOOD, "note": "x"}), "validation_error", "extra_forbidden"),
        ("лишнее поле в книге", json.dumps({"books": [{**first, "isbn": "1"}] + GOOD["books"][1:]}),
         "validation_error", "extra_forbidden"),
        ("пустой title", json.dumps({"books": [{**first, "title": ""}] + GOOD["books"][1:]}),
         "validation_error", "string_too_short"),
        ("year строкой", json.dumps({"books": [{**first, "year": "около 2000"}] + GOOD["books"][1:]}),
         "validation_error", "int_parsing"),
    ]
    for name, text, expected_status, expected_type in cases:
        result = ex.check_response_text(text)
        types_found = [e["type"] for e in (result["validation_errors"] or [])]
        ok = result["validation_status"] == expected_status and (expected_type is None or expected_type in types_found)
        check(f"валидация: {name} → {expected_status}" + (f" ({expected_type})" if expected_type else ""),
              ok, f"получено {result['validation_status']} {types_found}")

    for name, text, expected in [("пустой ответ", "", "empty"), ("Markdown", "**1. Книга**", "json_error"),
                                 ("JSON в ```", "```json\n" + GOOD_TEXT + "\n```", "json_error")]:
        result = ex.check_response_text(text)
        check(f"парсинг: {name} → {expected}", result["parse_status"] == expected, result["parse_status"])


def checks_model_argument():
    print("\n3. Аргумент --model и папки результатов")
    root = Path("/results-root")
    original = ex.RESULTS_DIR
    ex.RESULTS_DIR = root
    check("модель по умолчанию → results/", ex.results_dir_for(config.MODEL) == root)
    check("gemini-3.6-flash → results/model_gemini-3.6-flash/",
          ex.results_dir_for("gemini-3.6-flash") == root / "model_gemini-3.6-flash")
    check("gemini-3.5-flash-lite → results/model_gemini-3.5-flash-lite/",
          ex.results_dir_for("gemini-3.5-flash-lite") == root / "model_gemini-3.5-flash-lite")
    ex.RESULTS_DIR = original

    with tempfile.TemporaryDirectory() as tmp:
        for bad in ["../evil", "a/b", " ", "-x"]:
            code, calls, _ = run_main(["--model", bad], [], Path(tmp))
            check(f"недопустимое имя модели {bad!r} отклонено (exit 2, без вызовов API)",
                  code == 2 and not calls and not any(Path(tmp).iterdir()))

    with tempfile.TemporaryDirectory() as tmp:
        code, calls, out = run_main(["--model", "gemini-3.6-flash", "--dry-run"], [], Path(tmp))
        check("--dry-run с --model: без вызовов API и без файлов",
              code == 0 and not calls and not any(Path(tmp).iterdir()))
        check("--dry-run показывает модель и папку",
              "Модель: gemini-3.6-flash" in out and "model_gemini-3.6-flash" in out)


def checks_full_run_new_model():
    print("\n4. Полный прогон на заглушке: --model gemini-3.6-flash")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # В корне уже лежат результаты основной модели — они не должны мешать и не должны меняться
        (root / "raw_results.json").write_text("[]", encoding="utf-8")
        (root / "summary.md").write_text("# старый прогон\n", encoding="utf-8")
        before = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}

        code, calls, _ = run_main(["--model", "gemini-3.6-flash"], [], root)
        target = root / "model_gemini-3.6-flash"
        records = load_records(target / ex.RAW_RESULTS_NAME)
        check("завершён без ошибок, 9 вызовов, 9 записей", code == 0 and len(calls) == 9 and len(records) == 9)
        check("все вызовы API к gemini-3.6-flash", {c["model"] for c in calls} == {"gemini-3.6-flash"})
        check("поле model в записях = gemini-3.6-flash", {r["model"] for r in records} == {"gemini-3.6-flash"})
        check("порядок чередуется v1, v2, v3 × 3", [c["variant"] for c in calls] == ["v1", "v2", "v3"] * 3)
        expected_saved = list(range(9))
        actual_saved = [c["saved"].get("model_gemini-3.6-flash/raw_results.json", 0) for c in calls]
        check("raw_results.json сохраняется после каждого запуска", actual_saved == expected_saved, str(actual_saved))
        summary = (target / ex.SUMMARY_NAME).read_text(encoding="utf-8")
        check("summary.md: модель gemini-3.6-flash, статус «завершён»",
              "`gemini-3.6-flash`" in summary and "статус: завершён" in summary)
        after = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
        check("файлы основной модели в корне results/ не изменены", before == after)

        code, calls, out = run_main(["--model", "gemini-3.6-flash"], [], root)
        check("повторный запуск в ту же папку отклонён (exit 1, без вызовов API)",
              code == 1 and not calls and "уже есть результаты" in out)


def checks_default_model_guard():
    print("\n5. Модель по умолчанию: защита существующих результатов")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "raw_results.json").write_text("[1]", encoding="utf-8")
        code, calls, _ = run_main([], [], root)
        check("без --model при существующих results/raw_results.json — отказ без вызовов API",
              code == 1 and not calls and (root / "raw_results.json").read_text() == "[1]")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        code, calls, _ = run_main([], [], root)
        check("без --model: вызовы к gemini-3.8-flash, результаты в корне results/",
              code == 0 and {c["model"] for c in calls} == {"gemini-3.8-flash"}
              and (root / "raw_results.json").exists() and not list(root.glob("model_*")))


def checks_retries_and_stop():
    print("\n6. Повторы и остановка (на gemini-3.5-flash-lite)")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        script = [api_error(503), FakeResponse(),       # v1#1: 503 → ok (2 попытки)
                  api_error(504), api_error(504),       # v2#1: 504 → 504 (повтор исчерпан)
                  FakeResponse(),                       # v3#1: ok
                  api_error(400),                       # v1#2: 400 — без повтора, без остановки
                  api_error(503), api_error(429)]       # v2#2: 503 → 429 → остановка
        code, calls, out = run_main(["--model", "gemini-3.5-flash-lite"], script, root)
        records = load_records(root / "model_gemini-3.5-flash-lite" / ex.RAW_RESULTS_NAME)
        attempts = [(r["variant_id"][:2], r["run"], r["attempts_count"],
                     r["api_error"]["code"] if r["api_error"] else "ok") for r in records]
        check("попытки по запускам соответствуют сценарию",
              attempts == [("v1", 1, 2, "ok"), ("v2", 1, 2, 504), ("v3", 1, 1, "ok"),
                           ("v1", 2, 1, 400), ("v2", 2, 2, 429)], str(attempts))
        check("остановка на 429: всего 8 вызовов API, дальше запросов нет", len(calls) == 8)
        check("все неудачные попытки записаны в attempts",
              sum(len(r["attempts"]) for r in records) == 8)
        summary = (root / "model_gemini-3.5-flash-lite" / ex.SUMMARY_NAME).read_text(encoding="utf-8")
        check("summary.md: досрочная остановка и 4 невыполненных запуска",
              "остановлен досрочно" in summary and summary.count("- v") >= 4 and "v3_structured_output / запуск 3" in summary)


def main():
    print("Офлайн-проверки (сеть заблокирована, API не вызывается)")
    checks_config()
    checks_schema_and_validation()
    checks_model_argument()
    checks_full_run_new_model()
    checks_default_model_guard()
    checks_retries_and_stop()
    print(f"\nИтого: пройдено {len(PASSED)}, провалено {len(FAILED)}")
    if FAILED:
        print("Провалены:\n" + "\n".join(f"  - {name}" for name in FAILED))
        sys.exit(1)


if __name__ == "__main__":
    main()
