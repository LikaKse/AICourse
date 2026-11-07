"""Офлайн-проверки gemini-task4 — без единого запроса к API.

Запуск:
    ../venv/bin/python offline_checks.py

Сеть заблокирована на уровне сокетов, клиент API — заглушка, время моделируется,
результаты пишутся во временную папку (настоящая results/ не затрагивается).
"""

import sys

sys.dont_write_bytecode = True

import io
import json
import os
import socket
import subprocess
import tempfile
import types as pytypes
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path


def _no_network(*args, **kwargs):
    raise RuntimeError("Сетевое соединение запрещено в офлайн-проверках")


socket.socket.connect = _no_network
socket.create_connection = _no_network

from google import genai  # noqa: E402
from google.genai import errors, types  # noqa: E402

_client_created_on_import = []
_real_client = genai.Client
genai.Client = lambda *a, **k: _client_created_on_import.append(True)
PROJECT = Path(__file__).parent
PROJECT_RESULTS = PROJECT / "results"
_results_before = sorted(p.name for p in PROJECT_RESULTS.iterdir()) if PROJECT_RESULTS.exists() else None

import config  # noqa: E402
import demo  # noqa: E402
import experiment as ex  # noqa: E402
import summary  # noqa: E402
import validators as v  # noqa: E402
from tasks import TASKS, TASKS_BY_ID  # noqa: E402

_results_after = sorted(p.name for p in PROJECT_RESULTS.iterdir()) if PROJECT_RESULTS.exists() else None
genai.Client = _real_client

PASSED, FAILED = [], []
FAKE_KEY = "offline-FAKE-KEY-task4-do-not-leak"
CLIENT_KWARGS = []


def check(name, condition, details=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  {'✅' if condition else '❌'} {name}" + (f" — {details}" if details and not condition else ""))


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def install_clock():
    clock = FakeClock()
    ex.monotonic, ex.sleep = clock.monotonic, clock.sleep
    return clock


class FakeResponse:
    def __init__(self, text, usage="default", finish="STOP", latency=1.0):
        self.text, self.latency, self.model_version = text, latency, "fake"
        if usage == "default":
            usage = {"prompt_token_count": 60, "candidates_token_count": 40, "total_token_count": 100}
        self.usage_metadata = None if usage is None else pytypes.SimpleNamespace(**usage)
        self.candidates = [pytypes.SimpleNamespace(
            finish_reason=None if finish is None else pytypes.SimpleNamespace(name=finish))]


def single_json(answer="107"):
    return json.dumps({"answer": answer, "explanation": "29 + 30 + 31 + 17"}, ensure_ascii=False)


def names_json(names):
    return json.dumps({"names": names}, ensure_ascii=False)


def default_response(task_id, temperature, call_index):
    """Ответ по умолчанию: при t=0 — всегда одинаковый, иначе названия зависят от номера вызова."""
    if task_id == "days_count":
        return FakeResponse(single_json())
    if temperature == 0.0:
        return FakeResponse(names_json(["Байт", "Кофе-брейк", "Стек", "Деплой", "Капучино 404"]))
    return FakeResponse(names_json([f"Название {call_index}-{k}" for k in range(5)] if call_index % 2
                                   else ["Байт", f"Идея {call_index}", "Стек", f"Код {call_index}", "Пинг"]))


def make_fake_client(clock, script=None, results_root=None):
    script = list(script or [])
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = self
            CLIENT_KWARGS.append(kwargs)

        def generate_content(self, model, contents, config):
            saved = None
            if results_root is not None:
                raws = list(results_root.rglob(ex.RAW_RESULTS_NAME))
                saved = len(json.loads(raws[0].read_text())) if raws else 0
            task_id = next(t["id"] for t in TASKS if t["prompt"] == contents)
            calls.append({"model": model, "task_id": task_id, "prompt": contents, "start": clock.t, "saved": saved,
                          "config": config.model_dump(exclude_none=True, mode="json")})
            item = script.pop(0) if script else None
            if item is None:
                item = default_response(task_id, config.temperature, len(calls))
            if isinstance(item, BaseException):
                clock.t += 0.5
                raise item
            clock.t += item.latency
            return item

    return FakeClient, calls


def run_main(argv, root, script=None, env_key=FAKE_KEY):
    clock = install_clock()
    fake, calls = make_fake_client(clock, script, results_root=root)
    ex.genai.Client, ex.RESULTS_DIR = fake, root
    if env_key is None:
        os.environ.pop("GEMINI_API_KEY", None)
    else:
        os.environ["GEMINI_API_KEY"] = env_key
    out = io.StringIO()
    with redirect_stdout(out):
        code = ex.main(argv)
    ex.genai.Client = _real_client
    return code, calls, out.getvalue()


def api_error(code, message=None):
    return errors.APIError(code, {"error": {"code": code, "message": message or f"fake {code}", "status": "FAKE"}})


def load_raw(run_dir):
    p = run_dir / ex.RAW_RESULTS_NAME
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


# ---------- Проверки ----------

def checks_import_and_config():
    print("\n1. Импорт, конфигурация, совместимость SDK")
    check("импорт модулей не создаёт клиента API", not _client_created_on_import)
    check("импорт не меняет папку results/", _results_before == _results_after)
    proc = subprocess.run([sys.executable, "-B", "-c",
                           "import sys, demo, summary; print('google.genai' in sys.modules)"],
                          cwd=PROJECT, capture_output=True, text=True)
    check("demo.py и summary.py не импортируют google-genai", proc.stdout.strip() == "False", proc.stderr[-200:])
    check("модель gemini-3.5-flash-lite, thinking MINIMAL, max_output_tokens 2048",
          (config.MODEL, config.THINKING_LEVEL, config.MAX_OUTPUT_TOKENS) == ("gemini-3.5-flash-lite", "MINIMAL", 2048))
    check("температуры 0, 0.7, 1.0, 1.2; 5 повторов", config.TEMPERATURES == (0.0, 0.7, 1.0, 1.2)
          and config.RUNS_PER_COMBINATION == 5)
    dumps = {t: ex.describe_config(t) for t in config.TEMPERATURES}
    without_t = [{k: v_ for k, v_ in d.items() if k != "temperature"} for d in dumps.values()]
    check("конфиги отличаются только temperature",
          all(w == without_t[0] for w in without_t) and [d["temperature"] for d in dumps.values()] == [0.0, 0.7, 1.0, 1.2])
    check("в конфиге ровно temperature, max_output_tokens и thinking_level (без схемы и mime)",
          without_t[0] == {"max_output_tokens": 2048, "thinking_config": {"thinking_level": "MINIMAL"}})
    cfg = ex.build_config(1.2)
    check("SDK: GenerateContentConfig принимает temperature 1.2 и ThinkingLevel.MINIMAL",
          cfg.temperature == 1.2 and cfg.thinking_config.thinking_level == types.ThinkingLevel.MINIMAL
          and cfg.thinking_config.thinking_budget is None and cfg.response_schema is None
          and cfg.response_mime_type is None)
    a, b = date(2025, 3, 3), date(2025, 6, 17)
    check("эталон: с 3 марта по 17 июня 2025 включительно = 107 дней", (b - a).days + 1 == 107 == v.EXPECTED_DAYS)
    check("промпт задачи 1 явно говорит «включительно» и про оба конца",
          "включительно" in config.SINGLE_ANSWER_PROMPT and "считая и 3 марта, и 17 июня" in config.SINGLE_ANSWER_PROMPT)
    check("оба промпта содержат инструкцию JSON-формата",
          '"answer"' in config.SINGLE_ANSWER_PROMPT and '"names"' in config.GENERATION_PROMPT)


def checks_parsing():
    print("\n2. Разбор JSON и формат")
    good = single_json()
    cases = [("строгий JSON", good, "ok", False), ("```json-блок", f"```json\n{good}\n```", "fenced", True),
             ("текст вокруг JSON", f"Вот ответ: {good} Готово.", "extra_text", True),
             ("битый JSON", '{"answer": "107", "expl', "json_error", False),
             ("пустой", "  ", "empty", False), ("None", None, "empty", False)]
    for name, text, status, recovered in cases:
        p = v.parse_json_response(text)
        check(f"{name} → {status}", p["format_status"] == status and p["recovered"] == recovered,
              f"{p['format_status']}, recovered={p['recovered']}")


def checks_single_answer():
    print("\n3. Задача с одним ответом: правильность отдельно от формата")
    cases = [
        ("«107»", single_json("107"), "STOP", True, True, False),
        ("«107 дней»", single_json("107 дней"), "STOP", True, True, False),
        ("число 107 вместо строки", json.dumps({"answer": 107, "explanation": "x"}), "STOP", True, False, False),
        ("«106»", single_json("106"), "STOP", False, True, False),
        ("«106 или 107»", single_json("106 или 107"), "STOP", False, True, False),
        ("107 в ```-блоке", f"```json\n{single_json()}\n```", "STOP", True, False, False),
        ("обрезанный битый JSON", '{"answer": "10', "MAX_TOKENS", None, False, True),
        ("обрезан, но JSON целый", single_json(), "MAX_TOKENS", True, True, True),
        ("нет поля answer", json.dumps({"explanation": "x"}), "STOP", None, False, False),
        ("лишнее поле", json.dumps({"answer": "107", "explanation": "x", "confidence": 1}), "STOP", True, False, False),
    ]
    for name, text, finish, correct, fmt_ok, truncated in cases:
        e = v.evaluate_single(text, finish)
        check(f"{name}: is_correct={correct}, format_ok={fmt_ok}, truncated={truncated}",
              (e["is_correct"], e["format_ok"], e["truncated"]) == (correct, fmt_ok, truncated),
              f"{e['is_correct']}, {e['format_ok']}, {e['truncated']} ({e['check_reason']})")
    e = v.evaluate_single('{"answer": "10', "MAX_TOKENS")
    check("обрезка не считается неверным ответом (is_correct None, причина — MAX_TOKENS)",
          e["is_correct"] is None and "MAX_TOKENS" in e["check_reason"])


def checks_generation():
    print("\n4. Генерация: структура и повторы внутри списка")
    five = ["Байт", "Кофе-брейк", "Стек", "Деплой", "Капучино 404"]
    cases = [
        ("5 уникальных", names_json(five), "STOP", True, True),
        ("4 названия", names_json(five[:4]), "STOP", False, True),
        ("повтор с другим регистром и кавычками", names_json(["Байт", "«байт»", "Стек", "Деплой", "Пинг"]), "STOP", False, True),
        ("ё/е считаются одинаковыми", names_json(["Ёлка", "Елка", "Стек", "Деплой", "Пинг"]), "STOP", False, True),
        ("пустая строка", names_json(["Байт", " ", "Стек", "Деплой", "Пинг"]), "STOP", False, True),
        ("не строка", json.dumps({"names": ["Байт", 5, "Стек", "Деплой", "Пинг"]}), "STOP", False, False),
        ("нет names", json.dumps({"ideas": five}), "STOP", False, False),
        ("лишнее поле", json.dumps({"names": five, "comment": "x"}), "STOP", True, False),
        ("обрезан, JSON битый", '{"names": ["Байт", "Сте', "MAX_TOKENS", None, False),
    ]
    for name, text, finish, structure_ok, fmt_ok in cases:
        e = v.evaluate_generation(text, finish)
        check(f"{name}: structure_ok={structure_ok}, format_ok={fmt_ok}",
              (e["structure_ok"], e["format_ok"]) == (structure_ok, fmt_ok),
              f"{e['structure_ok']}, {e['format_ok']}, {e['structure_issues']}")
    e = v.evaluate_generation(names_json(["Байт", "«байт»", "Стек", "Деплой", "Пинг"]), "STOP")
    check("повтор внутри списка найден", e["duplicates_within"] == ["байт"], str(e["duplicates_within"]))


def make_record(task_id, t, run, text, finish="STOP", usage_total=100, latency=1.0, error=None):
    task = TASKS_BY_ID[task_id]
    return {"solution_id": ex.solution_id(task_id, t, run), "task_id": task_id, "kind": task["kind"],
            "temperature": t, "run": run, "raw_text": None if error else text, "finish_reason": None if error else finish,
            "total_token_count": None if error else usage_total, "candidates_token_count": None,
            "latency_sec": None if error else latency, "attempts": 1, "retries": 0, "attempt_log": [],
            "api_error": error, "expected_answer": task["expected"],
            "evaluation": None if error else task["evaluate"](text, finish)}


def checks_metrics():
    print("\n5. Метрики вариативности")
    same = [make_record("days_count", 0.0, i, single_json()) for i in range(1, 6)]
    m = summary.combo_metrics(same)
    check("5 одинаковых текстов: identical 5, distinct 1, схожесть 1.0",
          (m["identical_texts_max"], m["distinct_texts"], m["mean_similarity"]) == (5, 1, 1.0))
    mixed = [make_record("days_count", 1.2, 1, single_json("107")), make_record("days_count", 1.2, 2, single_json("106")),
             make_record("days_count", 1.2, 3, single_json("107 дней")),
             make_record("days_count", 1.2, 4, '{"answer": "10', finish="MAX_TOKENS"),
             make_record("days_count", 1.2, 5, None, error={"type": "APIError", "code": 503, "message": "x"})]
    m = summary.combo_metrics(mixed)
    check("смешанные: получено 4, ошибок API 1, верно 2, неверно 1, не оценено 1, обрезано 1",
          (m["received"], m["api_errors"], m["correct"], m["wrong"], m["not_evaluated"], m["truncated"]) == (4, 1, 2, 1, 1, 1),
          str({k: m[k] for k in ("received", "api_errors", "correct", "wrong", "not_evaluated", "truncated")}))
    check("различные ответы считаются по смыслу: «107» и «107 дней» — один ответ",
          m["answers"] == {"107": 2, "106": 1}, str(m["answers"]))
    check("ошибка API не считается неверным ответом", m["wrong"] == 1)
    check("среднее токенов без ошибки API (None не превращается в 0)", m["avg_total_tokens"] == 100)
    lists = [["A", "B", "C", "D", "E"], ["A", "B", "C", "F", "G"], ["H", "I", "J", "K", "L"]]
    recs = [make_record("cafe_names", 0.7, i + 1, names_json(n)) for i, n in enumerate(lists)]
    m = summary.combo_metrics(recs)
    check("уникальных названий 12 из 15", (m["unique_names"], m["total_names"]) == (12, 15))
    check("повторы между прогонами: a, b, c — в 2 прогонах", m["repeated_across_runs"] == {"a": 2, "b": 2, "c": 2},
          str(m["repeated_across_runs"]))
    expected_j = (3 / 7 + 0 + 0) / 3
    check("Jaccard между прогонами = (3/7 + 0 + 0) / 3", abs(m["mean_jaccard"] - expected_j) < 1e-9)
    check("одинаковый текст → схожесть 1, разные → < 1",
          summary.text_similarity("abc", "abc") == 1.0 and summary.text_similarity("abc", "xyz") < 1)


def checks_full_run():
    print("\n6. Полный прогон на заглушке (40 вызовов)")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        code, calls, out = run_main(["--full"], root)
        run_1 = root / "run_1"
        records = load_raw(run_1)
        check("exit 0, 40 вызовов, 40 записей", code == 0 and len(calls) == 40 and len(records) == 40,
              f"exit={code}, calls={len(calls)}, records={len(records)}")
        check("все вызовы к gemini-3.5-flash-lite", {c["model"] for c in calls} == {"gemini-3.5-flash-lite"})
        base = [{k: v_ for k, v_ in c["config"].items() if k != "temperature"} for c in calls]
        check("во всех 40 вызовах одинаковы все параметры, кроме temperature",
              all(b == base[0] for b in base) and base[0] == {"max_output_tokens": 2048,
                                                             "thinking_config": {"thinking_level": "MINIMAL"}})
        check("промпты не меняются между запусками (1 промпт на задачу)",
              {c["task_id"]: {x["prompt"] for x in calls if x["task_id"] == c["task_id"]} for c in calls}
              == {t["id"]: {t["prompt"]} for t in TASKS})
        combos = {(c["task_id"], c["config"]["temperature"]) for c in calls}
        per_combo = {k: sum(1 for c in calls if (c["task_id"], c["config"]["temperature"]) == k) for k in combos}
        check("8 комбинаций × 5 отдельных вызовов", len(combos) == 8 and set(per_combo.values()) == {5})
        order = [(c["task_id"], c["config"]["temperature"]) for c in calls[:8]]
        check("порядок: прогон 1 — обе задачи на всех температурах, затем прогон 2…",
              order == [(t["id"], temp) for t in TASKS for temp in config.TEMPERATURES])
        check("результат сохраняется сразу после каждого вызова (0, 1, …, 39)",
              [c["saved"] for c in calls] == list(range(40)))
        gaps = [b["start"] - a["start"] for a, b in zip(calls, calls[1:])]
        check("интервал между запросами ≥ 13 с", min(gaps) >= 13)
        check("timeout клиента 180 с", CLIENT_KWARGS[-1]["http_options"].timeout == 180_000)
        check("полный исходный текст сохранён в каждой записи",
              all(r["raw_text"] for r in records) and all(r["evaluation"] for r in records))
        leaked = [p.name for p in run_1.rglob("*") if p.is_file() and FAKE_KEY in p.read_text(encoding="utf-8")]
        check("API-ключ не попал в файлы и в консоль", not leaked and FAKE_KEY not in out)
        md = (run_1 / "summary.md").read_text(encoding="utf-8")
        check("summary.md: таблица, раздел про temperature = 0, примеры, все названия рядом",
              all(s in md for s in ("## Сводная таблица", "## Даёт ли temperature = 0", "## Примеры ответов",
                                    "## Все варианты названий", "### t = 0.0", "### t = 1.2", "| № | Прогон 1 |")))
        gen_names = {n for r in records if r["kind"] == "generation" for n in r["evaluation"]["names"]}
        check("в summary.md показаны все сгенерированные названия", all(n in md for n in gen_names))
        check("temperature = 0 на заглушке: «Да — все тексты совпали» для обеих задач",
              md.count("**Да — все тексты совпали побайтно.**") == 2)
        check("статус «завершён»", "статус: завершён" in md)

        before = {p.name: p.read_bytes() for p in run_1.iterdir()}
        run_main(["--full"], root)
        check("второй запуск → run_2, run_1 не изменён",
              (root / "run_2").is_dir() and {p.name: p.read_bytes() for p in run_1.iterdir()} == before)

        demo.RESULTS_DIR = root
        buf = io.StringIO()
        with redirect_stdout(buf):
            dcode = demo.main(["--run", "run_1", "--no-color"])
        dout = buf.getvalue()
        check("demo.py: работает по сохранённым результатам, все разделы",
              dcode == 0 and all(s in dout for s in ("Эксперимент «Температура»", "Эталон: 107",
                                                      "Даёт ли temperature = 0", "Сводка по метрикам")))
        check("demo.py: показывает все названия", all(n in dout for n in gen_names))
        code, calls, out = run_main([], root)
        check("без режима — exit 2, без вызовов", code == 2 and not calls and not (root / "run_3").exists())
        code, calls, out = run_main(["--full"], root, env_key=None)
        check("без GEMINI_API_KEY — exit 1, без вызовов", code == 1 and not calls and not (root / "run_3").exists())
        code, calls, out = run_main(["--dry-run"], root)
        check("--dry-run: без вызовов и папок, 40 вызовов в плане",
              code == 0 and not calls and not (root / "run_3").exists() and "вызовов API (без повторов): 40" in out)


def checks_errors_and_resume():
    print("\n7. Ошибки API, обрезка, повторы, Ctrl+C, продолжение")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        script = [api_error(400), FakeResponse('{"answer": "10', finish="MAX_TOKENS"),
                  api_error(503), None, TimeoutError("The read operation timed out")]
        code, calls, out = run_main(["--full"], root, script=script)
        records = load_raw(root / "run_1")
        r1, r2, r3, r4 = records[0], records[1], records[2], records[3]
        check("400: ошибка API, без повтора, evaluation None", r1["api_error"]["code"] == 400 and r1["attempts"] == 1
              and r1["evaluation"] is None)
        check("MAX_TOKENS: ответ получен, обрезан, is_correct None", r2["api_error"] is None
              and r2["evaluation"]["truncated"] and r2["evaluation"]["is_correct"] is None)
        check("503 → повтор → успех", r3["api_error"] is None and r3["attempts"] == 2 and r3["retries"] == 1)
        check("ReadTimeout записан как ошибка (тип и текст), без повтора",
              r4["api_error"]["type"] == "TimeoutError" and "timed out" in r4["api_error"]["message"] and r4["attempts"] == 1)
        md = (root / "run_1" / "summary.md").read_text(encoding="utf-8")
        check("ошибки API в отдельном разделе, не в «Неверных ответах»",
              "fake 400" in md.split("## Ошибки API")[1] and "fake 400" not in md.split("## Неверные ответы")[1].split("## Ошибки API")[0])
        check("статус «завершён с ошибками API»", "завершён с ошибками API" in md)
        kept = {r["solution_id"]: json.dumps(r, sort_keys=True) for r in records if r["api_error"] is None}
        code, calls, out = run_main(["--resume", "run_1"], root)
        records = load_raw(root / "run_1")
        sup = json.loads((root / "run_1" / ex.SUPERSEDED_NAME).read_text())
        check("--resume: 2 вызова, только неудачные (400 и таймаут), всего 40", len(calls) == 2 and len(records) == 40)
        check("--resume: успешные записи не изменены, неудачные — в superseded.json",
              all(json.dumps(r, sort_keys=True) == kept[r["solution_id"]] for r in records if r["solution_id"] in kept)
              and len(sup) == 2)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        code, calls, out = run_main(["--full"], root, script=[None, None, KeyboardInterrupt()])
        check("Ctrl+C: exit 130, сохранены 2 записи", code == 130 and len(load_raw(root / "run_1")) == 2)
        meta_path = root / "run_1" / ex.META_NAME
        meta = json.loads(meta_path.read_text())
        meta["tasks"]["days_count"]["prompt"] += " (изменено)"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False))
        code, calls, out = run_main(["--resume", "run_1"], root)
        check("--resume отказывается продолжать, если промпт отличается от сохранённого",
              code == 1 and not calls and "отличаются" in out)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        daily = "Generate content requests per day limit exceeded on Free Tier. Please retry in 13h2m."
        code, calls, out = run_main(["--full"], root, script=[None, api_error(429, daily)])
        check("дневная квота: остановка (exit 3) без повторов", code == 3 and len(calls) == 2
              and "исчерпана дневная квота" in (root / "run_1" / "summary.md").read_text())


def main():
    print("Офлайн-проверки gemini-task4 (сеть заблокирована, API не вызывается)")
    checks_import_and_config()
    checks_parsing()
    checks_single_answer()
    checks_generation()
    checks_metrics()
    checks_full_run()
    checks_errors_and_resume()
    print(f"\nИтого: пройдено {len(PASSED)}, провалено {len(FAILED)}")
    if FAILED:
        print("Провалены:\n" + "\n".join(f"  - {n}" for n in FAILED))
        sys.exit(1)


if __name__ == "__main__":
    main()
