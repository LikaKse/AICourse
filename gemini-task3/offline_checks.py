"""Офлайн-проверки — без единого запроса к API.

Запуск:
    ../venv/bin/python offline_checks.py

Гарантии отсутствия сети:
  * genai.Client подменяется заглушкой (ответы и ошибки задаются сценарием);
  * socket.socket.connect и socket.create_connection подменяются функцией, которая падает;
  * время моделируется (паузы не ждут по-настоящему);
  * результаты пишутся во временную папку, настоящая results/ не затрагивается.
"""

import sys

sys.dont_write_bytecode = True

import hashlib
import io
import json
import os
import socket
import tempfile
import types as pytypes
from contextlib import redirect_stdout
from decimal import Decimal
from pathlib import Path

# ---------- Блокировка сети (до импорта модулей проекта) ----------


def _no_network(*args, **kwargs):
    raise RuntimeError("Сетевое соединение запрещено в офлайн-проверках")


socket.socket.connect = _no_network
socket.create_connection = _no_network

from google import genai  # noqa: E402
from google.genai import _transformers as transformers  # noqa: E402
from google.genai import errors, types  # noqa: E402

# Импорт experiment не должен ничего запускать: если он создаст клиента — проверка упадёт
_client_created_on_import = []
_real_client = genai.Client
genai.Client = lambda *a, **k: _client_created_on_import.append(True)
PROJECT_RESULTS = Path(__file__).parent / "results"
_results_before_import = sorted(p.name for p in PROJECT_RESULTS.iterdir()) if PROJECT_RESULTS.exists() else None

import config  # noqa: E402
import experiment as ex  # noqa: E402
import summary  # noqa: E402
import validators as v  # noqa: E402
from tasks import TASKS, TASKS_BY_ID  # noqa: E402

_results_after_import = sorted(p.name for p in PROJECT_RESULTS.iterdir()) if PROJECT_RESULTS.exists() else None
genai.Client = _real_client

PASSED, FAILED = [], []
FAKE_KEY = "offline-FAKE-KEY-7f3a9c-do-not-leak"
CLIENT_KWARGS = []  # аргументы, с которыми создавался клиент-заглушка


def check(name, condition, details=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  {'✅' if condition else '❌'} {name}" + (f" — {details}" if details and not condition else ""))


# ---------- Моделируемое время ----------

class FakeClock:
    def __init__(self):
        self.t = 1000.0
        self.sleeps = []

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += seconds


def install_clock():
    clock = FakeClock()
    ex.monotonic = clock.monotonic
    ex.sleep = clock.sleep
    return clock


# ---------- Заглушки ответа и клиента ----------

class FakeResponse:
    def __init__(self, text, usage="default", finish="STOP", latency=1.0, model_version="fake"):
        self.text = text
        self.latency = latency
        self.model_version = model_version
        if usage == "default":
            usage = {"prompt_token_count": 10, "candidates_token_count": 20,
                     "thoughts_token_count": 5, "total_token_count": 35}
        self.usage_metadata = None if usage is None else pytypes.SimpleNamespace(**usage)
        self.candidates = ([] if finish == "no_candidates" else
                           [pytypes.SimpleNamespace(
                               finish_reason=None if finish is None else pytypes.SimpleNamespace(name=finish))])


def answer_json(answer, explanation="объяснение"):
    return json.dumps({"answer": answer, "explanation": explanation}, ensure_ascii=False)


CORRECT_ANSWERS = {
    "math_bat_ball": "0.05", "logic_syllogism": "Да", "combinatorics_digits": "24",
    "algorithm_loop": "14", "order_logic": "D, A, B, C",
}


def api_error(code, message=None):
    return errors.APIError(code, {"error": {"code": code, "message": message or f"fake {code}",
                                            "status": "FAKE"}})


def make_fake_client(clock, script=None, calls=None, results_root=None):
    """Клиент-заглушка. script — список ответов/исключений по порядку вызовов; после него —
    ответы по умолчанию (верный JSON для финального вызова, текст промпта для генерации)."""
    script = list(script or [])
    calls = calls if calls is not None else []

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.models = self
            CLIENT_KWARGS.append(kwargs)

        def generate_content(self, model, contents, config):
            saved = None
            if results_root is not None:
                raws = list(results_root.rglob(ex.RAW_RESULTS_NAME))
                saved = len(json.loads(raws[0].read_text())) if raws else 0
            structured = config.response_schema is not None
            task_id = next((t["id"] for t in TASKS if t["question"] in contents), None)
            calls.append({"model": model, "structured": structured, "task_id": task_id,
                          "thinking_level": config.thinking_config.thinking_level.name,
                          "thinking_budget": config.thinking_config.thinking_budget,
                          "max_output_tokens": config.max_output_tokens,
                          "start": clock.t, "saved": saved, "prompt": contents})
            item = script.pop(0) if script else None
            if item is None:
                item = (FakeResponse(answer_json(CORRECT_ANSWERS[task_id])) if structured
                        else FakeResponse("Реши задачу внимательно, проверь арифметику."))
            if isinstance(item, BaseException):
                clock.t += 0.5
                raise item
            clock.t += item.latency
            return item

    return FakeClient, calls


def run_main(argv, results_root, script=None, env_key=FAKE_KEY):
    clock = install_clock()
    fake_client, calls = make_fake_client(clock, script, results_root=results_root)
    ex.genai.Client = fake_client
    ex.RESULTS_DIR = results_root
    if env_key is None:
        os.environ.pop("GEMINI_API_KEY", None)
    else:
        os.environ["GEMINI_API_KEY"] = env_key
    out = io.StringIO()
    with redirect_stdout(out):
        try:
            code = ex.main(argv)
        except SystemExit as e:
            code = e.code
    ex.genai.Client = _real_client
    return code, calls, out.getvalue(), clock


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw(run_dir):
    path = run_dir / ex.RAW_RESULTS_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


# ---------- Проверки ----------

def checks_import_and_config():
    print("\n1. Импорт и конфигурация")
    check("импорт experiment не создаёт клиента API", not _client_created_on_import)
    check("импорт experiment не меняет папку results/", _results_before_import == _results_after_import)
    check("модель gemini-3.5-flash-lite", config.MODEL == "gemini-3.5-flash-lite")
    check("уровни MINIMAL и HIGH", config.THINKING_LEVELS == ("MINIMAL", "HIGH"))
    check("4 способа", config.STRATEGIES == ("direct", "step_by_step", "prompt_generation", "expert_group"))
    check("таймаут запроса 180 с", config.REQUEST_TIMEOUT_SEC == 180)
    check("5 задач", [t["id"] for t in TASKS] == ["math_bat_ball", "logic_syllogism", "combinatorics_digits",
                                                    "algorithm_loop", "order_logic"])
    ok_cfg = True
    for level in config.THINKING_LEVELS:
        for structured in (False, True):
            cfg = ex.build_config(level, structured)
            tc = cfg.thinking_config
            ok_cfg &= (tc.thinking_level == types.ThinkingLevel[level] and tc.thinking_budget is None
                       and cfg.max_output_tokens is None
                       and (cfg.response_schema is v.ApiSolution) == structured)
    check("thinking_level задан, thinking_budget и max_output_tokens не заданы (все 4 конфига)", ok_cfg)
    sources = "".join((Path(__file__).parent / f).read_text(encoding="utf-8")
                      for f in ("experiment.py", "config.py"))
    check("thinking_budget нигде не передаётся в коде", "thinking_budget=" not in sources)
    schema = transformers.t_schema(_real_client(api_key="offline"), v.ApiSolution).model_dump(
        exclude_none=True, mode="json")
    check("API-схема: answer и explanation обязательны, нет additional_properties",
          sorted(schema["required"]) == ["answer", "explanation"] and "additional_properties" not in json.dumps(schema))
    direct = ex.build_final_prompt("direct", TASKS[0])
    check("Direct: только задача и формат, без инструкций рассуждения",
          direct == f"{TASKS[0]['question']}\n\n{config.ANSWER_FORMAT}" and "пошагово" not in direct)
    check("формат ответа одинаков во всех финальных промптах",
          all(config.ANSWER_FORMAT in ex.build_final_prompt(s, TASKS[0], "GEN") for s in config.STRATEGIES))
    check("Step-by-step содержит инструкцию из задания",
          "Решай задачу пошагово. Проверь результат перед ответом." in ex.build_final_prompt("step_by_step", TASKS[0]))
    expert = ex.build_final_prompt("expert_group", TASKS[0])
    check("Expert group: аналитик, инженер, критик в одном промпте",
          all(role in expert for role in ("Аналитик", "Инженер", "Критик")))


def checks_reference_answers():
    print("\n2. Эталонные ответы и проверки по задачам")
    for task in TASKS:
        ok, reason = task["check"](task["expected"])
        check(f"эталон проходит проверку: {task['id']} = {task['expected']!r}", ok, reason)

    cases = {
        "math_bat_ball": (["0.05", "$0.05", "0,05 доллара", "5 центов", "5 cents", "0.05 доллара (5 центов)",
                           "Мяч стоит $0,05."],
                          ["0.10", "$0,1", "10 центов", "1.05", "0.05 или 0.10", "не знаю", ""]),
        "logic_syllogism": (["да", "Да", "Да, следует", "yes", "Следует", "ДА."],
                            ["нет", "Не следует", "да и нет", "no", "возможно", ""]),
        "combinatorics_digits": (["24", "24 числа", "Всего 24.", "24.0"],
                                 ["12", "4·3·2 = 24", "64", "двадцать четыре", ""]),
        "algorithm_loop": (["14", "total = 14", "14."],
                           ["13", "1 + 4 + 9 = 14", "30", ""]),
        "order_logic": (["D, A, B, C", "D → A → B → C", "DABC", "d-a-b-c", "Порядок: D, A, B, C",
                         "D, А, В, С (кириллица)"],
                        ["A, B, C, D", "D, A, C, B", "D, A, B", "порядок не определён", ""]),
    }
    for task_id, (good, bad) in cases.items():
        checker = TASKS_BY_ID[task_id]["check"]
        bad_ok = [b for b in good if not checker(b)[0]]
        check(f"{task_id}: засчитываются {len(good)} верных форм", not bad_ok, f"не засчитаны: {bad_ok}")
        wrong_ok = [b for b in bad if checker(b)[0]]
        check(f"{task_id}: отклоняются {len(bad)} неверных/неоднозначных", not wrong_ok, f"засчитаны: {wrong_ok}")


def checks_normalization_and_parsing():
    print("\n3. Нормализация и разбор JSON")
    check("normalize_text: регистр, пробелы, NBSP, пунктуация по краям",
          v.normalize_text("  Мяч  СТОИТ   $0,05.  ") == "мяч стоит $0,05")
    check("normalize_text(None) = ''", v.normalize_text(None) == "")
    nums = [n for n, _ in v.extract_numbers("$1,10 и 0.05")]
    check("extract_numbers: «1,10» и «0.05» как дроби", nums == [Decimal("1.10"), Decimal("0.05")], str(nums))
    nums = [n for n, _ in v.extract_numbers("цифры 1, 2, 3, 4")]
    check("extract_numbers: «1, 2, 3, 4» — четыре числа, а не дроби", nums == [1, 2, 3, 4], str(nums))
    check("extract_order: кириллические заглавные А, В, С распознаются",
          v.extract_order("D, А, В, С") == ["D", "A", "B", "C"])
    check("extract_order: русские слова и предлоги «а», «в», «с» не считаются буквами",
          v.extract_order("сначала D, а затем A, в конце B и с ним C") == ["D", "A", "B", "C"])

    parse_cases = [
        ("валидный JSON", answer_json("24"), "ok", "ok"),
        ("пустой ответ", "", "empty", "skipped"),
        ("None", None, "empty", "skipped"),
        ("Markdown", "**Ответ:** 24", "json_error", "skipped"),
        ("JSON в ```", "```json\n" + answer_json("24") + "\n```", "json_error", "skipped"),
        ("нет explanation", json.dumps({"answer": "24"}), "ok", "validation_error"),
        ("лишнее поле", json.dumps({"answer": "24", "explanation": "x", "confidence": 1}), "ok", "validation_error"),
        ("пустой answer", json.dumps({"answer": "", "explanation": "x"}), "ok", "validation_error"),
    ]
    for name, text, p_status, v_status in parse_cases:
        r = v.parse_solution(text)
        check(f"разбор: {name} → {p_status}/{v_status}",
              r["parse_status"] == p_status and r["validation_status"] == v_status,
              f"{r['parse_status']}/{r['validation_status']}")


def solve_with(script, strategy="direct", task_id="algorithm_loop", level="MINIMAL"):
    clock = install_clock()
    fake_client, calls = make_fake_client(clock, script)
    record = ex.solve(fake_client(), config.MODEL, TASKS_BY_ID[task_id], strategy, level,
                      ex.Pacer(config.MIN_INTERVAL_BETWEEN_REQUESTS_SEC))
    return record, calls, clock


def checks_missing_usage():
    print("\n4. Отсутствующие usage_metadata и finish_reason")
    r, _, _ = solve_with([FakeResponse(answer_json("14"), usage=None, finish=None)])
    check("usage_metadata = None → все поля токенов None (не 0)",
          all(r[f] is None for f in ex.USAGE_FIELDS), str({f: r[f] for f in ex.USAGE_FIELDS}))
    check("finish_reason отсутствует → None", r["finish_reason"] is None)
    check("при этом ответ проверен: верно", r["is_correct"] is True)
    r, _, _ = solve_with([FakeResponse(answer_json("14"), usage={"prompt_token_count": 7, "total_token_count": 9},
                                       finish="no_candidates")])
    check("частичный usage: известные поля сохранены, отсутствующие — None",
          r["prompt_token_count"] == 7 and r["total_token_count"] == 9
          and r["candidates_token_count"] is None and r["thoughts_token_count"] is None)
    check("нет candidates → finish_reason None", r["finish_reason"] is None)
    groups = summary.group_metrics([r])
    check("метрики: токены без отсутствующих значений не обнуляются",
          groups["avg_tokens"] == 9 and groups["avg_thoughts"] is None)


def checks_prompt_generation_sums():
    print("\n5. Prompt generation: суммирование токенов и времени")
    gen = FakeResponse("Сгенерированный промпт: внимательно посчитай сумму квадратов.",
                       usage={"prompt_token_count": 40, "candidates_token_count": 30,
                              "thoughts_token_count": 10, "total_token_count": 80}, latency=2.0)
    sol = FakeResponse(answer_json("14"), usage={"prompt_token_count": 90, "candidates_token_count": 25,
                                                 "thoughts_token_count": 15, "total_token_count": 130},
                       latency=3.5)
    r, calls, clock = solve_with([gen, sol], strategy="prompt_generation")
    check("2 вызова: генерация (без схемы) + решение (со схемой)",
          [c["structured"] for c in calls] == [False, True] and [c["stage"] for c in r["calls"]]
          == ["prompt_generation", "solve"])
    check("сгенерированный промпт подставлен во второй вызов",
          "внимательно посчитай сумму квадратов" in calls[1]["prompt"] and r["generated_prompt"].startswith("Сгенерированный"))
    check("оба вызова с одним thinking_level", [c["thinking_level"] for c in calls] == ["MINIMAL", "MINIMAL"])
    check("токены суммированы: total 80 + 130 = 210, prompt 130, candidates 55, thoughts 25",
          (r["total_token_count"], r["prompt_token_count"], r["candidates_token_count"], r["thoughts_token_count"])
          == (210, 130, 55, 25))
    check("время суммировано: 2.0 + 3.5 = 5.5 с", r["latency_sec"] == 5.5, str(r["latency_sec"]))
    check("метрики каждого вызова сохранены отдельно",
          r["calls"][0]["total_token_count"] == 80 and r["calls"][1]["total_token_count"] == 130
          and r["calls"][0]["latency_sec"] == 2.0 and r["calls"][1]["latency_sec"] == 3.5)
    check("между вызовами выдержан интервал ≥ 13 с", calls[1]["start"] - calls[0]["start"] >= 13)
    check("tokens_incomplete = False", r["tokens_incomplete"] is False)

    gen2 = FakeResponse("Промпт.", usage={"total_token_count": 80, "prompt_token_count": 40,
                                          "candidates_token_count": 30}, latency=2.0)
    sol2 = FakeResponse(answer_json("14"), usage=None, latency=3.0)
    r, _, _ = solve_with([gen2, sol2], strategy="prompt_generation")
    check("решение без usage: сумма = известная часть (80), tokens_incomplete = True",
          r["total_token_count"] == 80 and r["tokens_incomplete"] is True)
    r, _, _ = solve_with([FakeResponse("Промпт.", usage=None), FakeResponse(answer_json("14"), usage=None)],
                         strategy="prompt_generation")
    check("оба вызова без usage: сумма None (не 0)", r["total_token_count"] is None)

    r, calls, _ = solve_with([api_error(400)], strategy="prompt_generation")
    check("ошибка на генерации: решение не вызывается, is_correct None, ошибка с этапом",
          len(calls) == 1 and r["is_correct"] is None and r["api_error"]["stage"] == "prompt_generation")
    r, calls, _ = solve_with([FakeResponse("   ")], strategy="prompt_generation")
    check("пустой сгенерированный промпт: решение не вызывается, is_correct False",
          len(calls) == 1 and r["is_correct"] is False and "пустой промпт" in r["check_reason"])

    r, _, _ = solve_with([FakeResponse("Подсказка: ответ 0,05 доллара."), FakeResponse(answer_json("0.05"))],
                         strategy="prompt_generation", task_id="math_bat_ball")
    check("утечка ответа в промпт обнаружена", r["generated_prompt_leaks_answer"] is True)
    r, _, _ = solve_with(None, strategy="prompt_generation", task_id="math_bat_ball")
    check("без утечки — False", r["generated_prompt_leaks_answer"] is False)
    r, _, _ = solve_with(None, strategy="prompt_generation", task_id="logic_syllogism")
    check("для да/нет утечка не определяется — None", r["generated_prompt_leaks_answer"] is None)


def checks_retries():
    print("\n6. Повторы и ошибки API")
    r, calls, clock = solve_with([api_error(503), None])
    check("503 → ok: 2 попытки, пауза 15 с, верный ответ",
          r["attempts"] == 2 and r["retries"] == 1 and 15 in clock.sleeps and r["is_correct"] is True)
    r, calls, clock = solve_with([api_error(503), api_error(504), api_error(503)])
    check("503/504 ×3: 3 попытки (без бесконечных повторов), ошибка сохранена, is_correct None",
          len(calls) == 3 and r["api_error"]["code"] == 503 and r["is_correct"] is None)
    check("все неудачные попытки записаны", len(r["calls"][0]["attempts"]) == 3
          and all(a["error"] for a in r["calls"][0]["attempts"]))
    r, calls, clock = solve_with([api_error(429, "Requests per minute exceeded. Please retry in 45.5s."), None])
    check("429 per minute с подсказкой 45.5 с → пауза 45.5 с и повтор",
          r["is_correct"] is True and r["calls"][0]["attempts"][0]["retry_after_sec"] == 45.5)
    r, calls, clock = solve_with([api_error(429, "Requests per minute exceeded. Please retry in 200s."), None])
    check("подсказка 200 с ограничена MAX_RETRY_AFTER_SEC = 60",
          r["calls"][0]["attempts"][0]["retry_after_sec"] == 60)
    daily = ("Quota exceeded for metric: generate_content_free_tier_requests, limit: 20, "
             "description: Generate content requests per day limit exceeded on Free Tier. Please retry in 14h9m.")
    r, calls, clock = solve_with([api_error(429, daily)])
    check("429 per day → без повторов, daily_quota_exceeded = True",
          len(calls) == 1 and r["daily_quota_exceeded"] is True)
    r, calls, _ = solve_with([api_error(400)])
    check("400 → без повтора, ошибка не скрыта", len(calls) == 1 and r["api_error"]["code"] == 400)
    r, calls, _ = solve_with([TimeoutError("read timeout")])
    check("исключение клиента записано как ошибка (тип и текст)",
          r["api_error"]["type"] == "TimeoutError" and "read timeout" in r["api_error"]["message"])


def checks_metrics():
    print("\n7. Расчёт метрик")

    def rec(strategy, level, correct, tokens, latency, error=False, retries=0, incomplete=False):
        return {"solution_id": f"t|{strategy}|{level}", "task_id": "t", "strategy": strategy,
                "thinking_level": level, "is_correct": None if error else correct,
                "total_token_count": None if error else tokens, "thoughts_token_count": None,
                "latency_sec": None if error else latency, "api_error": {"code": 503} if error else None,
                "retries": retries, "attempts": 1 + retries, "tokens_incomplete": incomplete}

    records = [rec("direct", "MINIMAL", True, 100, 1.0), rec("direct", "MINIMAL", False, 200, 3.0),
               rec("direct", "MINIMAL", True, None, 2.0), rec("direct", "MINIMAL", None, None, None, error=True, retries=2)]
    m = summary.group_metrics(records)
    check("верных 2, всего 4, ответов 3", (m["correct"], m["total"], m["answered"]) == (2, 4, 3))
    check("точность от ответов 66.7%, от всех 50%",
          round(m["accuracy"], 1) == 66.7 and m["accuracy_of_all"] == 50)
    check("токены: сумма 300 по 2 известным, среднее 150 (None не считается нулём)",
          m["tokens_sum"] == 300 and m["tokens_known"] == 2 and m["avg_tokens"] == 150)
    check("среднее время 2.0 с по 3 ответам", m["avg_latency"] == 2.0 and m["latency_known"] == 3)
    check("ошибок API 1, повторов 2", m["api_errors"] == 1 and m["retries"] == 2)
    empty = summary.group_metrics([])
    check("пустая группа: точность и средние — None", empty["accuracy"] is None and empty["avg_tokens"] is None)

    groups = {(s, l): summary.group_metrics([]) for s in config.STRATEGIES for l in config.THINKING_LEVELS}
    groups[("direct", "MINIMAL")] = {**empty, "accuracy": 60}
    groups[("direct", "HIGH")] = {**empty, "accuracy": 100}
    groups[("step_by_step", "MINIMAL")] = {**empty, "accuracy": 80}
    groups[("step_by_step", "HIGH")] = {**empty, "accuracy": 100}
    text = "\n".join(summary.build_conclusions(groups))
    check("вывод: step-by-step помогает при MINIMAL (+20 п.п.), при HIGH преимущество исчезает",
          "+20 п.п." in text and "при HIGH преимущество исчезает" in text)
    check("вывод: при HIGH Direct = 100% → отмечен эффект потолка", "эффект потолка" in text)


def checks_pilot_run_and_saving():
    print("\n8. Пилотный прогон на заглушке, сохранение, защита от перезаписи, ключ")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        code, calls, out, clock = run_main(["--pilot"], root)
        run_1 = root / "run_1"
        records = load_raw(run_1)
        check("пилот: exit 0, 10 вызовов API, 8 решений", code == 0 and len(calls) == 10 and len(records) == 8,
              f"exit={code}, calls={len(calls)}, records={len(records)}")
        check("все вызовы — задача пилота и модель gemini-3.5-flash-lite",
              {c["task_id"] for c in calls} == {config.PILOT_TASK_ID} and {c["model"] for c in calls} == {config.MODEL})
        check("ни один вызов не передаёт thinking_budget и max_output_tokens",
              all(c["thinking_budget"] is None and c["max_output_tokens"] is None for c in calls))
        check("уровни: 5 вызовов MINIMAL, затем 5 HIGH",
              [c["thinking_level"] for c in calls] == ["MINIMAL"] * 5 + ["HIGH"] * 5)
        saved = [c["saved"] for c in calls]
        check("raw_results.json сохраняется после каждого решения (0,1,2,2,3,4,5,6,6,7)",
              saved == [0, 1, 2, 2, 3, 4, 5, 6, 6, 7], str(saved))
        gaps = [b["start"] - a["start"] for a, b in zip(calls, calls[1:])]
        check("интервал между началами запросов ≥ 13 с", min(gaps) >= 13, str(gaps))
        check("все 8 решений верны", all(r["is_correct"] for r in records))
        timeout_ms = CLIENT_KWARGS[-1]["http_options"].timeout
        check("клиент создан с таймаутом 180 с (180000 мс)", timeout_ms == 180_000, str(timeout_ms))
        check("поле model = gemini-3.5-flash-lite во всех решениях",
              {r["model"] for r in records} == {"gemini-3.5-flash-lite"})
        check("thinking_level в request_config каждого вызова совпадает с уровнем решения",
              all(c["request_config"]["thinking_config"] == {"thinking_level": r["thinking_level"]}
                  for r in records for c in r["calls"]))
        per_call_ok = all(all(f in c for f in ex.USAGE_FIELDS) for r in records for c in r["calls"])
        single_ok = all(len(r["calls"]) == 1 and all(r[f] == r["calls"][0][f] for f in ex.USAGE_FIELDS)
                        for r in records if r["strategy"] != "prompt_generation")
        pg_ok = all(len(r["calls"]) == 2
                    and r["total_token_count"] == sum(c["total_token_count"] for c in r["calls"])
                    and r["latency_sec"] == round(sum(c["latency_sec"] for c in r["calls"]), 3)
                    for r in records if r["strategy"] == "prompt_generation")
        check("usage_metadata сохранены в каждом вызове (calls[]) отдельно", per_call_ok)
        check("1 вызов: итоговые токены решения = токены этого вызова", single_ok)
        check("Prompt generation: итог = сумма 2 вызовов по токенам и времени", pg_ok)
        files = sorted(p.name for p in run_1.iterdir())
        check("в папке запуска: raw_results.json, run_meta.json, summary.md, нет .tmp",
              files == ["raw_results.json", "run_meta.json", "summary.md"], str(files))
        leaked = [p.name for p in run_1.rglob("*") if p.is_file() and FAKE_KEY in p.read_text(encoding="utf-8")]
        check("API-ключ не попал ни в один файл", not leaked, str(leaked))
        check("API-ключ не напечатан в консоль", FAKE_KEY not in out)
        md = (run_1 / "summary.md").read_text(encoding="utf-8")
        check("summary.md: итоговая таблица, MINIMAL vs HIGH, сравнение с Direct, статус «завершён»",
              all(s in md for s in ("| Способ | Reasoning | Верных ответов | Всего |", "## MINIMAL vs HIGH",
                                    "## Сравнение с Direct", "статус: завершён")))

        hash_before = {p.name: file_hash(p) for p in run_1.iterdir()}
        code, calls, _, _ = run_main(["--pilot"], root)
        check("второй запуск создаёт run_2, run_1 не изменён",
              (root / "run_2").is_dir() and {p.name: file_hash(p) for p in run_1.iterdir()} == hash_before)

        code, calls, out, _ = run_main([], root)
        check("без режима (--pilot/--full/--resume) — exit 2, без вызовов и без новых папок",
              code == 2 and not calls and not (root / "run_3").exists())
        code, calls, out, _ = run_main(["--pilot"], root, env_key=None)
        check("без GEMINI_API_KEY — exit 1, без вызовов и без новых папок",
              code == 1 and not calls and not (root / "run_3").exists())
        code, calls, out, _ = run_main(["--dry-run"], root)
        check("--dry-run: без вызовов API и без новых папок", code == 0 and not calls and not (root / "run_3").exists())
        check("--dry-run: 40 решений, 50 вызовов", "Решений: 40, ожидаемых вызовов API (без повторов): 50" in out)
        code, calls, out, _ = run_main(["--resume", "../escape"], root)
        check("--resume с недопустимым именем отклонён", code == 2 and not calls)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        no_usage = FakeResponse(answer_json("0.05"), usage=None, finish=None)
        partial = FakeResponse(answer_json("0.05"), usage={"prompt_token_count": 12, "total_token_count": 40})
        run_main(["--pilot"], root, script=[no_usage, partial])
        raw_text = (root / "run_1" / ex.RAW_RESULTS_NAME).read_text(encoding="utf-8")
        records = json.loads(raw_text)
        first, second = records[0], records[1]
        check("в файле: нет usage → null во всех полях токенов и finish_reason (не 0)",
              all(first[f] is None and first["calls"][0][f] is None for f in ex.USAGE_FIELDS)
              and first["finish_reason"] is None and first["calls"][0]["finish_reason"] is None)
        check("в файле: частичный usage → известные значения, остальные null",
              second["prompt_token_count"] == 12 and second["total_token_count"] == 40
              and second["candidates_token_count"] is None and second["thoughts_token_count"] is None)
        check('в raw_results.json записано именно "…_token_count": null',
              '"thoughts_token_count": null' in raw_text and '"total_token_count": null' in raw_text)


def checks_interrupt_and_resume():
    print("\n9. Ctrl+C, продолжение и дневная квота")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Вызовы: 1 direct, 2 step, 3 pg-генерация, 4 pg-решение → Ctrl+C
        code, calls, out, _ = run_main(["--pilot"], root, script=[None, None, None, KeyboardInterrupt()])
        run_dir = root / "run_1"
        records = load_raw(run_dir)
        check("Ctrl+C: exit 130, сохранены 2 завершённых решения",
              code == 130 and len(records) == 2, f"exit={code}, records={len(records)}")
        md = (run_dir / "summary.md").read_text(encoding="utf-8")
        check("summary.md: статус «прерван», подсказка --resume в консоли",
              "прерван пользователем" in md and "--resume run_1" in out)
        kept = {r["solution_id"]: json.dumps(r, sort_keys=True) for r in records}

        code, calls, out, _ = run_main(["--resume", "run_1"], root)
        records = load_raw(run_dir)
        check("--resume: выполнены оставшиеся 6 решений (8 вызовов), всего 8",
              code == 0 and len(calls) == 8 and len(records) == 8, f"calls={len(calls)}, records={len(records)}")
        check("ранее сохранённые решения не изменены",
              all(json.dumps(r, sort_keys=True) == kept[r["solution_id"]] for r in records if r["solution_id"] in kept))
        plan_order = [ex.solution_id(*item) for item in json.loads((run_dir / ex.META_NAME).read_text())["plan"]]
        check("порядок записей соответствует плану", [r["solution_id"] for r in records] == plan_order)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        daily = "Generate content requests per day limit exceeded on Free Tier. Please retry in 13h2m."
        code, calls, out, _ = run_main(["--pilot"], root, script=[None, api_error(429, daily)])
        run_dir = root / "run_1"
        records = load_raw(run_dir)
        check("дневная квота: остановка (exit 3) сразу после ошибки, 2 вызова",
              code == 3 and len(calls) == 2 and len(records) == 2)
        check("summary.md: статус «исчерпана дневная квота»",
              "исчерпана дневная квота" in (run_dir / "summary.md").read_text(encoding="utf-8"))
        ok_record = json.dumps(records[0], sort_keys=True)
        code, calls, out, _ = run_main(["--resume", "run_1"], root)
        records = load_raw(run_dir)
        superseded = json.loads((run_dir / ex.SUPERSEDED_NAME).read_text(encoding="utf-8"))
        check("--resume: неудачное решение перенесено в superseded.json и переделано",
              len(superseded) == 1 and superseded[0]["api_error"]["code"] == 429
              and len(records) == 8 and all(r["api_error"] is None for r in records))
        check("--resume: успешное решение не изменено", json.dumps(records[0], sort_keys=True) == ok_record)


def main():
    print("Офлайн-проверки gemini-task3 (сеть заблокирована, API не вызывается)")
    checks_import_and_config()
    checks_reference_answers()
    checks_normalization_and_parsing()
    checks_missing_usage()
    checks_prompt_generation_sums()
    checks_retries()
    checks_metrics()
    checks_pilot_run_and_saving()
    checks_interrupt_and_resume()
    print(f"\nИтого: пройдено {len(PASSED)}, провалено {len(FAILED)}")
    if FAILED:
        print("Провалены:\n" + "\n".join(f"  - {name}" for name in FAILED))
        sys.exit(1)


if __name__ == "__main__":
    main()
