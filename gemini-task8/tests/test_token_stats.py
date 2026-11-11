"""День 8: подсчёт токенов, статистика, история и обработка ошибок (офлайн, без API)."""

import io
import json
import logging
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from google.genai import types

import config
import experiments
import main
import token_stats
from agent import Agent, is_context_overflow_error
from history_store import HistoryStore
from tests.fakes import (FakeClient, api_error, call_response, fake_token_count, no_usage, text_response,
                         with_usage)

CONVERT_ARGS = {"amount": 150, "from_currency": "USD", "to_currency": "VND"}


def user(text):
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


class UsageParsingTest(unittest.TestCase):
    def test_all_fields(self):
        resp = with_usage(text_response("x"), prompt=1000, candidates=40, thoughts=60, cached=256)
        resp.usage_metadata.tool_use_prompt_token_count = 7
        u = token_stats.usage_from_response(resp)
        self.assertTrue(u.present)
        self.assertEqual((u.prompt_tokens, u.candidates_tokens, u.thoughts_tokens, u.cached_tokens), (1000, 40, 60, 256))
        self.assertEqual(u.input_tokens, 1007)          # prompt + tool_use_prompt
        self.assertEqual(u.output_tokens, 100)          # candidates + thinking

    def test_none_fields_are_zero_and_missing_usage(self):
        resp = text_response("x")
        resp.usage_metadata = types.GenerateContentResponseUsageMetadata(prompt_token_count=5)
        u = token_stats.usage_from_response(resp)
        self.assertEqual((u.prompt_tokens, u.output_tokens, u.cached_tokens), (5, 0, 0))
        missing = token_stats.usage_from_response(no_usage(text_response("x")))
        self.assertFalse(missing.present)
        self.assertEqual(missing.input_tokens, 0)


class HistoryShapeTest(unittest.TestCase):
    def test_counts_tool_parts(self):
        agent = Agent(FakeClient([
            call_response(("convert_currency", CONVERT_ARGS), text="Сейчас"),
            text_response("Ответ", thought="мысль"),
        ]))
        agent.run("150 USD?")
        shape = token_stats.history_shape(agent.history)
        self.assertEqual(shape.messages, 4)
        self.assertEqual((shape.function_call_parts, shape.function_response_parts), (1, 1))
        self.assertEqual((shape.user_text_parts, shape.model_text_parts, shape.thought_parts), (1, 2, 1))


class CostTest(unittest.TestCase):
    def test_known_model(self):
        cost = token_stats.estimate_cost(1_000_000, 1_000_000, "gemini-3.5-flash-lite")
        self.assertAlmostEqual(cost, config.PRICE_INPUT_PER_1M + config.PRICE_OUTPUT_PER_1M)

    def test_unknown_model_has_no_price(self):
        self.assertIsNone(token_stats.estimate_cost(10, 10, "some-unverified-model"))
        self.assertIn("не проверен", token_stats.pricing_note("some-unverified-model"))

    def test_turn_cost_uses_actual_usage(self):
        r = Agent(FakeClient([with_usage(text_response("x"), prompt=2000, candidates=100, thoughts=0)])).run("x")
        self.assertAlmostEqual(r.stats.cost_usd, (2000 * 0.30 + 100 * 2.50) / 1e6)


class AgentCountingTest(unittest.TestCase):
    def setUp(self):
        logging.getLogger("agent").setLevel(logging.CRITICAL)

    def test_request_and_history_counted_before_send(self):
        client = FakeClient([text_response("Привет, Аня")])
        r = Agent(client).run("Меня зовут Аня")
        # 1) только текущее сообщение, 2) вся история перед обращением
        self.assertEqual(len(client.models.count_requests), 2)
        self.assertEqual(client.models.count_requests[0], [user("Меня зовут Аня")])
        self.assertEqual(client.models.count_requests[1], client.requests[0]["contents"])
        self.assertEqual(r.stats.request_tokens, fake_token_count([user("Меня зовут Аня")]))
        self.assertEqual(r.stats.history_tokens, r.stats.request_tokens)  # первая реплика = вся история
        self.assertEqual(r.stats.response_tokens, 25)
        self.assertIsNone(r.stats.tool_exchange_tokens)      # инструментов не было

    def test_multi_call_turn_counts_each_call_and_totals(self):
        client = FakeClient([
            with_usage(call_response(("convert_currency", CONVERT_ARGS, "id-1")), prompt=300, candidates=15, thoughts=10),
            with_usage(call_response(("calculate", {"expression": "3750000 * 1.1"}, "id-2")),
                       prompt=360, candidates=12, thoughts=8),
            with_usage(text_response("4 125 000 VND"), prompt=420, candidates=30, thoughts=0),
        ])
        r = Agent(client).run("150 USD в VND и +10%")
        st = r.stats
        self.assertEqual(st.model_calls, 3)
        self.assertEqual([c.usage.prompt_tokens for c in st.calls], [300, 360, 420])
        self.assertEqual((st.input_tokens, st.output_tokens, st.thoughts_tokens), (1080, 75, 18))
        self.assertEqual((r.input_tokens, r.output_tokens), (1080, 75))     # совместимость с Днём 7
        self.assertEqual(st.response_tokens, 30)
        self.assertEqual(st.last_prompt_tokens, 420)
        # История растёт с каждым обращением и включает function_call/function_response.
        hist = [c.history_tokens for c in st.calls]
        self.assertEqual(hist, sorted(hist))
        self.assertLess(hist[0], hist[1])
        self.assertEqual([c.history_tokens for c in st.calls],
                         [fake_token_count(req["contents"]) for req in client.requests])
        self.assertEqual([(c.shape.function_call_parts, c.shape.function_response_parts) for c in st.calls],
                         [(0, 0), (1, 1), (2, 2)])
        self.assertEqual([c.function_calls_in_response for c in st.calls], [1, 1, 0])
        # Токены вызовов инструментов и их результатов этого хода — отдельным подсчётом.
        exchange = client.models.count_requests[-1]
        self.assertEqual([c.role for c in exchange], ["model", "user", "model", "user"])
        self.assertEqual(st.tool_exchange_tokens, fake_token_count(exchange))

    def test_overhead_is_prompt_minus_history(self):
        client = FakeClient([with_usage(text_response("x"), prompt=900)])
        r = Agent(client).run("один два три")
        self.assertEqual(r.stats.calls[0].overhead_tokens, 900 - r.stats.calls[0].history_tokens)

    def test_session_accumulates_and_survives_reset(self):
        agent = Agent(FakeClient([text_response("1"), text_response("2"), api_error(503)]))
        agent.run("a")
        agent.run("b")
        agent.reset()
        agent.run("c")             # неудачный ход тоже учитывается: обращение было
        s = agent.session_stats
        self.assertEqual((s.turns, s.model_calls), (3, 3))
        self.assertEqual((s.input_tokens, s.output_tokens), (200, 50))   # у ошибки usage нет
        self.assertIsNotNone(s.cost_usd)

    def test_history_grows_between_turns(self):
        agent = Agent(FakeClient([text_response("ответ номер один"), text_response("ответ номер два")]))
        first = agent.run("первый вопрос").stats.history_tokens
        second = agent.run("второй вопрос").stats.history_tokens
        self.assertGreater(second, first)

    def test_count_error_does_not_break_turn(self):
        client = FakeClient([text_response("ok")], count_error=api_error(429, "quota", "RESOURCE_EXHAUSTED"))
        agent = Agent(client, input_token_limit=10)
        r = agent.run("x")
        self.assertEqual(r.status, "ok")                 # без подсчёта локальная защита не срабатывает
        self.assertIsNone(r.stats.request_tokens)
        self.assertIsNone(r.stats.history_tokens)
        self.assertEqual(len(r.stats.count_errors), 2)
        self.assertIn("н/д", token_stats.format_turn(r.stats))
        self.assertEqual(r.input_tokens, 100)            # usage_metadata есть всегда

    def test_count_disabled(self):
        client = FakeClient([text_response("ok")])
        r = Agent(client, count_tokens=False).run("x")
        self.assertEqual(client.models.count_requests, [])
        self.assertIsNone(r.stats.request_tokens)

    def test_missing_usage_metadata(self):
        r = Agent(FakeClient([no_usage(text_response("ok"))])).run("x")
        self.assertEqual(r.status, "ok")
        self.assertEqual((r.input_tokens, r.output_tokens), (0, 0))
        self.assertIsNone(r.stats.calls[0].overhead_tokens)

    def test_api_key_not_in_count_error(self):
        fake_key = "FAKE-KEY-FOR-COUNT-TEST-1234567890ab"
        with mock.patch.dict("os.environ", {config.API_KEY_ENV: fake_key}):
            client = FakeClient([text_response("ok")], count_error=api_error(400, f"bad key {fake_key}", "INVALID_ARGUMENT"))
            r = Agent(client).run("x")
        self.assertNotIn(fake_key, " ".join(r.stats.count_errors))


class ContextOverflowTest(unittest.TestCase):
    def setUp(self):
        logging.getLogger("agent").setLevel(logging.CRITICAL)

    def test_local_guard_blocks_without_generate_and_rolls_back(self):
        client = FakeClient([text_response("короткий"), text_response("не должен понадобиться")])
        agent = Agent(client, input_token_limit=50)
        agent.run("привет")
        before = list(agent.history)
        r = agent.run("слово " * 200)
        self.assertEqual(r.status, "context_overflow")
        self.assertTrue(r.stats.blocked_locally)
        self.assertEqual(r.stats.model_calls, 0)
        self.assertEqual(len(client.requests), 1)        # второй generate_content не отправлялся
        self.assertEqual(agent.history, before)
        self.assertIn("лимит контекста", r.text)
        self.assertIn("НЕ отправлен", token_stats.format_turn(r.stats))

    def test_guard_disabled_sends_and_api_error_is_classified(self):
        err = api_error(400, "The input token count (60) exceeds the maximum number of tokens allowed (50).",
                        "INVALID_ARGUMENT")
        client = FakeClient([err, text_response("снова работаю")])
        agent = Agent(client, input_token_limit=50, enforce_context_limit=False)
        r = agent.run("слово " * 200)
        self.assertEqual(r.status, "context_overflow")
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(agent.history, [])
        self.assertIn("exceeds the maximum", r.stats.calls[0].error)
        self.assertEqual(agent.run("ок?").status, "ok")      # агент продолжает работать

    def test_unknown_limit_never_blocks(self):
        client = FakeClient([text_response("ok")])
        r = Agent(client, input_token_limit=None).run("слово " * 500)
        self.assertEqual(r.status, "ok")

    def test_error_classifier(self):
        self.assertTrue(is_context_overflow_error(api_error(400, "The input token count (2) exceeds the maximum "
                                                                 "number of tokens allowed (1).", "INVALID_ARGUMENT")))
        self.assertFalse(is_context_overflow_error(api_error(400, "API key not valid", "INVALID_ARGUMENT")))
        self.assertFalse(is_context_overflow_error(api_error(429, "token limit per minute", "RESOURCE_EXHAUSTED")))
        self.assertFalse(is_context_overflow_error(ValueError("token limit")))


class FormatTest(unittest.TestCase):
    def test_turn_report_contains_required_numbers(self):
        client = FakeClient([call_response(("calculate", {"expression": "2+2"})), text_response("4")])
        agent = Agent(client, input_token_limit=1_000_000)
        r = agent.run("2+2?")
        text = token_stats.format_turn(r.stats, agent.session_stats)
        for needle in ("Запрос пользователя", "История перед отправкой", "Ответ модели", "Итого за ход (2 обращ.)",
                       "вход 200", "выход 50", "function_call: 1", "Вызовы инструментов и их результаты",
                       "Стоимость хода: ≈ $", "/ 1 000 000", "Сессия: ходов 1"):
            self.assertIn(needle, text)

    def test_unknown_limit_and_price(self):
        self.assertIn("лимит неизвестен", token_stats.format_context(10, None))
        turn = token_stats.TurnStats()
        with mock.patch.object(config, "PRICING", {}):
            self.assertIn("тариф неизвестен", token_stats.format_turn(turn))

    def test_to_dict_is_json_serializable(self):
        r = Agent(FakeClient([call_response(("calculate", {"expression": "1"})), text_response("1")])).run("1?")
        data = json.loads(json.dumps(r.stats.to_dict()))
        self.assertEqual(data["input_tokens"], 200)
        self.assertEqual(len(data["calls"]), 2)
        self.assertIn("overhead_tokens", data["calls"][0])


class HistoryCompatibilityTest(unittest.TestCase):
    """История из Дня 7 (format_version 1) загружается и считается без изменений формата."""

    DAY7_FILE = {
        "format_version": 1, "sdk_version": "2.34.0", "saved_at": "2026-11-10T12:00:00",
        "history": [
            {"role": "user", "parts": [{"text": "Меня зовут Аня. Сколько будет 150 долларов в донгах?"}]},
            {"role": "model", "parts": [{"function_call": {"id": "call_1", "name": "convert_currency",
                                                           "args": CONVERT_ARGS},
                                         "thought_signature": "c2lnbmF0dXJl"}]},
            {"role": "user", "parts": [{"function_response": {"id": "call_1", "name": "convert_currency",
                                                              "response": {"result": {"converted_amount": 3750000}}}}]},
            {"role": "model", "parts": [{"text": "Аня, это 3 750 000 VND."}]},
        ],
    }

    def setUp(self):
        logging.getLogger("agent").setLevel(logging.CRITICAL)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "chat_history.json"
        self.path.write_text(json.dumps(self.DAY7_FILE, ensure_ascii=False), encoding="utf-8")

    def chat(self, inputs, script, **client_kwargs):
        client = FakeClient(script, **client_kwargs)
        out = io.StringIO()
        with mock.patch("builtins.input", side_effect=inputs), redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = main.cmd_chat(self.path, client=client)
        return code, out.getvalue(), client

    def test_restored_history_counted_with_tool_parts(self):
        history = HistoryStore(self.path).load()
        client = FakeClient([])
        agent = Agent(client)
        agent.history = history
        tokens = agent.count_history_tokens()
        self.assertEqual(tokens, fake_token_count(history))
        sent = client.models.count_requests[0]
        self.assertEqual(sent[1].parts[0].function_call.id, "call_1")
        self.assertEqual(sent[1].parts[0].thought_signature, b"signature")
        self.assertEqual(sent[2].parts[0].function_response.id, "call_1")

    def test_chat_shows_limit_restored_tokens_and_turn_stats(self):
        code, out, client = self.chat(["Как меня зовут?", "stats", "exit"], [text_response("Аня")],
                                      input_token_limit=1_048_576)
        self.assertEqual(code, 0)
        self.assertIn("Лимит контекста gemini-3.5-flash-lite: 1 048 576", out)
        self.assertIn("Восстановлена история", out)
        self.assertIn("Восстановленная история:", out)
        self.assertIn("─── Токены", out)
        self.assertIn("Сессия: ходов 1", out)
        self.assertIn("История сейчас: 6 сообщ.", out)
        # первым запросом в модель ушла вся восстановленная история
        self.assertEqual(len(client.requests[0]["contents"]), 5)
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(saved["format_version"], 1)
        self.assertEqual(len(saved["history"]), 6)
        self.assertNotIn("token", json.dumps(list(saved)))   # формат файла Дня 7 не изменился

    def test_chat_unknown_limit_is_reported(self):
        _, out, _ = self.chat(["exit"], [])
        self.assertIn("неизвестен", out)

    def test_overflow_in_chat_is_not_saved(self):
        before = self.path.read_text(encoding="utf-8")
        _, out, client = self.chat(["слово " * 300, "exit"], [], input_token_limit=100)
        self.assertIn("Запрос не отправлен", out)
        self.assertEqual(client.requests, [])
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)


class ExperimentsTest(unittest.TestCase):
    def setUp(self):
        logging.getLogger("agent").setLevel(logging.CRITICAL)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(experiments, "RESULTS_DIR", Path(self.tmp.name) / "results")
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher2 = mock.patch.object(experiments, "ROOT", Path(self.tmp.name))
        patcher2.start()
        self.addCleanup(patcher2.stop)

    def report(self, name):
        (path,) = (Path(self.tmp.name) / "results").glob(f"tokens_{name}_*.json")
        return json.loads(path.read_text(encoding="utf-8"))

    def run_quiet(self, fn, *args, **kwargs):
        with redirect_stdout(io.StringIO()) as out:
            code = fn(*args, **kwargs)
        return code, out.getvalue()

    def test_dialog_declined_makes_no_calls(self):
        with mock.patch.object(main, "make_client", side_effect=AssertionError("API не должен вызываться")):
            code, _ = self.run_quiet(experiments.run_dialog, "short", experiments.SHORT_DIALOG, False,
                                     confirm=lambda q, y: False)
        self.assertEqual(code, 1)

    def test_short_dialog_report(self):
        client = FakeClient([text_response(f"ответ {i}") for i in range(3)], input_token_limit=1_048_576)
        code, out = self.run_quiet(experiments.run_dialog, "short", experiments.SHORT_DIALOG, True, client=client)
        self.assertEqual(code, 0)
        self.assertIn("Рост истории по ходам", out)
        rep = self.report("short")
        self.assertEqual(rep["input_token_limit"], 1_048_576)
        hist = [t["token_stats"]["history_tokens"] for t in rep["turns"]]
        self.assertEqual(hist, sorted(hist))
        self.assertEqual(rep["session"]["turns"], 3)
        self.assertEqual(rep["final_history_messages"], 6)

    def test_long_dialog_is_long(self):
        self.assertGreaterEqual(len(experiments.LONG_DIALOG), 10)

    def test_overflow_stops_when_limit_unknown(self):
        client = FakeClient([])
        code, out = self.run_quiet(experiments.run_overflow, True, client=client)
        self.assertEqual(code, 1)
        self.assertIn("неизвестен", out)
        self.assertEqual(client.requests, [])
        self.assertEqual(self.report("overflow")["result"], "limit_unknown")

    def test_build_overflow_text_exceeds_limit_by_count_tokens(self):
        agent = Agent(FakeClient([]))
        text, tokens, steps = experiments.build_overflow_text(agent, 5_000)
        self.assertGreater(tokens, 5_000)
        self.assertEqual(tokens, fake_token_count([user(text)]))
        self.assertEqual(steps[0]["step"], "sample")

    def test_overflow_full_flow(self):
        limit = 2_000
        err = api_error(400, f"The input token count (2100) exceeds the maximum number of tokens allowed ({limit}).",
                        "INVALID_ARGUMENT")
        client = FakeClient([err, text_response("Да")], input_token_limit=limit)
        code, out = self.run_quiet(experiments.run_overflow, True, client=client)
        self.assertEqual(code, 0)
        rep = self.report("overflow")
        self.assertEqual(rep["guarded"]["status"], "context_overflow")
        self.assertEqual(rep["guarded"]["generate_content_calls"], 0)
        self.assertEqual(rep["api"]["status"], "context_overflow")
        self.assertIn("exceeds the maximum", rep["api"]["error"])
        self.assertEqual(rep["api"]["history_after"], 0)
        self.assertEqual(rep["after_error"]["status"], "ok")
        self.assertGreater(rep["overflow_message"]["count_tokens"], limit)
        self.assertEqual(len(client.requests), 2)      # только реальный отказ + проверочный запрос
        self.assertNotIn("Строка 1:", json.dumps(rep, ensure_ascii=False))   # огромный текст не в протоколе

    def test_overflow_declined_after_guard(self):
        client = FakeClient([], input_token_limit=2_000)
        code, _ = self.run_quiet(experiments.run_overflow, False, client=client, confirm=lambda q, y: False)
        self.assertEqual(code, 0)
        self.assertEqual(client.requests, [])
        self.assertEqual(self.report("overflow")["result"], "guard_only")

    def test_count_only_client_blocks_generate(self):
        wrapped = experiments._CountOnlyClient(FakeClient([text_response("x")]))
        with self.assertRaises(RuntimeError):
            wrapped.models.generate_content(model="m", contents=[], config=None)


if __name__ == "__main__":
    unittest.main()
