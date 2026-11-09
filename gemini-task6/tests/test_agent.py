"""Офлайн-тесты цикла агента на фейковом клиенте (без API)."""

import logging
import os
import unittest
from unittest import mock

from google.genai import types

import config
from agent import Agent
from tests.fakes import FakeClient, api_error, call_response, empty_response, text_response

CONVERT_ARGS = {"amount": 150, "from_currency": "USD", "to_currency": "VND"}


def function_responses(content: types.Content):
    return [p.function_response for p in content.parts if p.function_response]


class ConfigTest(unittest.TestCase):
    def test_manual_function_calling_config(self):
        cfg = Agent(client=None).build_config()
        self.assertTrue(cfg.automatic_function_calling.disable)
        self.assertEqual(cfg.tool_config.function_calling_config.mode, types.FunctionCallingConfigMode.AUTO)
        decls = cfg.tools[0].function_declarations
        self.assertEqual({d.name for d in decls}, {"get_current_time", "calculate", "convert_currency"})
        self.assertTrue(all(isinstance(d, types.FunctionDeclaration) for d in decls))
        self.assertEqual(cfg.system_instruction, config.SYSTEM_INSTRUCTION)

    def test_invalid_max_iterations(self):
        with self.assertRaises(ValueError):
            Agent(client=None, max_iterations=0)


class AgentLoopTest(unittest.TestCase):
    def setUp(self):
        logging.getLogger("agent").setLevel(logging.INFO)

    # --- сценарий 1: без инструментов ---
    def test_answer_without_tools(self):
        client = FakeClient([text_response("API — это способ, которым программы общаются.")])
        agent = Agent(client)
        r = agent.run("Объясни простыми словами, что такое API")
        self.assertEqual(r.status, "ok")
        self.assertEqual(r.iterations, 1)
        self.assertEqual(r.tool_calls, [])
        self.assertIn("API", r.text)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual([c.role for c in agent.history], ["user", "model"])
        self.assertEqual((r.input_tokens, r.output_tokens), (100, 25))

    # --- сценарий 2: один инструмент ---
    def test_single_tool_call(self):
        client = FakeClient([
            call_response(("convert_currency", CONVERT_ARGS, "id-1"), signature=b"SIG-1"),
            text_response("150 USD = 3 750 000 VND (фиксированный демонстрационный курс)."),
        ])
        agent = Agent(client)
        with self.assertLogs("agent", level="INFO") as logs:
            r = agent.run("Сколько будет 150 долларов в донгах?")

        self.assertEqual(r.status, "ok")
        self.assertEqual(r.iterations, 2)
        self.assertEqual([c.name for c in r.tool_calls], ["convert_currency"])
        self.assertEqual(r.tool_calls[0].response["result"]["converted_amount"], 3_750_000)

        # Второй запрос содержит: user → model(function_call + подпись) → user(function_response)
        second = client.requests[1]["contents"]
        self.assertEqual([c.role for c in second], ["user", "model", "user"])
        model_part = second[1].parts[0]
        self.assertEqual(model_part.function_call.name, "convert_currency")
        self.assertEqual(model_part.thought_signature, b"SIG-1")       # подпись не потеряна
        (fr,) = function_responses(second[2])
        self.assertEqual((fr.id, fr.name), ("id-1", "convert_currency"))  # id совпадает с вызовом
        self.assertEqual(fr.response["result"]["converted_amount"], 3_750_000)

        log_text = "\n".join(logs.output)
        self.assertIn("Вызов инструмента: convert_currency", log_text)
        self.assertIn('"amount": 150', log_text)
        self.assertIn("Результат convert_currency", log_text)
        self.assertIn("3750000", log_text)

    # --- сценарий 3: два инструмента последовательно ---
    def test_two_sequential_tools_use_previous_result(self):
        def second_step():
            # «Модель» строит выражение из результата, который реально пришёл из Python.
            fr = function_responses(client.requests[-1]["contents"][-1])[0]
            converted = fr.response["result"]["converted_amount"]
            return call_response(("calculate", {"expression": f"{converted} * 1.1"}, "id-2"), signature=b"SIG-2")

        client = FakeClient([
            call_response(("convert_currency", CONVERT_ARGS, "id-1"), signature=b"SIG-1"),
            second_step,
            text_response("Итого 4 125 000 VND. Курс фиксированный демонстрационный."),
        ])
        agent = Agent(client)
        r = agent.run("Переведи 150 долларов в донги, а затем добавь 10 процентов")

        self.assertEqual(r.status, "ok")
        self.assertEqual(r.iterations, 3)
        self.assertEqual([c.name for c in r.tool_calls], ["convert_currency", "calculate"])
        self.assertEqual([c.iteration for c in r.tool_calls], [1, 2])
        self.assertEqual(r.tool_calls[1].args, {"expression": "3750000 * 1.1"})
        self.assertEqual(r.tool_calls[1].response["result"]["result"], 4_125_000)

        third = client.requests[2]["contents"]
        self.assertEqual([c.role for c in third], ["user", "model", "user", "model", "user"])
        # обе подписи сохранены в истории в исходных частях
        self.assertEqual(third[1].parts[0].thought_signature, b"SIG-1")
        self.assertEqual(third[3].parts[0].thought_signature, b"SIG-2")
        self.assertEqual(function_responses(third[4])[0].response["result"]["result"], 4_125_000)

    def test_parallel_calls_answered_in_one_message_in_order(self):
        client = FakeClient([
            call_response(("get_current_time", {}, "a"), ("calculate", {"expression": "2*21"}, "b")),
            text_response("Готово"),
        ])
        r = Agent(client).run("Который час и сколько 2*21?")
        self.assertEqual(r.status, "ok")
        last = client.requests[1]["contents"][-1]
        frs = function_responses(last)
        self.assertEqual([(f.id, f.name) for f in frs], [("a", "get_current_time"), ("b", "calculate")])
        self.assertEqual(frs[1].response["result"]["result"], 42)
        # подпись только на первом вызове — оба вызова в истории в исходном виде
        model_parts = client.requests[1]["contents"][1].parts
        self.assertEqual([p.thought_signature for p in model_parts], [b"sig", None])

    def test_text_alongside_function_call_is_kept_in_history(self):
        client = FakeClient([
            call_response(("calculate", {"expression": "1+1"}), text="Сейчас посчитаю."),
            text_response("2"),
        ])
        Agent(client).run("1+1?")
        parts = client.requests[1]["contents"][1].parts
        self.assertEqual(parts[0].text, "Сейчас посчитаю.")
        self.assertIsNotNone(parts[1].function_call)

    def test_thoughts_are_excluded_from_answer(self):
        client = FakeClient([text_response("Ответ", thought="секретные рассуждения", signature=b"s")])
        agent = Agent(client)
        r = agent.run("вопрос")
        self.assertEqual(r.text, "Ответ")
        self.assertEqual(agent.history[-1].parts[1].thought_signature, b"s")

    # --- ошибки инструментов ---
    def test_tool_error_is_sent_back_and_model_can_recover(self):
        client = FakeClient([
            call_response(("calculate", {"expression": "1/0"})),
            call_response(("unknown_tool", {})),
            call_response(("convert_currency", {"amount": 1})),
            text_response("На ноль делить нельзя."),
        ])
        with self.assertLogs("agent", level="WARNING") as logs:
            r = Agent(client).run("1/0?")
        self.assertEqual(r.status, "ok")
        self.assertEqual([c.ok for c in r.tool_calls], [False, False, False])
        errs = [function_responses(client.requests[i]["contents"][-1])[0].response["error"] for i in (1, 2, 3)]
        self.assertIn("Деление на ноль", errs[0])
        self.assertIn("Неизвестный инструмент", errs[1])
        self.assertIn("Неверные аргументы", errs[2])
        self.assertIn("Ошибка инструмента calculate", "\n".join(logs.output))

    # --- ограничение итераций ---
    def test_max_iterations_stops_infinite_loop(self):
        client = FakeClient([call_response(("calculate", {"expression": "1+1"}))] * 10)
        agent = Agent(client, max_iterations=4)
        with self.assertLogs("agent", level="WARNING") as logs:
            r = agent.run("зацикленный запрос")
        self.assertEqual(r.status, "max_iterations")
        self.assertEqual(r.iterations, 4)
        self.assertEqual(len(client.requests), 4)       # ровно лимит, не больше
        self.assertEqual(len(r.tool_calls), 4)
        self.assertIn("4", r.text)
        self.assertEqual(agent.history, [])             # история откатилась
        self.assertIn("лимит итераций", "\n".join(logs.output))

    def test_default_limit_is_ten(self):
        client = FakeClient([call_response(("get_current_time", {}))] * 20)
        r = Agent(client).run("x")
        self.assertEqual(config.MAX_ITERATIONS, 10)
        self.assertEqual((r.status, len(client.requests)), ("max_iterations", 10))

    # --- ошибки API ---
    def test_api_error_rolls_back_and_next_request_works(self):
        client = FakeClient([
            text_response("Первый ответ"),
            call_response(("calculate", {"expression": "2+2"})),
            api_error(503),
            text_response("После ошибки всё работает"),
        ])
        agent = Agent(client)
        agent.run("первый")
        with self.assertLogs("agent", level="ERROR"):
            r = agent.run("второй")
        self.assertEqual(r.status, "api_error")
        self.assertIn("503", r.text)
        self.assertEqual(len(agent.history), 2)          # остался только первый обмен
        r3 = agent.run("третий")
        self.assertEqual(r3.status, "ok")
        self.assertEqual([c.role for c in client.requests[-1]["contents"]], ["user", "model", "user"])

    def test_network_error(self):
        client = FakeClient([TimeoutError("timed out")])
        r = Agent(client).run("x")
        self.assertEqual(r.status, "api_error")
        self.assertIn("TimeoutError", r.text)

    def test_empty_response(self):
        agent = Agent(FakeClient([empty_response("SAFETY")]))
        r = agent.run("x")
        self.assertEqual(r.status, "empty_response")
        self.assertIn("SAFETY", r.text)
        self.assertEqual(agent.history, [])

    def test_api_key_redacted_from_errors_and_logs(self):
        fake_key = "AIzaFAKE-KEY-FOR-TEST-1234567890abcdef"
        with mock.patch.dict(os.environ, {config.API_KEY_ENV: fake_key}):
            client = FakeClient([api_error(400, f"API key not valid: {fake_key}", "INVALID_ARGUMENT")])
            with self.assertLogs("agent", level="INFO") as logs:
                r = Agent(client).run("x")
        self.assertNotIn(fake_key, r.text)
        self.assertNotIn(fake_key, "\n".join(logs.output))
        self.assertIn("***", r.text)

    # --- история между запросами ---
    def test_history_persists_between_runs_and_reset(self):
        client = FakeClient([text_response("Меня зовут Агент"), text_response("Вы спрашивали про имя")])
        agent = Agent(client)
        agent.run("Как тебя зовут?")
        agent.run("О чём я спрашивал?")
        sent = client.requests[1]["contents"]
        self.assertEqual(len(sent), 3)
        self.assertEqual(sent[0].parts[0].text, "Как тебя зовут?")
        agent.reset()
        self.assertEqual(agent.history, [])


if __name__ == "__main__":
    unittest.main()
