"""Офлайн-тесты инструментов и диспетчера (без API)."""

import json
import unittest
from datetime import datetime, timezone

import config
import tools
from tools import ToolError, calculate, convert_currency, dispatch, get_current_time


class GetCurrentTimeTest(unittest.TestCase):
    def test_converts_to_ho_chi_minh(self):
        fixed = datetime(2026, 11, 10, 20, 30, 0, tzinfo=timezone.utc)
        r = get_current_time(now=fixed)
        self.assertEqual(r["timezone"], "Asia/Ho_Chi_Minh")
        self.assertEqual(r["date"], "2026-11-11")       # +07:00 → уже следующий день
        self.assertEqual(r["time"], "03:30:00")
        self.assertEqual(r["utc_offset"], "+0700")
        self.assertEqual(r["weekday"], "среда")
        self.assertEqual(r["iso"], "2026-11-11T03:30:00+07:00")

    def test_real_clock(self):
        r = get_current_time()
        self.assertTrue(r["iso"].endswith("+07:00"))


class CalculateTest(unittest.TestCase):
    def test_basic_operations(self):
        cases = {
            "2 + 3": 5, "10 - 4": 6, "6 * 7": 42, "7 / 2": 3.5, "7 // 2": 3,
            "7 % 3": 1, "2 ** 10": 1024, "-5 + 2": -3, "+4": 4, "(2 + 3) * 4": 20,
            "2 + 3 * 4": 14, "1.5 * 2": 3, "0.1 + 0.2": 0.3,
        }
        for expr, expected in cases.items():
            with self.subTest(expr=expr):
                self.assertEqual(calculate(expr)["result"], expected)

    def test_scenario_three_math_without_float_noise(self):
        # 3750000 * 1.1 в Python = 4125000.000000001 — инструмент нормализует
        self.assertEqual(calculate("3750000 * 1.1")["result"], 4125000)
        self.assertEqual(calculate("3750000 + 3750000 * 10 / 100")["result"], 4125000)
        self.assertIsInstance(calculate("3750000 * 1.1")["result"], int)

    def test_unicode_operators_and_separators(self):
        self.assertEqual(calculate("6 × 7")["result"], 42)
        self.assertEqual(calculate("8 ÷ 2")["result"], 4)
        self.assertEqual(calculate("10 − 3")["result"], 7)
        self.assertEqual(calculate("3_750_000 * 2")["result"], 7_500_000)

    def test_rejects_unsafe_and_unsupported(self):
        bad = [
            "__import__('os').system('echo hacked')", "open('x')", "abs(-1)", "x + 1",
            "[1, 2]", "lambda: 1", "'a' * 3", "True + 1", "1 if 1 else 2", "1 < 2",
            "(1).__class__", "2 ** 1000", "9 ** 9 ** 9", "10 ** 31", "1 / 0", "5 % 0",
            "0 ** -1", "(-8) ** 0.5", "", "   ", "2 +", "10%", "1" * 300,
        ]
        for expr in bad:
            with self.subTest(expr=expr):
                with self.assertRaises(ToolError):
                    calculate(expr)

    def test_non_string_rejected(self):
        with self.assertRaises(ToolError):
            calculate(123)

    def test_no_eval_in_source(self):
        import inspect, re
        src = inspect.getsource(tools)
        self.assertIsNone(re.search(r"(?<![\w.])eval\(", src))
        self.assertIsNone(re.search(r"(?<![\w.])exec\(", src))


class ConvertCurrencyTest(unittest.TestCase):
    def test_usd_to_vnd(self):
        r = convert_currency(150, "USD", "VND")
        self.assertEqual(r["converted_amount"], 3_750_000)
        self.assertTrue(r["rate_is_fixed_demo"])
        self.assertIn("НЕ актуальный рыночный курс", r["note"])
        self.assertIn("25 000", r["note"])

    def test_vnd_to_usd(self):
        self.assertEqual(convert_currency(3_750_000, "VND", "USD")["converted_amount"], 150)
        self.assertEqual(convert_currency(10_000, "vnd", "usd")["converted_amount"], 0.4)

    def test_fractional_and_case(self):
        self.assertEqual(convert_currency(1.5, " usd ", "Vnd")["converted_amount"], 37_500)
        self.assertEqual(convert_currency(12_345, "VND", "USD")["converted_amount"], 0.49)

    def test_rate_comes_from_config(self):
        self.assertEqual(convert_currency(1, "USD", "VND")["converted_amount"], config.USD_TO_VND_RATE)

    def test_errors(self):
        for args in [
            (-1, "USD", "VND"), ("150", "USD", "VND"), (True, "USD", "VND"),
            (float("nan"), "USD", "VND"), (float("inf"), "USD", "VND"),
            (1, "EUR", "VND"), (1, "USD", "RUB"), (1, "USD", "USD"),
        ]:
            with self.subTest(args=args):
                with self.assertRaises(ToolError):
                    convert_currency(*args)


class DispatchTest(unittest.TestCase):
    def test_success_wrapped_in_result(self):
        r = dispatch("convert_currency", {"amount": 150, "from_currency": "USD", "to_currency": "VND"})
        self.assertEqual(r["result"]["converted_amount"], 3_750_000)
        self.assertEqual(dispatch("calculate", {"expression": "2+2"})["result"]["result"], 4)
        self.assertIn("iso", dispatch("get_current_time", {})["result"])
        self.assertIn("iso", dispatch("get_current_time", None)["result"])

    def test_unknown_tool(self):
        r = dispatch("delete_files", {})
        self.assertIn("Неизвестный инструмент", r["error"])

    def test_missing_argument(self):
        r = dispatch("convert_currency", {"amount": 150, "from_currency": "USD"})
        self.assertIn("Неверные аргументы", r["error"])

    def test_unexpected_argument(self):
        self.assertIn("не принимает", dispatch("calculate", {"expression": "1", "x": 2})["error"])
        # служебный параметр now не доступен модели
        self.assertIn("не принимает", dispatch("get_current_time", {"now": "2020-01-01"})["error"])

    def test_tool_error_becomes_error_dict(self):
        self.assertEqual(dispatch("calculate", {"expression": "1/0"}), {"error": "Деление на ноль."})

    def test_unexpected_exception_is_caught(self):
        def boom(expression):
            raise RuntimeError("сломалось")
        original = tools.TOOL_FUNCTIONS["calculate"]
        tools.TOOL_FUNCTIONS["calculate"] = boom
        try:
            r = dispatch("calculate", {"expression": "1"})
        finally:
            tools.TOOL_FUNCTIONS["calculate"] = original
        self.assertIn("Внутренняя ошибка", r["error"])
        self.assertIn("RuntimeError", r["error"])

    def test_results_are_json_serializable(self):
        for name, args in [
            ("get_current_time", {}), ("calculate", {"expression": "1/3"}),
            ("convert_currency", {"amount": 1, "from_currency": "USD", "to_currency": "VND"}),
        ]:
            json.dumps(dispatch(name, args), ensure_ascii=False)


class DeclarationsTest(unittest.TestCase):
    def test_every_declaration_has_function_and_vice_versa(self):
        names = [d.name for d in tools.FUNCTION_DECLARATIONS]
        self.assertEqual(sorted(names), sorted(tools.TOOL_FUNCTIONS))
        self.assertEqual(len(names), 3)

    def test_schema_required_params_match_signatures(self):
        import inspect
        for d in tools.FUNCTION_DECLARATIONS:
            schema = d.parameters_json_schema or {"properties": {}, "required": []}
            sig = inspect.signature(tools.TOOL_FUNCTIONS[d.name])
            required = {p.name for p in sig.parameters.values() if p.default is inspect.Parameter.empty}
            with self.subTest(tool=d.name):
                self.assertEqual(set(schema.get("required", [])), required)
                self.assertTrue(set(schema["properties"]) <= set(sig.parameters))
                self.assertTrue(d.description)
                for prop in schema["properties"].values():
                    self.assertTrue(prop.get("description"))

    def test_declarations_are_not_python_callables(self):
        # В запрос уходят только описания: SDK не сможет выполнить функцию сам.
        for d in tools.FUNCTION_DECLARATIONS:
            self.assertFalse(callable(d))


if __name__ == "__main__":
    unittest.main()
