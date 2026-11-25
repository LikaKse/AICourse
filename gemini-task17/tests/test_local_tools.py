"""Офлайн-тесты локальных инструментов Дня 6 (перенесены без изменения логики)."""

import unittest
from datetime import datetime, timezone
from pathlib import Path

import config
import local_tools
from local_tools import ToolError, calculate, convert_currency, dispatch, get_current_time


class GetCurrentTimeTest(unittest.TestCase):
    def test_converts_to_ho_chi_minh(self):
        moment = datetime(2026, 11, 25, 0, 30, tzinfo=timezone.utc)
        result = get_current_time(moment)
        self.assertEqual(result["timezone"], "Asia/Ho_Chi_Minh")
        self.assertEqual(result["date"], "2026-11-25")
        self.assertEqual(result["time"], "07:30:00")
        self.assertEqual(result["utc_offset"], "+0700")

    def test_real_clock(self):
        result = get_current_time()
        self.assertEqual(result["timezone"], config.TIMEZONE)
        self.assertRegex(result["date"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertIn(result["weekday"], local_tools.WEEKDAYS_RU)


class CalculateTest(unittest.TestCase):
    def test_basic_operations(self):
        cases = {
            "2 + 3": 5,
            "10 - 4": 6,
            "6 * 7": 42,
            "8 / 2": 4,
            "9 // 2": 4,
            "9 % 2": 1,
            "2 ** 10": 1024,
            "(2 + 3) * 4": 20,
            "-5 + 2": -3,
            "+5": 5,
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(calculate(expression)["result"], expected)

    def test_scenario_three_math_without_float_noise(self):
        result = calculate("3750000 * 1.1")
        self.assertEqual(result["result"], 4_125_000)
        self.assertIsInstance(result["result"], int)

    def test_mixed_scenario_math(self):
        self.assertEqual(calculate("12 * 25000 * (1 + 10 / 100)")["result"], 330_000)

    def test_unicode_operators_and_separators(self):
        self.assertEqual(calculate("1 000 × 2")["result"], 2_000)
        self.assertEqual(calculate("1_000 ÷ 4")["result"], 250)
        self.assertEqual(calculate("10 − 3")["result"], 7)

    def test_rejects_unsafe_and_unsupported(self):
        cases = [
            "__import__('os')",
            "open('x')",
            "x + 1",
            "2 ** 101",
            "9 ** 9 ** 9",
            "1 / 0",
            "1 // 0",
            "1 % 0",
            "True + 1",
            "1 +",
            "",
            "   ",
            "1 + 2j",
        ]
        for expression in cases:
            with self.subTest(expression=expression):
                with self.assertRaises(ToolError):
                    calculate(expression)

    def test_non_string_rejected(self):
        with self.assertRaises(ToolError):
            calculate(123)

    def test_no_eval_in_source(self):
        source = Path(local_tools.__file__).read_text(encoding="utf-8")
        self.assertNotIn("eval(", source)
        self.assertNotIn("exec(", source)

    def test_too_long_expression(self):
        with self.assertRaises(ToolError):
            calculate("1+" * 200 + "1")


class ConvertCurrencyTest(unittest.TestCase):
    def test_usd_to_vnd(self):
        result = convert_currency(150, "usd", "vnd")
        self.assertEqual(result["converted_amount"], 3_750_000)
        self.assertEqual(result["rate"], "1 USD = 25000 VND")
        self.assertTrue(result["rate_is_fixed_demo"])
        self.assertIn("НЕ актуальный", result["note"])

    def test_vnd_to_usd(self):
        result = convert_currency(3_750_000, "VND", "USD")
        self.assertEqual(result["converted_amount"], 150)

    def test_errors(self):
        for args in [(-1, "USD", "VND"), (1, "EUR", "VND"), (1, "USD", "USD")]:
            with self.subTest(args=args):
                with self.assertRaises(ToolError):
                    convert_currency(*args)


class DispatchTest(unittest.TestCase):
    def test_success_wrapped_in_result(self):
        result = dispatch("calculate", {"expression": "2+2"})
        self.assertEqual(result, {"result": {"expression": "2+2", "result": 4}})

    def test_unknown_tool(self):
        self.assertIn("Неизвестный инструмент", dispatch("nope", {})["error"])

    def test_missing_argument(self):
        self.assertIn("Неверные аргументы", dispatch("calculate", {})["error"])

    def test_unexpected_argument(self):
        result = dispatch("get_current_time", {"x": 1})
        self.assertIn("не принимает аргументы", result["error"])

    def test_tool_error_becomes_error_dict(self):
        self.assertIn("Деление на ноль", dispatch("calculate", {"expression": "1/0"})["error"])

    def test_every_declaration_has_function_and_vice_versa(self):
        declaration_names = {declaration.name for declaration in local_tools.FUNCTION_DECLARATIONS}
        self.assertEqual(declaration_names, set(local_tools.TOOL_FUNCTIONS))

    def test_schema_required_params_match_signatures(self):
        import inspect

        for declaration in local_tools.FUNCTION_DECLARATIONS:
            function = local_tools.TOOL_FUNCTIONS[declaration.name]
            schema = declaration.parameters_json_schema or {}
            properties = set(schema.get("properties", {}))
            signature = inspect.signature(function)
            visible = {
                name for name, parameter in signature.parameters.items()
                if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
                and name != "now"
            }
            self.assertEqual(properties, visible, declaration.name)
            required = set(schema.get("required", []))
            for name in visible:
                has_default = signature.parameters[name].default is not inspect.Parameter.empty
                self.assertEqual(name in required, not has_default, f"{declaration.name}.{name}")


if __name__ == "__main__":
    unittest.main()
