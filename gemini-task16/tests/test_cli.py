"""Тесты CLI-проверок и бесплатных режимов без Gemini."""

import io
import logging
import os
import unittest
from contextlib import redirect_stderr
from unittest import mock

import config
import main
from agent import AgentResult, ToolCallRecord


def call(iteration, name, source, response, args=None):
    return ToolCallRecord(iteration, name, args or {}, response, source)


class MixedCheckTest(unittest.TestCase):
    def test_pass(self):
        result = AgentResult(
            "Итоговая стоимость: 330 000.",
            "ok",
            3,
            [
                call(1, "read_text_file", "mcp", {"content": "quantity=12"}),
                call(2, "calculate", "local", {"result": {"result": 330_000}},
                     {"expression": "12 * 25000 * 1.1"}),
            ],
        )
        self.assertEqual(main.check_mixed(result), [])

    def test_fail_wrong_first_tool(self):
        result = AgentResult("330000", "ok", 2, [
            call(1, "calculate", "local", {"result": {"result": 330_000}}),
        ])
        problems = main.check_mixed(result)
        self.assertTrue(any("read_text_file" in p for p in problems))

    def test_fail_wrong_total(self):
        result = AgentResult("Итог 1", "ok", 2, [
            call(1, "read_text_file", "mcp", {"content": "x"}),
            call(2, "calculate", "local", {"result": {"result": 1}}),
        ])
        problems = main.check_mixed(result)
        self.assertTrue(any("ожидалось 330000" in p for p in problems))

    def test_fail_no_calculate(self):
        result = AgentResult("330000", "ok", 2, [
            call(1, "read_text_file", "mcp", {"content": "x"}),
        ])
        self.assertTrue(any("calculate" in p for p in main.check_mixed(result)))

    def test_digits_normalization(self):
        self.assertEqual(main._digits("330 000 и 330,000"), "330000 и 330000")


class CliSafetyTest(unittest.TestCase):
    def test_demo_without_key_returns_2_without_mcp(self):
        with mock.patch.dict(os.environ, {}, clear=True), redirect_stderr(io.StringIO()) as err:
            code = main.main(["--demo"])
        self.assertEqual(code, 2)
        self.assertIn(config.API_KEY_ENV, err.getvalue())

    def test_confirm_refuses(self):
        with mock.patch("builtins.input", return_value="n"):
            self.assertFalse(main.confirm("Запустить?", False))
        self.assertTrue(main.confirm("Запустить?", True))

    def test_redact_filter(self):
        key = "AIzaFAKE-KEY-FOR-TEST-0123456789"
        with mock.patch.dict(os.environ, {config.API_KEY_ENV: key}):
            record = logging.LogRecord("agent", logging.INFO, __file__, 1, "key=%s", (key,), None)
            main.RedactSecretsFilter().filter(record)
            self.assertNotIn(key, record.getMessage())
            self.assertIn("***", record.getMessage())


if __name__ == "__main__":
    unittest.main()
