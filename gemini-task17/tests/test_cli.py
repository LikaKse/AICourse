"""Тесты CLI-проверок Дня 17 без Gemini."""

import io
import logging
import os
import unittest
from contextlib import redirect_stderr
from unittest import mock

import config
import main
from agent import AgentResult, ToolCallRecord
from tests.fakes import TASK_DATA


def call(iteration, name, source, response, args=None):
    return ToolCallRecord(iteration, name, args or {}, response, source)


class TaskCheckTest(unittest.TestCase):
    def test_pass(self):
        result = AgentResult(
            "Задача 5: «laboriosam mollitia et enim quasi adipisci quia provident illum». "
            "Статус: не выполнена.",
            "ok",
            2,
            [call(1, "get_task", "mcp", {"result": dict(TASK_DATA)}, {"task_id": 5})],
        )
        self.assertEqual(main.check_task(result), [])

    def test_fail_wrong_tool(self):
        result = AgentResult("5", "ok", 2, [
            call(1, "calculate", "local", {"result": {"result": 5}}),
        ])
        problems = main.check_task(result)
        self.assertTrue(any("get_task" in problem for problem in problems))

    def test_fail_wrong_id(self):
        data = dict(TASK_DATA)
        data["id"] = 6
        result = AgentResult("6", "ok", 2, [
            call(1, "get_task", "mcp", {"result": data}),
        ])
        self.assertTrue(any("id=" in problem for problem in main.check_task(result)))

    def test_fail_result_not_used(self):
        result = AgentResult("Не знаю.", "ok", 2, [
            call(1, "get_task", "mcp", {"result": dict(TASK_DATA)}),
        ])
        self.assertTrue(any("id 5" in problem for problem in main.check_task(result)))


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
