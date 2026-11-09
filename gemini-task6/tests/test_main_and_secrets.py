"""Офлайн-тесты CLI-проверок сценариев и отсутствия секретов в файлах проекта."""

import io
import logging
import os
import re
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import config
import main
from agent import AgentResult, ToolCallRecord

ROOT = Path(__file__).resolve().parent.parent
GOOGLE_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{35}")
SKIP_DIRS = {"__pycache__", ".git", "venv", ".venv"}


def project_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and not SKIP_DIRS & set(path.relative_to(ROOT).parts):
            yield path


class NoSecretsTest(unittest.TestCase):
    def test_no_google_api_key_pattern_in_files(self):
        for path in project_files():
            text = path.read_text(encoding="utf-8", errors="ignore")
            with self.subTest(file=str(path.relative_to(ROOT))):
                self.assertIsNone(GOOGLE_KEY_PATTERN.search(text))

    def test_real_key_value_not_in_files(self):
        key = os.environ.get(config.API_KEY_ENV)
        if not key:
            self.skipTest(f"{config.API_KEY_ENV} не задан — проверяется только шаблон AIza…")
        for path in project_files():
            with self.subTest(file=str(path.relative_to(ROOT))):
                self.assertNotIn(key, path.read_text(encoding="utf-8", errors="ignore"))

    def test_gitignore_covers_secrets_and_temp(self):
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for entry in (".env", "venv/", "__pycache__/", "logs/"):
            self.assertIn(entry, text)

    def test_log_filter_redacts_key(self):
        fake_key = "AIzaFAKE-KEY-FOR-LOG-FILTER-0123456789"
        with mock.patch.dict(os.environ, {config.API_KEY_ENV: fake_key}):
            record = logging.LogRecord("agent", logging.INFO, __file__, 1, "key=%s", (fake_key,), None)
            main.RedactSecretsFilter().filter(record)
            self.assertNotIn(fake_key, record.getMessage())

    def test_missing_key_exits_without_api_call(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit) as ctx:
                main.make_client()
        self.assertIn(config.API_KEY_ENV, str(ctx.exception))


class ScenarioCheckTest(unittest.TestCase):
    S1, S2, S3 = main.SCENARIOS

    def _call(self, it, name, args, result):
        return ToolCallRecord(it, name, args, {"result": result})

    def test_scenario_definitions(self):
        self.assertEqual([s["expected_tools"] for s in main.SCENARIOS],
                         [[], ["convert_currency"], ["convert_currency", "calculate"]])

    def test_pass_no_tools(self):
        r = AgentResult("API — это интерфейс", "ok", 1)
        self.assertEqual(main.check_scenario(self.S1, r), [])

    def test_fail_if_tool_used_when_not_expected(self):
        r = AgentResult("x", "ok", 2, [self._call(1, "calculate", {"expression": "1"}, {"result": 1})])
        self.assertTrue(main.check_scenario(self.S1, r))

    def test_pass_scenario_two(self):
        r = AgentResult("150 USD = 3 750 000 VND по фиксированному курсу", "ok", 2,
                        [self._call(1, "convert_currency", {}, {"converted_amount": 3750000})])
        self.assertEqual(main.check_scenario(self.S2, r), [])

    def test_scenario_two_requires_fixed_rate_disclaimer(self):
        r = AgentResult("150 USD = 3 750 000 VND", "ok", 2,
                        [self._call(1, "convert_currency", {}, {"converted_amount": 3750000})])
        self.assertTrue(any("фиксированный" in p for p in main.check_scenario(self.S2, r)))

    def test_pass_scenario_three(self):
        r = AgentResult("Итого 4,125,000 VND (демонстрационный курс)", "ok", 3, [
            self._call(1, "convert_currency", {}, {"converted_amount": 3750000}),
            self._call(2, "calculate", {"expression": "3750000 * 1.1"}, {"result": 4125000}),
        ])
        self.assertEqual(main.check_scenario(self.S3, r), [])

    def test_scenario_three_fails_if_parallel_or_not_using_result(self):
        r = AgentResult("4 125 000 фиксированный", "ok", 2, [
            self._call(1, "convert_currency", {}, {"converted_amount": 3750000}),
            self._call(1, "calculate", {"expression": "150 * 25000 * 1.1"}, {"result": 4125000}),
        ])
        problems = main.check_scenario(self.S3, r)
        self.assertTrue(any("последовательно" in p for p in problems))
        self.assertTrue(any("не использует" in p for p in problems))

    def test_digits_normalization(self):
        self.assertIn("4125000", main._digits("4 125 000 VND"))
        self.assertIn("4125000", main._digits("4.125.000 донгов"))
        self.assertIn("4125000", main._digits("4 125 000"))
        self.assertIn("3750000", main._digits("3,750,000.00"))

    def test_scenarios_declined_makes_no_api_call(self):
        with mock.patch("builtins.input", return_value="n"), \
             mock.patch.object(main, "make_client", side_effect=AssertionError("API не должен вызываться")):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main.cmd_scenarios([], assume_yes=False), 1)

    def test_dry_run_is_offline(self):
        out = io.StringIO()
        with mock.patch.object(main, "make_client", side_effect=AssertionError("API не должен вызываться")):
            with redirect_stdout(out):
                main.cmd_dry_run()
        self.assertIn("convert_currency", out.getvalue())
        self.assertIn("отключено: True", out.getvalue())


if __name__ == "__main__":
    unittest.main()
