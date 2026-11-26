"""Тесты CLI-парсера без запуска сети."""

import unittest

from main import SCENARIOS, build_parser


class ParserTest(unittest.TestCase):
    def test_scenarios(self):
        self.assertEqual(SCENARIOS, ("success", "summarize-error", "empty-search"))
        args = build_parser().parse_args(["--scenario", "success", "--query", "qui", "--filename", "x.txt"])
        self.assertEqual(args.scenario, "success")
        self.assertEqual(args.query, "qui")
        self.assertEqual(args.filename, "x.txt")

    def test_demo_and_list_tools(self):
        self.assertTrue(build_parser().parse_args(["--demo"]).demo)
        self.assertTrue(build_parser().parse_args(["--list-tools"]).list_tools)

    def test_one_mode_required(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args([])


if __name__ == "__main__":
    unittest.main()
