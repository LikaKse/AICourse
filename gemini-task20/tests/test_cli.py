"""Тесты CLI-парсера Дня 20."""

import unittest

from main import build_parser


class ParserTest(unittest.TestCase):
    def test_list_tools(self):
        args = build_parser().parse_args(["--list-tools"])
        self.assertTrue(args.list_tools)

    def test_scenarios(self):
        full = build_parser().parse_args(["--scenario", "full"])
        down = build_parser().parse_args(["--scenario", "catalog-down", "--query", "q"])
        self.assertEqual(full.scenario, "full")
        self.assertEqual(down.scenario, "catalog-down")
        self.assertEqual(down.query, "q")

    def test_demo(self):
        self.assertTrue(build_parser().parse_args(["--demo"]).demo)

    def test_mode_required(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args([])


if __name__ == "__main__":
    unittest.main()
