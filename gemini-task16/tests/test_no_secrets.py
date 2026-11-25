"""Проверки секретов, изоляции и зафиксированных версий."""

import json
import os
import re
import unittest
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parent.parent
GOOGLE_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{35}")
SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", ".tools", "logs", "results"}


def project_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and not SKIP_DIRS & set(path.relative_to(ROOT).parts):
            yield path


class NoSecretsTest(unittest.TestCase):
    def test_no_google_api_key_pattern(self):
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

    def test_gitignore_covers_env_and_isolated_tools(self):
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for entry in (".env", ".venv/", ".tools/", "logs/", "__pycache__/"):
            self.assertIn(entry, text)

    def test_env_example_has_no_value(self):
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("GEMINI_API_KEY=", text)
        self.assertNotRegex(text, r"GEMINI_API_KEY=\S+")


class VersionPinsTest(unittest.TestCase):
    def test_python_pins(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("mcp==2.3.0", requirements)
        self.assertIn("google-genai==2.34.0", requirements)

    def test_node_and_server_pins(self):
        self.assertEqual(config.NODE_VERSION, "24.22.3")
        self.assertEqual(config.MCP_SERVER_VERSION, "2.4.0")
        package = json.loads((ROOT / "scripts" / "mcp-server-package.json").read_text(encoding="utf-8"))
        self.assertEqual(
            package["dependencies"]["@modelcontextprotocol/server-filesystem"], "2.4.0"
        )

    def test_server_is_read_only_and_confined(self):
        self.assertTrue(config.MCP_READ_ONLY)
        self.assertTrue(config.MCP_CONFINE_ROOTS)
        client_args = (ROOT / "mcp_client.py").read_text(encoding="utf-8")
        self.assertIn("--read-only", client_args)
        self.assertIn("--confine-roots", client_args)


if __name__ == "__main__":
    unittest.main()
