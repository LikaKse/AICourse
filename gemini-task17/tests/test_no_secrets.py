"""Проверки секретов и фиксированных версий Дня 17."""

import os
import re
import unittest
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parent.parent
GOOGLE_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{35}")
SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "logs", "results"}


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

    def test_gitignore_covers_env_and_venv(self):
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for entry in (".env", ".venv/", "logs/", "__pycache__/"):
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

    def test_server_identity(self):
        self.assertEqual(config.SERVER_NAME, "jsonplaceholder-tasks")
        self.assertEqual(config.SERVER_VERSION, "0.1.0")
        self.assertEqual(config.MCP_EXPOSED_TOOLS, ("get_task",))
        self.assertIn("jsonplaceholder.typicode.com", config.TASKS_API_BASE)

    def test_server_and_client_use_stdio(self):
        server_source = (ROOT / "server.py").read_text(encoding="utf-8")
        client_source = (ROOT / "mcp_client.py").read_text(encoding="utf-8")
        self.assertIn('mcp.run(transport="stdio")', server_source)
        self.assertIn("ClientSession", client_source)
        self.assertIn("TASKS_API_BASE", client_source)


if __name__ == "__main__":
    unittest.main()
