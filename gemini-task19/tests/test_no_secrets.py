"""Проверки изоляции, отсутствия Gemini и фиксированных версий."""

import re
import unittest
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parent.parent
GOOGLE_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{35}")
SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "logs", "output"}


def project_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and not SKIP_DIRS & set(path.relative_to(ROOT).parts):
            yield path


class IsolationTest(unittest.TestCase):
    def test_no_google_key_pattern(self):
        for path in project_files():
            text = path.read_text(encoding="utf-8", errors="ignore")
            with self.subTest(file=str(path.relative_to(ROOT))):
                self.assertIsNone(GOOGLE_KEY_PATTERN.search(text))

    def test_no_gemini_dependency_or_import(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertNotIn("google-genai", requirements)
        for path in project_files():
            if path.suffix != ".py" or path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            with self.subTest(file=str(path.relative_to(ROOT))):
                self.assertNotIn("google.genai", text)
                self.assertNotIn("genai.Client", text)

    def test_gitignore_covers_output_and_venv(self):
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for entry in (".venv/", "output/*", "!output/.gitkeep", "logs/"):
            self.assertIn(entry, text)

    def test_versions_and_tools(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("mcp==2.3.0", requirements)
        self.assertIn("httpx2==2.16.2", requirements)
        self.assertIn("pydantic==2.14.4", requirements)
        self.assertEqual(config.SERVER_NAME, "content-chain")
        self.assertEqual(config.MCP_EXPOSED_TOOLS, ("search", "summarize", "save_to_file"))

    def test_fail_hook_is_test_only(self):
        client = (ROOT / "mcp_client.py").read_text(encoding="utf-8")
        self.assertIn('if self.fail_summarize:', client)
        server = (ROOT / "server.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.get("TASK19_FAIL_SUMMARIZE") == "1"', server)


if __name__ == "__main__":
    unittest.main()
