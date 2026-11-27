"""Проверки изоляции, отсутствия Gemini и аргументов Filesystem MCP."""

import re
import unittest
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parent.parent
GOOGLE_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{35}")
SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "logs", ".tools"}


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

    def test_gitignore_covers_environment_and_node(self):
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for entry in (".venv/", ".tools/", "logs/", "__pycache__/"):
            self.assertIn(entry, text)

    def test_fixed_versions(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("mcp==2.3.0", requirements)
        self.assertIn("pydantic==2.14.4", requirements)
        self.assertEqual(config.NODE_VERSION, "24.22.3")
        self.assertEqual(config.FILESYSTEM_SERVER_VERSION, "2.4.0")

    def test_filesystem_command_is_read_only_and_confined(self):
        specs = (ROOT / "server_specs.py").read_text(encoding="utf-8")
        self.assertIn('"--read-only"', specs)
        self.assertIn('"--confine-roots"', specs)
        self.assertIn("str(config.WORKSPACE)", specs)
        self.assertIn("verify_allowed_directory=config.WORKSPACE", specs)


if __name__ == "__main__":
    unittest.main()
