"""Тесты детерминированного резюме."""

import unittest

from models import Post
from summarizer import Summarizer, SummarizerError
from tests.fakes import POSTS


class SummarizerTest(unittest.TestCase):
    def test_summary_uses_only_input_data(self):
        items = [Post.model_validate(item) for item in POSTS]
        result = Summarizer().summarize(items)
        self.assertEqual(result.source_count, 2)
        self.assertIn("Найдено публикаций: 2", result.summary)
        self.assertIn("qui est esse", result.summary)
        self.assertIn("est rerum tempore vitae", result.summary)
        self.assertNotIn("выдуман", result.summary.lower())

    def test_empty_items_error(self):
        with self.assertRaises(SummarizerError):
            Summarizer().summarize([])


if __name__ == "__main__":
    unittest.main()
