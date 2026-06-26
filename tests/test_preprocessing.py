import unittest

from tourism_rag_assistant.utils.text import has_suspect_text, normalize_text


class PreprocessingTest(unittest.TestCase):
    def test_normalize_text_strips_hidden_whitespace(self) -> None:
        self.assertEqual(normalize_text("  Hello\u200b   World\xa0 "), "hello world")

    def test_has_suspect_text_detects_non_landmark_noise(self) -> None:
        self.assertTrue(has_suspect_text("A picture of cars near a billboard"))

    def test_has_suspect_text_allows_memorial_text(self) -> None:
        self.assertFalse(has_suspect_text("Мемориал в историческом центре города"))


if __name__ == "__main__":
    unittest.main()
