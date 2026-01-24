"""
test_repetition_logic.py
Unit tests for PromptRepetitionEngine.
"""

import unittest
from src.utils.repetition import PromptRepetitionEngine

class TestRepetitionLogic(unittest.TestCase):
    def setUp(self):
        self.engine = PromptRepetitionEngine()
        self.query = "What is the capital of France?"
        self.context = "France is a country in Europe."

    def test_baseline(self):
        res = self.engine.apply_repetition(self.query, self.context, style="baseline")
        self.assertEqual(res, f"{self.context}\n{self.query}")

    def test_2x_factor(self):
        res = self.engine.apply_repetition(self.query, self.context, factor=2)
        expected = f"{self.context}\n{self.query} {self.context}\n{self.query}"
        self.assertEqual(res, expected)

    def test_2x_style(self):
        res = self.engine.apply_repetition(self.query, self.context, style="2x")
        expected = f"{self.context}\n{self.query} {self.context}\n{self.query}"
        self.assertEqual(res, expected)

    def test_verbose_style(self):
        res = self.engine.apply_repetition(self.query, self.context, style="verbose")
        expected = f"{self.context}\n{self.query} Let me repeat that: {self.context}\n{self.query}"
        self.assertEqual(res, expected)

    def test_3x_factor(self):
        res = self.engine.apply_repetition(self.query, self.context, factor=3)
        # 3x maps to 3x style
        expected = f"{self.context}\n{self.query} Let me repeat that: {self.context}\n{self.query} Let me repeat that one more time: {self.context}\n{self.query}"
        self.assertEqual(res, expected)

    def test_no_context(self):
        res = self.engine.apply_repetition(self.query, factor=2)
        self.assertEqual(res, f"{self.query} {self.query}")

if __name__ == '__main__':
    unittest.main()
