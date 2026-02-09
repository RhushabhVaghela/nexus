"""
Test suite for SuffixDecoding implementation.

This module contains tests for the SuffixTrie, SuffixCache, and SuffixDecoder classes.
"""

import unittest
import tempfile
import os
from suffix_decoding import (
    SuffixTrie,
    SuffixCache,
    SuffixDecoder,
    SuffixDecodingIntegration,
)


class TestSuffixTrie(unittest.TestCase):
    """Test cases for SuffixTrie."""

    def setUp(self):
        """Set up test fixtures."""
        self.trie = SuffixTrie(max_nodes=1000, min_score=0.01)

    def test_insert_basic(self):
        """Test basic insertion."""
        self.trie.insert("hello", score=1.0)
        self.assertEqual(self.trie.node_count, 6)  # h-e-l-l-o

    def test_insert_multiple(self):
        """Test inserting multiple suffixes."""
        self.trie.insert("hello", score=1.0)
        self.trie.insert("world", score=0.8)
        self.trie.insert("help", score=0.9)

        self.assertEqual(self.trie.node_count, 6 + 6 + 4)

    def test_search_exact_match(self):
        """Test searching for exact match."""
        self.trie.insert("hello", score=1.0)
        self.trie.insert("world", score=0.8)

        matches = self.trie.search("hello")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0], "hello")
        self.assertEqual(matches[0][1], 1.0)

    def test_search_prefix_match(self):
        """Test searching with prefix."""
        self.trie.insert("hello", score=1.0)
        self.trie.insert("help", score=0.9)
        self.trie.insert("shell", score=0.7)

        matches = self.trie.search("hel", max_matches=10)
        suffixes = [s for s, _ in matches]

        self.assertIn("hello", suffixes)
        self.assertIn("help", suffixes)

    def test_match_suffixes(self):
        """Test the match_suffixes method."""
        self.trie.insert("hello", score=1.0)
        self.trie.insert("world", score=0.8)

        suffixes = self.trie.match_suffixes("hello", max_matches=5)
        self.assertEqual(suffixes, ["hello"])

    def test_tree_search_beam(self):
        """Test beam search functionality."""
        self.trie.insert("hello", score=1.0)
        self.trie.insert("help", score=0.9)
        self.trie.insert("hero", score=0.7)

        candidates = self.trie.tree_search("he", k=2)

        self.assertLessEqual(len(candidates), 2)
        self.assertTrue(all(c.startswith("he") for c in candidates))

    def test_prune(self):
        """Test pruning low-score nodes."""
        self.trie.insert("high_score", score=1.0)
        self.trie.insert("low_score", score=0.005)

        initial_count = self.trie.node_count
        removed = self.trie.prune(min_score=0.01)

        self.assertGreater(removed, 0)
        self.assertLess(self.trie.node_count, initial_count)

    def test_save_load(self):
        """Test persistence."""
        self.trie.insert("test", score=1.0)
        self.trie.insert("example", score=0.8)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name

        try:
            self.trie.save(path)
            loaded_trie = SuffixTrie.load(path)

            self.assertEqual(loaded_trie.node_count, self.trie.node_count)

            matches = loaded_trie.search("test")
            self.assertEqual(len(matches), 1)
        finally:
            os.unlink(path)

    def test_empty_prefix(self):
        """Test handling empty prefix."""
        self.trie.insert("test", score=1.0)

        matches = self.trie.search("")
        self.assertEqual(matches, [])

    def test_no_match(self):
        """Test when no match exists."""
        self.trie.insert("hello", score=1.0)

        matches = self.trie.search("xyz")
        self.assertEqual(matches, [])


class TestSuffixCache(unittest.TestCase):
    """Test cases for SuffixCache."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache = SuffixCache(max_size=10)

    def test_basic_operations(self):
        """Test basic cache operations."""
        self.cache.insert("test", ["esting", "extreme"])

        result = self.cache.get("test")
        self.assertEqual(result, ["esting", "extreme"])

    def test_cache_miss(self):
        """Test cache miss."""
        result = self.cache.get("unknown")
        self.assertIsNone(result)

    def test_lru_eviction(self):
        """Test LRU eviction."""
        for i in range(15):
            self.cache.insert(f"key{i}", [f"value{i}"])

        # First entries should be evicted
        self.assertIsNone(self.cache.get("key0"))

        # Recent entries should still exist
        self.assertIsNotNone(self.cache.get("key14"))

    def test_update_from_completions(self):
        """Test learning from completions."""
        self.cache.update_from_completions("test", ["testing complete"])

        result = self.cache.get("test")
        self.assertIsNotNone(result)
        self.assertIn("ing complete", result)

    def test_clear(self):
        """Test cache clearing."""
        self.cache.insert("test", ["value"])
        self.cache.clear()

        self.assertEqual(len(self.cache.cache), 0)
        self.assertEqual(len(self.cache.access_order), 0)


class TestSuffixDecoder(unittest.TestCase):
    """Test cases for SuffixDecoder."""

    def setUp(self):
        """Set up test fixtures."""
        self.decoder = SuffixDecoder(max_cache_size=100)

    def test_pre_populate(self):
        """Test pre-populating with training data."""
        texts = [
            "The quick brown fox",
            "jumps over the lazy dog",
            "Machine learning is great",
        ]

        self.decoder.pre_populate(texts)

        stats = self.decoder.get_statistics()
        self.assertGreater(stats["trie_stats"]["node_count"], 0)

    def test_generate(self):
        """Test generation functionality."""
        self.decoder.pre_populate(["hello world", "hello there"])

        result = self.decoder.generate("hello", max_tokens=5)

        self.assertTrue(result.startswith("hello"))

    def test_verify(self):
        """Test verification."""
        self.assertTrue(self.decoder.verify("test", "test"))
        self.assertFalse(self.decoder.verify("test", "different"))

    def test_accept_prefix(self):
        """Test learning from acceptance."""
        self.decoder.accept_prefix("test acceptance")

        stats = self.decoder.get_statistics()
        self.assertEqual(stats["total_accepted"], 1)

    def test_acceptance_rate_update(self):
        """Test acceptance rate calculation."""
        for _ in range(10):
            self.decoder.generate("test")

        self.decoder.accept_prefix("test accepted")

        stats = self.decoder.get_statistics()
        self.assertEqual(stats["total_generated"], 11)
        self.assertEqual(stats["total_accepted"], 1)
        self.assertAlmostEqual(stats["acceptance_rate"], 1 / 11, places=2)


class TestSuffixDecodingIntegration(unittest.TestCase):
    """Test cases for integration wrapper."""

    def setUp(self):
        """Set up test fixtures."""
        self.integration = SuffixDecodingIntegration()

    def test_draft_generation(self):
        """Test draft token generation."""
        self.integration.decoder.pre_populate(["hello world", "how are you"])

        drafts = self.integration.draft("hello", max_tokens=3)

        self.assertIsInstance(drafts, list)
        self.assertLessEqual(len(drafts), 3)

    def test_learn_from_completion(self):
        """Test learning from completion."""
        self.integration.learn_from_completion("test", "test completion")

        stats = self.integration.get_statistics()
        self.assertIn("speedup_estimate", stats)

    def test_speedup_estimate(self):
        """Test speedup estimation."""
        estimate = self.integration.get_speedup_estimate()

        self.assertGreater(estimate, 1.0)
        self.assertLessEqual(estimate, 2.5)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""

    def test_complete_workflow(self):
        """Test complete workflow from training to generation."""
        # Initialize decoder
        decoder = SuffixDecoder(max_cache_size=50)

        # Pre-populate with training data
        training_texts = [
            "The neural network processes data through layers",
            "Each layer transforms the input representation",
            "Training optimizes weights using backpropagation",
            "Gradient descent finds the minimum loss",
            "Stochastic gradient descent speeds up convergence",
        ]

        decoder.pre_populate(training_texts)

        # Generate
        result = decoder.generate("The neural", max_tokens=10)

        # Verify
        self.assertTrue(result.startswith("The neural"))

        # Accept and learn
        decoder.accept_prefix(result)

        # Check statistics
        stats = decoder.get_statistics()
        self.assertGreater(stats["total_generated"], 0)
        self.assertGreater(stats["acceptance_rate"], 0)


if __name__ == "__main__":
    unittest.main()
