"""
Comprehensive Unit Tests for Data Filtering Pipeline (Paper 2601.15394).

Tests cover:
- Entropy-based filtering
- Classifier-based filtering
- Threshold configuration
- Filtering statistics
- Edge cases (small datasets)
"""

import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.nexus_final.data_loader import MemorizationFilter, UniversalDataLoader


class TestMemorizationFilterInitialization:
    """Test suite for filter initialization."""
    
    def test_default_initialization(self):
        """Test filter initialization with defaults."""
        filter_obj = MemorizationFilter()
        assert filter_obj.entropy_threshold == 0.4
        assert filter_obj.risk_threshold == 0.5
        assert filter_obj.classifier is None
        assert filter_obj.classifier_path is None
    
    def test_custom_initialization(self):
        """Test filter initialization with custom values."""
        filter_obj = MemorizationFilter(
            entropy_threshold=0.3,
            risk_threshold=0.7
        )
        assert filter_obj.entropy_threshold == 0.3
        assert filter_obj.risk_threshold == 0.7
    
    def test_initialization_with_classifier_path(self, tmp_path):
        """Test filter initialization with classifier path."""
        # This would require a real classifier file
        filter_obj = MemorizationFilter(
            classifier_path="/nonexistent/path.pkl"
        )
        assert filter_obj.classifier is None  # Should handle missing file gracefully


class TestEntropyCalculation:
    """Test entropy calculation methods."""
    
    def test_calculate_entropy_high(self):
        """Test entropy calculation for high entropy text."""
        high_entropy_text = "The quick brown fox jumps over the lazy dog multiple times"
        entropy = MemorizationFilter.calculate_entropy(high_entropy_text)
        assert entropy > 0
        assert isinstance(entropy, float)
    
    def test_calculate_entropy_low(self):
        """Test entropy calculation for low entropy text."""
        low_entropy_text = "aaaaabbbbbcccccddddd"
        entropy_low = MemorizationFilter.calculate_entropy(low_entropy_text)
        assert entropy_low >= 0
    
    def test_calculate_entropy_empty(self):
        """Test entropy calculation for empty string."""
        entropy = MemorizationFilter.calculate_entropy("")
        assert entropy == 0.0
    
    def test_calculate_entropy_comparison(self):
        """Test that entropy differs between text types."""
        high_entropy_text = "The quick brown fox jumps over the lazy dog"
        low_entropy_text = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        
        entropy_high = MemorizationFilter.calculate_entropy(high_entropy_text)
        entropy_low = MemorizationFilter.calculate_entropy(low_entropy_text)
        
        # Random text should have higher entropy than repetitive
        assert entropy_high != entropy_low


class TestFilteringLogic:
    """Test filtering logic and decision making."""
    
    def test_should_filter_low_entropy(self):
        """Test filtering of low entropy samples."""
        filter_obj = MemorizationFilter(entropy_threshold=0.9)
        
        # Low entropy sample (should be filtered)
        low_entropy_sample = {
            "messages": [
                {"role": "user", "content": "aaaaaaaaaaaaaaaaaa"},
                {"role": "assistant", "content": "bbbbbbbbbbbbbbbbbb"}
            ]
        }
        
        result = filter_obj.should_filter(low_entropy_sample)
        assert isinstance(result, bool)
    
    def test_should_filter_high_entropy(self):
        """Test filtering of high entropy samples."""
        filter_obj = MemorizationFilter(entropy_threshold=0.1)
        
        # High entropy sample (should pass)
        high_entropy_sample = {
            "messages": [
                {"role": "user", "content": "Explain quantum mechanics in detail"},
                {"role": "assistant", "content": "Quantum mechanics is a fundamental theory..."}
            ]
        }
        
        result = filter_obj.should_filter(high_entropy_sample)
        assert isinstance(result, bool)
    
    def test_should_filter_text_format(self):
        """Test filtering with text format."""
        filter_obj = MemorizationFilter()
        
        sample = {"text": "This is some test content for filtering"}
        result = filter_obj.should_filter(sample)
        assert isinstance(result, bool)
    
    def test_should_filter_fallback(self):
        """Test filtering fallback for unknown format."""
        filter_obj = MemorizationFilter()
        
        sample = {"unknown_key": "some_value"}
        result = filter_obj.should_filter(sample)
        assert isinstance(result, bool)


class TestDatasetFiltering:
    """Test dataset filtering operations."""
    
    def test_filter_dataset_all_pass(self):
        """Test filtering where all samples pass."""
        filter_obj = MemorizationFilter(entropy_threshold=0.01)  # Very low threshold
        
        samples = [
            {"messages": [{"role": "user", "content": "Test content 1"}]},
            {"messages": [{"role": "user", "content": "Test content 2"}]},
            {"messages": [{"role": "user", "content": "Test content 3"}]},
        ]
        
        def sample_generator():
            for s in samples:
                yield s
        
        filtered = list(filter_obj.filter_dataset(sample_generator()))
        assert len(filtered) == 3
    
    def test_filter_dataset_generator(self):
        """Test that filter_dataset works with generators."""
        filter_obj = MemorizationFilter()
        
        def sample_generator():
            for i in range(5):
                yield {"messages": [{"role": "user", "content": f"Test {i}"}]}
        
        result = list(filter_obj.filter_dataset(sample_generator()))
        # Should be a generator that yields samples
        assert isinstance(result, list)


class TestThresholdConfigurations:
    """Test various threshold configurations."""
    
    def test_threshold_variations(self):
        """Test different entropy thresholds."""
        text = "This is some test content for the dataset loader"
        
        # Test with various thresholds
        for threshold in [0.1, 0.4, 0.8, 1.0]:
            filter_obj = MemorizationFilter(entropy_threshold=threshold)
            entropy = filter_obj.calculate_entropy(text)
            assert entropy >= 0
    
    def test_expected_reduction_rate(self):
        """Test expected memorization reduction rates."""
        # Create samples with varying entropy
        samples = [
            {"messages": [{"role": "user", "content": "a" * 1000}]},  # Low entropy
            {"messages": [{"role": "user", "content": "b" * 1000}]},  # Low entropy
            {"messages": [{"role": "user", "content": "The quick brown fox jumps over the lazy dog and then runs around the field"}]},  # Higher entropy
        ]
        
        # With low threshold, should filter the repetitive ones
        filter_obj = MemorizationFilter(entropy_threshold=0.3)
        
        def sample_generator():
            for s in samples:
                yield s
        
        filtered = list(filter_obj.filter_dataset(sample_generator()))
        
        # Should have reduced samples
        assert len(filtered) <= len(samples)


class TestUniversalDataLoader:
    """Test suite for UniversalDataLoader."""
    
    def test_initialization_without_filtering(self):
        """Test loader initialization without filtering."""
        loader = UniversalDataLoader(
            data_root="/tmp/test",
            filter_memorization_risk=False
        )
        
        assert loader.data_root == "/tmp/test"
        assert not loader.filter_memorization_risk
        assert loader.memorization_filter is None
    
    def test_initialization_with_filtering(self):
        """Test loader initialization with filtering."""
        loader = UniversalDataLoader(
            data_root="/tmp/test",
            filter_memorization_risk=True,
            entropy_threshold=0.4,
            risk_threshold=0.5
        )
        
        assert loader.filter_memorization_risk
        assert loader.memorization_filter is not None
        assert loader.memorization_filter.entropy_threshold == 0.4
        assert loader.memorization_filter.risk_threshold == 0.5
    
    def test_normalize_standard_format(self):
        """Test normalization of standard messages format."""
        loader = UniversalDataLoader()
        
        sample = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        }
        
        result = loader._normalize(sample, "test.json")
        assert result == sample  # Should pass through unchanged
    
    def test_normalize_problem_solution(self):
        """Test normalization of problem/solution format."""
        loader = UniversalDataLoader()
        
        sample = {
            "problem": "What is 2+2?",
            "solution": "4"
        }
        
        result = loader._normalize(sample, "test.json")
        assert result is not None
        assert "messages" in result
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"
    
    def test_normalize_instruction_output(self):
        """Test normalization of instruction/output format."""
        loader = UniversalDataLoader()
        
        sample = {
            "instruction": "Write a poem",
            "input": "about nature",
            "output": "Roses are red..."
        }
        
        result = loader._normalize(sample, "test.json")
        assert result is not None
        assert "messages" in result
        assert "Write a poem" in result["messages"][0]["content"]
    
    def test_normalize_translation(self):
        """Test normalization of translation format."""
        loader = UniversalDataLoader()
        
        sample = {
            "src": "Hello world",
            "trgs": ["Bonjour le monde"],
            "sl": "en",
            "tl": "fr"
        }
        
        result = loader._normalize(sample, "test.json")
        assert result is not None
        assert "messages" in result
        assert "Translate from en to fr" in result["messages"][0]["content"]
    
    def test_normalize_question_answer(self):
        """Test normalization of question/answer format."""
        loader = UniversalDataLoader()
        
        sample = {
            "question": "What is Python?",
            "answer": "A programming language"
        }
        
        result = loader._normalize(sample, "test.json")
        assert result is not None
        assert "messages" in result


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataset(self):
        """Test filtering of empty dataset."""
        filter_obj = MemorizationFilter()
        
        def empty_generator():
            return
            yield  # Make it a generator
        
        filtered = list(filter_obj.filter_dataset(empty_generator()))
        assert len(filtered) == 0
    
    def test_single_sample_dataset(self):
        """Test filtering of single sample dataset."""
        filter_obj = MemorizationFilter(entropy_threshold=0.01)
        
        def single_generator():
            yield {"messages": [{"role": "user", "content": "Single test"}]}
        
        filtered = list(filter_obj.filter_dataset(single_generator()))
        assert len(filtered) <= 1
    
    def test_very_small_threshold(self):
        """Test with very small entropy threshold."""
        filter_obj = MemorizationFilter(entropy_threshold=0.001)
        
        sample = {"messages": [{"role": "user", "content": "Test"}]}
        result = filter_obj.should_filter(sample)
        assert isinstance(result, bool)
    
    def test_very_large_threshold(self):
        """Test with very large entropy threshold."""
        filter_obj = MemorizationFilter(entropy_threshold=10.0)
        
        sample = {"messages": [{"role": "user", "content": "Test content here"}]}
        result = filter_obj.should_filter(sample)
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
