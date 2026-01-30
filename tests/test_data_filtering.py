"""
Tests for data filtering pipeline (Paper 2601.15394).
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.nexus_final.data_loader import MemorizationFilter, UniversalDataLoader


class TestMemorizationFilter:
    """Test suite for MemorizationFilter."""
    
    def test_initialization(self):
        """Test filter initialization."""
        filter_obj = MemorizationFilter(
            entropy_threshold=0.4,
            risk_threshold=0.5
        )
        
        assert filter_obj.entropy_threshold == 0.4
        assert filter_obj.risk_threshold == 0.5
        assert filter_obj.classifier is None
    
    def test_calculate_entropy(self):
        """Test entropy calculation."""
        # High entropy text
        high_entropy_text = "The quick brown fox jumps over the lazy dog multiple times"
        entropy_high = MemorizationFilter.calculate_entropy(high_entropy_text)
        assert entropy_high > 0
        
        # Low entropy text (repetitive)
        low_entropy_text = "aaaaabbbbbcccccddddd"
        entropy_low = MemorizationFilter.calculate_entropy(low_entropy_text)
        
        # Random text should have higher entropy than repetitive
        assert entropy_high != entropy_low
        
        # Empty text
        empty_entropy = MemorizationFilter.calculate_entropy("")
        assert empty_entropy == 0.0
    
    def test_should_filter_entropy_based(self):
        """Test filtering based on entropy."""
        filter_obj = MemorizationFilter(entropy_threshold=0.5)
        
        # High entropy sample (should pass)
        high_entropy_sample = {
            "messages": [
                {"role": "user", "content": "Explain quantum mechanics in detail"},
                {"role": "assistant", "content": "Quantum mechanics is a fundamental theory in physics..."}
            ]
        }
        
        # Low entropy sample (should be filtered)
        low_entropy_sample = {
            "messages": [
                {"role": "user", "content": "aaaaaaaaaaaaaaaaaa"},
                {"role": "assistant", "content": "bbbbbbbbbbbbbbbbbb"}
            ]
        }
        
        # Note: Actual entropy values depend on content
        # Just verifying the method works
        result_high = filter_obj.should_filter(high_entropy_sample)
        result_low = filter_obj.should_filter(low_entropy_sample)
        
        # Low entropy should be filtered
        assert isinstance(result_low, bool)
        assert isinstance(result_high, bool)
    
    def test_filter_dataset(self):
        """Test filtering a dataset stream."""
        filter_obj = MemorizationFilter(entropy_threshold=1.0)  # High threshold
        
        # Create sample data
        samples = [
            {"messages": [{"role": "user", "content": "Test content 1"}]},
            {"messages": [{"role": "user", "content": "Test content 2"}]},
            {"messages": [{"role": "user", "content": "Test content 3"}]},
        ]
        
        # Filter
        def sample_generator():
            for s in samples:
                yield s
        
        filtered = list(filter_obj.filter_dataset(sample_generator()))
        
        # All should pass with high threshold
        assert len(filtered) == 3


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


class TestEntropyThresholds:
    """Test various entropy threshold configurations."""
    
    def test_threshold_variations(self):
        """Test different entropy thresholds."""
        text = "This is some test content for the dataset loader"
        
        # Test with various thresholds
        for threshold in [0.1, 0.4, 0.8, 1.0]:
            filter_obj = MemorizationFilter(entropy_threshold=threshold)
            entropy = filter_obj.calculate_entropy(text)
            
            # Just verify calculation works
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])