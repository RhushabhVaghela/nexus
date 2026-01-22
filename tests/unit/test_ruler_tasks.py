#!/usr/bin/env python3
"""
Unit tests for RULER Benchmark tasks.
"""

import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.ruler_tasks import (
    TaskConfig, TaskSample, TaskCategory,
    SingleNIAH, MultiKeyNIAH, MultiValueNIAH, MultiQueryNIAH,
    VariableTracing, ChainFollowing,
    CommonWordCount, FrequentWord,
    get_task, get_all_tasks, RULER_TASKS
)


class TestTaskConfig:
    """Tests for TaskConfig."""
    
    def test_default_config(self):
        config = TaskConfig()
        assert config.context_length == 4096
        assert config.num_samples == 100
    
    def test_custom_config(self):
        config = TaskConfig(context_length=8192, num_samples=50)
        assert config.context_length == 8192


class TestSingleNIAH:
    """Tests for Single NIAH task."""
    
    def test_generate_sample(self):
        config = TaskConfig(context_length=1000)
        task = SingleNIAH(config)
        sample = task.generate_sample()
        
        assert isinstance(sample, TaskSample)
        assert len(sample.context) > 0
        assert "secret" in sample.question.lower()
        assert sample.expected_answer.startswith("SECRET-")
    
    def test_needle_in_context(self):
        config = TaskConfig(context_length=1000)
        task = SingleNIAH(config)
        sample = task.generate_sample()
        
        assert sample.expected_answer in sample.context
    
    def test_evaluate_correct(self):
        config = TaskConfig(context_length=1000)
        task = SingleNIAH(config)
        sample = task.generate_sample()
        
        is_correct, score = task.evaluate_response(sample.expected_answer, sample.expected_answer)
        assert is_correct
        assert score == 1.0
    
    def test_evaluate_contains(self):
        config = TaskConfig(context_length=1000)
        task = SingleNIAH(config)
        
        is_correct, score = task.evaluate_response("The answer is SECRET-1234", "secret-1234")
        assert is_correct
        assert score >= 0.9


class TestMultiKeyNIAH:
    """Tests for Multi-Key NIAH task."""
    
    def test_generate_sample(self):
        config = TaskConfig(context_length=2000)
        task = MultiKeyNIAH(config)
        sample = task.generate_sample()
        
        assert isinstance(sample, TaskSample)
        assert sample.expected_answer.startswith("CODE-")
        assert "target_city" in sample.metadata
    
    def test_multiple_codes_in_context(self):
        config = TaskConfig(context_length=2000)
        task = MultiKeyNIAH(config)
        sample = task.generate_sample()
        
        # Should have multiple codes
        code_count = sample.context.count("CODE-")
        assert code_count > 1


class TestMultiValueNIAH:
    """Tests for Multi-Value NIAH task."""
    
    def test_generate_sample(self):
        config = TaskConfig(context_length=1500)
        task = MultiValueNIAH(config)
        sample = task.generate_sample()
        
        assert isinstance(sample, TaskSample)
        assert "," in sample.expected_answer  # Multiple values
        assert "num_values" in sample.metadata
    
    def test_evaluate_partial(self):
        config = TaskConfig(context_length=1500)
        task = MultiValueNIAH(config)
        
        # Test partial match
        is_correct, score = task.evaluate_response("apple banana", "apple,banana,orange")
        assert 0 < score < 1


class TestMultiQueryNIAH:
    """Tests for Multi-Query NIAH task."""
    
    def test_generate_sample(self):
        config = TaskConfig(context_length=2000)
        task = MultiQueryNIAH(config)
        sample = task.generate_sample()
        
        assert isinstance(sample, TaskSample)
        assert sample.expected_answer.startswith("ID-")
        assert "all_pairs" in sample.metadata


class TestVariableTracing:
    """Tests for Variable Tracing task."""
    
    def test_generate_sample(self):
        config = TaskConfig(context_length=2000)
        task = VariableTracing(config)
        sample = task.generate_sample()
        
        assert isinstance(sample, TaskSample)
        assert sample.expected_answer.isdigit()
        assert "num_hops" in sample.metadata
    
    def test_context_contains_assignments(self):
        config = TaskConfig(context_length=2000)
        task = VariableTracing(config)
        sample = task.generate_sample()
        
        assert "Let" in sample.context
        assert "=" in sample.context


class TestChainFollowing:
    """Tests for Chain Following task."""
    
    def test_generate_sample(self):
        config = TaskConfig(context_length=2000)
        task = ChainFollowing(config)
        sample = task.generate_sample()
        
        assert isinstance(sample, TaskSample)
        assert "chain" in sample.metadata
        assert sample.expected_answer in sample.metadata["chain"]


class TestCommonWordCount:
    """Tests for Common Word Count task."""
    
    def test_generate_sample(self):
        config = TaskConfig(context_length=1000)
        task = CommonWordCount(config)
        sample = task.generate_sample()
        
        assert isinstance(sample, TaskSample)
        assert sample.expected_answer.isdigit()
        assert "target_word" in sample.metadata
    
    def test_word_count_matches(self):
        config = TaskConfig(context_length=1000)
        task = CommonWordCount(config)
        sample = task.generate_sample()
        
        target = sample.metadata["target_word"]
        actual_count = sample.context.upper().count(target)
        expected_count = int(sample.expected_answer)
        
        assert actual_count == expected_count


class TestFrequentWord:
    """Tests for Frequent Word task."""
    
    def test_generate_sample(self):
        config = TaskConfig(context_length=1000)
        task = FrequentWord(config)
        sample = task.generate_sample()
        
        assert isinstance(sample, TaskSample)
        assert sample.expected_answer in ["ALPHA", "BETA", "GAMMA"]


class TestGetTask:
    """Tests for task factory functions."""
    
    def test_get_valid_task(self):
        task = get_task("single_niah")
        assert isinstance(task, SingleNIAH)
    
    def test_get_invalid_task(self):
        with pytest.raises(ValueError):
            get_task("nonexistent_task")
    
    def test_get_all_tasks(self):
        tasks = get_all_tasks()
        assert len(tasks) == len(RULER_TASKS)
        assert "single_niah" in tasks


class TestTaskCategories:
    """Tests for task category assignments."""
    
    def test_retrieval_tasks(self):
        for name in ["single_niah", "multi_key_niah", "multi_value_niah", "multi_query_niah"]:
            task = get_task(name)
            assert task.category == TaskCategory.RETRIEVAL
    
    def test_multihop_tasks(self):
        for name in ["variable_tracing", "chain_following"]:
            task = get_task(name)
            assert task.category == TaskCategory.MULTI_HOP
    
    def test_aggregation_tasks(self):
        for name in ["common_word_count", "frequent_word"]:
            task = get_task(name)
            assert task.category == TaskCategory.AGGREGATION


class TestContextLengthScaling:
    """Tests for context length scaling."""
    
    def test_context_respects_length(self):
        for target_length in [500, 1000, 2000, 4000]:
            config = TaskConfig(context_length=target_length)
            task = SingleNIAH(config)
            sample = task.generate_sample()
            
            # Context should be approximately target length
            assert len(sample.context) >= target_length * 0.8
            assert len(sample.context) <= target_length * 1.2
    
    def test_samples_reproducible(self):
        config = TaskConfig(context_length=1000, seed=42)
        
        task1 = SingleNIAH(config)
        sample1 = task1.generate_sample()
        
        task2 = SingleNIAH(config)
        sample2 = task2.generate_sample()
        
        assert sample1.expected_answer == sample2.expected_answer
