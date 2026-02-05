"""
Unit tests for RULER Benchmark tasks.
"""

import pytest
from pathlib import Path
import sys
import json
from unittest.mock import MagicMock, patch

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
    def test_default_config(self):
        config = TaskConfig()
        assert config.context_length == 4096
        assert config.num_samples == 100
    
    def test_custom_config(self):
        config = TaskConfig(context_length=8192, num_samples=50)
        assert config.context_length == 8192

class TestRulerTasksAll:
    @pytest.mark.parametrize("task_cls", [
        SingleNIAH, MultiKeyNIAH, MultiValueNIAH, MultiQueryNIAH,
        VariableTracing, ChainFollowing, CommonWordCount, FrequentWord
    ])
    def test_task_lifecycle(self, task_cls):
        config = TaskConfig(context_length=500, num_samples=2)
        task = task_cls(config)
        
        # Test plural sample generation
        samples = task.generate_samples(2)
        assert len(samples) == 2
        assert isinstance(samples[0], TaskSample)
        
        # Test evaluation
        sample = samples[0]
        is_correct, score = task.evaluate_response(sample.expected_answer, sample.expected_answer)
        assert is_correct
        assert score == 1.0
        
        # Test failure evaluation
        is_correct, score = task.evaluate_response("totally wrong", sample.expected_answer)
        assert score < 0.5

    def test_get_task(self):
        for name in RULER_TASKS:
            task = get_task(name)
            assert task is not None
        
        with pytest.raises(ValueError):
            get_task("nonexistent")

    def test_get_all_tasks(self):
        tasks = get_all_tasks()
        assert len(tasks) == len(RULER_TASKS)

class TestTaskCategories:
    def test_categories(self):
        assert TaskCategory.RETRIEVAL.value == "retrieval"
        assert TaskCategory.MULTI_HOP.value == "multi_hop"
        assert TaskCategory.AGGREGATION.value == "aggregation"

class TestContextLengthScaling:
    def test_noise(self):
        config = TaskConfig(context_length=1000)
        task = SingleNIAH(config)
        noise = task._generate_noise(500)
        assert len(noise) >= 500
