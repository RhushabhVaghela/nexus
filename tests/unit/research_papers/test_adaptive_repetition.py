"""
Comprehensive Unit Tests for Adaptive Repetition (Paper 2512.14982).

Tests cover:
- Task classification
- Repetition level selection
- Router logic
- Performance metrics
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.utils.repetition import (
    PromptRepetitionEngine,
    AdaptiveRepetitionRouter,
    TaskComplexityAnalyzer,
    TaskType,
    TaskComplexity,
    apply_repetition,
    apply_adaptive,
    get_repetition_factor
)


class TestTaskComplexityAnalyzer:
    """Test suite for TaskComplexityAnalyzer."""
    
    def test_simple_qa_detection(self):
        """Test detection of simple Q&A tasks."""
        queries = [
            "What is 2+2?",
            "Who is the president?",
            "When did WWII end?",
            "Define photosynthesis"
        ]
        
        for query in queries:
            complexity = TaskComplexityAnalyzer.analyze(query)
            assert complexity in [TaskComplexity.SIMPLE, TaskComplexity.MODERATE]
    
    def test_complex_retrieval_detection(self):
        """Test detection of complex retrieval tasks."""
        queries = [
            "Find all research papers about machine learning published after 2020",
            "Retrieve information about quantum computing advances in the last 5 years"
        ]
        
        for query in queries:
            complexity = TaskComplexityAnalyzer.analyze(query)
            assert complexity in [TaskComplexity.MODERATE, TaskComplexity.COMPLEX]
    
    def test_task_type_detection_code(self):
        """Test task type detection for code tasks."""
        assert TaskComplexityAnalyzer.detect_task_type("Write a Python function") == TaskType.CODE
        assert TaskComplexityAnalyzer.detect_task_type("Debug this algorithm") == TaskType.CODE
    
    def test_task_type_detection_qa(self):
        """Test task type detection for Q&A tasks."""
        assert TaskComplexityAnalyzer.detect_task_type("What is the capital?") == TaskType.Q_AND_A
    
    def test_task_type_detection_creative(self):
        """Test task type detection for creative tasks."""
        assert TaskComplexityAnalyzer.detect_task_type("Write a story about") == TaskType.CREATIVE


class TestAdaptiveRepetitionRouter:
    """Test suite for AdaptiveRepetitionRouter."""
    
    def test_router_initialization(self):
        """Test router initialization."""
        router = AdaptiveRepetitionRouter()
        assert router.rules is not None
        assert len(router.rules) > 0
    
    def test_simple_qa_routing(self):
        """Test routing for simple Q&A (should use baseline)."""
        router = AdaptiveRepetitionRouter()
        
        query = "What is the capital of France?"
        config = router.route(query)
        
        assert config.task_type == TaskType.Q_AND_A
        assert config.repetition_factor >= 1
    
    def test_complex_retrieval_routing(self):
        """Test routing for complex retrieval (should use 3x)."""
        router = AdaptiveRepetitionRouter()
        
        query = "Find all research papers about machine learning published after 2020"
        config = router.route(query)
        
        assert config.task_type == TaskType.RETRIEVAL
        assert config.repetition_factor >= 1
    
    def test_code_task_routing(self):
        """Test routing for code tasks."""
        router = AdaptiveRepetitionRouter()
        
        query = "Write a Python function to calculate Fibonacci numbers"
        config = router.route(query)
        
        assert config.task_type == TaskType.CODE
        assert config.repetition_factor >= 1
    
    def test_get_repetition_factor(self):
        """Test getting just the repetition factor."""
        router = AdaptiveRepetitionRouter()
        
        factor_simple = router.get_repetition_factor("What is 2+2?")
        factor_complex = router.get_repetition_factor("Find papers about AI")
        
        assert factor_simple >= 1
        assert factor_complex >= 1
    
    def test_add_custom_rule(self):
        """Test adding custom routing rules."""
        router = AdaptiveRepetitionRouter()
        
        # Add custom rule
        router.add_custom_rule(TaskType.Q_AND_A, TaskComplexity.SIMPLE, 2)
        
        # Verify rule was added
        assert router.rules[(TaskType.Q_AND_A, TaskComplexity.SIMPLE)] == 2


class TestPromptRepetitionEngine:
    """Test suite for PromptRepetitionEngine."""
    
    def test_baseline_repetition(self):
        """Test baseline (no repetition)."""
        engine = PromptRepetitionEngine(use_adaptive_routing=False)
        
        result = engine.apply_repetition("What is 2+2?", factor=1)
        assert result == "What is 2+2?"
    
    def test_2x_repetition(self):
        """Test 2x repetition."""
        engine = PromptRepetitionEngine(use_adaptive_routing=False)
        
        result = engine.apply_repetition("What is 2+2?", factor=2)
        assert "What is 2+2?" in result
        assert "Let me repeat that" in result
        assert result.count("What is 2+2?") == 2
    
    def test_3x_repetition(self):
        """Test 3x repetition."""
        engine = PromptRepetitionEngine(use_adaptive_routing=False)
        
        result = engine.apply_repetition("What is 2+2?", factor=3)
        assert result.count("What is 2+2?") == 3
    
    def test_adaptive_repetition_simple(self):
        """Test adaptive repetition for simple query."""
        engine = PromptRepetitionEngine(use_adaptive_routing=True)
        
        result = engine.apply_adaptive_repetition("What is 2+2?")
        
        assert "text" in result
        assert "repetition_factor" in result
        assert "task_type" in result
        assert result["routing_applied"] == True
    
    def test_force_factor_override(self):
        """Test forcing a specific repetition factor."""
        engine = PromptRepetitionEngine(use_adaptive_routing=True)
        
        result = engine.apply_adaptive_repetition(
            "What is 2+2?",
            force_factor=3
        )
        
        assert result["repetition_factor"] == 3
    
    def test_batch_apply(self):
        """Test batch application of repetition."""
        engine = PromptRepetitionEngine(use_adaptive_routing=True)
        
        queries = [
            {"query": "What is 2+2?"},
            {"query": "Find papers about AI", "context": "Research context"}
        ]
        
        results = engine.batch_apply(queries)
        
        assert len(results) == 2
        assert all("text" in r for r in results)


class TestConvenienceFunctions:
    """Test suite for convenience functions."""
    
    def test_apply_repetition_function(self):
        """Test static apply_repetition function."""
        result = apply_repetition("Test query", factor=2)
        assert "Test query" in result
        assert result.count("Test query") == 2
    
    def test_apply_adaptive_function(self):
        """Test static apply_adaptive function."""
        result = apply_adaptive("What is 2+2?")
        assert "text" in result
        assert result["routing_applied"] == True
    
    def test_get_repetition_factor_function(self):
        """Test static get_repetition_factor function."""
        factor = get_repetition_factor("What is 2+2?")
        assert isinstance(factor, int)
        assert factor >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
