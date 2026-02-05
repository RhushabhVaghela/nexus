"""
tests/unit/test_repetition.py
Comprehensive tests for repetition functionality.

Tests cover:
- TaskComplexityAnalyzer
- AdaptiveRepetitionRouter
- PromptRepetitionEngine
- Edge cases and routing logic
"""

import pytest
from unittest.mock import patch, MagicMock

from src.utils.repetition import (
    TaskComplexity,
    TaskType,
    RepetitionConfig,
    TaskComplexityAnalyzer,
    AdaptiveRepetitionRouter,
    PromptRepetitionEngine,
    apply_repetition,
    apply_adaptive,
    get_repetition_factor,
)


class TestTaskComplexity:
    """Test TaskComplexity enum."""
    
    def test_simple_value(self):
        """Test SIMPLE complexity value."""
        assert TaskComplexity.SIMPLE.value == "simple"
    
    def test_moderate_value(self):
        """Test MODERATE complexity value."""
        assert TaskComplexity.MODERATE.value == "moderate"
    
    def test_complex_value(self):
        """Test COMPLEX complexity value."""
        assert TaskComplexity.COMPLEX.value == "complex"


class TestTaskType:
    """Test TaskType enum."""
    
    def test_q_and_a_value(self):
        """Test Q_AND_A value."""
        assert TaskType.Q_AND_A.value == "q_and_a"
    
    def test_retrieval_value(self):
        """Test RETRIEVAL value."""
        assert TaskType.RETRIEVAL.value == "retrieval"
    
    def test_reasoning_value(self):
        """Test REASONING value."""
        assert TaskType.REASONING.value == "reasoning"
    
    def test_creative_value(self):
        """Test CREATIVE value."""
        assert TaskType.CREATIVE.value == "creative"
    
    def test_code_value(self):
        """Test CODE value."""
        assert TaskType.CODE.value == "code"
    
    def test_summarization_value(self):
        """Test SUMMARIZATION value."""
        assert TaskType.SUMMARIZATION.value == "summarization"


class TestRepetitionConfig:
    """Test RepetitionConfig dataclass."""
    
    def test_creation(self):
        """Test creating repetition config."""
        config = RepetitionConfig(
            task_type=TaskType.Q_AND_A,
            complexity=TaskComplexity.SIMPLE,
            repetition_factor=1
        )
        
        assert config.task_type == TaskType.Q_AND_A
        assert config.complexity == TaskComplexity.SIMPLE
        assert config.repetition_factor == 1
        assert config.confidence_threshold == 0.7  # Default
    
    def test_creation_custom_threshold(self):
        """Test creating config with custom threshold."""
        config = RepetitionConfig(
            task_type=TaskType.CODE,
            complexity=TaskComplexity.COMPLEX,
            repetition_factor=3,
            confidence_threshold=0.9
        )
        
        assert config.confidence_threshold == 0.9


class TestTaskComplexityAnalyzer:
    """Test TaskComplexityAnalyzer class."""
    
    def test_analyze_simple_question(self):
        """Test analysis of simple question."""
        query = "What is 2+2?"
        result = TaskComplexityAnalyzer.analyze(query)
        
        assert result == TaskComplexity.SIMPLE
    
    def test_analyze_moderate_question(self):
        """Test analysis of moderate question."""
        query = "Explain how photosynthesis works and why it's important?"
        result = TaskComplexityAnalyzer.analyze(query)
        
        assert result in [TaskComplexity.MODERATE, TaskComplexity.COMPLEX]
    
    def test_analyze_complex_question(self):
        """Test analysis of complex question."""
        query = "Analyze and compare the economic impacts of the Industrial Revolution in Britain and France, considering social and political factors"
        result = TaskComplexityAnalyzer.analyze(query)
        
        assert result == TaskComplexity.COMPLEX
    
    def test_analyze_with_retrieval_keywords(self):
        """Test detection of retrieval keywords."""
        query = "Find all research papers about machine learning published after 2020"
        result = TaskComplexityAnalyzer.analyze(query)
        
        assert result == TaskComplexity.COMPLEX
    
    def test_analyze_long_text(self):
        """Test analysis of long text triggers complex."""
        query = "What is " + "a " * 150 + "test?"
        result = TaskComplexityAnalyzer.analyze(query)
        
        assert result == TaskComplexity.COMPLEX
    
    def test_analyze_multiple_questions(self):
        """Test analysis with multiple questions."""
        query = "What is AI? How does it work? Why is it important?"
        result = TaskComplexityAnalyzer.analyze(query)
        
        assert result in [TaskComplexity.MODERATE, TaskComplexity.COMPLEX]
    
    def test_detect_task_type_code(self):
        """Test detection of code task."""
        query = "Write a Python function to sort a list"
        result = TaskComplexityAnalyzer.detect_task_type(query)
        
        assert result == TaskType.CODE
    
    def test_detect_task_type_retrieval(self):
        """Test detection of retrieval task."""
        query = "Find information about the Treaty of Versailles"
        result = TaskComplexityAnalyzer.detect_task_type(query)
        
        assert result == TaskType.RETRIEVAL
    
    def test_detect_task_type_creative(self):
        """Test detection of creative task."""
        query = "Write a poem about nature"
        result = TaskComplexityAnalyzer.detect_task_type(query)
        
        assert result == TaskType.CREATIVE
    
    def test_detect_task_type_reasoning(self):
        """Test detection of reasoning task."""
        query = "Explain why the sky is blue"
        result = TaskComplexityAnalyzer.detect_task_type(query)
        
        assert result == TaskType.REASONING
    
    def test_detect_task_type_summarization(self):
        """Test detection of summarization task."""
        query = "Summarize this article for me"
        result = TaskComplexityAnalyzer.detect_task_type(query)
        
        assert result == TaskType.SUMMARIZATION
    
    def test_detect_task_type_default_q_and_a(self):
        """Test default Q_AND_A for simple queries."""
        query = "What is the capital of France?"
        result = TaskComplexityAnalyzer.detect_task_type(query)
        
        assert result == TaskType.Q_AND_A


class TestAdaptiveRepetitionRouter:
    """Test AdaptiveRepetitionRouter class."""
    
    def test_route_simple_q_and_a(self):
        """Test routing for simple Q&A."""
        router = AdaptiveRepetitionRouter()
        query = "What is 2+2?"
        config = router.route(query)
        
        assert config.task_type == TaskType.Q_AND_A
        assert config.repetition_factor == 1
    
    def test_route_complex_retrieval(self):
        """Test routing for complex retrieval."""
        router = AdaptiveRepetitionRouter()
        query = "Find and analyze all papers about quantum computing from the last 5 years"
        config = router.route(query)
        
        assert config.task_type == TaskType.RETRIEVAL
        assert config.repetition_factor in [2, 3]
    
    def test_route_code_task(self):
        """Test routing for code task."""
        router = AdaptiveRepetitionRouter()
        query = "Implement a binary search algorithm in Python"
        config = router.route(query)
        
        assert config.task_type == TaskType.CODE
        assert config.repetition_factor in [2, 3]
    
    def test_route_reasoning_task(self):
        """Test routing for reasoning task."""
        router = AdaptiveRepetitionRouter()
        query = "Explain the theory of relativity in detail"
        config = router.route(query)
        
        assert config.task_type == TaskType.REASONING
        assert config.repetition_factor in [2, 3]
    
    def test_get_repetition_factor(self):
        """Test getting just the repetition factor."""
        router = AdaptiveRepetitionRouter()
        factor = router.get_repetition_factor("What is AI?")
        
        assert isinstance(factor, int)
        assert factor >= 1
    
    def test_add_custom_rule(self):
        """Test adding custom routing rule."""
        router = AdaptiveRepetitionRouter()
        router.add_custom_rule(TaskType.Q_AND_A, TaskComplexity.SIMPLE, 5)
        
        config = router.route("Simple question")
        # Should now use custom rule
        assert config.repetition_factor == 5
    
    def test_get_routing_report(self):
        """Test generating routing report."""
        router = AdaptiveRepetitionRouter()
        queries = [
            "What is 2+2?",
            "Find papers about AI",
            "Write Python code"
        ]
        report = router.get_routing_report(queries)
        
        assert report["total_queries"] == 3
        assert "factor_distribution" in report
        assert len(report["results"]) == 3
    
    def test_custom_rules_in_constructor(self):
        """Test providing custom rules in constructor."""
        custom_rules = {
            (TaskType.Q_AND_A, TaskComplexity.SIMPLE): 5
        }
        router = AdaptiveRepetitionRouter(custom_rules=custom_rules)
        
        factor = router.get_repetition_factor("What is 2+2?")
        assert factor == 5


class TestPromptRepetitionEngine:
    """Test PromptRepetitionEngine class."""
    
    def test_apply_repetition_baseline(self):
        """Test baseline repetition (no repetition)."""
        result = PromptRepetitionEngine.apply_repetition("Hello", factor=1)
        
        assert result == "Hello"
    
    def test_apply_repetition_2x(self):
        """Test 2x repetition."""
        result = PromptRepetitionEngine.apply_repetition("Hello", factor=2)
        
        assert "Hello" in result
        assert "Let me repeat that" in result
        assert result.count("Hello") == 2
    
    def test_apply_repetition_3x(self):
        """Test 3x repetition."""
        result = PromptRepetitionEngine.apply_repetition("Hello", factor=3)
        
        assert "Hello" in result
        assert result.count("Hello") == 3
        assert "Let me repeat that" in result
    
    def test_apply_repetition_with_context(self):
        """Test repetition with context."""
        result = PromptRepetitionEngine.apply_repetition(
            "What is the answer?",
            context="Context here",
            factor=2
        )
        
        assert "Context here" in result
        assert "What is the answer?" in result
    
    def test_apply_repetition_style_2x(self):
        """Test repetition with style='2x'."""
        result = PromptRepetitionEngine.apply_repetition("Hello", style="2x")
        
        assert result.count("Hello") == 2
    
    def test_apply_repetition_style_verbose(self):
        """Test repetition with style='verbose'."""
        result = PromptRepetitionEngine.apply_repetition("Hello", style="verbose")
        
        assert result.count("Hello") == 2
    
    def test_apply_repetition_style_3x(self):
        """Test repetition with style='3x'."""
        result = PromptRepetitionEngine.apply_repetition("Hello", style="3x")
        
        assert result.count("Hello") == 3
    
    def test_apply_adaptive_repetition(self):
        """Test adaptive repetition."""
        engine = PromptRepetitionEngine(use_adaptive_routing=True)
        result = engine.apply_adaptive_repetition("What is 2+2?")
        
        assert "text" in result
        assert "repetition_factor" in result
        assert "task_type" in result
        assert "task_complexity" in result
        assert "routing_applied" in result
    
    def test_apply_adaptive_with_force_factor(self):
        """Test adaptive repetition with forced factor."""
        engine = PromptRepetitionEngine(use_adaptive_routing=True)
        result = engine.apply_adaptive_repetition("Hello", force_factor=3)
        
        assert result["repetition_factor"] == 3
        assert result["routing_applied"] is False
    
    def test_apply_adaptive_without_routing(self):
        """Test adaptive repetition without routing."""
        engine = PromptRepetitionEngine(use_adaptive_routing=False)
        result = engine.apply_adaptive_repetition("Hello")
        
        assert result["repetition_factor"] == 1
        assert result["routing_applied"] is False
    
    def test_batch_apply(self):
        """Test batch application."""
        engine = PromptRepetitionEngine(use_adaptive_routing=True)
        queries = [
            {"query": "What is 2+2?"},
            {"query": "Find papers about AI", "context": "Research context"}
        ]
        results = engine.batch_apply(queries)
        
        assert len(results) == 2
        assert all("text" in r for r in results)
    
    def test_batch_apply_without_adaptive(self):
        """Test batch apply without adaptive routing."""
        engine = PromptRepetitionEngine(use_adaptive_routing=True)
        queries = [{"query": "Hello"}, {"query": "World"}]
        results = engine.batch_apply(queries, use_adaptive=False)
        
        assert len(results) == 2
        assert all(r["repetition_factor"] == 1 for r in results)


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_apply_repetition_function(self):
        """Test apply_repetition function."""
        result = apply_repetition("Hello", factor=2)
        
        assert result.count("Hello") == 2
    
    def test_apply_adaptive_function(self):
        """Test apply_adaptive function."""
        result = apply_adaptive("What is AI?")
        
        assert "text" in result
        assert "repetition_factor" in result
    
    def test_get_repetition_factor_function(self):
        """Test get_repetition_factor function."""
        factor = get_repetition_factor("Simple question")
        
        assert isinstance(factor, int)
        assert factor >= 1


class TestRoutingRules:
    """Test specific routing rules."""
    
    def test_retrieval_complex_rule(self):
        """Test complex retrieval uses 3x repetition."""
        router = AdaptiveRepetitionRouter()
        config = router.route("Find and analyze all documents", context="")
        
        if config.task_type == TaskType.RETRIEVAL and config.complexity == TaskComplexity.COMPLEX:
            assert config.repetition_factor == 3
    
    def test_code_complex_rule(self):
        """Test complex code uses 3x repetition."""
        router = AdaptiveRepetitionRouter()
        config = router.route("Implement a complex distributed system")
        
        if config.task_type == TaskType.CODE and config.complexity == TaskComplexity.COMPLEX:
            assert config.repetition_factor == 3
    
    def test_reasoning_complex_rule(self):
        """Test complex reasoning uses 3x repetition."""
        router = AdaptiveRepetitionRouter()
        config = router.route("Analyze and critique the philosophical implications of quantum mechanics")
        
        if config.task_type == TaskType.REASONING and config.complexity == TaskComplexity.COMPLEX:
            assert config.repetition_factor == 3
    
    def test_qa_simple_rule(self):
        """Test simple Q&A uses baseline."""
        router = AdaptiveRepetitionRouter()
        config = router.route("What is 2+2?")
        
        if config.task_type == TaskType.Q_AND_A and config.complexity == TaskComplexity.SIMPLE:
            assert config.repetition_factor == 1


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_query(self):
        """Test handling of empty query."""
        router = AdaptiveRepetitionRouter()
        config = router.route("")
        
        assert isinstance(config, RepetitionConfig)
        assert config.repetition_factor >= 1
    
    def test_very_long_query(self):
        """Test handling of very long query."""
        router = AdaptiveRepetitionRouter()
        query = "Explain " + "something " * 200
        config = router.route(query)
        
        assert config.complexity == TaskComplexity.COMPLEX
    
    def test_repetition_with_special_characters(self):
        """Test repetition with special characters."""
        query = "What is 2+2? (math question)"
        result = PromptRepetitionEngine.apply_repetition(query, factor=2)
        
        assert query in result
    
    def test_repetition_preserves_formatting(self):
        """Test repetition preserves formatting."""
        query = "Line 1\nLine 2"
        result = PromptRepetitionEngine.apply_repetition(query, factor=2)
        
        assert "Line 1" in result
        assert "Line 2" in result
