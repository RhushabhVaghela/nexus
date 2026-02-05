#!/usr/bin/env python3
"""
Unit tests for the GRPO Reward Functions module.
"""

import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reasoning.reward_functions import (
    RewardConfig, RewardResult, RewardType,
    CorrectnessReward, FormatReward, LengthReward,
    ConsistencyReward, ProcessReward, CombinedReward,
    create_reward_function
)


class TestRewardConfig:
    """Tests for RewardConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = RewardConfig()
        
        assert config.correctness_weight == 0.4
        assert config.format_weight == 0.2
        assert config.think_start_token == "<think>"
        assert config.think_end_token == "</think>"
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = RewardConfig(
            correctness_weight=0.6,
            format_weight=0.4,
            min_thinking_length=100
        )
        
        assert config.correctness_weight == 0.6
        assert config.min_thinking_length == 100


class TestCorrectnessReward:
    """Tests for CorrectnessReward."""
    
    def test_exact_numeric_match(self):
        """Test exact numeric answer match."""
        reward_fn = CorrectnessReward()
        
        result = reward_fn.compute(
            response="<think>Working...</think>\n\nThe answer is 42.",
            reference="42"
        )
        
        assert result.reward == 1.0
        assert result.reward_type == RewardType.CORRECTNESS
    
    def test_approximate_numeric_match(self):
        """Test approximate numeric match (within 1%)."""
        reward_fn = CorrectnessReward()
        
        result = reward_fn.compute(
            response="The answer is 99.5",
            reference="100"
        )
        
        assert result.reward >= 0.8
    
    def test_incorrect_answer(self):
        """Test completely wrong answer."""
        reward_fn = CorrectnessReward()
        
        result = reward_fn.compute(
            response="The answer is 1000",
            reference="42"
        )
        
        assert result.reward < 0.5
    
    def test_string_exact_match(self):
        """Test exact string match."""
        reward_fn = CorrectnessReward()
        
        result = reward_fn.compute(
            response="Paris",
            reference="Paris",
            answer_type="string"
        )
        
        assert result.reward == 1.0
    
    def test_string_contains_match(self):
        """Test string contains match."""
        reward_fn = CorrectnessReward()
        
        result = reward_fn.compute(
            response="The capital of France is Paris.",
            reference="Paris",
            answer_type="string"
        )
        
        assert result.reward >= 0.7


class TestFormatReward:
    """Tests for FormatReward."""
    
    def test_proper_thinking_format(self):
        """Test response with proper thinking format."""
        reward_fn = FormatReward()
        
        result = reward_fn.compute(
            response="<think>\nStep 1: Analyze\nStep 2: Solve\nTherefore, the answer is X.\n</think>\n\nThe answer is X."
        )
        
        assert result.reward >= 0.6
        assert result.details.get("thinking_tags") == "present"
    
    def test_missing_thinking_tags(self):
        """Test response without thinking tags."""
        reward_fn = FormatReward()
        
        result = reward_fn.compute(
            response="The answer is 42."
        )
        
        assert result.reward < 0.5
        assert result.details.get("thinking_tags") == "missing"
    
    def test_wrong_tag_order(self):
        """Test response with wrong tag order."""
        reward_fn = FormatReward()
        
        result = reward_fn.compute(
            response="</think>Some text<think>"
        )
        
        assert result.details.get("tag_order") == "incorrect"


class TestLengthReward:
    """Tests for LengthReward."""
    
    def test_optimal_length(self):
        """Test response with optimal thinking length."""
        reward_fn = LengthReward()
        
        # Generate content around optimal length (500 chars)
        thinking = "Step 1: " + "x" * 450 + "\nStep 2: Result"
        response = f"<think>\n{thinking}\n</think>\n\nAnswer"
        
        result = reward_fn.compute(response)
        
        assert result.reward >= 0.7
    
    def test_too_short(self):
        """Test response with too short thinking."""
        config = RewardConfig(min_thinking_length=100)
        reward_fn = LengthReward(config)
        
        result = reward_fn.compute("<think>Short</think>\n\nAnswer")
        
        assert result.reward < 0.5
        assert result.details.get("status") == "too_short"
    
    def test_too_long(self):
        """Test response with too long thinking."""
        config = RewardConfig(max_thinking_length=100)
        reward_fn = LengthReward(config)
        
        long_thinking = "<think>" + "x" * 500 + "</think>\n\nAnswer"
        result = reward_fn.compute(long_thinking)
        
        assert result.details.get("status") == "too_long"


class TestConsistencyReward:
    """Tests for ConsistencyReward."""
    
    def test_self_verification(self):
        """Test response with self-verification."""
        reward_fn = ConsistencyReward()
        
        result = reward_fn.compute(
            response="<think>\nLet me solve this.\n2 + 2 = 4\nLet me verify: 4 is correct!\n</think>\n\n4"
        )
        
        assert result.details.get("self_verification") == True
        assert result.reward >= 0.3
    
    def test_logical_connectors(self):
        """Test response with logical connectors."""
        reward_fn = ConsistencyReward()
        
        result = reward_fn.compute(
            response="<think>\nBecause X, therefore Y. Thus, the answer is Z.\n</think>\n\nZ"
        )
        
        assert result.details.get("logical_connectors") >= 2


class TestProcessReward:
    """Tests for ProcessReward."""
    
    def test_multiple_steps(self):
        """Test response with multiple reasoning steps."""
        reward_fn = ProcessReward()
        
        result = reward_fn.compute(
            response="<think>\n1. First step\n2. Second step\n3. Third step\n</think>\n\nAnswer"
        )
        
        assert result.details.get("steps_found") >= 3
        assert result.reward >= 0.5


class TestCombinedReward:
    """Tests for CombinedReward."""
    
    def test_combined_reward(self):
        """Test combined reward computation."""
        reward_fn = CombinedReward()
        
        response = """<think>
Let me solve this step by step.

Step 1: Identify the problem - we need to add 23 and 17.
Step 2: Perform the calculation: 23 + 17 = 40.
Step 3: Let me verify: 23 + 17 = 40. Correct!

Therefore, the answer is 40.
</think>

The answer is 40."""
        
        result = reward_fn.compute(
            response=response,
            reference="40",
            problem="What is 23 + 17?"
        )
        
        assert result.reward_type == RewardType.COMBINED
        assert "correctness" in result.details
        assert "format" in result.details
        assert "consistency" in result.details
        assert result.reward >= 0.5


class TestCreateRewardFunction:
    """Tests for create_reward_function factory."""
    
    def test_create_combined(self):
        """Test creating combined reward function."""
        reward_fn = create_reward_function("combined")
        assert isinstance(reward_fn, CombinedReward)
    
    def test_create_correctness(self):
        """Test creating correctness reward function."""
        reward_fn = create_reward_function("correctness")
        assert isinstance(reward_fn, CorrectnessReward)
    
    def test_create_with_config(self):
        """Test creating with custom config."""
        config = RewardConfig(correctness_weight=0.8)
        reward_fn = create_reward_function("combined", config)
        assert reward_fn.config.correctness_weight == 0.8
    
    def test_invalid_type(self):
        """Test invalid reward type raises error."""
        with pytest.raises(ValueError):
            create_reward_function("invalid_type")
