#!/usr/bin/env python3
"""
Unit tests for the CoT Generator module.
"""

import json
import pytest
import tempfile
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reasoning.cot_generator import (
    CoTGenerator, CoTConfig, ThinkingTrace, ReasoningType
)


class TestThinkingTrace:
    """Tests for ThinkingTrace dataclass."""
    
    def test_create_thinking_trace(self):
        """Test creating a ThinkingTrace."""
        trace = ThinkingTrace(
            problem="What is 2 + 2?",
            thinking="Let me add: 2 + 2 = 4",
            answer="4",
            reasoning_type=ReasoningType.MATH
        )
        assert trace.problem == "What is 2 + 2?"
        assert trace.answer == "4"
        assert trace.reasoning_type == ReasoningType.MATH
    
    def test_to_messages(self):
        """Test converting trace to chat messages."""
        trace = ThinkingTrace(
            problem="Test problem",
            thinking="Step 1\nStep 2",
            answer="Test answer"
        )
        messages = trace.to_messages()
        
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Test problem"
        assert messages[1]["role"] == "assistant"
        assert "<think>" in messages[1]["content"]
        assert "</think>" in messages[1]["content"]
    
    def test_to_dict(self):
        """Test converting trace to dictionary."""
        trace = ThinkingTrace(
            problem="Problem",
            thinking="Thinking",
            answer="Answer",
            reasoning_type=ReasoningType.CODE
        )
        result = trace.to_dict()
        
        assert "messages" in result
        assert result["reasoning_type"] == "code"


class TestCoTConfig:
    """Tests for CoTConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CoTConfig()
        
        assert config.think_start_token == "<think>"
        assert config.think_end_token == "</think>"
        assert config.max_thinking_length == 2048
        assert config.output_format == "jsonl"
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = CoTConfig(
            think_start_token="<reasoning>",
            think_end_token="</reasoning>",
            max_thinking_length=4096
        )
        
        assert config.think_start_token == "<reasoning>"
        assert config.max_thinking_length == 4096


class TestCoTGenerator:
    """Tests for CoTGenerator."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = CoTGenerator()
        assert generator.config is not None
        assert generator._reasoning_templates is not None
    
    def test_generator_with_custom_config(self):
        """Test generator with custom config."""
        config = CoTConfig(max_thinking_length=1024)
        generator = CoTGenerator(config)
        assert generator.config.max_thinking_length == 1024
    
    def test_convert_to_cot_math(self):
        """Test converting a math problem to CoT format."""
        generator = CoTGenerator()
        
        trace = generator.convert_to_cot(
            problem="Calculate 15 + 27",
            answer="42",
            reasoning_type=ReasoningType.MATH
        )
        
        assert trace.problem == "Calculate 15 + 27"
        assert trace.answer == "42"
        assert trace.reasoning_type == ReasoningType.MATH
        assert len(trace.thinking) > 0
    
    def test_convert_to_cot_code(self):
        """Test converting a code problem to CoT format."""
        generator = CoTGenerator()
        
        trace = generator.convert_to_cot(
            problem="Write a function to add two numbers",
            answer="def add(a, b): return a + b",
            reasoning_type=ReasoningType.CODE
        )
        
        assert trace.reasoning_type == ReasoningType.CODE
        assert "input" in trace.thinking.lower() or "output" in trace.thinking.lower()
    
    def test_convert_to_cot_with_explicit_steps(self):
        """Test converting with explicit reasoning steps."""
        generator = CoTGenerator()
        
        steps = [
            "First, identify the numbers",
            "Then, add them together",
            "The result is 10"
        ]
        
        trace = generator.convert_to_cot(
            problem="5 + 5 = ?",
            answer="10",
            reasoning_steps=steps
        )
        
        assert "Step 1" in trace.thinking
        assert "Step 2" in trace.thinking
    
    def test_generate_synthetic_math(self, tmp_path):
        """Test generating synthetic math problems."""
        generator = CoTGenerator()
        output_path = tmp_path / "synthetic_math.jsonl"
        
        count = generator.generate_synthetic_math(output_path, num_samples=10)
        
        assert count == 10
        assert output_path.exists()
        
        # Verify format
        with open(output_path) as f:
            lines = f.readlines()
        
        assert len(lines) == 10
        sample = json.loads(lines[0])
        assert "messages" in sample
    
    def test_generate_from_dataset(self, tmp_path):
        """Test generating CoT from existing dataset."""
        # Create input dataset
        input_path = tmp_path / "input.jsonl"
        with open(input_path, 'w') as f:
            f.write(json.dumps({"question": "What is 1+1?", "answer": "2"}) + "\n")
            f.write(json.dumps({"question": "What is 2+2?", "answer": "4"}) + "\n")
        
        output_path = tmp_path / "output.jsonl"
        generator = CoTGenerator()
        
        count = generator.generate_from_dataset(
            input_path,
            output_path,
            reasoning_type=ReasoningType.MATH,
            problem_key="question",
            answer_key="answer"
        )
        
        assert count == 2
        assert output_path.exists()


class TestReasoningType:
    """Tests for ReasoningType enum."""
    
    def test_reasoning_types_exist(self):
        """Test all reasoning types are defined."""
        assert ReasoningType.MATH.value == "math"
        assert ReasoningType.CODE.value == "code"
        assert ReasoningType.LOGIC.value == "logic"
        assert ReasoningType.PLANNING.value == "planning"
        assert ReasoningType.TOOL_USE.value == "tool_use"
        assert ReasoningType.GENERAL.value == "general"
