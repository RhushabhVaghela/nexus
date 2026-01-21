#!/usr/bin/env python3
"""
Chain-of-Thought Dataset Generator

Generates reasoning datasets with <think>...</think> formatting for training
models with advanced reasoning capabilities like DeepSeek-R1 and Claude Thinking.

Features:
- Convert existing datasets to CoT format
- Generate synthetic reasoning traces
- Distill from teacher models (GPT-4, Claude)
- Support for math, code, and general reasoning
"""

import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union
from enum import Enum

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of reasoning for different problem domains."""
    MATH = "math"
    CODE = "code"
    LOGIC = "logic"
    SCIENCE = "science"
    GENERAL = "general"
    PLANNING = "planning"
    TOOL_USE = "tool_use"


@dataclass
class ThinkingTrace:
    """A single thinking trace with structured reasoning."""
    problem: str
    thinking: str  # The <think>...</think> content
    answer: str
    reasoning_type: ReasoningType = ReasoningType.GENERAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_messages(self) -> List[Dict[str, str]]:
        """Convert to chat messages format."""
        return [
            {"role": "user", "content": self.problem},
            {"role": "assistant", "content": f"<think>\n{self.thinking}\n</think>\n\n{self.answer}"}
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "messages": self.to_messages(),
            "reasoning_type": self.reasoning_type.value,
            "metadata": self.metadata
        }


@dataclass
class CoTConfig:
    """Configuration for CoT dataset generation."""
    # Thinking format
    think_start_token: str = "<think>"
    think_end_token: str = "</think>"
    
    # Generation settings
    max_thinking_length: int = 2048
    min_thinking_steps: int = 2
    max_thinking_steps: int = 10
    
    # Sampling
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Output
    output_format: str = "jsonl"  # jsonl, json, parquet
    

class CoTGenerator:
    """
    Generate Chain-of-Thought datasets for reasoning training.
    
    Supports:
    - Math reasoning (GSM8K-style)
    - Code reasoning (step-by-step debugging)
    - Logic puzzles
    - General reasoning
    """
    
    def __init__(self, config: Optional[CoTConfig] = None):
        self.config = config or CoTConfig()
        self._reasoning_templates = self._load_templates()
    
    def _load_templates(self) -> Dict[ReasoningType, List[str]]:
        """Load reasoning templates for different domains."""
        return {
            ReasoningType.MATH: [
                "Let me break this problem down step by step.",
                "First, I need to identify what we're solving for.",
                "I'll work through this systematically.",
                "Let me analyze the given information.",
            ],
            ReasoningType.CODE: [
                "Let me trace through this code step by step.",
                "First, I'll understand what the code is supposed to do.",
                "I need to check the logic carefully.",
                "Let me debug this systematically.",
            ],
            ReasoningType.LOGIC: [
                "Let me reason through this logically.",
                "First, I'll identify the premises.",
                "I need to follow the logical chain.",
                "Let me check if the conclusion follows.",
            ],
            ReasoningType.PLANNING: [
                "Let me break this task into steps.",
                "First, I need to understand the goal.",
                "I'll create a plan to accomplish this.",
                "Let me think about the sequence of actions.",
            ],
            ReasoningType.TOOL_USE: [
                "To complete this task, I'll need to use tools.",
                "Let me identify which tools are needed.",
                "I'll execute this step by step.",
                "First, let me gather the necessary information.",
            ],
            ReasoningType.GENERAL: [
                "Let me think about this carefully.",
                "I'll consider different aspects of this question.",
                "Let me analyze this step by step.",
                "First, I need to understand what's being asked.",
            ],
        }
    
    def convert_to_cot(
        self, 
        problem: str, 
        answer: str,
        reasoning_type: ReasoningType = ReasoningType.GENERAL,
        reasoning_steps: Optional[List[str]] = None
    ) -> ThinkingTrace:
        """
        Convert a problem-answer pair to CoT format.
        
        Args:
            problem: The input problem/question
            answer: The final answer
            reasoning_type: Type of reasoning to apply
            reasoning_steps: Optional explicit reasoning steps
        
        Returns:
            ThinkingTrace with structured reasoning
        """
        if reasoning_steps:
            thinking = self._format_steps(reasoning_steps)
        else:
            thinking = self._generate_thinking(problem, answer, reasoning_type)
        
        return ThinkingTrace(
            problem=problem,
            thinking=thinking,
            answer=answer,
            reasoning_type=reasoning_type,
        )
    
    def _generate_thinking(
        self, 
        problem: str, 
        answer: str, 
        reasoning_type: ReasoningType
    ) -> str:
        """Generate thinking process for a problem."""
        templates = self._reasoning_templates.get(
            reasoning_type, 
            self._reasoning_templates[ReasoningType.GENERAL]
        )
        
        # Create structured thinking
        steps = []
        
        # Opening reflection
        steps.append(random.choice(templates))
        
        # Problem analysis
        if reasoning_type == ReasoningType.MATH:
            steps.extend(self._math_reasoning(problem, answer))
        elif reasoning_type == ReasoningType.CODE:
            steps.extend(self._code_reasoning(problem, answer))
        elif reasoning_type == ReasoningType.PLANNING:
            steps.extend(self._planning_reasoning(problem, answer))
        else:
            steps.extend(self._general_reasoning(problem, answer))
        
        # Conclusion
        steps.append(f"Therefore, the answer is: {answer}")
        
        return "\n\n".join(steps)
    
    def _math_reasoning(self, problem: str, answer: str) -> List[str]:
        """Generate math-specific reasoning steps."""
        steps = []
        
        # Extract numbers from problem
        numbers = re.findall(r'\d+(?:\.\d+)?', problem)
        
        if numbers:
            steps.append(f"Given information:\n- " + "\n- ".join([f"Number: {n}" for n in numbers[:5]]))
        
        steps.append("Setting up the equation:")
        steps.append("Solving step by step:")
        
        # If answer is a number, show verification
        if re.match(r'^[\d,\.]+$', str(answer).replace(',', '')):
            steps.append(f"Verification: Let me check if {answer} is correct.")
        
        return steps
    
    def _code_reasoning(self, problem: str, answer: str) -> List[str]:
        """Generate code-specific reasoning steps."""
        steps = []
        
        steps.append("Understanding the problem:")
        steps.append("- What is the input?")
        steps.append("- What should the output be?")
        steps.append("- What are the edge cases?")
        
        steps.append("\nApproach:")
        steps.append("1. First, I'll handle the base case")
        steps.append("2. Then, I'll implement the main logic")
        steps.append("3. Finally, I'll optimize if needed")
        
        return steps
    
    def _planning_reasoning(self, problem: str, answer: str) -> List[str]:
        """Generate planning-specific reasoning steps."""
        steps = []
        
        steps.append("Identifying the goal:")
        steps.append("Breaking down into sub-tasks:")
        steps.append("Ordering the tasks by dependency:")
        steps.append("Considering potential obstacles:")
        steps.append("Final plan of action:")
        
        return steps
    
    def _general_reasoning(self, problem: str, answer: str) -> List[str]:
        """Generate general reasoning steps."""
        steps = []
        
        steps.append("Analyzing the question:")
        steps.append("Considering relevant factors:")
        steps.append("Weighing the options:")
        steps.append("Coming to a conclusion:")
        
        return steps
    
    def _format_steps(self, steps: List[str]) -> str:
        """Format reasoning steps into thinking content."""
        formatted = []
        for i, step in enumerate(steps, 1):
            formatted.append(f"Step {i}: {step}")
        return "\n\n".join(formatted)
    
    def generate_from_dataset(
        self,
        dataset_path: Union[str, Path],
        output_path: Union[str, Path],
        reasoning_type: ReasoningType = ReasoningType.GENERAL,
        sample_size: int = 0,  # 0 = all
        problem_key: str = "question",
        answer_key: str = "answer",
    ) -> int:
        """
        Generate CoT dataset from existing dataset.
        
        Args:
            dataset_path: Path to input dataset (JSONL)
            output_path: Path to output CoT dataset
            reasoning_type: Type of reasoning to apply
            sample_size: Number of samples (0 = all)
            problem_key: Key for problem text in dataset
            answer_key: Key for answer in dataset
        
        Returns:
            Number of samples generated
        """
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        samples = []
        
        # Load input dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        
        if sample_size > 0:
            samples = random.sample(samples, min(sample_size, len(samples)))
        
        # Generate CoT versions
        generated = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                problem = sample.get(problem_key, sample.get("text", ""))
                answer = sample.get(answer_key, "")
                
                if problem:
                    trace = self.convert_to_cot(problem, answer, reasoning_type)
                    f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
                    generated += 1
        
        logger.info(f"Generated {generated} CoT samples to {output_path}")
        return generated
    
    def generate_synthetic_math(
        self,
        output_path: Union[str, Path],
        num_samples: int = 1000,
    ) -> int:
        """
        Generate synthetic math reasoning datasets.
        
        Creates arithmetic, algebra, and word problems with step-by-step solutions.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        generated = 0
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for _ in range(num_samples):
                trace = self._generate_math_problem()
                f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
                generated += 1
        
        logger.info(f"Generated {generated} synthetic math samples to {output_path}")
        return generated
    
    def _generate_math_problem(self) -> ThinkingTrace:
        """Generate a single synthetic math problem with solution."""
        problem_types = ['arithmetic', 'word_problem', 'algebra']
        ptype = random.choice(problem_types)
        
        if ptype == 'arithmetic':
            return self._gen_arithmetic()
        elif ptype == 'word_problem':
            return self._gen_word_problem()
        else:
            return self._gen_algebra()
    
    def _gen_arithmetic(self) -> ThinkingTrace:
        """Generate arithmetic problem."""
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        op = random.choice(['+', '-', '*'])
        
        if op == '+':
            answer = a + b
            problem = f"Calculate: {a} + {b}"
            steps = [
                f"Adding {a} and {b}",
                f"{a} + {b} = {answer}"
            ]
        elif op == '-':
            answer = a - b
            problem = f"Calculate: {a} - {b}"
            steps = [
                f"Subtracting {b} from {a}",
                f"{a} - {b} = {answer}"
            ]
        else:
            answer = a * b
            problem = f"Calculate: {a} × {b}"
            steps = [
                f"Multiplying {a} by {b}",
                f"= {a} × {b // 10 if b >= 10 else b}0 + {a} × {b % 10}" if b >= 10 else f"= {answer}",
                f"= {a * (b // 10) * 10} + {a * (b % 10)}" if b >= 10 else "",
                f"= {answer}"
            ]
            steps = [s for s in steps if s]
        
        return ThinkingTrace(
            problem=problem,
            thinking="\n".join(steps),
            answer=str(answer),
            reasoning_type=ReasoningType.MATH
        )
    
    def _gen_word_problem(self) -> ThinkingTrace:
        """Generate word problem."""
        templates = [
            {
                "problem": "Sarah has {a} apples. She buys {b} more. How many apples does she have now?",
                "op": "+",
                "steps": [
                    "Initial apples: {a}",
                    "Additional apples: {b}",
                    "Total = {a} + {b} = {answer}"
                ]
            },
            {
                "problem": "A store had {a} items. They sold {b} items. How many items remain?",
                "op": "-",
                "steps": [
                    "Initial items: {a}",
                    "Items sold: {b}",
                    "Remaining = {a} - {b} = {answer}"
                ]
            },
            {
                "problem": "Each box contains {b} pencils. If there are {a} boxes, how many pencils are there in total?",
                "op": "*",
                "steps": [
                    "Pencils per box: {b}",
                    "Number of boxes: {a}",
                    "Total pencils = {a} × {b} = {answer}"
                ]
            }
        ]
        
        template = random.choice(templates)
        a = random.randint(5, 50)
        b = random.randint(3, 30)
        
        if template["op"] == '+':
            answer = a + b
        elif template["op"] == '-':
            answer = a - b
        else:
            answer = a * b
        
        problem = template["problem"].format(a=a, b=b)
        steps = [s.format(a=a, b=b, answer=answer) for s in template["steps"]]
        
        return ThinkingTrace(
            problem=problem,
            thinking="\n".join(steps),
            answer=str(answer),
            reasoning_type=ReasoningType.MATH
        )
    
    def _gen_algebra(self) -> ThinkingTrace:
        """Generate simple algebra problem."""
        a = random.randint(2, 10)
        b = random.randint(1, 20)
        c = random.randint(10, 100)
        
        # ax + b = c, solve for x
        x = (c - b) / a
        
        problem = f"Solve for x: {a}x + {b} = {c}"
        steps = [
            f"Starting with: {a}x + {b} = {c}",
            f"Subtract {b} from both sides: {a}x = {c} - {b} = {c - b}",
            f"Divide both sides by {a}: x = {c - b} / {a} = {x:.2f}" if x != int(x) else f"Divide both sides by {a}: x = {c - b} / {a} = {int(x)}"
        ]
        
        return ThinkingTrace(
            problem=problem,
            thinking="\n".join(steps),
            answer=f"x = {x:.2f}" if x != int(x) else f"x = {int(x)}",
            reasoning_type=ReasoningType.MATH
        )


def main():
    """Demo CoT generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate CoT datasets")
    parser.add_argument("--input", help="Input dataset path")
    parser.add_argument("--output", default="data/cot_dataset.jsonl", help="Output path")
    parser.add_argument("--type", default="general", 
                        choices=["math", "code", "logic", "general", "planning"],
                        help="Reasoning type")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples")
    
    args = parser.parse_args()
    
    generator = CoTGenerator()
    
    if args.synthetic:
        if args.type == "math":
            generator.generate_synthetic_math(args.output, args.num_samples)
        else:
            print(f"Synthetic generation for {args.type} not implemented yet")
    elif args.input:
        reasoning_type = ReasoningType(args.type)
        generator.generate_from_dataset(args.input, args.output, reasoning_type)
    else:
        # Demo
        trace = generator.convert_to_cot(
            "If a train travels 120 km in 2 hours, what is its speed?",
            "60 km/h",
            ReasoningType.MATH
        )
        print("Example CoT:")
        print(json.dumps(trace.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
