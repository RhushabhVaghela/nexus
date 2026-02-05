#!/usr/bin/env python3
"""
GRPO Reward Functions for Reasoning Training

Implements reward functions for Group Relative Policy Optimization (GRPO)
training, following DeepSeek-R1's approach to emergent reasoning.

Reward Types:
- Correctness: Verify math/code answers
- Format: Check valid <think>...</think> structure
- Length: Penalize excessive verbosity
- Consistency: Check reasoning coherence
- Process: Reward intermediate reasoning steps
"""

import re
import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from enum import Enum
import subprocess
import tempfile
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class RewardType(Enum):
    """Types of rewards for GRPO training."""
    CORRECTNESS = "correctness"
    FORMAT = "format"
    LENGTH = "length"
    CONSISTENCY = "consistency"
    PROCESS = "process"
    COMBINED = "combined"


@dataclass
class RewardConfig:
    """Configuration for reward computation."""
    # Weights for combined reward
    correctness_weight: float = 0.4
    format_weight: float = 0.2
    length_weight: float = 0.1
    consistency_weight: float = 0.2
    process_weight: float = 0.1
    
    # Format settings
    think_start_token: str = "<think>"
    think_end_token: str = "</think>"
    required_sections: List[str] = field(default_factory=list)
    
    # Length settings
    min_thinking_length: int = 50
    max_thinking_length: int = 4096
    optimal_thinking_length: int = 500
    
    # Code execution settings
    code_timeout: int = 10  # seconds
    allow_code_execution: bool = True


@dataclass
class RewardResult:
    """Result of reward computation."""
    reward: float
    reward_type: RewardType
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"RewardResult(type={self.reward_type.value}, reward={self.reward:.4f})"


class RewardFunction(ABC):
    """
    Base class for reward functions.
    
    All reward functions should return a value in [-1, 1] range.
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
    
    @abstractmethod
    def compute(
        self, 
        response: str, 
        reference: Optional[str] = None,
        problem: Optional[str] = None,
        **kwargs
    ) -> RewardResult:
        """Compute reward for a response."""
        pass


class CorrectnessReward(RewardFunction):
    """
    Verify correctness of answers.
    
    Supports:
    - Numeric answers (exact and approximate)
    - Code execution verification
    - String matching
    """
    
    def compute(
        self, 
        response: str, 
        reference: Optional[str] = None,
        problem: Optional[str] = None,
        answer_type: str = "auto",
        **kwargs
    ) -> RewardResult:
        """
        Compute correctness reward.
        
        Args:
            response: Model response (may include thinking)
            reference: Expected answer
            problem: Original problem (for context)
            answer_type: "numeric", "code", "string", or "auto"
        
        Returns:
            RewardResult with correctness score
        """
        if reference is None:
            return RewardResult(0.0, RewardType.CORRECTNESS, {"reason": "No reference provided"})
        
        # Extract answer from response (after </think>)
        extracted_answer = self._extract_answer(response)
        
        # Auto-detect answer type
        if answer_type == "auto":
            answer_type = self._detect_answer_type(reference)
        
        # Compute correctness based on type
        if answer_type == "numeric":
            score, details = self._check_numeric(extracted_answer, reference)
        elif answer_type == "code":
            score, details = self._check_code(extracted_answer, reference, problem)
        else:
            score, details = self._check_string(extracted_answer, reference)
        
        return RewardResult(score, RewardType.CORRECTNESS, details)
    
    def _extract_answer(self, response: str) -> str:
        """Extract the final answer from response (after thinking)."""
        # Look for answer after </think>
        if self.config.think_end_token in response:
            parts = response.split(self.config.think_end_token)
            if len(parts) > 1:
                return parts[-1].strip()
        
        # No thinking tags, return last line
        lines = response.strip().split('\n')
        return lines[-1].strip()
    
    def _detect_answer_type(self, reference: str) -> str:
        """Detect the type of answer."""
        # Check if numeric
        cleaned = re.sub(r'[,\s\$%]', '', str(reference))
        try:
            float(cleaned)
            return "numeric"
        except ValueError:
            pass
        
        # Check if code
        if any(kw in str(reference).lower() for kw in ['def ', 'return ', 'class ', 'import ']):
            return "code"
        
        return "string"
    
    def _check_numeric(self, response: str, reference: str) -> Tuple[float, Dict]:
        """Check numeric answer correctness."""
        # Extract numbers from both
        resp_nums = re.findall(r'-?\d+(?:\.\d+)?', response.replace(',', ''))
        ref_nums = re.findall(r'-?\d+(?:\.\d+)?', reference.replace(',', ''))
        
        if not ref_nums:
            return 0.0, {"reason": "No reference number found"}
        
        if not resp_nums:
            return 0.0, {"reason": "No number in response"}
        
        ref_val = float(ref_nums[-1])
        resp_val = float(resp_nums[-1])
        
        # Exact match
        if abs(ref_val - resp_val) < 1e-6:
            return 1.0, {"match": "exact", "reference": ref_val, "response": resp_val}
        
        # Close match (within 1%)
        if ref_val != 0 and abs(ref_val - resp_val) / abs(ref_val) < 0.01:
            return 0.9, {"match": "approximate", "reference": ref_val, "response": resp_val}
        
        # Partial credit for order of magnitude
        if ref_val != 0 and resp_val != 0:
            ratio = resp_val / ref_val
            if 0.1 <= ratio <= 10:
                score = max(0, 1 - abs(math.log10(ratio)))
                return score * 0.5, {"match": "partial", "reference": ref_val, "response": resp_val}
        
        return 0.0, {"match": "incorrect", "reference": ref_val, "response": resp_val}
    
    def _check_code(self, response: str, reference: str, problem: Optional[str]) -> Tuple[float, Dict]:
        """Check code correctness via execution."""
        if not self.config.allow_code_execution:
            return self._check_string(response, reference)
        
        # Extract code blocks
        code_pattern = r'```(?:python)?\n?(.*?)```'
        resp_codes = re.findall(code_pattern, response, re.DOTALL)
        
        if not resp_codes:
            # Try to find code without blocks
            resp_codes = [response]
        
        # Try to execute and compare
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(resp_codes[-1])
                f.flush()
                
                result = subprocess.run(
                    ['python', f.name],
                    capture_output=True,
                    text=True,
                    timeout=self.config.code_timeout
                )
                
                Path(f.name).unlink()
                
                if result.returncode == 0:
                    output = result.stdout.strip()
                    if reference.strip() in output:
                        return 1.0, {"execution": "success", "output": output}
                    return 0.5, {"execution": "ran", "output": output, "expected": reference}
                
                return 0.0, {"execution": "error", "error": result.stderr}
                
        except subprocess.TimeoutExpired:
            return 0.0, {"execution": "timeout"}
        except Exception as e:
            return 0.0, {"execution": "exception", "error": str(e)}
    
    def _check_string(self, response: str, reference: str) -> Tuple[float, Dict]:
        """Check string answer correctness."""
        resp_clean = response.lower().strip()
        ref_clean = reference.lower().strip()
        
        if resp_clean == ref_clean:
            return 1.0, {"match": "exact"}
        
        if ref_clean in resp_clean:
            return 0.8, {"match": "contains"}
        
        # Jaccard similarity
        resp_words = set(resp_clean.split())
        ref_words = set(ref_clean.split())
        
        if resp_words and ref_words:
            intersection = len(resp_words & ref_words)
            union = len(resp_words | ref_words)
            similarity = intersection / union
            return similarity * 0.5, {"match": "partial", "similarity": similarity}
        
        return 0.0, {"match": "no_match"}


class FormatReward(RewardFunction):
    """
    Check valid thinking format structure.
    
    Rewards:
    - Proper <think>...</think> tags
    - Clear step separation
    - Logical structure
    """
    
    def compute(
        self, 
        response: str, 
        reference: Optional[str] = None,
        problem: Optional[str] = None,
        **kwargs
    ) -> RewardResult:
        """Compute format reward."""
        score = 0.0
        details = {}
        
        # Check for thinking tags
        has_start = self.config.think_start_token in response
        has_end = self.config.think_end_token in response
        
        if has_start and has_end:
            score += 0.4
            details["thinking_tags"] = "present"
            
            # Check proper order
            start_idx = response.index(self.config.think_start_token)
            end_idx = response.index(self.config.think_end_token)
            
            if start_idx < end_idx:
                score += 0.2
                details["tag_order"] = "correct"
            else:
                details["tag_order"] = "incorrect"
        else:
            details["thinking_tags"] = "missing"
        
        # Extract thinking content
        thinking = self._extract_thinking(response)
        
        if thinking:
            # Check for step-by-step structure
            step_patterns = [
                r'step\s*\d+',
                r'\d+\.',
                r'first|second|third|then|next|finally',
                r'let me|i will|i need to',
            ]
            
            step_count = 0
            for pattern in step_patterns:
                if re.search(pattern, thinking.lower()):
                    step_count += 1
            
            step_score = min(step_count / 3, 1.0) * 0.2
            score += step_score
            details["step_structure"] = step_count
            
            # Check for conclusion
            has_conclusion = any(kw in thinking.lower() for kw in [
                'therefore', 'so', 'thus', 'answer is', 'result is', 'conclusion'
            ])
            if has_conclusion:
                score += 0.1
                details["has_conclusion"] = True
            else:
                details["has_conclusion"] = False
            
            # Penalize empty thinking
            if len(thinking.strip()) < 20:
                score -= 0.3
                details["thinking_too_short"] = True
        else:
            details["thinking_content"] = "empty"
        
        # Check for answer after thinking
        if self.config.think_end_token in response:
            after_think = response.split(self.config.think_end_token)[-1].strip()
            if len(after_think) > 0:
                score += 0.1
                details["has_answer"] = True
            else:
                details["has_answer"] = False
        
        return RewardResult(max(0, min(1, score)), RewardType.FORMAT, details)
    
    def _extract_thinking(self, response: str) -> str:
        """Extract content between thinking tags."""
        pattern = f"{re.escape(self.config.think_start_token)}(.*?){re.escape(self.config.think_end_token)}"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1)
        return ""


class LengthReward(RewardFunction):
    """
    Penalize excessive verbosity while encouraging sufficient reasoning.
    
    Uses a bell curve centered on optimal length.
    """
    
    def compute(
        self, 
        response: str, 
        reference: Optional[str] = None,
        problem: Optional[str] = None,
        **kwargs
    ) -> RewardResult:
        """Compute length reward."""
        # Extract thinking content
        thinking = self._extract_thinking(response)
        length = len(thinking)
        
        details = {"thinking_length": length}
        
        # Too short
        if length < self.config.min_thinking_length:
            score = length / self.config.min_thinking_length * 0.5
            details["status"] = "too_short"
            return RewardResult(score, RewardType.LENGTH, details)
        
        # Too long
        if length > self.config.max_thinking_length:
            over_ratio = (length - self.config.max_thinking_length) / self.config.max_thinking_length
            score = max(0, 1 - over_ratio)
            details["status"] = "too_long"
            return RewardResult(score, RewardType.LENGTH, details)
        
        # Bell curve around optimal
        optimal = self.config.optimal_thinking_length
        distance = abs(length - optimal) / optimal
        score = math.exp(-0.5 * distance ** 2)
        
        details["status"] = "optimal" if distance < 0.3 else "acceptable"
        return RewardResult(score, RewardType.LENGTH, details)
    
    def _extract_thinking(self, response: str) -> str:
        """Extract content between thinking tags."""
        pattern = f"{re.escape(self.config.think_start_token)}(.*?){re.escape(self.config.think_end_token)}"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1)
        return response


class ConsistencyReward(RewardFunction):
    """
    Check reasoning coherence and self-consistency.
    
    Detects:
    - Contradictions
    - Logical flow
    - Self-verification
    """
    
    def compute(
        self, 
        response: str, 
        reference: Optional[str] = None,
        problem: Optional[str] = None,
        **kwargs
    ) -> RewardResult:
        """Compute consistency reward."""
        score = 0.0
        details = {}
        
        thinking = self._extract_thinking(response)
        
        # Check for self-verification patterns
        verification_patterns = [
            r"let me (check|verify|double-check)",
            r"(checking|verifying) (my|the) (answer|result|work)",
            r"(this|the answer) (is|seems) (correct|right)",
            r"substitut(e|ing) back",
            r"sanity check",
        ]
        
        has_verification = False
        for pattern in verification_patterns:
            if re.search(pattern, thinking.lower()):
                has_verification = True
                break
        
        if has_verification:
            score += 0.3
            details["self_verification"] = True
        else:
            details["self_verification"] = False
        
        # Check for contradiction patterns (negative reward)
        contradiction_patterns = [
            r"(wait|no),?\s*(that's|i was) (wrong|incorrect)",
            r"actually,?\s*i made a mistake",
            r"let me (redo|restart|try again)",
        ]
        
        has_contradiction = False
        for pattern in contradiction_patterns:
            if re.search(pattern, thinking.lower()):
                has_contradiction = True
                break
        
        if has_contradiction:
            # Slight penalty but not too much (fixing mistakes is good)
            score -= 0.1
            details["contradictions"] = True
        else:
            score += 0.2
            details["contradictions"] = False
        
        # Check for logical connectors (shows flow)
        connectors = [
            r'\btherefore\b', r'\bthus\b', r'\bso\b', r'\bhence\b',
            r'\bbecause\b', r'\bsince\b', r'\bgiven that\b',
            r'\bif\b.*\bthen\b', r'\bfirst\b.*\bthen\b',
        ]
        
        connector_count = sum(1 for p in connectors if re.search(p, thinking.lower()))
        flow_score = min(connector_count / 4, 1.0) * 0.3
        score += flow_score
        details["logical_connectors"] = connector_count
        
        # Check answer consistency with reasoning
        answer = self._extract_answer(response)
        if answer and thinking:
            # Simple check: answer value appears in reasoning
            answer_nums = re.findall(r'\d+', answer)
            if answer_nums:
                final_num = answer_nums[-1]
                if final_num in thinking:
                    score += 0.2
                    details["answer_in_reasoning"] = True
                else:
                    details["answer_in_reasoning"] = False
        
        return RewardResult(max(0, min(1, score)), RewardType.CONSISTENCY, details)
    
    def _extract_thinking(self, response: str) -> str:
        """Extract content between thinking tags."""
        pattern = f"{re.escape(self.config.think_start_token)}(.*?){re.escape(self.config.think_end_token)}"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1)
        return response
    
    def _extract_answer(self, response: str) -> str:
        """Extract the final answer from response."""
        if self.config.think_end_token in response:
            parts = response.split(self.config.think_end_token)
            if len(parts) > 1:
                return parts[-1].strip()
        lines = response.strip().split('\n')
        return lines[-1].strip()


class ProcessReward(RewardFunction):
    """
    Reward intermediate reasoning steps (process reward model approach).
    
    Used for step-level RLHF instead of just outcome-based rewards.
    """
    
    def compute(
        self, 
        response: str, 
        reference: Optional[str] = None,
        problem: Optional[str] = None,
        **kwargs
    ) -> RewardResult:
        """Compute process reward."""
        thinking = self._extract_thinking(response)
        
        if not thinking:
            return RewardResult(0.0, RewardType.PROCESS, {"reason": "No thinking content"})
        
        # Score each "step"
        steps = self._parse_steps(thinking)
        
        if not steps:
            return RewardResult(0.3, RewardType.PROCESS, {"steps_found": 0})
        
        step_scores = []
        step_details = []
        
        for i, step in enumerate(steps):
            score, detail = self._score_step(step, i, problem)
            step_scores.append(score)
            step_details.append(detail)
        
        avg_score = sum(step_scores) / len(step_scores)
        
        # Bonus for multiple good steps
        if len(steps) >= 3 and avg_score > 0.5:
            avg_score = min(1.0, avg_score * 1.1)
        
        return RewardResult(
            avg_score, 
            RewardType.PROCESS, 
            {"steps_found": len(steps), "step_scores": step_scores}
        )
    
    def _parse_steps(self, thinking: str) -> List[str]:
        """Parse thinking into individual steps."""
        # Try numbered steps first
        numbered = re.split(r'\n\s*\d+[.):]\s*', thinking)
        if len(numbered) > 2:
            return [s.strip() for s in numbered if s.strip()]
        
        # Try paragraph breaks
        paragraphs = thinking.split('\n\n')
        if len(paragraphs) > 1:
            return [p.strip() for p in paragraphs if p.strip()]
        
        # Single block
        sentences = re.split(r'[.!?]\s+', thinking)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _score_step(self, step: str, step_idx: int, problem: Optional[str]) -> Tuple[float, str]:
        """Score an individual reasoning step."""
        score = 0.5  # Base score
        
        # Length check
        if len(step) < 10:
            return 0.1, "too_short"
        if len(step) > 500:
            score -= 0.1
        
        # Contains reasoning indicators
        if any(kw in step.lower() for kw in ['because', 'therefore', 'so', 'thus', 'since']):
            score += 0.2
        
        # Contains math/calculation
        if re.search(r'\d+\s*[\+\-\*\/\=]\s*\d+', step):
            score += 0.2
        
        # Early steps should set up problem
        if step_idx == 0:
            if any(kw in step.lower() for kw in ['given', 'need to', 'find', 'solve', 'let']):
                score += 0.1
        
        return min(1.0, score), "scored"
    
    def _extract_thinking(self, response: str) -> str:
        """Extract content between thinking tags."""
        pattern = f"{re.escape(self.config.think_start_token)}(.*?){re.escape(self.config.think_end_token)}"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1)
        return response


class CombinedReward(RewardFunction):
    """
    Combine multiple reward functions with weighted averaging.
    
    This is the main reward function used for GRPO training.
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        super().__init__(config)
        self.correctness = CorrectnessReward(config)
        self.format = FormatReward(config)
        self.length = LengthReward(config)
        self.consistency = ConsistencyReward(config)
        self.process = ProcessReward(config)
    
    def compute(
        self, 
        response: str, 
        reference: Optional[str] = None,
        problem: Optional[str] = None,
        **kwargs
    ) -> RewardResult:
        """Compute combined reward from all reward functions."""
        # Compute individual rewards
        results = {
            "correctness": self.correctness.compute(response, reference, problem, **kwargs),
            "format": self.format.compute(response, reference, problem, **kwargs),
            "length": self.length.compute(response, reference, problem, **kwargs),
            "consistency": self.consistency.compute(response, reference, problem, **kwargs),
            "process": self.process.compute(response, reference, problem, **kwargs),
        }
        
        # Weighted combination
        combined_score = (
            self.config.correctness_weight * results["correctness"].reward +
            self.config.format_weight * results["format"].reward +
            self.config.length_weight * results["length"].reward +
            self.config.consistency_weight * results["consistency"].reward +
            self.config.process_weight * results["process"].reward
        )
        
        details = {
            name: {"reward": r.reward, "details": r.details}
            for name, r in results.items()
        }
        
        return RewardResult(combined_score, RewardType.COMBINED, details)


def create_reward_function(
    reward_type: str = "combined",
    config: Optional[RewardConfig] = None
) -> RewardFunction:
    """Factory function to create reward functions."""
    config = config or RewardConfig()
    
    reward_map = {
        "correctness": CorrectnessReward,
        "format": FormatReward,
        "length": LengthReward,
        "consistency": ConsistencyReward,
        "process": ProcessReward,
        "combined": CombinedReward,
    }
    
    if reward_type not in reward_map:
        raise ValueError(f"Unknown reward type: {reward_type}. Available: {list(reward_map.keys())}")
    
    return reward_map[reward_type](config)


def main():
    """Demo reward functions."""
    # Example response with thinking
    response = """<think>
Let me solve this step by step.

Step 1: I need to calculate 23 × 17.
Step 2: I can break this down: 23 × 17 = 23 × (10 + 7)
Step 3: = 230 + 161 = 391

Let me verify: 23 × 17... 20×17 = 340, 3×17 = 51, so 340 + 51 = 391. Correct!
</think>

The answer is 391.
"""
    
    # Create combined reward function
    reward_fn = create_reward_function("combined")
    
    # Compute reward
    result = reward_fn.compute(
        response=response,
        reference="391",
        problem="Calculate 23 × 17"
    )
    
    print(f"Combined Reward: {result.reward:.4f}")
    print("\nDetailed scores:")
    for name, data in result.details.items():
        print(f"  {name}: {data['reward']:.4f}")


if __name__ == "__main__":
    main()
