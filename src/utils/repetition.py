"""
repetition.py
Core utility for Prompt Repetition (arXiv:2512.14982).
Handles text-based repetition styles and adaptive repetition routing.
"""

from typing import Literal, Union, Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)

RepetitionStyle = Literal["baseline", "2x", "verbose", "3x"]


class TaskComplexity(Enum):
    """Task complexity levels for adaptive repetition."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class TaskType(Enum):
    """Task types for repetition routing."""
    Q_AND_A = "q_and_a"
    RETRIEVAL = "retrieval"
    REASONING = "reasoning"
    CREATIVE = "creative"
    CODE = "code"
    SUMMARIZATION = "summarization"


@dataclass
class RepetitionConfig:
    """Configuration for adaptive repetition."""
    task_type: TaskType
    complexity: TaskComplexity
    repetition_factor: int
    confidence_threshold: float = 0.7


class TaskComplexityAnalyzer:
    """
    Analyzes task complexity to determine appropriate repetition level.
    """
    
    # Complexity indicators
    COMPLEX_KEYWORDS = [
        "explain", "analyze", "compare", "contrast", "evaluate",
        "synthesize", "critique", "justify", "investigate", "derive"
    ]
    
    RETRIEVAL_KEYWORDS = [
        "find", "retrieve", "search", "locate", "extract",
        "look up", "fetch", "get information about", "what is the"
    ]
    
    SIMPLE_KEYWORDS = [
        "what is", "who is", "when", "where", "define",
        "list", "name", "yes or no", "true or false"
    ]
    
    CODE_INDICATORS = [
        "code", "function", "algorithm", "implement", "write a program",
        "debug", "python", "javascript", "java", "c++"
    ]
    
    @classmethod
    def analyze(cls, query: str, context: str = "") -> TaskComplexity:
        """
        Analyze the complexity of a task.
        
        Args:
            query: The main query/instruction
            context: Additional context
            
        Returns:
            TaskComplexity enum value
        """
        combined_text = f"{query} {context}".lower()
        
        # Count complexity indicators
        complex_score = sum(1 for kw in cls.COMPLEX_KEYWORDS if kw in combined_text)
        retrieval_score = sum(1 for kw in cls.RETRIEVAL_KEYWORDS if kw in combined_text)
        simple_score = sum(1 for kw in cls.SIMPLE_KEYWORDS if kw in combined_text)
        
        # Check for multiple questions
        question_count = combined_text.count("?")
        
        # Check length and structure
        word_count = len(combined_text.split())
        has_multiple_parts = bool(re.search(r'\n\s*\d+\.', combined_text))
        
        # Determine complexity
        if retrieval_score > 0 or complex_score >= 2 or word_count > 100 or has_multiple_parts:
            return TaskComplexity.COMPLEX
        elif complex_score == 1 or question_count > 1 or word_count > 50:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE
    
    @classmethod
    def detect_task_type(cls, query: str, context: str = "") -> TaskType:
        """
        Detect the type of task.
        
        Args:
            query: The main query
            context: Additional context
            
        Returns:
            TaskType enum value
        """
        combined_text = f"{query} {context}".lower()
        
        # Check for code tasks
        if any(indicator in combined_text for indicator in cls.CODE_INDICATORS):
            return TaskType.CODE
        
        # Check for retrieval tasks
        if any(kw in combined_text for kw in cls.RETRIEVAL_KEYWORDS):
            return TaskType.RETRIEVAL
        
        # Check for creative tasks
        creative_keywords = ["write", "create", "generate", "compose", "draft"]
        if any(kw in combined_text for kw in creative_keywords):
            return TaskType.CREATIVE
        
        # Check for reasoning tasks
        reasoning_keywords = ["explain", "analyze", "why", "how does", "reason"]
        if any(kw in combined_text for kw in reasoning_keywords):
            return TaskType.REASONING
        
        # Check for summarization
        summary_keywords = ["summarize", "summary", "tl;dr", "brief overview"]
        if any(kw in combined_text for kw in summary_keywords):
            return TaskType.SUMMARIZATION
        
        # Default to Q&A
        return TaskType.Q_AND_A


class AdaptiveRepetitionRouter:
    """
    Router that decides repetition level based on task characteristics.
    Based on paper 2512.14982 recommendations.
    
    - 3x repetition for complex retrieval tasks
    - Baseline for simple Q&A
    - Adaptive based on task complexity
    """
    
    # Default routing rules
    DEFAULT_RULES = {
        # Complex retrieval tasks benefit most from 3x repetition
        (TaskType.RETRIEVAL, TaskComplexity.COMPLEX): 3,
        (TaskType.RETRIEVAL, TaskComplexity.MODERATE): 2,
        (TaskType.RETRIEVAL, TaskComplexity.SIMPLE): 2,
        
        # Reasoning tasks benefit from 2x repetition
        (TaskType.REASONING, TaskComplexity.COMPLEX): 3,
        (TaskType.REASONING, TaskComplexity.MODERATE): 2,
        (TaskType.REASONING, TaskComplexity.SIMPLE): 1,
        
        # Code tasks benefit from 2x repetition
        (TaskType.CODE, TaskComplexity.COMPLEX): 3,
        (TaskType.CODE, TaskComplexity.MODERATE): 2,
        (TaskType.CODE, TaskComplexity.SIMPLE): 2,
        
        # Simple Q&A uses baseline
        (TaskType.Q_AND_A, TaskComplexity.SIMPLE): 1,
        (TaskType.Q_AND_A, TaskComplexity.MODERATE): 1,
        (TaskType.Q_AND_A, TaskComplexity.COMPLEX): 2,
        
        # Creative writing
        (TaskType.CREATIVE, TaskComplexity.COMPLEX): 2,
        (TaskType.CREATIVE, TaskComplexity.MODERATE): 1,
        (TaskType.CREATIVE, TaskComplexity.SIMPLE): 1,
        
        # Summarization
        (TaskType.SUMMARIZATION, TaskComplexity.COMPLEX): 2,
        (TaskType.SUMMARIZATION, TaskComplexity.MODERATE): 2,
        (TaskType.SUMMARIZATION, TaskComplexity.SIMPLE): 1,
    }
    
    def __init__(self, custom_rules: Optional[Dict] = None):
        """
        Initialize router with optional custom rules.
        
        Args:
            custom_rules: Dictionary mapping (TaskType, TaskComplexity) to repetition factor
        """
        self.rules = custom_rules or self.DEFAULT_RULES.copy()
        self.analyzer = TaskComplexityAnalyzer()
    
    def route(self, query: str, context: str = "") -> RepetitionConfig:
        """
        Determine repetition configuration for a query.
        
        Args:
            query: The main query/instruction
            context: Additional context
            
        Returns:
            RepetitionConfig with recommended settings
        """
        # Analyze task
        task_type = self.analyzer.detect_task_type(query, context)
        complexity = self.analyzer.analyze(query, context)
        
        # Look up repetition factor
        key = (task_type, complexity)
        repetition_factor = self.rules.get(key, 1)
        
        logger.debug(f"Task: {task_type.value}, Complexity: {complexity.value}, Factor: {repetition_factor}")
        
        return RepetitionConfig(
            task_type=task_type,
            complexity=complexity,
            repetition_factor=repetition_factor
        )
    
    def get_repetition_factor(self, query: str, context: str = "") -> int:
        """Get just the repetition factor for a query."""
        config = self.route(query, context)
        return config.repetition_factor
    
    def add_custom_rule(self, task_type: TaskType, complexity: TaskComplexity, factor: int):
        """Add or update a routing rule."""
        self.rules[(task_type, complexity)] = factor
        logger.info(f"Added rule: {task_type.value} + {complexity.value} = {factor}x")
    
    def get_routing_report(self, queries: List[str]) -> Dict[str, Any]:
        """Generate a report on routing decisions for a set of queries."""
        results = []
        factor_counts = {1: 0, 2: 0, 3: 0}
        
        for query in queries:
            config = self.route(query)
            factor_counts[config.repetition_factor] += 1
            results.append({
                "query": query[:100] + "..." if len(query) > 100 else query,
                "task_type": config.task_type.value,
                "complexity": config.complexity.value,
                "repetition_factor": config.repetition_factor
            })
        
        return {
            "total_queries": len(queries),
            "factor_distribution": factor_counts,
            "results": results
        }


class PromptRepetitionEngine:
    """
    Enhanced prompt repetition engine with adaptive routing.
    """
    
    def __init__(self, use_adaptive_routing: bool = True):
        """
        Initialize the repetition engine.
        
        Args:
            use_adaptive_routing: Whether to use adaptive repetition routing
        """
        self.use_adaptive_routing = use_adaptive_routing
        self.router = AdaptiveRepetitionRouter() if use_adaptive_routing else None
    
    @staticmethod
    def apply_repetition(
        query: str, 
        context: str = "", 
        factor: int = 1,
        style: Union[RepetitionStyle, str] = "baseline"
    ) -> str:
        """
        Apply repetition to a prompt based on the specified factor or style.
        
        Args:
            query: The main question/instruction.
            context: The context or data (optional).
            factor: Integer repetition factor (1, 2, 3). Overrides style if > 1.
            style: One of 'baseline', '2x', 'verbose', '3x'.
            
        Returns:
            Formatted string with repetition applied.
        """
        full_query = f"{context}\n{query}" if context else query
        
        # Override style if factor is explicitly set to a multi-repetition value
        if factor == 2:
            style = "2x"
        elif factor == 3:
            style = "3x"
        
        # If explicitly told to use baseline or factor is 1 and style is baseline, return as is
        if style == "baseline" and factor <= 1:
            return full_query
            
        # Apply styles
        if style == "2x":
            return f"{full_query} Let me repeat that: {full_query}"
        elif style == "verbose":
            return f"{full_query} Let me repeat that: {full_query}"
        elif style == "3x":
            return (f"{full_query} Let me repeat that: {full_query} "
                   f"Let me repeat that one more time: {full_query}")
        
        return full_query
    
    def apply_adaptive_repetition(
        self,
        query: str,
        context: str = "",
        force_factor: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Apply repetition with adaptive routing.
        
        Args:
            query: The main query
            context: Additional context
            force_factor: Optional forced repetition factor (overrides routing)
            
        Returns:
            Dictionary with repeated text and metadata
        """
        # Determine repetition factor
        if force_factor is not None:
            factor = force_factor
            task_type = TaskType.Q_AND_A
            complexity = TaskComplexity.SIMPLE
        elif self.use_adaptive_routing and self.router:
            config = self.router.route(query, context)
            factor = config.repetition_factor
            task_type = config.task_type
            complexity = config.complexity
        else:
            factor = 1
            task_type = TaskType.Q_AND_A
            complexity = TaskComplexity.SIMPLE
        
        # Apply repetition
        repeated_text = self.apply_repetition(query, context, factor)
        
        return {
            "text": repeated_text,
            "repetition_factor": factor,
            "task_type": task_type.value,
            "task_complexity": complexity.value,
            "routing_applied": self.use_adaptive_routing and force_factor is None
        }
    
    def batch_apply(self, 
                   queries: List[Dict[str, str]],
                   use_adaptive: bool = True) -> List[Dict[str, Any]]:
        """
        Apply repetition to a batch of queries.
        
        Args:
            queries: List of dicts with 'query' and optional 'context' keys
            use_adaptive: Whether to use adaptive routing
            
        Returns:
            List of results with repetition applied
        """
        results = []
        for item in queries:
            query = item.get("query", "")
            context = item.get("context", "")
            
            if use_adaptive and self.use_adaptive_routing:
                result = self.apply_adaptive_repetition(query, context)
            else:
                result = {
                    "text": self.apply_repetition(query, context, factor=1),
                    "repetition_factor": 1,
                    "task_type": "unknown",
                    "task_complexity": "unknown",
                    "routing_applied": False
                }
            
            results.append(result)
        
        return results


# Convenience functions for direct use
def apply_repetition(query: str, context: str = "", factor: int = 1) -> str:
    """Static function to apply repetition."""
    return PromptRepetitionEngine.apply_repetition(query, context, factor)


def apply_adaptive(query: str, context: str = "") -> Dict[str, Any]:
    """Static function to apply adaptive repetition."""
    engine = PromptRepetitionEngine(use_adaptive_routing=True)
    return engine.apply_adaptive_repetition(query, context)


def get_repetition_factor(query: str, context: str = "") -> int:
    """Get the recommended repetition factor for a query."""
    router = AdaptiveRepetitionRouter()
    return router.get_repetition_factor(query, context)


# Example usage and testing
if __name__ == "__main__":
    # Test adaptive routing
    test_queries = [
        "What is 2+2?",  # Simple Q&A - should use baseline
        "Find the research papers about machine learning published after 2020",  # Complex retrieval - 3x
        "Explain the theory of relativity in detail",  # Complex reasoning - 3x
        "Write a Python function to calculate fibonacci",  # Code - 2x
        "Who is the president of the United States?",  # Simple Q&A - baseline
    ]
    
    engine = PromptRepetitionEngine(use_adaptive_routing=True)
    
    print("Adaptive Repetition Routing Examples:")
    print("=" * 60)
    
    for query in test_queries:
        result = engine.apply_adaptive_repetition(query)
        print(f"\nQuery: {query}")
        print(f"  Task Type: {result['task_type']}")
        print(f"  Complexity: {result['task_complexity']}")
        print(f"  Repetition Factor: {result['repetition_factor']}")
        print(f"  Routing Applied: {result['routing_applied']}")
        print(f"  Result Length: {len(result['text'])} chars")
