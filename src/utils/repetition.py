"""
repetition.py
Core utility for Prompt Repetition (arXiv:2512.14982).
Handles text-based repetition styles.
"""

from typing import Literal, Union

RepetitionStyle = Literal["baseline", "2x", "verbose", "3x"]

class PromptRepetitionEngine:
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
            Formatted string.
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
            return f"{full_query} {full_query}"
        elif style == "verbose":
            return f"{full_query} Let me repeat that: {full_query}"
        elif style == "3x":
            return f"{full_query} Let me repeat that: {full_query} Let me repeat that one more time: {full_query}"
        
        return full_query
