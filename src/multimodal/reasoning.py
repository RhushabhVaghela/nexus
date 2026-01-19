import enum
from typing import List, Dict, Optional, Union
import torch
import logging

logger = logging.getLogger(__name__)

class ReasoningLevel(enum.Enum):
    LOW = "low"       # Direct answer, concise
    MEDIUM = "medium" # Brief thinking process (<think>...</think>)
    HIGH = "high"     # Deep detailed analysis, step-by-step validation

REASONING_PROMPTS = {
    ReasoningLevel.LOW: "",
    ReasoningLevel.MEDIUM: (
        "You are a helpful assistant. Before answering, briefly think about the problem "
        "inside <think> tags to organize your thoughts."
    ),
    ReasoningLevel.HIGH: (
        "You are an expert AI assistant. You MUST think deeply before answering.\n"
        "1. Analyze the user's request carefully.\n"
        "2. Break down the problem into steps inside <think> tags.\n"
        "3. Consider edge cases and verify your logic.\n"
        "4. Provide the final answer after the closing </think> tag."
    )
}

class ReasoningWrapper:
    """
    Wraps a model's generation process to inject reasoning capabilities via System Prompts.
    Does NOT require a separate fine-tuned model, but leverages the base model's instruction following.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def prepare_messages(
        self, 
        messages: List[Dict[str, str]], 
        level: Union[str, ReasoningLevel] = ReasoningLevel.LOW
    ) -> List[Dict[str, str]]:
        """
        Injects the appropriate system prompt for the requested reasoning level.
        """
        if isinstance(level, str):
            level = ReasoningLevel(level.lower())
            
        if level == ReasoningLevel.LOW:
            return messages

        system_prompt = REASONING_PROMPTS[level]
        
        # Check if system message already exists
        if messages and messages[0]["role"] == "system":
            # Append reasoning instruction to existing system prompt
            messages[0]["content"] += f"\n\n[Reasoning Instruction]\n{system_prompt}"
            return messages
        else:
            # Prepend new system message
            return [{"role": "system", "content": system_prompt}] + messages

    def generate(
        self, 
        messages: List[Dict[str, str]], 
        level: Union[str, ReasoningLevel] = ReasoningLevel.LOW,
        **kwargs
    ):
        """
        Generates a response with the specified reasoning level.
        """
        enhanced_messages = self.prepare_messages(messages, level)
        
        # Convert to model input format (assumes model has apply_chat_template or similar)
        # For OmniMultimodalLM, we likely pass the messages to its generation method
        # or we format them here if it expects raw text/tokens.
        
        # NOTE: This assumes the wrapped model has a valid chat template.
        # If using the 'OmniMultimodalLM' class we built, we might need to handle tokenization here.
        
        text_input = self.tokenizer.apply_chat_template(
            enhanced_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text_input, return_tensors="pt").to(self.model.device)
        
        # Forward minimal kwargs suitable for Qwen generation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **kwargs
            )
            
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-processing: If we want to return just the answer or keep thoughts, depends on UX.
        # For now, we return the full raw text (including <think> tags).
        return decoded
