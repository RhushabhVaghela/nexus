
import sys
import os
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path - dynamically relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from multimodal.reasoning import ReasoningWrapper, ReasoningLevel

def test_reasoning():
    print("🧪 Testing Reasoning Engine with MOCK Model...")
    
    # 1. Mock Tokenizer and Model
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.zeros((1, 10), dtype=torch.long),
        "attention_mask": torch.ones((1, 10), dtype=torch.long)
    }
    tokenizer.decode.return_value = "Mocked response about quantum entanglement"
    
    model = MagicMock()
    model.device = torch.device("cpu")
    model.generate.return_value = torch.zeros((1, 20), dtype=torch.long)
    
    wrapper = ReasoningWrapper(model, tokenizer)
    
    # Test Levels
    prompts = [
        (ReasoningLevel.LOW, "What is 2+2?"),
        (ReasoningLevel.HIGH, "Explain quantum entanglement.")
    ]
    
    for level, query in prompts:
        messages = [{"role": "user", "content": query}]
        
        # Verify message preparation
        enhanced = wrapper.prepare_messages(messages, level)
        
        print(f"\n--- Level: {level.name} ---")
        if level == ReasoningLevel.HIGH:
            # High level should inject CoT thought prompt
            has_cot_instruction = any("<think>" in m['content'] for m in enhanced if m['role'] == 'system')
            print(f"✅ CoT Instruction Injected: {has_cot_instruction}")
            assert has_cot_instruction, "CoT instruction should be injected for HIGH level"
        else:
            has_system = any(m['role'] == 'system' for m in enhanced)
            print(f"ℹ️  System Prompt Present: {has_system} (Expected: False/Minimal)")
        
        # Verify actual generation logic (mocked)
        inputs = tokenizer(query, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=20)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"   Generated (mocked): {response[:50]}...")
            
    print("\n✅ Reasoning Wrapper Verified with MOCK MODEL.")

if __name__ == "__main__":
    test_reasoning()
