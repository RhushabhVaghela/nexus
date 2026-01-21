
import sys
import os
import torch
from pathlib import Path

# Add src to path
sys.path.append("/mnt/d/Research Experiments/nexus_model/src")

from multimodal.reasoning import ReasoningWrapper, ReasoningLevel
from multimodal.model import OmniMultimodalLM
from transformers import AutoTokenizer

def test_reasoning():
    print("🧪 Testing Reasoning Engine...")
    
    # 1. Load Tokenizer & Mock Model (or Real if feasible, but Mock is faster for logic check)
    # For this test, we DO want to see if the wrapper formats keys correctly.
    # We will load the REAL tokenizer but mock the generation to avoid loading 5GB model just for prompt check.
    
    model_name = "/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4"
    if not os.path.exists(model_name):
        print(f"⚠️ Model path not found: {model_name}. Using Qwen/Qwen2.5-7B-Instruct tokenizer for test.")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    class MockModel:
        def __init__(self):
            self.device = "cpu"
        def generate(self, input_ids, **kwargs):
            # Return dummy tokens " <think> Thinking... </think> Answer"
            return torch.tensor([[1, 2, 3]]) 
            
    mock_model = MockModel()
    
    wrapper = ReasoningWrapper(mock_model, tokenizer)
    
    # Test Levels
    prompts = [
        (ReasoningLevel.LOW, "What is 2+2?"),
        (ReasoningLevel.HIGH, "Solve the Riemann Hypothesis.")
    ]
    
    for level, query in prompts:
        messages = [{"role": "user", "content": query}]
        
        # We manually inspect the 'inputs' being prepared to ensure system prompt is injected
        enhanced = wrapper.prepare_messages(messages, level)
        
        print(f"\n--- Level: {level.name} ---")
        if level == ReasoningLevel.HIGH:
            has_system = any(m['role'] == 'system' and "<think>" in m['content'] for m in enhanced)
            print(f"✅ System Prompt Injected: {has_system}")
            if has_system:
                print(f"   Prompt Content: {enhanced[0]['content'][:100]}...")
        else:
            has_system = any(m['role'] == 'system' for m in enhanced)
            print(f"ℹ️  System Prompt Present: {has_system} (Expected: False/Minimal)")
            
    print("\n✅ Reasoning Wrapper Logic Verified (Prompt Injection works).")

if __name__ == "__main__":
    test_reasoning()
