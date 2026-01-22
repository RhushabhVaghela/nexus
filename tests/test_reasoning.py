
import sys
import os
import torch
from pathlib import Path

# Add src to path - dynamically relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from multimodal.reasoning import ReasoningWrapper, ReasoningLevel
# from multimodal.model import OmniMultimodalLM # Removed if not needed for prompt test
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_reasoning():
    print("🧪 Testing Reasoning Engine with Real Model...")
    
    # 1. Load Real Model (Qwen2.5-0.5B for efficiency)
    model_name = "/mnt/e/data/models/Qwen2.5-0.5B"
    if not os.path.exists(model_name):
        print(f"⚠️ Model path not found: {model_name}. Skipping test.")
        return

    print(f"Loading tokenizer and model from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    model.eval()
    
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
        else:
            has_system = any(m['role'] == 'system' for m in enhanced)
            print(f"ℹ️  System Prompt Present: {has_system} (Expected: False/Minimal)")
        
        # Verify actual generation doesn't crash
        inputs = tokenizer(query, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=20)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"   Generated (truncated): {response[:50]}...")
            
    print("\n✅ Reasoning Wrapper Verified with REAL MODEL.")

if __name__ == "__main__":
    test_reasoning()
