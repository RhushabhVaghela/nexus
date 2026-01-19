
import sys
import torch
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from multimodal.model import OmniMultimodalLM
from multimodal.reasoning import ReasoningWrapper, ReasoningLevel
from multimodal.tools import get_default_executor
from transformers import AutoTokenizer

def audit_capabilities():
    print("🕵️ CAPABILITY AUDIT: Checking Agentic IQ...")
    print("-------------------------------------------")
    
    model_path = "/mnt/e/data/base-model/Qwen2.5-Omni-7B-GPTQ-Int4"
    
    # 1. Load Model
    print(f"📦 Loading Model from {model_path}...")
    try:
        # We need the tokenizer separately for chat templates
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = OmniMultimodalLM(model_path)
    except Exception as e:
        print(f"❌ FATAL: Model load failed: {e}")
        return

    # 2. Test Reasoning (CoT)
    print("\n🧠 TEST 1: Reasoning (Chain-of-Thought)")
    reasoner = ReasoningWrapper(model.wrapper.llm, tokenizer)
    
    question = "If I have 3 apples and eat one, then buy two more, how many do I have?"
    messages = [{"role": "user", "content": question}]
    
    print(f"   Query: {question}")
    print("   ... Generating with ReasoningLevel.HIGH ...")
    
    try:
        response = reasoner.generate(messages, level=ReasoningLevel.HIGH, max_new_tokens=256)
        print(f"   Output:\n{response}")
        
        if "<think>" in response and "</think>" in response:
            print("   ✅ PASS: Model generated thought tags!")
        else:
            print("   ⚠️  FAIL: Model ignored reasoning prompt (No <think> tags).")
            print("      -> Recommendation: Fine-Tuning Required.")
            
    except Exception as e:
        print(f"   ❌ Error during generation: {e}")

    # 3. Test Tool Use
    print("\n🔧 TEST 2: Tool Use (Function Calling)")
    # Prepare a tool-use prompt manually since we haven't implemented automatic tool binding in the wrapper yet
    # We check if it can Output JSON when asked.
    
    tools = get_default_executor()
    tool_schema = tools.get_default_executor().get_schema() if hasattr(tools, 'get_default_executor') else tools.get_schema()
    
    tool_prompt = f"""You have access to the following tools:
{tool_schema}

To use a tool, output ONLY a JSON object with "name" and "arguments".
User: Calculate 12345 multiplied by 67890.
"""
    messages_tool = [{"role": "user", "content": tool_prompt}]
    
    print("   ... Generating with Tool Prompt ...")
    try:
        # Use basic generation
        text_input = tokenizer.apply_chat_template(messages_tool, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text_input, return_tensors="pt").to(model.wrapper.llm.device)
        
        with torch.no_grad():
            outputs = model.wrapper.llm.generate(**inputs, max_new_tokens=128, temperature=0.1)
            
        tool_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"   Output:\n{tool_response}")
        
        # Simple heuristic check for JSON
        if "{" in tool_response and "}" in tool_response and "calculator" in tool_response:
             print("   ✅ PASS: Model attempted JSON tool call.")
        else:
             print("   ⚠️  FAIL: Model output valid natural language but failed JSON constraint.")
             print("      -> Recommendation: Fine-Tuning Required.")

    except Exception as e:
         print(f"   ❌ Error during tool test: {e}")

    print("\n-------------------------------------------")
    print("Audit Complete.")

if __name__ == "__main__":
    audit_capabilities()
