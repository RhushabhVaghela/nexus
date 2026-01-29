
import torch
from transformers import AutoModelForCausalLM, AutoConfig
import sys
import os

model_path = "/mnt/e/data/models/stepfun-ai_Step3-VL-10B"

def test_load(quantize=False):
    print(f"--- Testing Load (Quantize={quantize}) ---")
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "device_map": "auto"
    }
    if quantize:
        kwargs["load_in_4bit"] = True
    
    try:
        # We use AutoModelForCausalLM as specified in auto_map
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        print(f"SUCCESS: Model loaded as {type(model).__name__}")
        return True
    except Exception as e:
        print(f"FAILURE: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test 1: Full precision
    success_fp = test_load(quantize=False)
    
    if success_fp:
        print("\nFull precision load succeeded. The issue IS quantization related.")
    else:
        print("\nFull precision load FAILED. The issue is likely in the architecture definition or AutoModel mapping.")
