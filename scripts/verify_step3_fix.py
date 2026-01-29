
import sys
import os
import torch
from pathlib import Path

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.omni.loader import OmniModelLoader

def test_loading():
    model_path = "/mnt/e/data/models/stepfun-ai_Step3-VL-10B"
    print(f"Testing loader for {model_path}...")
    
    loader = OmniModelLoader(model_path)
    
    # Try loading in 4-bit to replicate user environment
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "load_in_4bit": True,
        "device_map": "auto"
    }
    
    try:
        model, tokenizer = loader.load(mode="full", **load_kwargs)
        print(f"SUCCESS: Model loaded as {type(model).__name__}")
    except Exception as e:
        print(f"FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loading()
