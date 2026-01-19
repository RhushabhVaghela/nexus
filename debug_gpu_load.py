
import torch
from transformers import AutoModelForCausalLM, AutoConfig
import time
import os

model_path = "/mnt/d/Research Experiments/manus_model/base-model/gpt-oss-20b"

print(f"Checking model path: {model_path}")

print("\n--- Attempting Direct GPU Load (device_map='cuda:0') ---")
try:
    st = time.time()
    # device_map="cuda:0" forces direct loading to GPU, bypassing RAM if safetensors
    # low_cpu_mem_usage=True is implied by device_map
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda:0", 
        trust_remote_code=True
    )
    print(f"SUCCESS: Loaded in {time.time()-st:.2f}s")
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
except Exception as e:
    print(f"FAILED: {e}")
