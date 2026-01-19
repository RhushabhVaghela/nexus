
import torch
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import time
import os

model_path = "/mnt/d/Research Experiments/manus_model/base-model/gpt-oss-20b"

print(f"Checking model path: {model_path}")

print("\n--- Attempting 4-bit Load with CPU Offload & Skip Modules ---")
try:
    # Try enabling CPU offload for 4-bit to solve dispatch error
    # Also valid to try skipping q_proj if SCB error persists, but let's try just offload first
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True # Allow CPU offloading for quantized modules
    )
    
    # Force some layers to CPU to test offloading
    device_map = {
        "model.layers.0": "cpu", 
        "model.layers.1": "cpu",
        "model.layers.2": "cpu"
    }
    # Or just use "auto"
    
    st = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map="auto", # Enable dispatching
        trust_remote_code=True
    )
    print(f"SUCCESS: Loaded in 4-bit in {time.time()-st:.2f}s")
    print(f"Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
except Exception as e:
    print(f"FAILED 4-bit: {e}")
