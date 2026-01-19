
import torch
from transformers import AutoModelForCausalLM, AutoConfig
import time
import os

model_path = "/mnt/d/Research Experiments/manus_model/base-model/gpt-oss-20b"

print(f"Checking model path: {model_path}")
print(f"Files: {os.listdir(model_path)}")

print("\n--- Load Config ---")
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
print(config)

print("\n--- Attempting 4-bit Load (BNB) ---")
try:
    from transformers import BitsAndBytesConfig
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )
    print("SUCCESS: Loaded in 4-bit")
    print(f"Memory: {torch.cuda.memory_allocated()/1e9} GB")
except Exception as e:
    print(f"FAILED 4-bit: {e}")

print("\n--- Attempting FP16 Load ---")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print("SUCCESS: Loaded in FP16")
    print(f"Memory: {torch.cuda.memory_allocated()/1e9} GB")
except Exception as e:
    print(f"FAILED FP16: {e}")
