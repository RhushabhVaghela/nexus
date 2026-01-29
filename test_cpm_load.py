import torch
import sys
print("Importing transformers...")
from transformers import AutoTokenizer, AutoModelForCausalLM
print("Loading model...")
model_path = "/mnt/e/data/models/AgentCPM-Explore"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Tokenizer loaded.")
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    print("Model loaded successfully!")
    print(f"Model type: {type(model)}")
except Exception as e:
    import traceback
    print(f"FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)
