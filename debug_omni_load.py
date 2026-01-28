from transformers import AutoModel, AutoConfig, AutoModelForCausalLM
import torch

model_path = "/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4"

print(f"Loading {model_path}...")

try:
    print("Attempt 1: AutoModel with trust_remote_code=True (No Quant Config)")
    model = AutoModel.from_pretrained(
        model_path, 
        device_map="auto", 
        trust_remote_code=True
    )
    print("Success: Loaded with AutoModel")
except Exception as e:
    print(f"Failed Attempt 1: {e}")

    try:
        print("\nAttempt 2: AutoModelForCausalLM with trust_remote_code=True")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            trust_remote_code=True
        )
        print("Success: Loaded with AutoModelForCausalLM")
    except Exception as e:
        print(f"Failed Attempt 2: {e}")
        
        try:
            print("\nAttempt 3: Qwen2ForCausalLM (Hard Force)")
            from transformers import Qwen2ForCausalLM
            model = Qwen2ForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True
            )
            print("Success: Loaded with Qwen2ForCausalLM")
        except Exception as e:
            print(f"Failed Attempt 3: {e}")
