import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time

MODEL_PATH = "/mnt/e/data/models/AgentCPM-Explore"

def get_nf4_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

def debug_run():
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    print("Loading model...", flush=True)
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=get_nf4_config(),
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    print(f"Model loaded in {time.time() - start:.2f}s", flush=True)
    
    # Test cases
    cases = [
        "What is 25 + 75?",
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    ]
    
    for q in cases:
        print(f"\n--- Question: {q} ---", flush=True)
        # Try simple chat template if exists
        if tokenizer.chat_template:
            messages = [{"role": "user", "content": q}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"Question: {q}\nAnswer:"
        
        print(f"Prompt: {prompt}", flush=True)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Response: {response}", flush=True)

if __name__ == "__main__":
    debug_run()
