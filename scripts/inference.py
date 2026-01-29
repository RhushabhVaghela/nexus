import sys
import os
import argparse
import json

# Ensure we can import from src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))

try:
    import unsloth
except ImportError:
    pass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from nexus_core.student.core import NexusStudentCore, NexusStudentConfig

# Default Release Path
RELEASE_PATH = "nexus-release-v1"

def load_student(release_path):
    # PRE-FLIGHT: Cleanup previous runs to free VRAM
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Loading Nexus Student from {release_path}...")
    
    # 0. Path Resolution: Direct Loading
    actual_model_path = release_path
    if not os.path.exists(os.path.join(actual_model_path, "config.json")):
        print(f"[Error] config.json not found in {actual_model_path}")

    # 1. High-Performance Load (Transformers with 4-bit)
    try:
        print(f"[Memory] Loading via Transformers (4-bit/Offload)...")
        tokenizer = AutoTokenizer.from_pretrained(actual_model_path)
        model = AutoModelForCausalLM.from_pretrained(
            actual_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        return model, tokenizer
    except Exception as e:
        print(f"Standard load failed: {e}. Trying NexusStudentCore fallback...")
        
    # 2. Fallback (Universal Custom Architecture Recovery)
    print(f"Trying NexusStudentCore fallback for {actual_model_path}...")
    try:
        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(actual_model_path)
    except:
        print("[Warn] Tokenizer not found in path. Using GPT-2 fallback.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

    try:
        conf_path = os.path.join(actual_model_path, "config.json") 
        with open(conf_path, 'r') as f:
            cfg_dict = json.load(f)
        config = NexusStudentConfig(**cfg_dict)
    except:
        print("[Warn] Model config not found or invalid. Using default 2B architecture.")
        config = NexusStudentConfig(vocab_size=128256, hidden_size=2048, num_hidden_layers=16)

    # MEMORY FIX: Initialize on META device to avoid RAM OOM
    print("[Memory] Initializing Model Skeletons (Meta-Device)...")
    try:
        try:
            from accelerate import init_empty_weights
            with init_empty_weights():
                model = NexusStudentCore(config)
        except ImportError:
            # Manual meta-device fallback
            print("[Warn] 'accelerate' missing. Trying manual meta-dispatch...")
            with torch.device("meta"):
                model = NexusStudentCore(config)
    except Exception as e:
        print(f"[Critical] Meta-init failed: {e}. Falling back to eager RAM (⚠️ Risk of OOM).")
        model = NexusStudentCore(config)

    # 3. Load Weights (Direct to Device)
    from safetensors.torch import load_file
    safe_path = os.path.join(actual_model_path, "model.safetensors")
    bin_path = os.path.join(actual_model_path, "pytorch_model.bin")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        if os.path.exists(safe_path):
            print(f"[Memory] Streaming weights from {os.path.basename(safe_path)} to {device}...")
            state_dict = load_file(safe_path)
            model = model.to_empty(device=device) # Allocate real memory on GPU
            model.load_state_dict(state_dict, strict=False)
            print("[Success] Loaded Nexus Architecture from safetensors.")
        elif os.path.exists(bin_path):
            print(f"[Memory] Streaming weights from {os.path.basename(bin_path)} to {device}...")
            state_dict = torch.load(bin_path, map_location=device)
            model = model.to_empty(device=device)
            model.load_state_dict(state_dict, strict=False)
            print("[Success] Loaded Nexus Architecture from bin.")
        else:
            print("[Warning] No weights found! Running on dummy weights.")
            model = model.to(device)
            
        return model, tokenizer

    except Exception as e:
        print(f"[Critical] Weight loading failed: {e}")
        return None, None

def load_knowledge_engine(release_path, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    index_path = os.path.join(release_path, "knowledge_index.faiss")
    if not os.path.exists(index_path):
        print(f"[Info] No Knowledge Index found at {index_path}. Running without RAG.")
        return None
        
    print(f"Loading Knowledge Engine from {index_path}...")
    from nexus_final.knowledge import KnowledgeTower
    import faiss
    
    # Initialize Tower
    tower = KnowledgeTower(embedding_model=embedding_model, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Load FAISS Index
    try:
        tower.index = faiss.read_index(index_path)
        # We need the original documents to map indices back to text.
        # In a real deployed system, these would be in a DB. 
        # For now, we check if there's a documents.json or we rely on the index simply existing 
        # but warn that we can't retrieve text without the source doc map.
        doc_path = os.path.join(release_path, "knowledge_docs.json")
        if os.path.exists(doc_path):
             with open(doc_path, 'r') as f:
                 tower.documents = json.load(f)
             print(f"✅ Knowledge Base Loaded ({tower.index.ntotal} vectors, {len(tower.documents)} docs).")
             return tower
        else:
            print("[Warn] Index found but 'knowledge_docs.json' is missing. Context retrieval will be empty.")
            return None
    except Exception as e:
        print(f"[Error] Failed to load index: {e}")
        return None

import time
import psutil

def get_gpu_memory():
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = total - free
        return used / 1024**3, total / 1024**3
    return 0, 0

def print_diagnostic_dashboard():
    ram = psutil.virtual_memory()
    cpu = psutil.cpu_percent()
    vram_used, vram_total = get_gpu_memory()
    
    print(f"\n{'-'*60}")
    print(f"📊 NEXUS DIAGNOSTIC DASHBOARD")
    print(f"{'-'*60}")
    print(f"💻 CPU: {cpu}% | 🧠 RAM: {ram.percent}% ({ram.used/1024**3:.1f}GB / {ram.total/1024**3:.1f}GB)")
    if torch.cuda.is_available():
        print(f"🚀 VRAM: {vram_used:.1f}GB / {vram_total:.1f}GB")
    print(f"{'-'*60}\n")

def generate_response(model, tokenizer, prompt, knowledge_tower=None, max_new_tokens=256):
    final_prompt = prompt
    
    # RAG: Indexation-based Context Injection
    if knowledge_tower:
        try:
            context_docs = knowledge_tower.retrieve_text_context(prompt, top_k=2)
            if context_docs:
                context_str = "\n".join([f"- {d}" for d in context_docs])
                final_prompt = f"Context:\n{context_str}\n\nUser: {prompt}"
                print(f"[RAG] Context Injected.")
        except Exception as e:
            print(f"[RAG Error] {e}")

    inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)
    
    # Metrics Initialization
    start_time = time.time()
    tokens_generated = 0
    
    print("Nexus: ", end="", flush=True)
    
    # We use manual loop for real-time tokens/s and metric display
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
    
    generated_ids = input_ids
    
    # Sampling Parameters (Optimized for Early-Stage Coherence)
    temp = 0.7
    top_p = 0.9
    top_k = 50
    repetition_penalty = 1.1
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            outputs = model(input_ids=generated_ids, attention_mask=attention_mask)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply Repetition Penalty (Corrected for mixed-sign logits)
            for token_id in set(generated_ids[0].tolist()):
                if next_token_logits[:, token_id] > 0:
                    next_token_logits[:, token_id] /= repetition_penalty
                else:
                    next_token_logits[:, token_id] *= repetition_penalty

            # Apply Temperature
            next_token_logits = next_token_logits / max(temp, 1e-5)
            
            # Top-P (Nucleus) Sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            indices_to_remove = sorted_indices[cumulative_probs > top_p]
            next_token_logits[:, indices_to_remove] = -float('Inf')
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            tokens_generated += 1
            
            # Decode for streaming
            token_str = tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(token_str, end="", flush=True)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    end_time = time.time()
    duration = end_time - start_time
    tps = tokens_generated / max(duration, 0.0001)
    
    print(f"\n\n[Stats] Gen Speed: {tps:.2f} tok/s | Total Tokens: {tokens_generated} | Duration: {duration:.2f}s")
    if tps < 1.0:
        print(f"[Hint] Low speed? Check if 'bitsandbytes' is installed for optimized 4-bit loading.")
    if duration > 0 and tokens_generated > 0:
         print(f"[Convergence Note] Initial loss was high (~14). '!!!' or jibberish indicates the model needs more training steps to align its logic.")
    print_diagnostic_dashboard()
    
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=RELEASE_PATH)
    args = parser.parse_args()
    
    model, tokenizer = load_student(args.model_path)
    if tokenizer is None:
        print("[Error] Tokenizer required for inference.")
        return
        
    knowledge_tower = load_knowledge_engine(args.model_path)
    
    print_diagnostic_dashboard()
    print("\n✅ Nexus Student Ready. Type 'exit' to quit.\n")
    
    while True:
        try:
            prompt = input("User: ")
        except EOFError:
            break
            
        if prompt.lower() in ["exit", "quit"]:
            break
        elif prompt.strip() == "":
            continue
            
        generate_response(model, tokenizer, prompt, knowledge_tower)
        print("")

if __name__ == "__main__":
    main()
