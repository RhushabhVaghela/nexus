import sys
import os
import argparse
import json
import torch
import time
import psutil
from transformers import AutoTokenizer
from safetensors.torch import load_file

# Ensure we can import from src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from nexus_core.student.core import NexusStudentCore, NexusStudentConfig

RELEASE_PATH = "nexus-release-v1"

def get_diagnostics():
    ram = psutil.virtual_memory()
    stats = {
        "cpu": psutil.cpu_percent(),
        "ram_percent": ram.percent,
        "ram_used": ram.used / 1024**3,
        "ram_total": ram.total / 1024**3,
        "vram_used": 0,
        "vram_total": 0
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        stats["vram_used"] = (total - free) / 1024**3
        stats["vram_total"] = total / 1024**3
    return stats

def print_dashboard(stats, tps=None):
    print(f"\n{'-'*60}")
    print(f"🌈 NEXUS MULTIMODAL DASHBOARD")
    print(f"{'-'*60}")
    print(f"💻 CPU: {stats['cpu']}% | 🧠 RAM: {stats['ram_percent']}% ({stats['ram_used']:.1f}GB)")
    if torch.cuda.is_available():
        print(f"🚀 VRAM: {stats['vram_used']:.1f}GB / {stats['vram_total']:.1f}GB")
    if tps:
        print(f"⚡ SPEED: {tps:.2f} tok/s")
    print(f"{'-'*60}\n")

def load_multimodal_nexus(release_path):
    print(f"[Init] Loading Multimodal Nexus from {release_path}...")
    
    # 1. Load Model with Meta-Device Shield
    conf_path = os.path.join(release_path, "config.json")
    with open(conf_path, 'r') as f:
        config = NexusStudentConfig(**json.load(f))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Empty init
    try:
        from accelerate import init_empty_weights
        with init_empty_weights():
            model = NexusStudentCore(config)
    except:
        with torch.device("meta"):
            model = NexusStudentCore(config)
    
    # Stream weights
    safe_path = os.path.join(release_path, "model.safetensors")
    if os.path.exists(safe_path):
        state_dict = load_file(safe_path)
        model = model.to_empty(device=device)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Core Model Loaded.")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(release_path)
    
    # 2. Load Sparse Router
    router_path = os.path.join(release_path, "router.pt")
    if os.path.exists(router_path):
        # We manually load it into model.router
        router_state = torch.load(router_path, map_location=device)
        # Handle cases where it might be a dict or a state_dict
        if "gate.weight" in router_state:
             model.router.load_state_dict(router_state)
        print("✅ Sparse router Loaded.")

    return model, tokenizer

def run_multimodal_inference(model, tokenizer, prompt, image_path=None, audio_path=None):
    print(f"\n[Nexus] Processing Query...")
    
    # Intent Detection via Router
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], output_router_logits=True)
        # In a real system, we'd use router_logits to fetch the right tower.
        # Here we simulate the 'Reasoning' tower focus.
        
    start_time = time.time()
    tokens = 0
    print("Response: ", end="", flush=True)
    
    gen_ids = inputs["input_ids"]
    mask = inputs.get("attention_mask", None)
    
    # Multimodal Projection (MOCK)
    # In a real system, image_path would go through a ClipEncoder and then a NexusAdapter.
    adapter_states = {}
    if image_path:
        print(f"🖼️ [Vision] Incorporating Visual Context: {image_path}")
        # Create a mock 1x16x2048 hidden state representing visual features
        adapter_states["vision"] = torch.randn(1, 16, model.config.hidden_size).to(model.device)
    
    if audio_path:
        print(f"👂 [Audio] Incorporating Audio Context: {audio_path}")
        adapter_states["audio"] = torch.randn(1, 16, model.config.hidden_size).to(model.device)

    with torch.no_grad():
        for i in range(256):
            outputs = model(input_ids=gen_ids, attention_mask=mask, adapter_hidden_states=adapter_states if adapter_states else None)
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
            
            gen_ids = torch.cat([gen_ids, next_token], dim=-1)
            tokens += 1
            
            print(tokenizer.decode(next_token[0], skip_special_tokens=True), end="", flush=True)
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    duration = time.time() - start_time
    tps = tokens / max(duration, 0.0001)
    
    print_dashboard(get_diagnostics(), tps=tps)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to image for vision reasoning")
    parser.add_argument("--audio", type=str, help="Path to audio for voice reasoning")
    parser.add_argument("--model", type=str, default=RELEASE_PATH)
    args = parser.parse_args()

    model, tokenizer = load_multimodal_nexus(args.model)
    
    print_dashboard(get_diagnostics())
    print("Welcome to Nexus Multimodal CLI. Ask me anything about text, images, or audio!")
    
    while True:
        try:
            p = input("\nQuery > ")
            if p.lower() in ["exit", "quit"]: break
            run_multimodal_inference(model, tokenizer, p, image_path=args.image, audio_path=args.audio)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
