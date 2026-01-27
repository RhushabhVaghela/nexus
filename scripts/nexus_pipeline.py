import os
import sys
import json
import signal
import time
import argparse
import subprocess
from datetime import datetime

# Pipeline State File
STATE_FILE = ".pipeline_state.json"

# Master Teacher Configuration
# Master Teacher Configuration
TEACHER_CONFIG = {
    # REASONING & AGENTIC
    "reasoning_core": {"model": "openbmb/AgentCPM-Explore", "type": "nf4", "active": True},
    "logic_heavy":    {"model": "zai-org/GLM-4.7-Flash", "type": "nf4", "active": False}, # Enable in Phase 2
    "interpretability": {"model": "google/gemma-scope-2-27b-pt", "type": "nf4", "active": False},

    # VISION
    "vision_main":    {"model": "stepfun-ai/Step3-VL-10B", "type": "nf4", "active": False},
    "vision_enc":     {"model": "siglip2-so400m-patch16-512", "type": "base", "active": False},
    "video_enc":      {"model": "MCG-NJU/videomae-large", "type": "base", "active": False},
    
    # AUDIO
    "omni_speech":    {"model": "nvidia/personaplex-7b-v1", "type": "nf4", "active": False},
    "asr_long":       {"model": "microsoft/VibeVoice-ASR", "type": "base", "active": False},
    "asr_fast":       {"model": "nvidia/parakeet-tdt-0.6b-v3", "type": "base", "active": False},
    "tts_custom":     {"model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "type": "nf4", "active": False},
    "tts_design":     {"model": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "type": "nf4", "active": False},
    "tts_tokenizer":  {"model": "Qwen/Qwen3-TTS-Tokenizer-12Hz", "type": "base", "active": False},
    
    # GENERATION
    "image_gen":      {"model": "stabilityai/stable-diffusion-3-medium-diffusers", "type": "base", "active": False},
    "video_gen":      {"model": "stabilityai/stable-video-diffusion-img2vid-xt-1-1", "type": "base", "active": False}
}

class NexusPipeline:
    def __init__(self):
        self.state = self._load_state()
        self.paused = False
        
        # Signal Handling for Graceful Pause
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _load_state(self):
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        return {"current_stage": "init", "completed_stages": [], "config": {}}

    def _save_state(self):
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)
        print(f"[Pipeline] State saved to {STATE_FILE}")

    def _handle_interrupt(self, signum, frame):
        print("\n[Pipeline] Pause signal received! Finishing current step and saving state...")
        self.paused = True
        self._save_state()
        sys.exit(0)

    def run_command(self, cmd):
        print(f"[Exec] {cmd}")
        # In a real scenario, use subprocess.Popen to allow interrupting sub-process
        # For now, os.system is simple blocking
        ret = os.system(cmd)
        if ret != 0:
            print(f"[Error] Command failed with code {ret}")
            sys.exit(ret)

    def stage_profiling(self):
        if "profiling" in self.state["completed_stages"]:
            print("[Skip] Profiling already complete.")
            return

        print("\n=== STAGE 1 & 2: NIWT PROFILING & ACTIVATION ANALYSIS ===")
        print("Using NIWT Core for Feature Bitmask Extraction...")
        
        # Import dynamically to avoid circular issues during init
        # Updated to use new src/nexus_core structure
        sys.path.append(os.path.join(os.getcwd(), 'src'))
        from nexus_core.profiling.niwt import NIWTCore
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        # Config for NF4 (as defined in chats)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        for name, conf in TEACHER_CONFIG.items():
            if not conf["active"]:
                continue
                
            print(f"\n[Profiler] Target: {name} ({conf['model']})")
            
            # 1. Load Teacher (Mock for CI/CD speed, Real for production)
            # In a real run, we would load the model here:
            # model = AutoModelForCausalLM.from_pretrained(conf['model'], quantization_config=bnb_config)
            # tokenizer = AutoTokenizer.from_pretrained(conf['model'])
            # For now, we continue to call the separate script OR use the class if we want single-process
            # Let's stick to the script approach for isolation, but updated to use niwt_core
            
            # Robust import: Add scripts/ to PYTHONPATH so 'from niwt_core' works
            cmd = f"export PYTHONPATH=$PYTHONPATH:$(pwd)/scripts && /home/rhushabh/miniconda3/envs/nexus/bin/python -c 'from niwt_core import NIWTCore; print(\"NIWT Core Loaded for {name}\")'"
            self.run_command(cmd)

            # In production, we would invoke:
            # niwt = NIWTCore(model, tokenizer, {})
            # niwt.run_stage_1_perturbation(loading_real_gsm8k_data())
            # niwt.run_stage_2_activation_analysis(...)
        
        self.state["completed_stages"].append("profiling")
        self.state["current_stage"] = "training"
        self._save_state()

    def stage_training(self):
        if "training" in self.state["completed_stages"]:
            print("[Skip] Training already complete.")
            return

        print("\n=== STAGE 2: DISTILLATION LOOP ===")
        print("Distilling Teacher Knowledge to Student Core...")
        
        # Locate profile
        profile_path = "results/niwt_profiling/mock_critical_layers.json" # Default
        # Logic to find latest profile
        results_dir = "results/niwt_profiling"
        if os.path.exists(results_dir):
            files = sorted([f for f in os.listdir(results_dir) if f.endswith(".json")], reverse=True)
            if files:
                profile_path = os.path.join(results_dir, files[0])
                print(f"[Info] Found profile: {profile_path}")

        self.run_command(f"/home/rhushabh/miniconda3/envs/nexus/bin/python train.py --epochs 1 --profile_path '{profile_path}'")
        
        self.state["completed_stages"].append("training")
        self.state["current_stage"] = "router_training"
        self._save_state()

    def stage_router_training(self):
        if "router_training" in self.state["completed_stages"]:
            print("[Skip] Router Training already complete.")
            return

        print("\n=== STAGE 3: ROUTER TRAINING ===")
        print("Updates SparseIntentRouter weights...")
        
        # Execute the newly created router training script
        # Using the same python environment
        cmd = f"/home/rhushabh/miniconda3/envs/nexus/bin/python scripts/train_router.py"
        self.run_command(cmd)
        
        self.state["completed_stages"].append("router_training")
        self.state["current_stage"] = "done"
        self._save_state()

    def run(self):
        print("Nexus Automation Pipeline Initialized.")
        print(f"Current State: {self.state['current_stage']}")
        
        stages = {
            "init": self.stage_profiling,
            "profiling": self.stage_profiling,
            "training": self.stage_training,
            "router_training": self.stage_router_training
        }
        
        # Determine start point
        start_key = self.state.get("current_stage", "init")
        
        # Simple sequence
        if start_key == "init" or start_key == "profiling":
            self.stage_profiling()
            if self.paused: return
            
        if self.state["current_stage"] == "training":
            self.stage_training()
            if self.paused: return

        if self.state["current_stage"] == "router_training":
            self.stage_router_training()
            if self.paused: return

        print("\n=== PIPELINE COMPLETE ===")

if __name__ == "__main__":
    pipeline = NexusPipeline()
    pipeline.run()
