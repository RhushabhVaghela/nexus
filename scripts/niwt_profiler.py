import torch
import os
import json
import random
import numpy as np
import time
import argparse
import logging
import shutil
import subprocess
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import gc

from src.data.niwt_loader import NIWTDataLoader

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# NEXUS CONFIGURATION & PATHS
# ==========================================
DEFAULT_MODEL_PATH = "/mnt/e/data/models/AgentCPM-Explore"
DEFAULT_BENCHMARK_PATH = "/mnt/e/data/benchmarks"
OUTPUT_DIR = "/mnt/d/Research Experiments/nexus/results/niwt_profiling"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SEED = 42

# Hardware Safety Constants
SAFE_TEMP_THRESHOLD = 83  # Celsius
COOLDOWN_PERIOD = 60 # Seconds
BATCH_SIZE_DEFAULT = 4

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ThermalProtection:
    """
    Hardware Safety Monitor.
    Pauses execution if GPU temperature exceeds critical threshold.
    """
    def __init__(self, threshold=SAFE_TEMP_THRESHOLD, cooldown_sec=COOLDOWN_PERIOD):
        self.threshold = threshold
        self.cooldown_sec = cooldown_sec

    def check(self):
        # Only check if nvidia-smi is available
        if shutil.which('nvidia-smi'):
            try:
                result = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], 
                    encoding='utf-8'
                )
                temp = int(result.strip())
                if temp >= self.threshold:
                    logger.warning(f"🔥 [CRITICAL] GPU Temp {temp}°C exceeds limit {self.threshold}°C! Cooling down for {self.cooldown_sec}s...")
                    time.sleep(self.cooldown_sec)
                    return False
            except Exception:
                pass 
        return True

def get_nf4_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

def load_teacher(model_path):
    logger.info(f"Loading Teacher from: {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=get_nf4_config(),
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

class NIWTProfiler:
    def __init__(self, model, tokenizer, batch_size=BATCH_SIZE_DEFAULT):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.critical_layers = []
        self.thermal = ThermalProtection()

    def format_prompts(self, questions: list[str]) -> list[str]:
        prompts = []
        for q in questions:
            if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
                messages = [{"role": "user", "content": q}]
                try:
                    prompts.append(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
                except:
                    # Fallback if template fails
                    prompts.append(f"Question: {q}\nAnswer:")
            else:
                prompts.append(f"Question: {q}\nLet's think step by step.\nAnswer:")
        return prompts

    @torch.no_grad()
    def evaluate_batch(self, prompts: list[str], targets: list[str]) -> float:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        
        # Optimize generation for speed
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=512, 
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,
            temperature=0.0
        )
        
        # Slice output to remove prompt for cleaner checking
        # But tokenizer.decode usually handles the whole thing. 
        # We'll just check if target is in the decoded string.
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        batch_score = 0
        for i, response in enumerate(decoded):
            # Basic containment check. For math, we might need more robust checking.
            # Assuming normalized targets are substrings of desired answer.
            if str(targets[i]).strip() in response:
                batch_score += 1.0
            
        return batch_score / len(prompts) if prompts else 0

    def profile_layers_batched(self, calibration_data, resume_from=None):
        """
        calibration_data: List of (prompts, targets) tuples, where prompts is a list of strings
        """
        logger.info("\n--- NIWT STAGE 1: BATCHED PERTURBATION PROFILING ---")
        
        if not calibration_data:
            logger.error("No calibration data provided!")
            return [], []

        # Baseline
        logger.info("Calculating Baseline...")
        scores = []
        for prompts, targets in tqdm(calibration_data, desc="Baseline Inference"):
            self.thermal.check()
            scores.append(self.evaluate_batch(prompts, targets))
        
        baseline = sum(scores) / len(scores) if scores else 0
        logger.info(f"Baseline Accuracy: {baseline:.2%}")

        # Layer Profiling
        # Auto-detect layer structure
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            layers = self.model.layers
        else:
            logger.error("Could not find layers in model structure.")
            return [], []
            
        layer_scores = []
        start_layer = 0
        if resume_from:
             logger.info(f"Resuming from layer {resume_from}...")
             start_layer = resume_from

        for i in range(start_layer, len(layers)):
            layer = layers[i]
            
            # Perturb: Identity Bypass
            original_forward = layer.forward
            layer.forward = lambda x, *args, **kwargs: x
            
            batch_scores = []
            for prompts, targets in tqdm(calibration_data, desc=f"Profiling Layer {i}", leave=False):
                self.thermal.check()
                batch_scores.append(self.evaluate_batch(prompts, targets))
            
            score = sum(batch_scores) / len(batch_scores) if batch_scores else 0
            drop = (baseline - score) / (baseline + 1e-9)
            
            logger.info(f"Layer {i:02d}: Acc={score:.2%} | Drop={drop:+.2%}")
            
            if drop > 0.15:
                logger.info(f"  >>> CRITICAL LAYER DETECTED")
                self.critical_layers.append({"layer": i, "drop": drop, "accuracy": score})
            
            layer_scores.append({"layer": i, "drop": drop, "accuracy": score})
            
            # Restore
            layer.forward = original_forward
            
            # Checkpoint
            if i % 2 == 0:
                self.save_checkpoint(i, layer_scores)

            # Cleanup
            if i % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        return self.critical_layers, layer_scores

    def save_checkpoint(self, layer_idx, scores):
        ckpt_file = os.path.join(OUTPUT_DIR, "niwt_checkpoint.json")
        with open(ckpt_file, 'w') as f:
            json.dump({
                "last_layer": layer_idx,
                "scores": scores,
                "critical": self.critical_layers
            }, f)

def main():
    parser = argparse.ArgumentParser(description="NIWT Profiler")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--benchmarks_path", type=str, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--capability", type=str, default="reasoning", choices=["reasoning", "agentic", "vision", "coding"])
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--num_batches", type=int, default=5, help="Number of batches to use for profiling (-1 for all)")
    args = parser.parse_args()

    set_seed(SEED)
    
    # Load Model
    teacher, tokenizer = load_teacher(args.model_path)
    if not teacher: return

    # Init Loader
    loader = NIWTDataLoader(args.benchmarks_path, batch_size=args.batch_size)
    
    # Init Profiler
    profiler = NIWTProfiler(teacher, tokenizer, batch_size=args.batch_size)
    
    logger.info(f"Profiling capability: {args.capability}")
    
    # Load Data Batches
    calibration_data = []
    iterator = loader.load_capability(args.capability)
    
    logger.info("Loading data batches...")
    for i, (batch, meta) in enumerate(iterator):
        if args.num_batches != -1 and i >= args.num_batches: break
        prompts, targets = loader.get_prompt_target_batch(batch)
        if prompts:
            calibration_data.append((prompts, targets))
            
    if not calibration_data:
        logger.error(f"No data found for capability '{args.capability}'. Check paths and schema mappings.")
        return

    # Checkpoint Resume Logic
    ckpt_path = os.path.join(OUTPUT_DIR, "niwt_checkpoint.json")
    resume_layer = None
    existing_scores = []
    
    if os.path.exists(ckpt_path):
        logger.info("Found checkpoint. Resuming...")
        try:
            with open(ckpt_path, 'r') as f:
                ckpt = json.load(f)
                resume_layer = ckpt.get("last_layer", 0) + 1
                profiler.critical_layers = ckpt.get("critical", [])
                existing_scores = ckpt.get("scores", [])
        except:
            logger.warning("Failed to load checkpoint, starting fresh.")
    
    # Run Profiling
    critical, new_scores = profiler.profile_layers_batched(calibration_data, resume_from=resume_layer)
    all_scores = existing_scores + new_scores
    
    # Save Results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = os.path.basename(args.model_path)
    output_file = os.path.join(OUTPUT_DIR, f"{model_name}_{args.capability}_niwt_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump({
            "config": vars(args),
            "critical_layers": critical, 
            "all_scores": all_scores
        }, f, indent=2)
    logger.info(f"Results saved to {output_file}")
    
    # Cleanup Checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

if __name__ == "__main__":
    main()
