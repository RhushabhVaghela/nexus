import argparse
import torch
import os
import json
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.nexus_core.config import NexusConfig

# ==========================================
# CONFIGURATION
# ==========================================
MODELS_ROOT = "/mnt/e/data/models"
BENCHMARKS_ROOT = "/mnt/e/data/benchmarks"
CSV_PATH = "new-plan-conversation-files/ModelName-Parameters-Category-BestFeature.csv"
OUTPUT_DIR = "/mnt/d/Research Experiments/nexus/results/niwt_profiling"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SEED = 42

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class ReasoningDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, num_samples=100, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        try:
            df = pd.read_parquet(parquet_path)
            # Standardize column names if needed, usually 'question', 'answer'
            # Fallback for different datasets
            q_col = 'question' if 'question' in df.columns else 'prompt'
            a_col = 'answer' if 'answer' in df.columns else 'completion'
            
            for _, row in df.head(num_samples).iterrows():
                self.samples.append({
                    "question": row[q_col], 
                    "answer": row[a_col].split('####')[-1].strip() if '####' in str(row[a_col]) else str(row[a_col])
                })
        except Exception as e:
            print(f"Error loading dataset {parquet_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # We return raw text here, tokenization happens in collate_fn or before generation
        return self.samples[idx]

def get_nf4_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

class NIWTBatchProfiler:
    def __init__(self, model, tokenizer, batch_size=8):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.tokenizer.padding_side = 'left' # Important for batch generation
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def format_prompts(self, questions):
        prompts = []
        for q in questions:
            if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
                messages = [{"role": "user", "content": q}]
                prompts.append(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            else:
                prompts.append(f"Question: {q}\nLet's think step by step.\nAnswer:")
        return prompts

    @torch.no_grad()
    def evaluate_batch(self, batch):
        questions = [b['question'] for b in batch]
        targets = [b['answer'] for b in batch]
        
        prompts = self.format_prompts(questions)
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=256, # Enough for answer check, saving VRAM
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False
        )
        
        # Decode only new tokens? Or full? 
        # Usually decode full and check suffix, but let's decode everything for simplicity
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        scores = []
        for resp, target in zip(responses, targets):
            scores.append(1.0 if target in resp else 0.0)
            
        return sum(scores)

    def run_evaluation_loop(self, dataloader, desc="Evaluating"):
        total_score = 0
        total_samples = 0
        
        for batch in tqdm(dataloader, desc=desc, leave=False):
            score = self.evaluate_batch(batch)
            total_score += score
            total_samples += len(batch)
            
        return total_score / total_samples if total_samples > 0 else 0

    def profile(self, dataloader):
        print("\n--- NIWT BATCH PROFILING ---")
        
        # 1. Baseline
        baseline = self.run_evaluation_loop(dataloader, desc="Baseline")
        print(f"Baseline Accuracy: {baseline:.2%}")
        
        layers = self.model.model.layers
        layer_results = []
        critical_layers = []

        for i, layer in enumerate(tqdm(layers, desc="Layer Profiling")):
            # Perturb: Identity Bypass
            original_forward = layer.forward
            layer.forward = lambda x, *args, **kwargs: x
            
            # Evaluate
            score = self.run_evaluation_loop(dataloader, desc=f"Layer {i}")
            drop = (baseline - score) / (baseline + 1e-9)
            
            layer_results.append({
                "layer": i,
                "accuracy": score,
                "drop": drop
            })
            
            if drop > 0.15:
                critical_layers.append(i)
                
            # Restore
            layer.forward = original_forward
            
            # GC
            if i % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        return critical_layers, layer_results, baseline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Name of model from CSV")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()

    set_seed(SEED)
    
    # Load Config
    config = NexusConfig(CSV_PATH, MODELS_ROOT, BENCHMARKS_ROOT)
    
    # Determine Model to Load
    # If no arg provided, list available models and exit (or pick first)
    if not args.model_name:
        print("Available models:")
        for m in config.models:
            print(f" - {m}")
        return

    model_info = config.get_model_info(args.model_name)
    if not model_info:
        print(f"Model {args.model_name} not found in config.")
        return

    print(f"Profiling Model: {model_info.name}")
    print(f"Local Path: {model_info.path}")
    
    # Load Model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_info.path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_info.path,
            quantization_config=get_nf4_config(),
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Load Data
    bench_path = config.get_benchmark_path(args.model_name)
    print(f"Benchmark: {bench_path}")
    dataset = ReasoningDataset(bench_path, tokenizer, num_samples=args.samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: x)
    
    # Run Profiler
    profiler = NIWTBatchProfiler(model, tokenizer, batch_size=args.batch_size)
    critical, results, baseline = profiler.profile(dataloader)
    
    # Save Results
    output_file = os.path.join(OUTPUT_DIR, f"{model_info.name.replace('/','_')}_profile.json")
    with open(output_file, 'w') as f:
        json.dump({
            "model": model_info.name,
            "baseline": baseline,
            "critical_layers": critical,
            "results": results
        }, f, indent=2)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
