import argparse
import torch
import json
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))
# Ensure scripts is in path (for niwt_core local import if needed)
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

from niwt_core import NIWTCore

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_id", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--sample_size", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="results/niwt_profiling")
    args = parser.parse_args()

    print(f"\n[NIWT Profiler] Starting analysis for {args.teacher_id}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Hardware] Mode: {device}")

    # 1. Load Model (4-bit for VRAM safety)
    print(f"[Loader] Loading {args.model_path} in 4-bit...")
    try:
        # fix_mistral_regex is needed for some newer models/tokenizers
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, fix_mistral_regex=True)
            print("[Loader] Tokenizer loaded with fix_mistral_regex=True")
        except TypeError as e:
            print(f"[Warn] fix_mistral_regex not supported or failed: {e}. Falling back.")
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        sys.exit(1)

    # 2. Initialize NIWT Core
    config = {"alpha": 1.0} # Default config
    niwt = NIWTCore(model, tokenizer, config)

    # Load Dataset Stimulus
    calibration_data = ["Hello world"] # Default fallback
    test_cases = [("The capital of France is", "Paris")]

    if args.dataset_name:
        try:
            from datasets import load_dataset
            print(f"[Loader] Loading stimulus dataset: {args.dataset_name} (Sample Size: {args.sample_size})...")
            
            # Determine load strategy
            if os.path.isdir(args.dataset_name):
                # Check for dataset_dict.json (Arrow) or files (JSON/Parquet)
                if os.path.exists(os.path.join(args.dataset_name, "dataset_dict.json")):
                    from datasets import load_from_disk
                    ds = load_from_disk(args.dataset_name)
                    # Handle splits if 'train' exists, else take first
                    if 'train' in ds: ds = ds['train']
                    # Convert to iterable for consistency if needed, or just slice
                    # If it's arrow, we can just slice. But let's unify interface.
                else:
                    # Assume JSON/JSONL directory
                    ds = load_dataset("json", data_dir=args.dataset_name, split="train", streaming=True)
            else:
                # Load streaming from Hub
                ds = load_dataset(args.dataset_name, split="train", streaming=True)
            
            # Extract samples based on sample_size
            samples = []
            
            # Handle both IterableDataset (streaming) and Dataset (map-style)
            # take() works on IterableDataset. For Dataset, slice.
            if hasattr(ds, "take"):
                iterable_ds = ds.take(args.sample_size)
            else:
                iterable_ds = ds.select(range(min(len(ds), args.sample_size)))

            for item in iterable_ds:
                # Try to find text content in common fields
                text = item.get("text") or item.get("content") or item.get("instruction") or str(item)
                samples.append(text[:512]) # Truncate for speed
            
            if samples:
                calibration_data = samples
                # Create synthetic test cases from data (Source -> Source prefix)
                test_cases = [(s[:50], s[50:60]) for s in samples[:5] if len(s) > 60]
                print(f"[Loader] Loaded {len(samples)} samples for stimulus.")
        except Exception as e:
            print(f"[Warn] Failed to load dataset {args.dataset_name}: {e}. Using fallback.")

    # 3. Stage 1: Perturbation
    critical_layers = niwt.run_stage_1_perturbation(test_cases)
    
    # 4. Stage 2: Activation Analysis
    niwt.run_stage_2_activation_analysis(calibration_data)
    
    # 5. Stage 3: Spectral Analysis
    rank = niwt.run_stage_3_spectral()
    
    # 6. Save Profile
    # Ensure full path for nested teacher IDs (e.g. Qwen/Coder) exists
    profile_path = os.path.join(args.output_dir, f"{args.teacher_id}_profile.json")
    os.makedirs(os.path.dirname(profile_path), exist_ok=True)
    
    profile_data = {
        "teacher_id": args.teacher_id,
        "critical_layers": [cl['layer'] for cl in critical_layers],
        "intrinsic_dimension": rank,
        "status": "complete"
    }
    
    with open(profile_path, 'w') as f:
        json.dump(profile_data, f, indent=2)
        
    print(f"[NIWT] Profile saved to {profile_path}")

if __name__ == "__main__":
    main()
