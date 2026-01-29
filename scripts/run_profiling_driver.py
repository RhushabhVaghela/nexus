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
from nexus_core.data.sanitizer import UniversalSanitizer
import logging

# Suppress noisy transformers generation warnings (e.g. "top_p is ignored when do_sample=False")
logging.getLogger("transformers").setLevel(logging.ERROR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_id", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--sample_size", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="results/niwt_profiling")
    parser.add_argument("--no_quant", action="store_true", help="Disable 4-bit quantization (useful for small models/CPU)")
    args = parser.parse_args()

    print(f"\n[NIWT Profiler] Starting analysis for {args.teacher_id}...")
    
    # 0. Hardware Optimization (Beast Mode: RTX 5080)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Hardware] Mode: {device}")

    # 1. Load Model (4-bit for VRAM safety, unless disabled)
    print(f"[Loader] Loading {args.model_path} (Quantization: {'Disabled' if args.no_quant else '4-bit'})...")
    # Strong suppression for experimental/beta models that might have noisy configs
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    
    # Datatype logic: Use BF16 for Blackwell/Ada hardware
    load_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Load Model using Universal OmniModelLoader
    from src.omni.loader import OmniModelLoader
    
    print(f"[Loader] Initializing Universal Loader for {args.teacher_id}...")
    loader = OmniModelLoader(args.model_path)
    
    try:
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": load_dtype
        }
        if not args.no_quant:
            load_kwargs["load_in_4bit"] = True
            
        model, tokenizer = loader.load(mode="full", **load_kwargs)
        print(f"[Loader] Model loaded successfully using class: {model.__class__.__name__}")
    except Exception as e:
        print(f"[Error] Universal Loader failed for {args.teacher_id}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Re-suppress if needed, but keeping it INFO for now to catch issues
    # logging.getLogger("transformers").setLevel(logging.ERROR)

    # Load Dataset Stimulus
    calibration_data = ["Hello world"] * 50 # Default fallback
    
    # Use robust static calibration prompts instead of fragile next-token prediction
    test_cases = [
        ("The capital of France is", "Paris"),
        ("The command to list files in Linux is", "ls"),
        ("10 + 10 =", "20"),
        ("The largest planet in solar system is", "Jupiter"),
        ("Python is a programming", "language"),
        ("Hello, how are", "you"),
        ("Red, Green and", "Blue"),
        ("A cat says", "meow"),
        ("Opposite of Up is", "Down"),
        ("Water boils at 100 degrees", "Celsius")
    ]

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
                # Use UniversalSanitizer to handle messy fragments reliably
                clean_text = UniversalSanitizer.sanitize(item)
                if clean_text:
                    samples.append(clean_text)
            
            if samples:
                calibration_data = samples
                print(f"[Loader] Loaded {len(samples)} sanitized samples for (Stage 2) activation analysis.")
                
        except Exception as e:
            print(f"[Warn] Failed to load dataset {args.dataset_name}: {e}. Using fallback calibration data.")

    # stage 1: Perturbation
    # For profiling, we force greedy search (do_sample=False) but must unset sampling flags
    # to avoid transformers warnings/errors
    def run_safe_generate(model, tokenizer, **kwargs):
        # Remove sampling params if do_sample is False
        if not kwargs.get("do_sample", False):
            kwargs.pop("top_p", None)
            kwargs.pop("top_k", None)
            kwargs.pop("temperature", None)
        return model.generate(**kwargs)

    try:
        # 2. Initialize NIWT Core
        print("[NIWT] Initializing NIWT Core...")
        config = {"alpha": 1.0} # Default config
        niwt = NIWTCore(model, tokenizer, config)

        # 3. Stage 1: Perturbation
        print("[NIWT] Running Stage 1: Perturbation...")
        critical_layers = niwt.run_stage_1_perturbation(test_cases)
        
        # 4. Stage 2: Activation Analysis
        print("[NIWT] Running Stage 2: Activation Analysis...")
        niwt.run_stage_2_activation_analysis(calibration_data)
        
        # 5. Stage 3: Spectral Analysis
        print("[NIWT] Running Stage 3: Spectral Analysis...")
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

    except Exception as e:
        print(f"[Error] Profiling failed for {args.teacher_id}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
