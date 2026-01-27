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
    parser.add_argument("--output_dir", type=str, default="results/niwt_profiling")
    args = parser.parse_args()

    print(f"\n[NIWT Profiler] Starting analysis for {args.teacher_id}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Hardware] Mode: {device}")

    # 1. Load Model (4-bit for VRAM safety)
    print(f"[Loader] Loading {args.model_path} in 4-bit...")
    try:
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

    # 3. Stage 1: Perturbation
    # Minimal test case for proving ground
    test_cases = [
        ("The capital of France is", "Paris"),
        ("2 + 2 =", "4")
    ]
    critical_layers = niwt.run_stage_1_perturbation(test_cases)
    
    # 4. Stage 2: Activation Analysis
    calibration_data = ["Hello world"] # Minimal calibration
    niwt.run_stage_2_activation_analysis(calibration_data)
    
    # 5. Stage 3: Spectral Analysis
    rank = niwt.run_stage_3_spectral()
    
    # 6. Save Profile
    os.makedirs(args.output_dir, exist_ok=True)
    profile_path = os.path.join(args.output_dir, f"{args.teacher_id}_profile.json")
    
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
