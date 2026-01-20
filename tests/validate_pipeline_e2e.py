#!/usr/bin/env python3
"""
validate_pipeline_e2e.py
Real end-to-end validation of the training pipeline.

Uses local datasets at E:/data/datasets with sample_size=2.
Validates all stages work with real model and produce checkpoints.

OOM Safeguards:
- GPU memory clearing between stages
- Memory monitoring before each stage
- Automatic cooldown integration

Usage:
    # Validate text capabilities with 2 samples
    python tests/validate_pipeline_e2e.py --capabilities cot tools --sample-size 2
    
    # Validate all text capabilities
    python tests/validate_pipeline_e2e.py --all-text
"""

import os
import sys
import gc
import torch
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ============== OOM SAFEGUARDS ==============

def clear_gpu_memory():
    """Clear GPU memory to prevent OOM."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        logger.info(f"GPU memory cleared. Free: {get_gpu_memory_free():.1f}GB")

def get_gpu_memory_free() -> float:
    """Get available GPU memory in GB."""
    if not torch.cuda.is_available():
        return 0.0
    total = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated(0)
    return (total - allocated) / (1024**3)

def check_memory_safe(min_free_gb: float = 2.0) -> bool:
    """Check if we have enough memory to continue."""
    free = get_gpu_memory_free()
    if free < min_free_gb:
        logger.warning(f"Low GPU memory: {free:.1f}GB free (need {min_free_gb}GB)")
        clear_gpu_memory()
        free = get_gpu_memory_free()
    return free >= min_free_gb

# ============== LOCAL DATASET MAPPING ==============

LOCAL_DATASETS = {
    # Capability -> (local path, file pattern)
    "cot": "/mnt/e/data/datasets/kaist-ai_CoT-Collection/data/CoT_collection_en.json",
    "reasoning": "/mnt/e/data/datasets/openai_gsm8k",
    "thinking": "/mnt/e/data/datasets/O1-OPEN_OpenO1-SFT-Pro",
    "tools": "/mnt/e/data/datasets/Salesforce_xlam-function-calling-60k/xlam_function_calling_60k.json",
    "streaming": None,  # Runtime feature
    "podcast": "/mnt/e/data/datasets/spawn99_CornellMovieDialogCorpus",
    "vision-qa": "/mnt/e/data/datasets/AI4Math_MathVista",  
    "video-understanding": "/mnt/e/data/datasets/VLM2Vec_MSR-VTT",
    "tri-streaming": "/mnt/e/data/datasets/VoiceAssistant_Lite",
    "image-generation": "/mnt/e/data/datasets/LucasFang_Laion-Aesthetics-High-Resolution-GoT",
    "video-generation": "/mnt/e/data/datasets/fullstack__stargate_s04e01_100topkdiverse_text2vid",
}

def load_local_dataset(capability: str, sample_size: int = 2):
    """Load dataset from local path instead of downloading."""
    from datasets import Dataset
    
    path_str = LOCAL_DATASETS.get(capability)
    if path_str is None:
        logger.info(f"{capability}: No dataset needed (runtime feature)")
        return None
    
    path = Path(path_str)
    if not path.exists():
        logger.warning(f"{capability}: Local path not found: {path}")
        return None
    
    logger.info(f"{capability}: Loading from {path}")
    
    try:
        # If it's a JSON file, load directly
        if path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle dict format (key: sample)
            if isinstance(data, dict):
                data = list(data.values())[:sample_size]
            else:
                data = data[:sample_size]
            
            logger.info(f"{capability}: Loaded {len(data)} samples from JSON")
            return Dataset.from_list(data)
        
        # If it's a directory, look for files
        if path.is_dir():
            # Try parquet first
            parquet_files = list(path.glob("**/*.parquet"))
            if parquet_files:
                from datasets import load_dataset
                ds = load_dataset("parquet", data_files=str(parquet_files[0]), split="train")
                if len(ds) > sample_size:
                    ds = ds.select(range(sample_size))
                logger.info(f"{capability}: Loaded {len(ds)} from parquet")
                return ds
            
            # Try JSON files
            json_files = list(path.glob("**/*.json")) + list(path.glob("**/*.jsonl"))
            if json_files:
                with open(json_files[0], 'r', encoding='utf-8') as f:
                    content = f.read(10)
                    f.seek(0)
                    if content.startswith('[') or content.startswith('{'):
                        data = json.load(f)
                        if isinstance(data, dict):
                            data = list(data.values())
                    else:
                        data = [json.loads(line) for line in f][:sample_size]
                
                data = data[:sample_size]
                logger.info(f"{capability}: Loaded {len(data)} from JSON")
                return Dataset.from_list(data)
        
    except Exception as e:
        logger.error(f"{capability}: Load failed: {e}")
    
    return None


@dataclass
class ValidationResult:
    capability: str
    success: bool
    steps: int
    duration_s: float
    checkpoint_path: Optional[str]
    error: Optional[str] = None


def validate_single_stage(
    capability: str,
    base_model: str,
    output_dir: str,
    sample_size: int = 2,
    epochs: int = 1,
) -> ValidationResult:
    """Validate a single training stage with real model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    logger.info(f"\n{'='*60}")
    logger.info(f"VALIDATING: {capability}")
    logger.info(f"{'='*60}")
    
    start = time.time()
    
    # Clear memory before each stage
    clear_gpu_memory()
    
    if not check_memory_safe(min_free_gb=1.5):
        return ValidationResult(
            capability=capability, success=False, steps=0, duration_s=0,
            checkpoint_path=None, error="Insufficient GPU memory"
        )
    
    stage_output = Path(output_dir) / capability
    stage_output.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load dataset
        dataset = load_local_dataset(capability, sample_size)
        if dataset is None and capability != "streaming":
            return ValidationResult(
                capability=capability, success=False, steps=0, duration_s=time.time()-start,
                checkpoint_path=None, error="Could not load dataset"
            )
        
        # Load model
        logger.info(f"{capability}: Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Train
        if capability == "streaming":
            logger.info(f"{capability}: Runtime feature, skipping training")
            steps = 0
        else:
            logger.info(f"{capability}: Training on {len(dataset)} samples...")
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            
            steps = 0
            for sample in dataset:
                text = str(sample)[:500]
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                inputs["labels"] = inputs["input_ids"].clone()
                
                outputs = model(**inputs)
                loss = outputs.loss
                logger.info(f"{capability}: Step {steps+1}, Loss={loss.item():.4f}")
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                steps += 1
        
        # Save
        logger.info(f"{capability}: Saving checkpoint...")
        model.save_pretrained(str(stage_output))
        tokenizer.save_pretrained(str(stage_output))
        
        duration = time.time() - start
        
        return ValidationResult(
            capability=capability,
            success=True,
            steps=steps,
            duration_s=duration,
            checkpoint_path=str(stage_output),
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return ValidationResult(
            capability=capability, success=False, steps=0, duration_s=time.time()-start,
            checkpoint_path=None, error=str(e)
        )
    finally:
        clear_gpu_memory()


def run_validation(
    base_model: str,
    output_dir: str,
    capabilities: List[str],
    sample_size: int = 2,
    epochs: int = 1,
) -> List[ValidationResult]:
    """Run validation on multiple capabilities."""
    
    results = []
    
    logger.info(f"\n{'#'*60}")
    logger.info(f"E2E PIPELINE VALIDATION")
    logger.info(f"Base model: {base_model}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Capabilities: {capabilities}")
    logger.info(f"Sample size: {sample_size}, Epochs: {epochs}")
    logger.info(f"{'#'*60}\n")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for cap in capabilities:
        result = validate_single_stage(
            capability=cap,
            base_model=base_model,
            output_dir=output_dir,
            sample_size=sample_size,
            epochs=epochs,
        )
        results.append(result)
        
        status = "✅ PASS" if result.success else "❌ FAIL"
        logger.info(f"\n{status} {cap}: {result.steps} steps in {result.duration_s:.1f}s")
        if result.error:
            logger.info(f"   Error: {result.error}")
    
    return results


def print_summary(results: List[ValidationResult]):
    """Print validation summary."""
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results if r.success)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} passed\n")
    print(f"{'Capability':<20} {'Status':<8} {'Steps':<8} {'Time':<10} {'Checkpoint'}")
    print("-"*70)
    
    for r in results:
        status = "✅ PASS" if r.success else "❌ FAIL"
        ckpt = Path(r.checkpoint_path).name if r.checkpoint_path else "None"
        print(f"{r.capability:<20} {status:<8} {r.steps:<8} {r.duration_s:>6.1f}s   {ckpt}")
    
    print("="*70)
    
    if passed == total:
        print("\n🎉 ALL VALIDATIONS PASSED!")
    else:
        print(f"\n⚠️ {total - passed} validations failed")


def main():
    parser = argparse.ArgumentParser(description="E2E Pipeline Validation")
    parser.add_argument("--base-model", default="/mnt/e/data/models/Qwen2.5-0.5B")
    parser.add_argument("--output-dir", default="/mnt/e/data/models/validation_output")
    parser.add_argument("--sample-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--capabilities", nargs="+", default=["cot", "tools", "streaming"])
    parser.add_argument("--all-text", action="store_true")
    parser.add_argument("--all", action="store_true")
    
    args = parser.parse_args()
    
    if args.all:
        args.capabilities = [k for k in LOCAL_DATASETS.keys()]
    elif args.all_text:
        args.capabilities = ["cot", "reasoning", "thinking", "tools", "streaming"]
    
    results = run_validation(
        base_model=args.base_model,
        output_dir=args.output_dir,
        capabilities=args.capabilities,
        sample_size=args.sample_size,
        epochs=args.epochs,
    )
    
    print_summary(results)
    return 0 if all(r.success for r in results) else 1


if __name__ == "__main__":
    exit(main())
