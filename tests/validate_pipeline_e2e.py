#!/usr/bin/env python3
"""
validate_pipeline_e2e.py
Real end-to-end validation of the training pipeline.

Uses Universal Data Loader for all dataset formats (JSON, JSONL, parquet, etc.)
Validates all stages work with real model and produce checkpoints.

Directory Structure Explanation:
- validation_output/   -> E2E test checkpoints (one folder per capability)
                          Each contains complete model files (safetensors + tokenizer)
- Each checkpoint is a COMPLETE loadable model for inference or further training

Usage:
    # Validate specific capabilities
    python tests/validate_pipeline_e2e.py --capabilities cot tools --sample-size 2
    
    # Validate all text capabilities
    python tests/validate_pipeline_e2e.py --all-text
    
    # Validate all capabilities
    python tests/validate_pipeline_e2e.py --all
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

from src.data.universal_loader import load_dataset_universal
from src.omni.loader import OmniModelLoader
from src.metrics_tracker import ALL_DATASETS

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
    # Capability -> local dataset path (supports any format via universal loader)
    "cot": "/mnt/e/data/datasets/kaist-ai_CoT-Collection/data/CoT_collection_en.json",
    "reasoning": "/mnt/e/data/datasets/openai_gsm8k",
    "thinking": "/mnt/e/data/datasets/O1-OPEN_OpenO1-SFT-Pro",  # Now works with universal loader!
    "tools": "/mnt/e/data/datasets/Salesforce_xlam-function-calling-60k/xlam_function_calling_60k.json",
    "streaming": None,  # Runtime feature, no training
    "podcast": "/mnt/e/data/datasets/spawn99_CornellMovieDialogCorpus",
    "vision-qa": "/mnt/e/data/datasets/AI4Math_MathVista",  
    "video-understanding": "/mnt/e/data/datasets/VLM2Vec_MSR-VTT",
    "tri-streaming": "/mnt/e/data/datasets/VoiceAssistant_Lite",
    "image-generation": "/mnt/e/data/datasets/LucasFang_Laion-Aesthetics-High-Resolution-GoT",
    "video-generation": "/mnt/e/data/datasets/fullstack__stargate_s04e01_100topkdiverse_text2vid",
}


def load_dataset_for_capability(capability: str, sample_size: int = 2):
    """Load dataset using dynamic discovery or fallback mapping."""
    # 1. Try dynamic discovery first
    paths = ALL_DATASETS.get(capability, [])
    
    if not paths:
        # 2. Try legacy mapping if not found dynamically
        legacy_path = LOCAL_DATASETS.get(capability)
        if legacy_path:
            paths = [legacy_path]
            
    if not paths and capability != "streaming":
        logger.warning(f"{capability}: No datasets found via discovery or legacy mapping")
        return None
        
    if capability == "streaming":
        logger.info(f"{capability}: Runtime feature, no dataset needed")
        return None
        
    # Combine all found datasets for this capability
    from datasets import concatenate_datasets
    datasets = []
    
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
            
        logger.info(f"{capability}: Loading from {path}")
        try:
            result = load_dataset_universal(p, sample_size=sample_size)
            if result.dataset:
                datasets.append(result.dataset)
        except Exception as e:
            logger.error(f"{capability}: Failed to load {path}: {e}")
            
    if not datasets:
        return None
        
    if len(datasets) == 1:
        return datasets[0]
        
    return concatenate_datasets(datasets)


@dataclass
class ValidationResult:
    """Result of validating a single capability."""
    capability: str
    success: bool
    steps: int
    duration_s: float
    checkpoint_path: Optional[str]
    dataset_format: str = ""
    samples_loaded: int = 0
    initial_loss: float = 0.0
    final_loss: float = 0.0
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
    losses = []
    
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
        # Load dataset using universal loader
        dataset = load_dataset_for_capability(capability, sample_size)
        dataset_format = "none"
        samples_loaded = 0
        
        if dataset is None and capability != "streaming":
            return ValidationResult(
                capability=capability, success=False, steps=0, duration_s=time.time()-start,
                checkpoint_path=None, error="Could not load dataset"
            )
        
        if dataset is not None:
            samples_loaded = len(dataset)
            # Try to detect format
            dataset_format = "universal"
        
        # Load model (smart loader handles Omni vs standard models)
        logger.info(f"{capability}: Loading model...")
        
        # Check if Omni model
        is_omni = OmniModelLoader.is_omni_model(base_model)
        
        if is_omni:
            logger.info(f"{capability}: Detected Omni model, using OmniModelLoader")
            loader = OmniModelLoader()
            model, tokenizer = loader.load_for_training(base_model)
        else:
            # Standard text model
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
            
            # Training params optimized for small batch validation
            # VERY conservative to prevent NaN
            base_lr = 1e-7  # Ultra-low LR for small batches
            warmup_steps = max(2, len(dataset) // 3)  # 33% warmup
            accumulation_steps = 4  # Higher accumulation for stability
            max_loss_value = 10.0  # Skip steps with loss > this
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, eps=1e-6)
            
            steps = 0
            accumulated_loss = 0
            valid_loss_count = 0
            nan_count = 0
            
            for sample in dataset:
                # Use longer text for better gradients
                text = str(sample)[:2048]  # Even longer text
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                inputs["labels"] = inputs["input_ids"].clone()
                
                try:
                    outputs = model(**inputs)
                    loss = outputs.loss
                    
                    # Check for NaN/Inf and skip
                    if torch.isnan(loss) or torch.isinf(loss):
                        nan_count += 1
                        logger.warning(f"{capability}: Step {steps+1} - Skipping NaN/Inf loss")
                        optimizer.zero_grad()
                        steps += 1
                        continue
                    
                    # Clip loss magnitude to prevent explosion
                    if loss.item() > max_loss_value:
                        logger.warning(f"{capability}: Step {steps+1} - Loss {loss.item():.2f} > {max_loss_value}, clipping")
                        loss = loss.clamp(max=max_loss_value)
                    
                    # Scale for accumulation
                    loss = loss / accumulation_steps
                    accumulated_loss += loss.item()
                    valid_loss_count += 1
                    
                    loss.backward()
                    
                except RuntimeError as e:
                    logger.error(f"{capability}: Step {steps+1} - Runtime error: {e}")
                    optimizer.zero_grad()
                    steps += 1
                    continue
                
                # Update weights every accumulation_steps
                if (steps + 1) % accumulation_steps == 0 or (steps + 1) == len(dataset):
                    # Aggressive gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    
                    # Skip update if gradients are NaN
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        logger.warning(f"{capability}: Skipping update - grad_norm is NaN/Inf")
                        optimizer.zero_grad()
                        accumulated_loss = 0
                        valid_loss_count = 0
                        steps += 1
                        continue
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Learning rate warmup
                    current_step = (steps + 1) // accumulation_steps
                    if current_step < warmup_steps:
                        lr_scale = (current_step + 1) / warmup_steps
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = base_lr * lr_scale
                    
                    avg_loss = accumulated_loss / max(valid_loss_count, 1) * accumulation_steps
                    logger.info(f"{capability}: Step {steps+1}, Loss={avg_loss:.4f}, GradNorm={grad_norm:.4f}")
                    losses.append(avg_loss)
                    accumulated_loss = 0
                    valid_loss_count = 0
                
                steps += 1
            
            if nan_count > 0:
                logger.warning(f"{capability}: Total NaN/Inf steps skipped: {nan_count}")
        
        # Save checkpoint (creates complete model directory)
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
            dataset_format=dataset_format,
            samples_loaded=samples_loaded,
            initial_loss=losses[0] if losses else 0.0,
            final_loss=losses[-1] if losses else 0.0,
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
    logger.info(f"{'#'*60}")
    logger.info(f"Base model: {base_model}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Capabilities: {capabilities}")
    logger.info(f"Sample size: {sample_size}")
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
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    passed = sum(1 for r in results if r.success)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} passed\n")
    print(f"{'Capability':<20} {'Status':<8} {'Steps':<6} {'Samples':<8} {'Loss':<12} {'Time':<8} {'Checkpoint'}")
    print("-"*80)
    
    for r in results:
        status = "✅" if r.success else "❌"
        ckpt = Path(r.checkpoint_path).name if r.checkpoint_path else "None"
        loss_str = f"{r.initial_loss:.2f}→{r.final_loss:.2f}" if r.initial_loss else "N/A"
        print(f"{r.capability:<20} {status:<8} {r.steps:<6} {r.samples_loaded:<8} {loss_str:<12} {r.duration_s:>5.1f}s   {ckpt}")
    
    print("="*80)
    
    if passed == total:
        print("\n🎉 ALL VALIDATIONS PASSED!")
    else:
        print(f"\n⚠️ {total - passed} validations failed")


def main():
    parser = argparse.ArgumentParser(description="E2E Pipeline Validation")
    parser.add_argument("--base-model", default="/mnt/e/data/models/Qwen2.5-0.5B")
    parser.add_argument("--output-dir", default="/mnt/e/data/output/validation")
    parser.add_argument("--sample-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--capabilities", nargs="+", default=["cot", "tools", "streaming"])
    parser.add_argument("--all-text", action="store_true", help="Validate all text capabilities")
    parser.add_argument("--all", action="store_true", help="Validate all capabilities")
    
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
