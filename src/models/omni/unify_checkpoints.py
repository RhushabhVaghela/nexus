#!/usr/bin/env python3
"""
unify_checkpoints.py
Unify multiple independently trained capability checkpoints into a single LLM.

Strategies:
1. Linear Merging (Weighted Average): Best for merging multiple fine-tuned deltas.
2. Sequential Merging: Best if capabilities were trained as adapters (LoRA).
3. Task Arithmetic: Adding deltas (Checkpoints - Base) to the base model.
4. TIES Merging: Trim, Elect, and Sign - Reduces interference between capabilities.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Optional
import shutil
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def merge_checkpoints(
    base_model_path: str,
    checkpoint_paths: List[str],
    output_path: str,
    method: str = "linear",
    weights: Optional[List[float]] = None,
    density: float = 0.2,
    dry_run: bool = False
):
    """
    Merge multiple capability checkpoints into a single unified model.
    """
    logger.info(f"🚀 Unifying {len(checkpoint_paths)} checkpoints using method: {method}")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if weights is None:
        weights = [1.0 / len(checkpoint_paths)] * len(checkpoint_paths)
    
    # Check weights sum
    if abs(sum(weights) - 1.0) > 1e-6:
        logger.warning(f"Weights sum to {sum(weights)}, normally they should sum to 1.0.")
    
    # 1. Load the Base Model (as the skeleton)
    logger.info(f"Loading Base Model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=torch.bfloat16,
        device_map="cpu"  # Keep on CPU to save VRAM during merging
    )
    base_state_dict = base_model.state_dict()
    
    # 2. Result state dict
    if method in ["task_arithmetic", "ties"]:
        logger.info(f"Using {method.upper()} (calculating deltas)...")
        # Start with base and add deltas
        merged_state_dict = {k: v.clone().to(torch.float32) for k, v in base_state_dict.items()}
        deltas = []
    else:
        # Start with zeros for weighted average
        merged_state_dict = {k: v.clone().to(torch.float32) * 0.0 for k, v in base_state_dict.items()}
    
    # 3. Accumulate weighted checkpoints
    for i, (path, weight) in enumerate(zip(checkpoint_paths, weights)):
        logger.info(f"  [#{i+1}] Processing {path} (Weight: {weight})")
        
        # We load just the state dict to save memory
        # Check for safetensors vs pytorch_model.bin
        sd_path = Path(path) / "model.safetensors"
        if sd_path.exists():
            from safetensors.torch import load_file
            current_sd = load_file(str(sd_path))
        else:
            sd_path = Path(path) / "pytorch_model.bin"
            if sd_path.exists():
                current_sd = torch.load(str(sd_path), map_location="cpu")
            else:
                 # Fallback to loading full model if files aren't in expected location
                 logger.warning(f"  Standard weights not found in {path}, loading via Transformers...")
                 model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, device_map="cpu")
                 current_sd = model.state_dict()
        
        current_deltas = {}
        for k, v in tqdm(current_sd.items(), desc=f"Processing {Path(path).name}", leave=False):
            if k in merged_state_dict:
                val = v.to(torch.float32)
                if method in ["task_arithmetic", "ties"]:
                    # delta = FineTuned - Base
                    delta = val - base_state_dict[k].to(torch.float32)
                    if method == "ties":
                        current_deltas[k] = delta
                    else:
                        merged_state_dict[k] += delta * weight
                else:
                    # Accumulate: merged = sum(weight_i * weight_dict_i)
                    merged_state_dict[k] += val * weight
            else:
                logger.warning(f"Skipping key {k} from {path} (not in base model)")
                
        if method == "ties":
            deltas.append(current_deltas)
            
        # Clean up current state dict
        del current_sd
        torch.cuda.empty_cache()

    # 3.5 TIES Specific Logic
    if method == "ties":
        logger.info(f"Running TIES post-processing (Density: {density})...")
        for k in tqdm(merged_state_dict.keys(), desc="TIES Merging Layers"):
            # Get all deltas for this layer
            layer_deltas = [d[k] for d in deltas if k in d]
            if not layer_deltas: continue
            
            # Step 1: Trim (Keep only top 'density' weights by magnitude)
            trimmed_deltas = []
            for d in layer_deltas:
                flat_d = d.flatten()
                k_val = int(len(flat_d) * density)
                if k_val > 0:
                    topk_values, _ = torch.topk(torch.abs(flat_d), k_val)
                    threshold = topk_values[-1]
                    mask = (torch.abs(d) >= threshold).float()
                    trimmed_deltas.append(d * mask)
                else:
                    trimmed_deltas.append(torch.zeros_like(d))
            
            # Step 2: Elect (Check sign dominance)
            # We use the sign of the sum of deltas to determine the dominant direction
            final_sign = torch.sign(torch.sum(torch.stack(trimmed_deltas), dim=0))
            
            # Step 3: Disjoint Merge
            # Filter deltas to only those matching the dominant sign
            merged_delta = torch.zeros_like(merged_state_dict[k])
            count = torch.zeros_like(merged_state_dict[k])
            
            for i, d in enumerate(trimmed_deltas):
                mask = (torch.sign(d) == final_sign).float() * (d != 0).float()
                merged_delta += d * mask * weights[i]
                count += mask
            
            # Average survived weights
            divisor = torch.clamp(count, min=1.0)
            merged_state_dict[k] += merged_delta / divisor

    # 4. Cast back to original dtype and load into model
    logger.info("Finishing merge and casting to bfloat16...")
    for k in merged_state_dict:
        merged_state_dict[k] = merged_state_dict[k].to(torch.bfloat16)
        
    base_model.load_state_dict(merged_state_dict)
    
    # 5. Save the unified model
    if dry_run:
        logger.info("🧪 DRY-RUN: Model merged but NOT saving to disk.")
        # Calculate avg magnitude of change for confirmation
        total_delta = 0.0
        for k, v in merged_state_dict.items():
            total_delta += torch.norm(v - base_state_dict[k].to(torch.float32)).item()
        logger.info(f"Total Model Weight Delta Magnitude: {total_delta:.4f}")
        return
        
    logger.info(f"Saving unified model to {output_path}")
    base_model.save_pretrained(str(output_path))
    
    # 6. Copy tokenizer from first checkpoint (usually identical)
    tokenizer_source = checkpoint_paths[0]
    logger.info(f"Copying tokenizer and config from {tokenizer_source}")
    for file in Path(tokenizer_source).iterdir():
        if "model.safetensors" not in file.name and "pytorch_model.bin" not in file.name:
            if file.is_file():
                shutil.copy2(file, output_dir / file.name)
    
    logger.info("✅ Model unification complete!")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Unify multiple checkpoints into a single model")
    parser.add_argument("--base", required=True, help="Path to the original base model")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="List of checkpoint directories to merge")
    parser.add_argument("--output", required=True, help="Output directory for unified model")
    parser.add_argument("--weights", nargs="+", type=float, help="Weights for each checkpoint (default: equal)")
    parser.add_argument("--method", choices=["linear", "task_arithmetic", "ties"], default="linear", help="Merging method")
    parser.add_argument("--density", type=float, default=0.2, help="Density for TIES merging (fraction of weights to keep)")
    parser.add_argument("--dry-run", action="store_true", help="Perform merge but don't save model")
    
    args = parser.parse_args()
    
    merge_checkpoints(
        args.base, 
        args.checkpoints, 
        args.output, 
        weights=args.weights, 
        method=args.method,
        density=args.density,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
