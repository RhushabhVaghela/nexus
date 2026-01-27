import torch
import torch.nn as nn
import json
import os
import time
import gc
from typing import List, Dict, Tuple, Optional, Any
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class ThermalProtection:
    """
    Hardware Safety Monitor.
    Pauses execution if GPU temperature exceeds critical threshold (83C).
    """
    def __init__(self, threshold=83.0, cooldown_sec=30):
        self.threshold = threshold
        self.cooldown_sec = cooldown_sec

    def check(self):
        import subprocess
        import shutil
        
        if shutil.which('nvidia-smi'):
            try:
                # Query GPU 0 Temperature
                result = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], 
                    encoding='utf-8'
                )
                temp = int(result.strip())
                if temp >= self.threshold:
                    print(f"\n[CRITICAL] GPU Temp {temp}C exceeds limit {self.threshold}C! Cooling down for {self.cooldown_sec}s...")
                    time.sleep(self.cooldown_sec)
            except Exception as e:
                # Silent fail on environments where nvidia-smi might be weird
                pass

class EvaluationDataset(Dataset):
    """Simple Dataset wrapper for batched evaluation."""
    def __init__(self, samples: List[Tuple[str, str]]):
        self.samples = samples  # List of (prompt, target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class NIWTCore:
    """
    The Neural Information-Weighted Tower (NIWT) Engine.
    Implements the 4-Stage extraction pipeline with Optimized Batch Processing.
    """
    def __init__(self, model, tokenizer, config: Dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.critical_layers = []
        self.neuron_mask = {} 
        self.thermal = ThermalProtection()
        
        # Optimization Config
        self.batch_size = config.get("batch_size", 4)
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # =========================================================
    # STAGE 1: PERTURBATION (Layer-Level)
    # =========================================================
    def run_stage_1_perturbation(self, test_cases: List[Tuple[str, str]]) -> List[Dict]:
        """
        Surgically disable layers to see which tasks break.
        Returns a list of Critical Layers.
        """
        print(f"\n[NIWT Stage 1] Starting Layer Perturbation Analysis (BS={self.batch_size})...")
        
        # 1. Establish Baseline
        print("Calculating Optimized Baseline...")
        baseline_score = self._evaluate_capability_batched(test_cases)
        print(f"[Baseline] Score: {baseline_score:.2%}")
        
        results = []
        
        # 2. Iterate Layers
        layers = self._get_model_layers()
        
        # Create a progress bar
        pbar = tqdm(range(len(layers)), desc="Layer Profiling")
        
        for i in pbar:
            # Thermal check every few layers
            if i % 5 == 0: self.thermal.check()

            # Hook mechanism to bypass layer
            # We use a simple identity bypass for the "Perturbation"
            original_forward = layers[i].forward
            layers[i].forward = lambda *args, **kwargs: args[0] # Identity: Pass hidden states through
            
            # Evaluate
            score = self._evaluate_capability_batched(test_cases)
            drop = (baseline_score - score) / (baseline_score + 1e-9)
            
            # Restore
            layers[i].forward = original_forward
            
            # Classify
            is_critical = drop > 0.15 # 15% threshold from docs
            
            status = "CRITICAL" if is_critical else "OK"
            pbar.set_postfix({"Layer": i, "Drop": f"{drop:+.2%}", "Status": status})
            
            if is_critical:
                self.critical_layers.append({"layer": i, "drop": drop, "score": score})
            
            results.append({"layer": i, "drop": drop, "critical": is_critical, "score": score})
            
            # Cleanup VRAM
            if i % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            
        print(f"[Stage 1] Complete. Found {len(self.critical_layers)} critical layers.")
        return self.critical_layers

    # =========================================================
    # STAGE 2: ACTIVATION ANALYSIS (Neuron-Level)
    # =========================================================
    def run_stage_2_activation_analysis(self, calibration_data: List[str]):
        """
        For Critical Layers, find specifically which neurons fire.
        Generates the 'Feature Bitmask'.
        """
        print("\n[NIWT Stage 2] Starting Activation Analysis...")
        if not self.critical_layers:
            print("[Skip] No critical layers found in Stage 1.")
            return

        # Hooks to capture activations
        activations = {}
        def get_activation(name):
            def hook(model, input, output):
                # Simple Max-over-time pooling for firing detection
                # output shape: (batch, seq, hidden)
                # We want to see which hidden dim is active average over batch/seq
                if isinstance(output, tuple):
                    output = output[0]
                activations[name] = output.detach().abs().mean(dim=(0, 1))
            return hook

        # Register hooks on Critical Layers
        handles = []
        layers = self._get_model_layers()
        for cl in self.critical_layers:
            idx = cl['layer']
            handle = layers[idx].register_forward_hook(get_activation(f"layer_{idx}"))
            handles.append(handle)

        # Pass Data (Batched)
        print(f"Analyzing activations with {len(calibration_data)} samples...")
        
        # Simple batching for calibration
        loader = DataLoader(calibration_data, batch_size=self.batch_size, shuffle=False)
        
        for batch_prompts in tqdm(loader, desc="Activation Scan"):
            inputs = self.tokenizer(batch_prompts, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                self.model(**inputs)
            
            # Clear VRAM after batch
            del inputs
            torch.cuda.empty_cache()

        # Analyze & Threshold
        for name, act_tensor in activations.items():
            # Threshold: Top 30% active neurons
            threshold = torch.quantile(act_tensor, 0.70)
            mask = act_tensor > threshold
            layer_idx = int(name.split('_')[1])
            self.neuron_mask[layer_idx] = mask.nonzero().squeeze().tolist()
            print(f"  [Layer {layer_idx}] Masked {len(self.neuron_mask[layer_idx])} / {len(act_tensor)} neurons")

        # Cleanup
        for h in handles:
            h.remove()

    # =========================================================
    # HELPERS
    # =========================================================
    def _get_model_layers(self):
        # Auto-detect layer list based on common architectures
        if hasattr(self.model, "model"):
            if hasattr(self.model.model, "layers"):
                return self.model.model.layers
        if hasattr(self.model, "layers"):
             return self.model.layers
        # Try generic "h" (common in some HF models)
        if hasattr(self.model, "h"):
            return self.model.h
        if hasattr(self.model, "transformer"):
             if hasattr(self.model.transformer, "h"):
                 return self.model.transformer.h
        raise ValueError("Could not locate Transformer layers in model.")

    def _evaluate_capability_batched(self, test_cases: List[Tuple[str, str]]) -> float:
        """
        Evaluate average success on test cases using Batch Processing.
        """
        dataset = EvaluationDataset(test_cases)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        total_correct = 0
        total_samples = 0
        
        for prompts, targets in loader:
            inputs = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                # Generate small sample
                # Note: max_new_tokens is small for profiling speed (16-32 is usually enough to check CoT start or answer)
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=32, 
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    do_sample=False # Deterministic
                )
            
            decoded_batch = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Check correctness
            for i, response in enumerate(decoded_batch):
                # Simple containment check
                if targets[i].lower() in response.lower():
                    total_correct += 1
            
            total_samples += len(prompts)
            
            del inputs, outputs
        
        return total_correct / (total_samples + 1e-9)
