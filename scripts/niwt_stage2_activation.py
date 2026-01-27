import argparse
import torch
import os
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.nexus_core.config import NexusConfig
from scripts.niwt_batch_profiler import get_nf4_config, ReasoningDataset
from torch.utils.data import DataLoader

class NIWTActivationMapper:
    def __init__(self, model, tokenizer, critical_layers):
        self.model = model
        self.tokenizer = tokenizer
        self.critical_layers = critical_layers
        self.activations = {}
        self.handles = []

    def _hook_fn(self, layer_idx):
        def hook(module, input, output):
            # Capture activations (output of the layer)
            # output is usually (hidden_states, ...)
            hidden = output[0] if isinstance(output, tuple) else output
            # We want to capture the mean activation or max activation per neuron
            # Shape: [batch, seq, hidden_dim] -> [hidden_dim] (aggregated)
            
            # Simple aggregation: Mean across batch and sequence
            mean_act = hidden.abs().mean(dim=(0, 1)).detach().cpu()
            
            if layer_idx not in self.activations:
                self.activations[layer_idx] = mean_act
            else:
                # Running mean update could be better, but simple accumulation for now
                self.activations[layer_idx] += mean_act
        return hook

    def register_hooks(self):
        for layer_idx in self.critical_layers:
            layer = self.model.model.layers[layer_idx]
            handle = layer.register_forward_hook(self._hook_fn(layer_idx))
            self.handles.append(handle)

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    @torch.no_grad()
    def map_activations(self, dataloader):
        print("\n--- NIWT STAGE 2: ACTIVATION MAPPING ---")
        self.register_hooks()
        
        count = 0
        for batch in tqdm(dataloader, desc="Mapping Activations"):
            questions = [b['question'] for b in batch]
            prompts = [f"Question: {q}\nAnswer:" for q in questions] # Simple prompt
            
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.model.device)
            self.model(**inputs)
            count += 1
            
        self.remove_hooks()
        
        # Normalize
        feature_bitmask = {}
        for layer_idx, act_sum in self.activations.items():
            avg_act = act_sum / count
            # Thresholding: Keep top 20% neurons or those above mean?
            # Strategy: Top 25% neurons are "Active"
            threshold = torch.quantile(avg_act, 0.75)
            mask = (avg_act > threshold).int().tolist()
            feature_bitmask[layer_idx] = mask
            
        return feature_bitmask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--profile_result", type=str, required=True, help="Path to Stage 1 JSON")
    args = parser.parse_args()

    # Load Config & Model (Simplified for brevity, assuming similar setup to Stage 1)
    config = NexusConfig("new-plan-conversation-files/ModelName-Parameters-Category-BestFeature.csv", "/mnt/e/data/models", "/mnt/e/data/benchmarks")
    model_info = config.get_model_info(args.model_name)
    
    # Load Stage 1 Results
    with open(args.profile_result, 'r') as f:
        profile_data = json.load(f)
    critical_layers = profile_data['critical_layers']
    
    if not critical_layers:
        print("No critical layers found in Stage 1. Exiting.")
        return

    print(f"Mapping Activations for {args.model_name} on layers: {critical_layers}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_info.path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_info.path, quantization_config=get_nf4_config(), device_map="auto", trust_remote_code=True)

    # Load Data
    bench_path = config.get_benchmark_path(args.model_name)
    dataset = ReasoningDataset(bench_path, tokenizer, num_samples=20) # Small sample for mapping
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=lambda x: x)
    
    mapper = NIWTActivationMapper(model, tokenizer, critical_layers)
    bitmask = mapper.map_activations(dataloader)
    
    output_file = args.profile_result.replace("_profile.json", "_bitmask.json")
    with open(output_file, 'w') as f:
        json.dump({"model": args.model_name, "feature_bitmask": bitmask}, f)
    print(f"Bitmask saved to {output_file}")

if __name__ == "__main__":
    main()
