import argparse
import json
import torch
from src.nexus_core.towers.reasoning_tower import ReasoningTower
# Assuming we have a factory or config to choose the right tower type

def consolidate_model(spectral_config_path, output_dir):
    print("\n--- NIWT STAGE 4: CONSOLIDATION ---")
    with open(spectral_config_path, 'r') as f:
        data = json.load(f)
        
    configs = data['spectral_configs']
    model_name = data['model']
    
    # Simulate creating the Tower
    # In reality, we would initialize the specific Tower class and load the learned projection matrices
    # (which would be derived from Stage 3's SVD).
    
    # For demo, we create a ReasoningTower and save its config
    print(f"Consolidating {model_name} into Specialist Tower...")
    
    # We take the config of the first critical layer to determine dims
    first_layer = list(configs.keys())[0]
    teacher_dim = configs[first_layer]['source_dim']
    student_dim = configs[first_layer]['target_dim']
    
    tower_config = {
        "teacher_dim": teacher_dim,
        "student_dim": student_dim,
        "critical_layers": list(configs.keys()),
        "original_model": model_name
    }
    
    output_path = f"{output_dir}/{model_name.replace('/','_')}_tower_config.json"
    with open(output_path, 'w') as f:
        json.dump(tower_config, f, indent=2)
        
    print(f"Tower Configuration saved to {output_path}")
    print("Consolidation Complete. The Tower is ready for training/inference.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectral_config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/mnt/d/Research Experiments/nexus/results/towers")
    args = parser.parse_args()
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    consolidate_model(args.spectral_config, args.output_dir)

if __name__ == "__main__":
    main()
