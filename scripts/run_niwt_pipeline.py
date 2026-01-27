import argparse
import subprocess
import os
import sys
from datetime import datetime

# Path to python interpreter in current env
PYTHON_EXE = sys.executable

def run_command(cmd, stage_name):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting {stage_name}...")
    print(f"Command: {cmd}")
    try:
        subprocess.check_call(cmd, shell=True)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {stage_name} Completed Successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {stage_name} FAILED with error code {e.returncode}.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Nexus NIWT Orchestrator")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model from the CSV")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = "/mnt/d/Research Experiments/nexus/results/niwt_profiling"
    
    # Files
    profile_json = os.path.join(output_dir, f"{args.model_name.replace('/','_')}_profile.json")
    bitmask_json = profile_json.replace("_profile.json", "_bitmask.json")
    spectral_json = profile_json.replace("_profile.json", "_spectral_config.json")
    
    # Stage 1: Batch Profiling
    cmd1 = f"{PYTHON_EXE} {os.path.join(script_dir, 'niwt_batch_profiler.py')} --model_name '{args.model_name}' --batch_size {args.batch_size} --samples {args.samples}"
    if not run_command(cmd1, "Stage 1: Profiling"): return

    # Stage 2: Activation Mapping
    cmd2 = f"{PYTHON_EXE} {os.path.join(script_dir, 'niwt_stage2_activation.py')} --model_name '{args.model_name}' --profile_result '{profile_json}'"
    if not run_command(cmd2, "Stage 2: Activation Mapping"): return

    # Stage 3: Spectral Analysis
    cmd3 = f"{PYTHON_EXE} {os.path.join(script_dir, 'niwt_stage3_spectral.py')} --bitmask_file '{bitmask_json}'"
    if not run_command(cmd3, "Stage 3: Spectral Analysis"): return

    # Stage 4: Consolidation
    cmd4 = f"{PYTHON_EXE} {os.path.join(script_dir, 'niwt_stage4_consolidation.py')} --spectral_config '{spectral_json}'"
    if not run_command(cmd4, "Stage 4: Consolidation"): return

    print("\n" + "="*50)
    print(f"NIWT Pipeline Complete for {args.model_name}")
    print("="*50)

if __name__ == "__main__":
    main()
