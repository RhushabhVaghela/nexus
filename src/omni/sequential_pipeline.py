#!/usr/bin/env python3
"""
sequential_pipeline.py
Orchestrate sequential fine-tuning of multiple capabilities.
Each stage uses the output of the previous stage as its starting point.

Usage:
    python src/omni/sequential_pipeline.py \\
        --base-model "/path/to/base" \\
        --stages cot tools reasoning \\
        --output-dir "E:/data/output/sequential_unified"
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def run_sequential_training(base_model: str, stages: list, output_dir: str, sample_size: int = 0):
    """
    Run training stages sequentially.
    """
    current_base = base_model
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🏁 Starting Sequential Training Pipeline")
    logger.info(f"   Base Model: {base_model}")
    logger.info(f"   Stages: {' -> '.join(stages)}")
    
    for i, capability in enumerate(stages):
        stage_num = i + 1
        logger.info(f"\n" + "="*60)
        logger.info(f"🚀 STAGE {stage_num}/{len(stages)}: {capability}")
        logger.info(f"="*60)
        
        stage_output = output_root / f"stage_{stage_num}_{capability}"
        
        # Construct command
        # We assume each stage has its own script or can be run via a generic stage runner
        # Based on the codebase, we can use the capability_registry to find the script
        
        from src.capability_registry import CapabilityRegistry
        registry = CapabilityRegistry()
        cap_info = registry.get(capability)
        
        if not cap_info:
            logger.error(f"❌ Unknown capability: {capability}. Skipping.")
            continue
            
        training_script = cap_info.training_script
        
        cmd = [
            "conda", "run", "-n", "nexus",
            "python3", training_script,
            "--base-model", str(current_base),
            "--output-dir", str(stage_output),
            "--sample-size", str(sample_size)
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            # Run the training script
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output
            for line in process.stdout:
                print(f"[{capability}] {line.strip()}")
                
            process.wait()
            
            if process.returncode != 0:
                logger.error(f"❌ Stage {capability} failed with return code {process.returncode}")
                # Optional: Continue or stop? Usually stop.
                sys.exit(1)
                
            logger.info(f"✅ Stage {capability} completed successfully.")
            
            # Update base model for NEXT stage
            # We need to find the "final_model_..." or last checkpoint in stage_output
            # BaseStage saves to final_model_{timestamp}
            final_models = list(stage_output.glob("final_model_*"))
            if final_models:
                # Get the most recent one
                final_models.sort(key=os.path.getmtime)
                current_base = final_models[-1]
            else:
                # Fallback to the directory itself if it looks like a model
                current_base = stage_output
                
            logger.info(f"📍 Next stage will start from: {current_base}")
            
        except Exception as e:
            logger.error(f"❌ Error during stage {capability}: {e}")
            sys.exit(1)

    logger.info(f"\n" + "="*60)
    logger.info(f"🏆 SEQUENTIAL PIPELINE COMPLETE!")
    logger.info(f"   Final Unified Model: {current_base}")
    logger.info(f"="*60)

def main():
    parser = argparse.ArgumentParser(description="Sequential Model Unification Pipeline")
    parser.add_argument("--base-model", required=True, help="Initial base model path")
    parser.add_argument("--stages", nargs="+", required=True, help="Capabilities to train in sequence")
    parser.add_argument("--output-dir", required=True, help="Root output directory")
    parser.add_argument("--sample-size", type=int, default=100, help="Samples per stage (for testing/limited runs)")
    
    args = parser.parse_args()
    
    # Ensure src is in path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    run_sequential_training(args.base_model, args.stages, args.output_dir, args.sample_size)

if __name__ == "__main__":
    main()
