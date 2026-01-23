import argparse
import subprocess
import os
import sys
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Nexus Universal Explainer CLI")
    parser.add_argument("prompt", help="What do you want to explain? (e.g. 'How a neuron works')")
    parser.add_argument("--model", default="/mnt/e/data/output/trained/remotion-explainer", help="Path to trained explainer model")
    parser.add_argument("--output", default="explanation.mp4", help="Output filename")
    parser.add_argument("--narrate", action="store_true", help="Generate audio narration")
    
    args = parser.parse_args()

    print(f"\033[94m🚀 Nexus Explainer:\033[0m {args.prompt}")

    # 1. Initialize Engine
    from src.inference.remotion_engine import RemotionExplainerEngine
    
    try:
        engine = RemotionExplainerEngine(model_path=args.model)
        
        # 2. Generate and Render
        video_path = engine.generate_video(
            prompt=args.prompt,
            output_name=args.output,
            narrate=args.narrate
        )
        
        print(f"\n\033[92m✅ Explanation generated successfully: {video_path}\033[0m")
        
    except Exception as e:
        print(f"\n\033[91m❌ Error during generation: {e}\033[0m")
        sys.exit(1)

if __name__ == "__main__":
    main()
