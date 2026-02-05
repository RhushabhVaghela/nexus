#!/usr/bin/env python3
"""
Nexus Explain CLI
Entry point for generating explanations via CLI.
"""
import sys
import argparse
from src.nexus.optimization.remotion_engine import RemotionExplainerEngine, OmniInference

def main():
    parser = argparse.ArgumentParser(description="Nexus Explainer CLI")
    parser.add_argument("prompt", type=str, help="What do you want to explain?")
    parser.add_argument("--narrate", action="store_true", help="Generate narration")
    
    args = parser.parse_args()
    
    # Initialize engine
    # Note: This is a stub for the test to utilize
    model_path = "/mnt/e/data/models/Qwen2.5-0.5B" 
    engine = RemotionExplainerEngine(model_path=model_path)
    
    # Generate video
    video_path = engine.generate_video(args.prompt, narrate=args.narrate)
    print(f"Explanation generated at: {video_path}")

if __name__ == "__main__":
    main()
