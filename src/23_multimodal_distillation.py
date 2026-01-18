#!/usr/bin/env python3
"""
23_multimodal_distillation.py
Run the Multimodal Data Processor to format and split real data (Vision/Audio).
Note: "Distillation" name is kept for script compatibility, but it now performs formatting/splitting.

Usage:
  python 23_multimodal_distillation.py --input-dir /path/to/raw
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from multimodal import MultimodalDataProcessor
from utils.logging_config import setup_logger, log_header, log_completion

logger = setup_logger(__name__, "logs/multimodal_distillation.log")

CONFIG = {
    "default_input_base": "/mnt/e/data/multimodal",
}

def main():
    parser = argparse.ArgumentParser()
    # Modality arg is optional now as processor handles all found in dir, 
    # but we keep args for compatibility or specific targeting if we want to expand processor later.
    parser.add_argument("--modality", choices=["vision", "audio", "video"], help="Optional: specific modality (processor handles all found)")
    parser.add_argument("--teacher", default="mock-teacher", help="Ignored (Historic)")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None, help="Ignored (Processor uses internal structure)")
    
    args = parser.parse_args()
    
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = Path(CONFIG["default_input_base"])
        
    log_header(logger, "MULTIMODAL DATA PROCESSING", {
        "Input": str(input_dir),
        "Action": "Format & Split (Train/Val/Test)"
    })
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        logger.info("Please run 'run_multimodal_pipeline.sh download' first.")
        sys.exit(1)
        
    processor = MultimodalDataProcessor(str(input_dir))
    
    # Process specific or all
    if args.modality == "vision":
        if (input_dir / "vision").exists():
            processor.process_vision(input_dir / "vision")
    elif args.modality == "audio":
        if (input_dir / "audio").exists():
            processor.process_audio(input_dir / "audio")
    else:
        # Run all found
        processor.run()
    
    log_completion(logger, "Multimodal Processing", 0, 0, 0, 0, 0, 0.0)

if __name__ == "__main__":
    main()
