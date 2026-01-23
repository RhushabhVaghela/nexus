#!/usr/bin/env python3
import os
import sys
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import multiprocessing

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.universal_loader import UniversalDataLoader
from src.multimodal.decoders import OmniDecoder
from src.utils.corruption_tracker import tracker as corruption_tracker

class AssetVerifier:
    def __init__(self, data_dir: str, parallel: int = 4):
        self.base_dir = Path(data_dir)
        self.parallel = parallel
        self.decoder = OmniDecoder()
        self.summary = []

    def verify_encoders(self):
        print("🛠️ Verifying Encoders & Decoders...")
        try:
            # Test image decoder with a dummy path or real one if exists
            # Just checking if AutoProcessors loaded is enough for this phase
            print(f"   - SigLIP 2: {'✅ Loaded' if self.decoder.image.processor else '❌ Failed'}")
            print(f"   - Whisper V3 Turbo: {'✅ Loaded' if self.decoder.audio.processor else '❌ Failed'}")
        except Exception as e:
            print(f"   - Error: {e}")

    def verify_datasets(self, filter_name: Optional[str] = None):
        dataset_dirs = [
            self.base_dir / "datasets/multimodal",
            self.base_dir / "datasets/code",
            self.base_dir / "datasets/general",
            self.base_dir / "datasets/uncensored",
            self.base_dir / "benchmarks"
        ]
        
        all_dataset_paths = []
        for d in dataset_dirs:
            if d.exists():
                all_dataset_paths.extend([p for p in d.iterdir() if p.is_dir()])

        if filter_name:
            all_dataset_paths = [p for p in all_dataset_paths if filter_name.lower() in p.name.lower()]

        print(f"📂 Found {len(all_dataset_paths)} datasets/benchmarks. Starting verification...")

        for ds_path in all_dataset_paths:
            self._verify_single_dataset(ds_path)

    def _verify_single_dataset(self, ds_path: Path):
        print(f"\n🔍 Checking {ds_path.name}...")
        loader = UniversalDataLoader(ds_path)
        fmt = loader.detect_format()
        
        if fmt == "unknown":
            print(f"   ⚠️ Unknown format for {ds_path.name}. Skipping.")
            self.summary.append({"name": ds_path.name, "status": "Unknown Format", "samples": 0})
            return

        try:
            # We don't know total count without scanning, but for sharded we can get it from GlobalIndexMap
            # For this test, we'll just try to load indices 0, 1, 2, 3, 4
            samples_loaded = 0
            for i in range(5):
                try:
                    sample = loader.load_sample(i)
                    if sample:
                        # Test processing
                        self._test_processing(sample)
                        samples_loaded += 1
                except Exception as e:
                    print(f"   ❌ Sample {i} failed: {e}")
            
            status = "OK" if samples_loaded == 5 else "Partial"
            if samples_loaded == 0: status = "FAILED"
            
            print(f"   ✅ Loaded {samples_loaded}/5 samples. Format: {fmt}")
            self.summary.append({"name": ds_path.name, "status": status, "samples": samples_loaded, "format": fmt})
            
        except Exception as e:
            print(f"   ❌ Dataset failed: {e}")
            self.summary.append({"name": ds_path.name, "status": "ERROR", "error": str(e)})

    def _test_processing(self, sample: Dict[str, Any]):
        """Attempts full tensor processing for a sample."""
        modalities = sample.get("modalities", {})
        
        for mod, assets in modalities.items():
            for asset in assets:
                path = asset.get("path")
                if path and os.path.exists(path):
                    # Trigger decode
                    res = self.decoder.decode(path, mod)
                    # Clear memory
                    if "pixel_values" in res: del res["pixel_values"]
                    if "input_features" in res: del res["input_features"]
                elif path:
                    corruption_tracker.log_corrupted(path, "File missing from disk")

    def print_report(self):
        print("\n" + "="*80)
        print(f"{'DATASET NAME':<40} | {'FORMAT':<15} | {'STATUS':<10} | {'SAMPLES'}")
        print("-" * 80)
        for entry in self.summary:
            print(f"{entry['name'][:40]:<40} | {entry.get('format', 'N/A'):<15} | {entry['status']:<10} | {entry.get('samples', 0)}/5")
        print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/mnt/e/data")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--filter", default=None)
    args = parser.parse_args()

    verifier = AssetVerifier(args.data_dir, args.parallel)
    verifier.verify_encoders()
    verifier.verify_datasets(filter_name=args.filter)
    verifier.print_report()
