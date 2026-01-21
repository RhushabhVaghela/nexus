#!/usr/bin/env python3
"""
verify_all_datasets.py
Smoke test to verify that ALL defined/discovered datasets can be loaded and iterated.
Satisfies user requirement: "take sample size of all the datasets (and not one) ... test them"
"""

import sys
import os
from pathlib import Path
import logging
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics_tracker import ALL_DATASETS, MetricsTracker, TestDetailMetrics
from src.data.streaming_trainer import StreamingDatasetLoader, StreamingConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_datasets():
    logger.info("🔍 Verifying ALL Datasets (Smoke Test)...")
    
    tracker = MetricsTracker()
    results = []
    
    # Flatten datasets
    all_paths = []
    for cat, paths in ALL_DATASETS.items():
        all_paths.extend([(cat, p) for p in paths])
        
    logger.info(f"Found {len(all_paths)} total dataset paths across {len(ALL_DATASETS)} categories.")
    
    passes = 0
    failures = 0
    
    for cat, path_str in all_paths:
        path = Path(path_str)
        test_id = f"dataset::{cat}::{path.name}"
        logger.info(f"Testing: {path.name} ({cat})")
        
        try:
            # Use Streaming Loader to test loading 1 sample
            # This handles JSONL, Text, Giant files, Videos etc.
            config = StreamingConfig(buffer_size=10, max_samples=1)
            loader = StreamingDatasetLoader([str(path)], config)
            ds = loader.get_streaming_dataset()
            
            # Try to grab one sample
            sample = next(iter(ds))
            
            if sample:
                logger.info(f"   ✅ Success! Sample keys: {list(sample.keys())}")
                tracker.log_test_detail(TestDetailMetrics(
                    test_id=test_id,
                    name=f"load_{path.name}",
                    outcome="passed",
                    duration_s=0.1
                ))
                passes += 1
            else:
                logger.warning(f"   ⚠️  Empty dataset: {path}")
                tracker.log_test_detail(TestDetailMetrics(
                    test_id=test_id,
                    name=f"load_{path.name}",
                    outcome="failed",
                    error="Empty dataset",
                    duration_s=0.1
                ))
                failures += 1
                
        except Exception as e:
            logger.error(f"   ❌ Failed: {e}")
            tracker.log_test_detail(TestDetailMetrics(
                test_id=test_id,
                name=f"load_{path.name}",
                outcome="failed",
                error=str(e),
                duration_s=0.1
            ))
            failures += 1

    logger.info("="*60)
    logger.info(f"Summary: {passes} Passed, {failures} Failed")
    logger.info("Results logged to test_details.csv")
    
    if failures > 0:
        sys.exit(1)

if __name__ == "__main__":
    if "src" not in sys.modules:
        pass # Already handled by sys.path
    verify_datasets()
