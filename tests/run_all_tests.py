#!/usr/bin/env python3
"""
Test runner for all research paper implementation tests.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

TEST_FILES = [
    "test_memorization_classifier.py",
    "test_data_filtering.py",
    "test_temperature_scheduling.py",
    "test_adaptive_repetition.py",
    "test_kv_cache.py",
    "test_multimodal_repetition.py",
]


def run_test(test_file: str, verbose: bool = True) -> bool:
    """Run a single test file."""
    test_path = Path(__file__).parent / test_file
    
    if not test_path.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Running: {test_file}")
    print('='*60)
    
    cmd = [sys.executable, "-m", "pytest", str(test_path)]
    if verbose:
        cmd.append("-v")
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running {test_file}: {e}")
        return False


def run_all_tests():
    """Run all test files."""
    print("="*60)
    print("RESEARCH PAPER IMPLEMENTATION TEST SUITE")
    print("="*60)
    print("\nFeatures being tested:")
    print("  1. Pre-distillation Memorization Classifier (Paper 2601.15394)")
    print("  2. Data Filtering Pipeline (Paper 2601.15394)")
    print("  3. Temperature Scheduling (Paper 2601.15394)")
    print("  4. Hard vs Soft Distillation Analysis (Paper 2601.15394)")
    print("  5. Adaptive Repetition (Paper 2512.14982)")
    print("  6. KV-Cache Optimization (Paper 2512.14982)")
    print("  7. Multimodal Repetition (Paper 2512.14982)")
    
    results = {}
    passed = 0
    failed = 0
    
    for test_file in TEST_FILES:
        success = run_test(test_file)
        results[test_file] = success
        
        if success:
            passed += 1
            print(f"✅ {test_file} PASSED")
        else:
            failed += 1
            print(f"❌ {test_file} FAILED")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total:  {len(TEST_FILES)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


def run_with_coverage():
    """Run tests with coverage report."""
    print("Running tests with coverage...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        str(Path(__file__).parent),
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term"
    ]
    
    subprocess.run(cmd)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run research paper implementation tests")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    parser.add_argument("--test", type=str, help="Run specific test file")
    
    args = parser.parse_args()
    
    if args.coverage:
        run_with_coverage()
    elif args.test:
        success = run_test(args.test)
        sys.exit(0 if success else 1)
    else:
        sys.exit(run_all_tests())