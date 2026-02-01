#!/usr/bin/env python3
"""
Nexus Test Runner

A comprehensive test runner that intelligently categorizes and skips tests
that require real models, distributed systems, or GPU resources by default.

Usage:
    # Run all tests except real model and distributed tests (default)
    python scripts/run_tests.py

    # Include tests requiring real models
    python scripts/run_tests.py --real-models

    # Include distributed/cluster tests
    python scripts/run_tests.py --distributed

    # Include GPU-specific tests
    python scripts/run_tests.py --gpu

    # Run everything
    python scripts/run_tests.py --all

    # Only unit tests
    python scripts/run_tests.py --unit-only

    # Only integration tests
    python scripts/run_tests.py --integration-only

    # Generate test report
    python scripts/run_tests.py --report

    # Run with verbose output
    python scripts/run_tests.py -v

    # Run specific test file
    python scripts/run_tests.py tests/unit/test_example.py

    # Run with coverage
    python scripts/run_tests.py --coverage
"""

import argparse
import subprocess
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Set, Optional
import warnings

# Test categories with their markers
TEST_CATEGORIES = {
    "real_model": {
        "marker": "real_model",
        "description": "Tests requiring downloading real models from HuggingFace",
        "examples": [
            "tests/integration/test_end_to_end_real_models.py",
            "tests/conftest.py (real_text_model fixture)",
        ],
        "default_skip": True,
    },
    "distributed": {
        "marker": "distributed",
        "description": "Tests requiring distributed/multi-node setup (torch.distributed, mpi4py)",
        "examples": [
            "tests/unit/test_orchestration_scripts_3.py (distributed training)",
            "tests/unit/test_ring_attention.py (multi-GPU ring attention)",
        ],
        "default_skip": True,
    },
    "gpu": {
        "marker": "gpu",
        "description": "Tests requiring GPU/CUDA",
        "examples": [
            "tests/unit_streaming/test_streaming_trainer.py",
            "tests/integration/test_multimodal_encoders.py",
        ],
        "default_skip": True,
    },
    "slow": {
        "marker": "slow",
        "description": "Slow tests that take a long time to run",
        "examples": [
            "tests/integration/test_load_performance.py",
            "tests/integration/test_end_to_end_real_models.py",
        ],
        "default_skip": True,
    },
    "integration": {
        "marker": "integration",
        "description": "Integration tests that test multiple components",
        "examples": [
            "tests/integration/test_*.py",
        ],
        "default_skip": False,
    },
    "e2e": {
        "marker": "e2e",
        "description": "End-to-end tests",
        "examples": [
            "tests/e2e/test_*.py",
        ],
        "default_skip": False,
    },
    "benchmark": {
        "marker": "benchmark",
        "description": "Performance benchmark tests",
        "examples": [
            "benchmarks/test_*.py",
        ],
        "default_skip": True,
    },
    "chaos": {
        "marker": "chaos",
        "description": "Chaos engineering tests (fault injection)",
        "examples": [
            "tests/chaos/test_*.py",
        ],
        "default_skip": True,
    },
}


def get_project_root() -> Path:
    """Get the project root directory."""
    script_path = Path(__file__).resolve()
    return script_path.parent.parent


def detect_cuda() -> bool:
    """Detect if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def build_pytest_args(args: argparse.Namespace) -> List[str]:
    """
    Build pytest arguments based on command-line options.

    Args:
        args: Parsed command-line arguments

    Returns:
        List of pytest arguments
    """
    pytest_args = ["pytest"]

    # Verbose output
    if args.verbose:
        pytest_args.append("-v")

    # Collect-only mode (just list tests)
    if args.collect_only:
        pytest_args.append("--collect-only")

    # Coverage
    if args.coverage:
        pytest_args.extend(["--cov=src", "--cov-report=html", "--cov-report=term-missing"])

    # Test paths - if specific paths provided, use those
    if args.test_paths:
        pytest_args.extend(args.test_paths)
    else:
        # Default test paths
        if args.unit_only:
            pytest_args.append("tests/unit")
        elif args.integration_only:
            pytest_args.append("tests/integration")
        elif args.benchmark_only:
            pytest_args.append("benchmarks")
        else:
            pytest_args.append("tests")

    # Determine what to skip
    markers_to_skip = []

    # If --all is specified, don't skip anything
    if not args.all:
        # Check each category
        if not args.real_models:
            markers_to_skip.append("real_model")

        if not args.distributed:
            markers_to_skip.append("distributed")

        if not args.gpu:
            markers_to_skip.append("gpu")

        if not args.slow:
            markers_to_skip.append("slow")

        if not args.benchmark:
            markers_to_skip.append("benchmark")

        if not args.chaos:
            markers_to_skip.append("chaos")

        # Skip e2e tests unless explicitly requested or running all
        if not args.e2e and not args.integration_only:
            markers_to_skip.append("e2e")

    # Build the skip expression
    if markers_to_skip:
        skip_expr = " or ".join(markers_to_skip)
        pytest_args.extend(["-m", f"not ({skip_expr})"])

    # Pass through real-models flag to conftest.py if enabled
    if args.real_models or args.all:
        pytest_args.append("--use-real-models")

    if args.small_model:
        pytest_args.append("--small-model")

    if args.full_tests:
        pytest_args.append("--full-tests")

    # Additional pytest arguments
    if args.pytest_args:
        pytest_args.extend(args.pytest_args)

    # JUnit XML output for CI
    if args.junit_xml:
        pytest_args.extend(["--junitxml", args.junit_xml])

    return pytest_args


def print_test_plan(args: argparse.Namespace, pytest_args: List[str]):
    """
    Print a summary of what will be tested and what will be skipped.

    Args:
        args: Parsed command-line arguments
        pytest_args: Built pytest arguments
    """
    print("=" * 70)
    print("NEXUS TEST RUNNER")
    print("=" * 70)
    print()

    # Print what's being included
    print("Test Categories:")
    print("-" * 40)

    categories_status = []

    for cat_id, cat_info in TEST_CATEGORIES.items():
        # Determine if this category is enabled
        enabled = False

        if args.all:
            enabled = True
        elif cat_id == "real_model" and args.real_models:
            enabled = True
        elif cat_id == "distributed" and args.distributed:
            enabled = True
        elif cat_id == "gpu" and args.gpu:
            enabled = True
        elif cat_id == "slow" and args.slow:
            enabled = True
        elif cat_id == "benchmark" and (args.benchmark or args.benchmark_only):
            enabled = True
        elif cat_id == "chaos" and args.chaos:
            enabled = True
        elif cat_id == "integration":
            if args.integration_only or (not args.unit_only and not args.benchmark_only):
                enabled = True
        elif cat_id == "e2e":
            if args.e2e or args.integration_only:
                enabled = True
        elif cat_id == "unit":
            if args.unit_only:
                enabled = True

        # Default behavior
        if cat_id not in ["integration", "e2e"] and not any([
            args.all, args.real_models, args.distributed, args.gpu,
            args.slow, args.benchmark, args.chaos, args.unit_only,
            args.integration_only, args.benchmark_only, args.e2e
        ]):
            if cat_id in ["integration"]:
                enabled = True

        status = "✓ ENABLED" if enabled else "✗ SKIPPED"
        categories_status.append((cat_info["marker"], status, cat_info["description"]))
        print(f"  {status:12} {cat_info['marker']:15} - {cat_info['description']}")

    print()

    # Print test paths
    print("Test Paths:")
    print("-" * 40)
    if args.test_paths:
        for path in args.test_paths:
            print(f"  {path}")
    elif args.unit_only:
        print("  tests/unit")
    elif args.integration_only:
        print("  tests/integration")
        print("  tests/e2e")
    elif args.benchmark_only:
        print("  benchmarks")
    else:
        print("  tests")
    print()

    # Print CUDA status
    cuda_available = detect_cuda()
    print("Hardware:")
    print("-" * 40)
    print(f"  CUDA Available: {'Yes' if cuda_available else 'No'}")
    if cuda_available:
        try:
            import torch
            print(f"  CUDA Devices: {torch.cuda.device_count()}")
            print(f"  CUDA Version: {torch.version.cuda}")
        except:
            pass
    print()

    # Print pytest command
    print("Pytest Command:")
    print("-" * 40)
    print(f"  {' '.join(pytest_args)}")
    print()
    print("=" * 70)
    print()


def generate_report(result: subprocess.CompletedProcess, args: argparse.Namespace, duration: float):
    """
    Generate a test report in JSON format.

    Args:
        result: CompletedProcess from running pytest
        args: Command-line arguments
        duration: Test duration in seconds
    """
    if not args.report:
        return

    report_data = {
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": duration,
        "exit_code": result.returncode,
        "configuration": {
            "real_models": args.real_models or args.all,
            "distributed": args.distributed or args.all,
            "gpu": args.gpu or args.all,
            "slow": args.slow or args.all,
            "benchmark": args.benchmark or args.all or args.benchmark_only,
            "chaos": args.chaos or args.all,
            "unit_only": args.unit_only,
            "integration_only": args.integration_only,
        },
        "cuda_available": detect_cuda(),
    }

    # Write report
    report_dir = Path("test_reports")
    report_dir.mkdir(exist_ok=True)

    report_file = report_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)

    print(f"\nReport saved to: {report_file}")


def print_summary(result: subprocess.CompletedProcess, duration: float):
    """
    Print a summary of test results.

    Args:
        result: CompletedProcess from running pytest
        duration: Test duration in seconds
    """
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Duration: {duration:.2f} seconds")
    print(f"Exit Code: {result.returncode}")

    if result.returncode == 0:
        print("Status: ✓ PASSED")
    elif result.returncode == 5:
        print("Status: ⚠ NO TESTS COLLECTED")
    else:
        print("Status: ✗ FAILED")

    print("=" * 70)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Nexus Test Runner - Run tests with intelligent categorization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all default tests (excludes real models, distributed, GPU, slow)
  %(prog)s

  # Include real model tests
  %(prog)s --real-models

  # Run only unit tests
  %(prog)s --unit-only

  # Run everything including real models and distributed tests
  %(prog)s --all

  # Run specific test file
  %(prog)s tests/unit/test_example.py -v

  # Generate coverage report
  %(prog)s --coverage --report
        """
    )

    # Test category flags
    category_group = parser.add_argument_group("Test Categories")
    category_group.add_argument(
        "--real-models",
        action="store_true",
        help="Include tests requiring real models from HuggingFace"
    )
    category_group.add_argument(
        "--distributed",
        action="store_true",
        help="Include tests requiring distributed setup (torch.distributed, mpi4py)"
    )
    category_group.add_argument(
        "--gpu",
        action="store_true",
        help="Include tests requiring GPU/CUDA"
    )
    category_group.add_argument(
        "--slow",
        action="store_true",
        help="Include slow tests"
    )
    category_group.add_argument(
        "--benchmark",
        action="store_true",
        help="Include benchmark tests"
    )
    category_group.add_argument(
        "--chaos",
        action="store_true",
        help="Include chaos engineering tests"
    )
    category_group.add_argument(
        "--e2e",
        action="store_true",
        help="Include end-to-end tests"
    )
    category_group.add_argument(
        "--all",
        action="store_true",
        help="Run all tests including real models, distributed, and GPU tests"
    )

    # Test selection flags
    selection_group = parser.add_argument_group("Test Selection")
    selection_group.add_argument(
        "--unit-only",
        action="store_true",
        help="Run only unit tests"
    )
    selection_group.add_argument(
        "--integration-only",
        action="store_true",
        help="Run only integration and e2e tests"
    )
    selection_group.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Run only benchmark tests"
    )

    # Conftest.py passthrough flags
    conftest_group = parser.add_argument_group("Model Options (passed to conftest)")
    conftest_group.add_argument(
        "--small-model",
        "-S",
        action="store_true",
        help="Use small models for testing (passed to conftest)"
    )
    conftest_group.add_argument(
        "--full-tests",
        "-F",
        action="store_true",
        help="Run full test suite including slow tests (passed to conftest)"
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    output_group.add_argument(
        "--report",
        action="store_true",
        help="Generate JSON test report"
    )
    output_group.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    output_group.add_argument(
        "--junit-xml",
        metavar="PATH",
        help="Generate JUnit XML report for CI"
    )
    output_group.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect tests, don't run them"
    )

    # Positional arguments for test paths
    parser.add_argument(
        "test_paths",
        nargs="*",
        help="Specific test files or directories to run"
    )

    # Additional pytest arguments
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to pytest (use --pytest-args -- [args])"
    )

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Ensure we're in the project root
    os.chdir(get_project_root())

    # Build pytest arguments
    pytest_args = build_pytest_args(args)

    # Print test plan
    print_test_plan(args, pytest_args)

    # If collect-only, just list tests and exit
    if args.collect_only:
        result = subprocess.run(pytest_args)
        return result.returncode

    # Run tests
    print("Running tests...\n")
    start_time = datetime.now()

    try:
        result = subprocess.run(pytest_args)
    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user.")
        return 130

    duration = (datetime.now() - start_time).total_seconds()

    # Print summary
    print_summary(result, duration)

    # Generate report if requested
    generate_report(result, args, duration)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
