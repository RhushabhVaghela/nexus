import subprocess
import pytest
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
NEXUS_EXPLAIN = PROJECT_ROOT / "nexus_explain.py"

def run_command(args):
    """Utility to run a command and return its output and exit code."""
    # Use the same python executable to ensure environment consistency
    result = subprocess.run(
        [sys.executable, str(NEXUS_EXPLAIN)] + args,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT)
    )
    return result

def test_nexus_explain_help():
    """Verify nexus_explain.py --help works."""
    result = run_command(["--help"])
    assert result.returncode == 0
    assert "usage: nexus_explain.py" in result.stdout
    # Check for some expected arguments in help
    assert "--model" in result.stdout
    assert "prompt" in result.stdout

def test_nexus_explain_invalid_arg():
    """Verify nexus_explain.py fails on invalid arguments."""
    result = run_command(["--invalid-argument-xyz"])
    assert result.returncode != 0

def test_nexus_explain_version():
    """Verify nexus_explain.py can report version or at least run without query."""
    # We test with a simple query and dry-run
    result = run_command(["--model", "dummy", "What is Nexus?", "--dry-run"])
    # Dry run should succeed even without a real model
    assert result.returncode == 0
    assert "Dry run mode" in result.stdout

def test_numbered_scripts_execution_help():
    """Verify a sample numbered script can run with --help."""
    script_path = PROJECT_ROOT / "src" / "01_download_real_datasets.py"
    if script_path.exists():
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout or "help" in result.stdout.lower()
