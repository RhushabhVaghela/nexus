"""
conftest.py - Pytest configuration with proper import resolution
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

try:
    if torch.__spec__ is None:
        from importlib.machinery import ModuleSpec

        torch.__spec__ = ModuleSpec(
            name="torch", loader=None, origin=getattr(torch, "__file__", None)
        )
except Exception:
    pass
