#!/usr/bin/env python3
"""Setup script for Nexus package.

This file provides backwards compatibility for older tools
and editable installs. Primary configuration is in pyproject.toml.

Usage:
    pip install -e .        # Editable install
    pip install .           # Regular install
"""

from setuptools import setup

# All configuration lives in pyproject.toml
setup()
