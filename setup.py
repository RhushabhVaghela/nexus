#!/usr/bin/env python3
"""Setup script for Nexus package.

This file provides backwards compatibility for older tools
and editable installs. Primary configuration is in pyproject.toml.

Usage:
    pip install -e .        # Editable install
    pip install .           # Regular install
    python setup.py sdist   # Build source distribution
"""

from setuptools import setup

# Primary configuration in pyproject.toml
setup(
    # These are fallback values if pyproject.toml is not present
    name="nexus",
    use_scm_version={
        "write_to": "src/nexus/_version.py",
        "fallback_version": "6.1.0",
    },
    setup_requires=["setuptools_scm>=6.2"],
)