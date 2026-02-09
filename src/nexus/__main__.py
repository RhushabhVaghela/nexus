"""
Allow running Nexus as a module: python -m nexus

Delegates to the CLI entry point.
"""

import sys

from nexus.cli.nexus_cli import main

sys.exit(main())
