"""
Compatibility shim: expose the internal `src` package under the
`nexus` top-level package name so tests and imports that use
`import nexus` continue to work.

This module maps `nexus` to `src` by inserting the already-imported
`src` package into `sys.modules` as `nexus`. This is intentionally
lightweight and non-invasive.
"""

from __future__ import annotations

import importlib
import sys


def _ensure_nexus_alias() -> None:
    """Import the `src` package and register it as `nexus` in sys.modules.

    Many tests and modules import `nexus.*` while the repository places
    package code under the `src` package. Registering the alias makes
    both import styles work without editing test code.
    """
    try:
        src_pkg = importlib.import_module("src")
    except Exception:
        # If `src` is not importable yet, skip aliasing; normal import
        # errors will surface when running tests in the correct env.
        return

    # Insert alias so `import nexus` returns the `src` package module
    # object. Also ensure future imports like `import nexus.models`
    # resolve using the same underlying package paths.
    sys.modules.setdefault("nexus", src_pkg)


_ensure_nexus_alias()
