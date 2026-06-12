#!/usr/bin/env python3
"""Compatibility entrypoint for the replay dashboard demo.

The canonical dashboard lives at ``scripts/replay_dashboard.py``. Keep this
thin wrapper so ``streamlit run scripts/demo/replay_dashboard.py`` cannot drift
behind the real implementation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_dashboard_module():
    """Import and return the canonical scripts/replay_dashboard.py by path so this demo-folder shim re-exports its main() without duplicating the implementation."""
    dashboard_path = Path(__file__).resolve().parents[1] / "replay_dashboard.py"
    if not dashboard_path.exists():
        raise FileNotFoundError(f"Canonical replay dashboard not found: {dashboard_path}")

    spec = importlib.util.spec_from_file_location("_canonical_replay_dashboard", dashboard_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load replay dashboard from {dashboard_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


main = _load_dashboard_module().main


if __name__ == "__main__":
    main()
