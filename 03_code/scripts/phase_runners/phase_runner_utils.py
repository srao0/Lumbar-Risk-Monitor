#!/usr/bin/env python3
"""
Small helpers for examiner-facing phase runner scripts.

These helpers keep the phase scripts thin: each runner orchestrates existing
project entry points instead of duplicating signal-processing or ML logic.
"""

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_step(name: str, args: list[str], dry_run: bool = False) -> None:
    """Print and run one project command from the repository root."""
    print(f"\n[{name}]")
    print("  " + " ".join(args))
    if dry_run:
        return

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    completed = subprocess.run(args, cwd=ROOT, env=env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def python_cmd(module_or_script: str, *args: str) -> list[str]:
    """Build a Python command using the current interpreter."""
    return [sys.executable, module_or_script, *args]


def python_module(module: str, *args: str) -> list[str]:
    """Build a `python -m module` command using the current interpreter."""
    return [sys.executable, "-m", module, *args]
