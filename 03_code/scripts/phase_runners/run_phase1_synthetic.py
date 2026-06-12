#!/usr/bin/env python3
"""
Phase I runner: synthetic software-pipeline validation.

This entry point is intentionally examiner-facing. It runs the shared synthetic
generator, shared signal-processing pipeline, and shared ML/evaluation scripts
without introducing a separate synthetic-only processing implementation.
"""

from __future__ import annotations

# Make the project package importable when this file is run directly.
import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = _Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PKG_ROOT))


import argparse

from scripts.phase_runners.phase_runner_utils import python_cmd, python_module, run_step


def main() -> None:
    """Orchestrate the Phase I synthetic run end to end.

    Ordered steps (each skippable for re-runs): generate synthetic sessions →
    extract shared pipeline features → train/evaluate models with session-level
    CV → emit evaluation plots. Produces the synthetic combined_features table,
    trained model artefacts, and evaluation figures under ``--data_dir`` and the
    shared ml/ output folders. Pass-through to the shared scripts only — there is
    deliberately no synthetic-specific processing here.
    """
    parser = argparse.ArgumentParser(
        description="Run Phase I synthetic pipeline validation."
    )
    parser.add_argument("--data_dir", default="data/synthetic")
    parser.add_argument("--n_sessions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_generate", action="store_true")
    parser.add_argument("--skip_pipeline", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print("\nPhase I: Synthetic Pipeline Validation")
    print("This phase checks the software pipeline on controlled synthetic data.")
    print("It is not real participant or real hardware performance evidence.")

    if not args.skip_generate:
        run_step(
            "Generate synthetic sessions",
            python_cmd(
                "scripts/conversion/synthetic_generator.py",
                "--n_sessions", str(args.n_sessions),
                "--seed", str(args.seed),
                "--output_dir", args.data_dir,
            ),
            dry_run=args.dry_run,
        )

    if not args.skip_pipeline:
        run_step(
            "Extract shared pipeline features",
            python_module(
                "signal_processing.pipeline",
                "--data_dir", args.data_dir,
                "--label_source", "protocol",
            ),
            dry_run=args.dry_run,
        )

    if not args.skip_training:
        run_step(
            "Train/evaluate synthetic models",
            python_cmd(
                "ml/training/train_classifier.py",
                "--data_dir", args.data_dir,
                "--label_source", "protocol",
                "--cv_group", "session",
                "--seed", str(args.seed),
            ),
            dry_run=args.dry_run,
        )

    if not args.skip_plots:
        run_step(
            "Generate evaluation plots",
            python_cmd("ml/evaluation/evaluate.py"),
            dry_run=args.dry_run,
        )
        run_step(
            "Generate extra plots",
            python_cmd("ml/evaluation/generate_extra_plots.py"),
            dry_run=args.dry_run,
        )

    print("\nPhase I complete.")


if __name__ == "__main__":
    main()
