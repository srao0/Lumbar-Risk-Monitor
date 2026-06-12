#!/usr/bin/env python3
"""
Phase II.C IMU-only fallback route: frozen varied-session testing.

Loads frozen fallback RF_IMU models only. It has no retraining path and writes
results separately from the full-hybrid Phase II.C evaluator.
"""

from __future__ import annotations

# Make the project package importable when this file is run directly.
import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = _Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PKG_ROOT))


import argparse

from scripts.phase_runners.phase_runner_utils import python_cmd, run_step


def main() -> None:
    """Orchestrate held-out varied-session evaluation of the frozen fallback models.

    Ordered steps: validate the fallback varied-test session files/labels →
    aggregate evaluation of the frozen RF_IMU models against those sessions.
    Strictly evaluation-only — there is no retraining path — and writes results
    under ``--results_dir`` separately from the full-hybrid Phase II.C output.
    """
    parser = argparse.ArgumentParser(
        description="Run Phase II.C held-out evaluation for IMU-only fallback models."
    )
    parser.add_argument("--data_dir", default="data/real/varied_test_fallback")
    parser.add_argument("--models_dir", default="ml/models/phase2_fallback_protocol")
    parser.add_argument("--results_dir", default="results/phase2_varied_test_fallback")
    parser.add_argument("--expected_participants", type=int, default=1)
    parser.add_argument("--skip_pipeline", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # Every path (data, models, results) must be self-evidently a fallback
    # folder so contingency evaluation output can never be mixed with hybrid.
    for label, value in {
        "data_dir": args.data_dir,
        "models_dir": args.models_dir,
        "results_dir": args.results_dir,
    }.items():
        if "fallback" not in value.lower():
            parser.error(f"{label} must be a clearly labelled fallback folder.")

    print("\nPhase II.C: IMU-Only Fallback Varied-Session Evaluation")
    print("Contingency evidence only: frozen RF_IMU, no EMG, no FIS, no retraining.")
    print("Report separately from full_hybrid results.")

    run_step(
        "Validate fallback varied-test session files and labels",
        python_cmd(
            "scripts/datasets/validate_phase2_dataset.py",
            "--data_dir", args.data_dir,
            "--expected_participants", str(args.expected_participants),
            "--phase", "Phase II.C",
            "--mode", "imu_only_fallback",
            "--skip_combined",
        ),
        dry_run=args.dry_run,
    )

    evaluator_args = [
        "scripts/evaluation/evaluate_phase2_varied_test.py",
        "--data_dir", args.data_dir,
        "--models_dir", args.models_dir,
        "--results_dir", args.results_dir,
        "--mode", "imu_only_fallback",
    ]
    if args.skip_pipeline:
        evaluator_args.append("--skip_pipeline")
    if args.force:
        evaluator_args.append("--force")
    run_step(
        "Aggregate frozen fallback RF_IMU evaluation",
        python_cmd(*evaluator_args),
        dry_run=args.dry_run,
    )

    print("\nPhase II.C fallback route complete.")


if __name__ == "__main__":
    main()
