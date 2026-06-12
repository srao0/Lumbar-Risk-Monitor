#!/usr/bin/env python3
"""
Phase II.C runner: held-out varied-movement generalisation test.

This phase evaluates independently labelled varied-movement sessions using the
aggregate frozen-model evaluator. It contains no training path.
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
    """Orchestrate the held-out varied-movement generalisation test (Phase II.C).

    Ordered steps: validate the varied-test session files/labels → aggregate
    evaluation of the declared frozen system (full-hybrid or fallback) →
    optional single-session replay demonstration. Strictly evaluation-only: the
    runner never retrains on Phase II.C data, since this phase measures how the
    frozen Phase II.A system generalises to unseen movement. Writes aggregate
    results under ``--results_dir``.
    """
    parser = argparse.ArgumentParser(
        description="Run Phase II.C held-out varied-movement evaluation with frozen models."
    )
    parser.add_argument("--data_dir", default="data/real/varied_test")
    parser.add_argument(
        "--models_dir",
        default="ml/models/phase2_protocol",
        help="Directory containing frozen models trained only on Phase II.A protocol data.",
    )
    parser.add_argument("--results_dir", default="results/phase2_varied_test")
    parser.add_argument("--expected_participants", type=int, default=1)
    parser.add_argument("--skip_pipeline", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mode", choices=["full_hybrid", "imu_only_fallback"], default="full_hybrid")
    parser.add_argument(
        "--demo_session",
        default=None,
        help="Optional single held-out session to replay after aggregate evaluation.",
    )
    parser.add_argument("--demo_speed", type=float, default=1000.0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print("\nPhase II.C: Held-Out Varied-Movement Evaluation")
    if args.mode == "full_hybrid":
        print("Final system: frozen RF_IMU + frozen LR_EMG + fixed Mamdani FIS.")
    else:
        print("Contingency system: IMU-only fallback using frozen RF_IMU; reported separately.")
    print("This runner never retrains models on Phase II.C data.")

    run_step(
        "Validate varied-test session files and labels",
        python_cmd(
            "scripts/datasets/validate_phase2_dataset.py",
            "--data_dir", args.data_dir,
            "--expected_participants", str(args.expected_participants),
            "--phase", "Phase II.C",
            "--mode", args.mode,
            "--skip_combined",
        ),
        dry_run=args.dry_run,
    )

    evaluator_args = [
        "scripts/evaluation/evaluate_phase2_varied_test.py",
        "--data_dir", args.data_dir,
        "--models_dir", args.models_dir,
        "--results_dir", args.results_dir,
        "--mode", args.mode,
    ]
    if args.skip_pipeline:
        evaluator_args.append("--skip_pipeline")
    if args.force:
        evaluator_args.append("--force")
    run_step(
        "Aggregate evaluation of declared frozen system mode",
        python_cmd(*evaluator_args),
        dry_run=args.dry_run,
    )

    if args.demo_session:
        run_step(
            "Optional replay demonstration",
            python_cmd(
                "scripts/demo/demo_risk_monitor.py",
                "--session", args.demo_session,
                "--models_dir", args.models_dir,
                "--speed", str(args.demo_speed),
                "--mode", args.mode,
                "--no_plot",
            ),
            dry_run=args.dry_run,
        )

    print("\nPhase II.C complete.")


if __name__ == "__main__":
    main()
