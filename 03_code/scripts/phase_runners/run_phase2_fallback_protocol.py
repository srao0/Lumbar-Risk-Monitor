#!/usr/bin/env python3
"""
Phase II.A IMU-only fallback route: protocol training/checking.

This route trains the RF_IMU probability estimator on protocol-labelled
IMU-only fallback sessions. The deployed fallback decision layer is then:

    RF_IMU -> R_IMU + interpretable IMU features -> IMU Mamdani FIS

Outputs remain separate from the full-hybrid Phase II.A route.
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
    """Orchestrate the IMU-only fallback training route for Phase II.A.

    Ordered steps: validate fallback session files/protocol labels → extract
    shared IMU features → validate the combined feature table → train the
    fallback RF_IMU stage → validate frozen-model provenance. Produces frozen
    fallback models under ``--models_dir`` and evaluation outputs under
    ``--eval_dir``, both kept separate from the full-hybrid route so the
    contingency evidence is never conflated with the headline hybrid result.
    """
    parser = argparse.ArgumentParser(
        description="Run Phase II.A IMU-only fallback protocol checks/training."
    )
    parser.add_argument("--data_dir", default="data/real/protocol_train_fallback")
    parser.add_argument("--models_dir", default="ml/models/phase2_fallback_protocol")
    parser.add_argument("--eval_dir", default="ml/evaluation/phase2_fallback_protocol")
    parser.add_argument("--expected_participants", type=int, default=1)
    parser.add_argument("--cv_group", choices=["session", "participant"], default="session")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_pipeline", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # Guard against pointing the fallback route at full-hybrid folders: every
    # path must be self-evidently a fallback folder so outputs can never be
    # silently mixed with the hybrid evidence.
    if "fallback" not in args.data_dir.lower():
        parser.error("Fallback data_dir must be a clearly labelled fallback folder.")
    if "fallback" not in args.models_dir.lower() or "fallback" not in args.eval_dir.lower():
        parser.error("Fallback model/evaluation outputs must use clearly labelled fallback folders.")

    print("\nPhase II.A: IMU-Only Fallback Protocol Route")
    print("This is part of Phase II.A, reported separately from full-hybrid evidence.")
    print("Fallback route: RF_IMU probability estimator + IMU-only Mamdani FIS.")
    print("Full-hybrid route remains RF_IMU + LR_EMG + Mamdani FIS when EMG exists.")
    print(f"CV grouping: {args.cv_group}")

    run_step(
        "Validate fallback session files and protocol labels",
        python_cmd(
            "scripts/datasets/validate_phase2_dataset.py",
            "--data_dir", args.data_dir,
            "--expected_participants", str(args.expected_participants),
            "--phase", "Phase II.A",
            "--mode", "imu_only_fallback",
            "--skip_combined",
        ),
        dry_run=args.dry_run,
    )

    if not args.skip_pipeline:
        pipeline_args = [
            "signal_processing.pipeline",
            "--data_dir", args.data_dir,
            "--label_source", "protocol",
            "--phase", "Phase II.A",
            "--mode", "imu_only_fallback",
        ]
        if args.force:
            pipeline_args.append("--force")
        run_step(
            "Extract shared IMU fallback features",
            python_module(*pipeline_args),
            dry_run=args.dry_run,
        )

    run_step(
        "Validate fallback combined feature table",
        python_cmd(
            "scripts/datasets/validate_phase2_dataset.py",
            "--data_dir", args.data_dir,
            "--expected_participants", str(args.expected_participants),
            "--phase", "Phase II.A",
            "--mode", "imu_only_fallback",
        ),
        dry_run=args.dry_run,
    )

    if not args.skip_training:
        training_args = [
            "ml/training/train_classifier.py",
            "--data_dir", args.data_dir,
            "--label_source", "protocol",
            "--cv_group", args.cv_group,
            "--seed", str(args.seed),
            "--models_dir", args.models_dir,
            "--eval_dir", args.eval_dir,
            "--write_phase2_provenance",
            "--operating_mode", "imu_only_fallback",
            "--fallback_rf_imu_only",
        ]
        if args.force:
            training_args.append("--force")
        run_step(
            "Train fallback RF_IMU stage",
            python_cmd(*training_args),
            dry_run=args.dry_run,
        )
        run_step(
            "Validate fallback frozen model provenance",
            python_cmd(
                "scripts/datasets/model_provenance.py",
                "--models_dir", args.models_dir,
                "--mode", "imu_only_fallback",
            ),
            dry_run=args.dry_run,
        )

    print("\nPhase II.A fallback route complete.")


if __name__ == "__main__":
    main()
