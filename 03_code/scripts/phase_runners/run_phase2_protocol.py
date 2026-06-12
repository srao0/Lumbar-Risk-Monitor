#!/usr/bin/env python3
"""
Phase II.A runner: real protocol data collection and model training/fine-tuning.

The runner validates the real protocol dataset, runs the shared feature
pipeline, validates the combined feature table, and trains with participant-
level cross-validation. It refuses signal-derived training labels through the
shared training code.
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
    """Orchestrate the full-hybrid Phase II.A real-protocol training run.

    Ordered steps: validate session files/labels → extract shared pipeline
    features → validate the combined Phase II feature table → train with
    participant-level leave-one-group-out CV → validate frozen-model provenance.
    Produces the trained hybrid models under ``--models_dir`` and evaluation
    output under ``--eval_dir``. Training uses researcher/protocol labels only;
    signal-derived labels stay diagnostic and are refused by the shared trainer.
    """
    parser = argparse.ArgumentParser(
        description="Run Phase II.A real protocol processing and training."
    )
    parser.add_argument("--data_dir", default="data/real/protocol_train")
    parser.add_argument("--models_dir", default="ml/models/phase2_protocol")
    parser.add_argument("--eval_dir", default="ml/evaluation/phase2_protocol")
    parser.add_argument("--expected_participants", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_pipeline", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mode", choices=["full_hybrid", "imu_only_fallback"], default="full_hybrid")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    # Refuse to let the hybrid runner train fallback models: fallback is a
    # separately reported contingency and has its own dedicated runner.
    if args.mode == "imu_only_fallback" and not args.skip_training:
        parser.error(
            "imu_only_fallback is a separately reported contingency path and must not train "
            "the hybrid Phase II.A models here; use the Phase II.A fallback runner for fallback training."
        )

    print("\nPhase II.A: Real Protocol Data Collection and Model Training/Fine-Tuning")
    print("Uses researcher/protocol labels only. Signal-derived labels are diagnostic.")
    print("Preferred evaluation: participant-level leave-one-group-out CV.")
    print(f"Declared mode: {args.mode}")

    run_step(
        "Validate session files and labels before feature extraction",
        python_cmd(
            "scripts/datasets/validate_phase2_dataset.py",
            "--data_dir", args.data_dir,
            "--expected_participants", str(args.expected_participants),
            "--phase", "Phase II.A",
            "--mode", args.mode,
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
            "--mode", args.mode,
        ]
        if args.force:
            pipeline_args.append("--force")
        run_step(
            "Extract shared pipeline features",
            python_module(*pipeline_args),
            dry_run=args.dry_run,
        )

    run_step(
        "Validate combined Phase II feature table",
        python_cmd(
            "scripts/datasets/validate_phase2_dataset.py",
            "--data_dir", args.data_dir,
            "--expected_participants", str(args.expected_participants),
            "--phase", "Phase II.A",
            "--mode", args.mode,
        ),
        dry_run=args.dry_run,
    )

    if not args.skip_training:
        training_args = [
            "ml/training/train_classifier.py",
            "--data_dir", args.data_dir,
            "--label_source", "protocol",
            "--cv_group", "participant",
            "--seed", str(args.seed),
            "--models_dir", args.models_dir,
            "--eval_dir", args.eval_dir,
            "--write_phase2_provenance",
        ]
        if args.force:
            training_args.append("--force")
        run_step(
            "Train with participant-level CV",
            python_cmd(*training_args),
            dry_run=args.dry_run,
        )
        run_step(
            "Validate frozen model provenance",
            python_cmd(
                "scripts/datasets/model_provenance.py",
                "--models_dir", args.models_dir,
            ),
            dry_run=args.dry_run,
        )

    print("\nPhase II.A complete.")


if __name__ == "__main__":
    main()
