#!/usr/bin/env python3
"""
Train final IMU-only fallback models from the frozen cleaned analysis sets.

Trains the RF stage only. The fallback Mamdani FIS is a fixed,
rule-based decision layer used downstream:

    IMU features -> RF -> R_IMU
    R_IMU + interpretable IMU features -> IMUFallbackFIS -> R_total
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRIMARY_4IMU_FEATURES = [
    "imu_trunk_angle_peak",
    "imu_trunk_angle_mean",
    "imu_angvel_peak",
    "imu_angvel_mean",
    "imu_time_in_risk_zone",
    "imu_time_high_velocity",
    "imu_ldlj",
    "imu_jerk_rms",
    "imu_jerk_peak",
    "imu_ldlj_multiaxis",
    "imu_compensation_index",
    "imu_lumbopelv_ratio",
    "imu_pelvis_angle_peak",
    "imu_pelvis_angle_mean",
    "imu_z_flex",
    "imu_z_vel",
    "imu_z_ldlj",
]

REDUCED_PELVIS_L3_FEATURES = [
    "imu_angvel_peak",
    "imu_angvel_mean",
    "imu_time_high_velocity",
    "imu_ldlj",
    "imu_jerk_rms",
    "imu_jerk_peak",
    "imu_pelvis_angle_peak",
    "imu_pelvis_angle_mean",
    "imu_l3_accel_tilt_peak",
    "imu_l3_accel_tilt_mean",
    "imu_l3_accel_tilt_range",
    "imu_z_vel",
    "imu_z_ldlj",
]


def _display_path(path: Path) -> str:
    """Render a path repo-relative for the metadata (absolute only if it lies outside the project), so the provenance record is portable."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _git_commit_hash() -> str | None:
    """Best-effort current commit hash for provenance; None if git is absent or this is not a checkout."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _load_training_frame(path: Path, features: list[str], target_col: str) -> pd.DataFrame:
    """Load a frozen analysis set and clean it for fitting — drop -1 UNLABELLED rows, drop NaNs in the feature/target columns, and fail loudly if a class or column is missing."""
    df = pd.read_csv(path)
    missing = [col for col in [*features, target_col, "participant_id"] if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df[df[target_col] != -1].copy()
    df = df.dropna(subset=[*features, target_col])
    df[target_col] = df[target_col].astype(int)
    if df[target_col].nunique() != 2:
        raise ValueError(f"{path} does not contain both safe and risky classes after cleaning.")
    return df


def _train_rf(df: pd.DataFrame, features: list[str], target_col: str, seed: int):
    """Fit the deployment RF — the manifest-locked config (500 trees, min_samples_leaf=3, balanced, fixed seed) that the frozen models must reproduce."""
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(df[features], df[target_col].astype(int))
    return model


def _write_feature_importance(model, features: list[str], path: Path) -> None:
    """Dump the RF's feature importances (descending) alongside the model — the evidence behind the reduced Pelvis-L3 feature selection."""
    if not hasattr(model, "feature_importances_"):
        return
    rows = sorted(
        zip(features, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )
    pd.DataFrame(rows, columns=["feature", "importance"]).to_csv(path, index=False)


def _train_one(
    name: str,
    csv_path: Path,
    features: list[str],
    output_path: Path,
    target_col: str,
    seed: int,
) -> dict:
    """Train one model end to end — load+clean, fit, persist the joblib and importances — and return its metadata block (counts, participants, features) for the manifest."""
    from joblib import dump

    df = _load_training_frame(csv_path, features, target_col)
    model = _train_rf(df, features, target_col, seed)
    dump(model, output_path)
    _write_feature_importance(
        model,
        features,
        output_path.with_name(output_path.stem + "_feature_importance.csv"),
    )

    return {
        "model_name": name,
        "model_file": _display_path(output_path),
        "training_csv": _display_path(csv_path),
        "n_windows": int(len(df)),
        "n_safe": int((df[target_col] == 0).sum()),
        "n_risky": int((df[target_col] == 1).sum()),
        "participants": sorted(df["participant_id"].dropna().unique().tolist()),
        "features": features,
    }


def main() -> None:
    """Train both fallback RFs (primary 4-IMU comparator + reduced Pelvis-L3 deployment model) from the frozen sets and write the provenance metadata JSON."""
    parser = argparse.ArgumentParser(
        description="Train final RF models for the IMU-only fallback route.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--analysis_dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "fallback_analysis_sets",
    )
    parser.add_argument(
        "--models_dir",
        type=Path,
        default=PROJECT_ROOT / "ml" / "models" / "fallback_final",
    )
    parser.add_argument("--target_col", default="risk_class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Keep sklearn/joblib imports inside main so the file can still be inspected
    # on machines where the ML environment has not been activated yet.
    try:
        import joblib  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "scikit-learn and joblib are required to train the fallback RF models. "
            "Activate the project Python environment, then rerun this script."
        ) from exc

    args.models_dir.mkdir(parents=True, exist_ok=True)
    primary_csv = args.analysis_dir / "primary_4imu_cleaned_features.csv"
    reduced_csv = args.analysis_dir / "reduced_pelvis_l3_features.csv"

    trained = [
        _train_one(
            "primary_4imu_cleaned",
            primary_csv,
            PRIMARY_4IMU_FEATURES,
            args.models_dir / "rf_primary_4imu.joblib",
            args.target_col,
            args.seed,
        ),
        _train_one(
            "reduced_pelvis_l3",
            reduced_csv,
            REDUCED_PELVIS_L3_FEATURES,
            args.models_dir / "rf_reduced_pelvis_l3.joblib",
            args.target_col,
            args.seed,
        ),
    ]

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "system_route": "imu_only_fallback_rf_imu_mamdani_fis",
        "trained_stage": "RF_IMU_probability_estimator",
        "fixed_decision_layer": "ml.fuzzy.mamdani_fis.IMUFallbackFIS",
        "target_col": args.target_col,
        "random_seed": args.seed,
        "git_commit_hash": _git_commit_hash(),
        "analysis_dir": str(args.analysis_dir.resolve()),
        "models": trained,
        "feature_columns": {
            "primary_4imu_cleaned": PRIMARY_4IMU_FEATURES,
            "reduced_pelvis_l3": REDUCED_PELVIS_L3_FEATURES,
        },
        "recommended_deployment_model": "rf_reduced_pelvis_l3.joblib",
        "notes": [
            "Reduced Pelvis-L3 is the deployment recommendation.",
            "Primary 4-IMU model is retained as a comparator.",
            "FIS rules are fixed and should be reported separately from RF training.",
        ],
        "command_used": " ".join([Path(sys.executable).name, *sys.argv]),
    }
    metadata_path = args.models_dir / "fallback_model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved final fallback models to {args.models_dir}")
    for item in trained:
        print(
            f"  {item['model_name']}: {item['n_windows']} windows, "
            f"{len(item['participants'])} participants -> {item['model_file']}"
        )
    print(f"  metadata -> {metadata_path}")


if __name__ == "__main__":
    main()
