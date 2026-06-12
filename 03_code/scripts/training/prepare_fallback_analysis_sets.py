#!/usr/bin/env python3
"""
Prepare analysis-ready IMU-only fallback feature sets.

Outputs:
  1. primary_4imu_cleaned_features.csv
     Cleaned 4-IMU fallback table after data-quality exclusions.

  2. reduced_pelvis_l3_features.csv
     Reduced lower-back/sagittal table that can include P02 despite T12/T4
     dropout, because it only uses Pelvis-L3 and L3 accelerometer features.

  3. p03_operating_envelope_summary.csv
     Per-movement summary for P03 as a sub-threshold/operating-envelope case.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ID_COLS = [
    "session_id",
    "participant_id",
    "window_centre_ms",
    "movement_label",
    "risk_class_protocol",
    "risk_class",
]

REDUCED_FEATURES = [
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

PRIMARY_EXCLUDE_PARTICIPANTS = {"participant_02"}
OPERATING_ENVELOPE_PARTICIPANTS = {"participant_03"}
PELVIS_PHYSIO_CAP_DEG = 60.0


def _load_session_feature(path: Path, participant_id: str, session_id: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.insert(0, "participant_id", participant_id)
    df.insert(1, "session_id", f"{participant_id}__{session_id}")
    return df


def _load_p03(p03_dir: Path) -> pd.DataFrame:
    path = p03_dir / "feature_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing P03 feature matrix: {path}")
    return _load_session_feature(path, "participant_03", "session_001")


def _labelled(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only protocol-labelled (0/1) windows, preferring the protocol label
    over any pre-existing risk_class, and normalise it into risk_class."""
    target = "risk_class_protocol" if "risk_class_protocol" in df.columns else "risk_class"
    out = df[df[target].isin([0, 1])].copy()
    out["risk_class"] = out[target].astype(int)
    return out


def _cap_pelvis_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clip physiologically impossible pelvis angles (Madgwick drift artefacts)
    to a 60 deg ceiling, recording which windows were capped for transparency."""
    out = df.copy()
    report = {}
    for col in ["imu_pelvis_angle_peak", "imu_pelvis_angle_mean"]:
        if col not in out.columns:
            continue
        flag_col = f"{col}_capped"
        values = out[col].astype(float)
        mask = values > PELVIS_PHYSIO_CAP_DEG
        out[flag_col] = mask
        out[col] = values.clip(upper=PELVIS_PHYSIO_CAP_DEG)
        report[col] = {
            "cap_deg": PELVIS_PHYSIO_CAP_DEG,
            "n_capped": int(mask.sum()),
            "pct_capped": float(mask.mean() * 100.0) if len(mask) else 0.0,
        }
    return out, report


def _summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, group in df.groupby("participant_id"):
        labelled = _labelled(group)
        rows.append({
            "participant_id": pid,
            "total_windows": int(len(group)),
            "labelled_windows": int(len(labelled)),
            "safe_windows": int((labelled["risk_class"] == 0).sum()),
            "risky_windows": int((labelled["risk_class"] == 1).sum()),
            "movements": int(labelled["movement_label"].nunique()) if "movement_label" in labelled else 0,
            "max_pelvis_angle_peak": float(labelled.get("imu_pelvis_angle_peak", pd.Series(dtype=float)).max()),
            "max_l3_accel_tilt_peak": float(labelled.get("imu_l3_accel_tilt_peak", pd.Series(dtype=float)).max()),
        })
    return pd.DataFrame(rows)


def _p03_envelope(p03: pd.DataFrame) -> pd.DataFrame:
    """Per-movement summary for P03, the sub-threshold/outlier mover, used to
    characterise the operating envelope rather than to train or score a model."""
    labelled = _labelled(p03)
    rows = []
    for movement, group in labelled.groupby("movement_label"):
        rows.append({
            "movement_label": movement,
            "n_windows": int(len(group)),
            "risk_class": int(group["risk_class"].mode().iloc[0]),
            "pelvis_angle_peak_max": float(group["imu_pelvis_angle_peak"].max()),
            "pelvis_angle_peak_mean": float(group["imu_pelvis_angle_peak"].mean()),
            "trunk_angle_peak_max": float(group["imu_trunk_angle_peak"].max()),
            "time_in_risk_zone_mean": float(group["imu_time_in_risk_zone"].mean()),
            "angvel_peak_max": float(group["imu_angvel_peak"].max()),
        })
    return pd.DataFrame(rows).sort_values(["risk_class", "movement_label"])


def prepare(data_dir: Path, p03_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = data_dir / "combined_features.csv"
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Missing cleaned combined feature table: {cleaned_path}")
    cleaned = pd.read_csv(cleaned_path)

    primary = cleaned[~cleaned["participant_id"].isin(PRIMARY_EXCLUDE_PARTICIPANTS)].copy()
    primary = primary[primary["risk_class"].isin([0, 1])].copy()
    primary, primary_cap_report = _cap_pelvis_features(primary)
    primary_path = out_dir / "primary_4imu_cleaned_features.csv"
    primary.to_csv(primary_path, index=False)

    p03 = _load_p03(p03_dir)
    p02_backup = data_dir / "participant_02" / "session_001" / "feature_matrix.pre_quality_exclusions_20260528_112742.csv"
    if not p02_backup.exists():
        p02_backup = data_dir / "participant_02" / "session_001" / "feature_matrix.csv"
    p02 = _load_session_feature(p02_backup, "participant_02", "session_001")

    reduced_base = pd.concat([cleaned, p02, p03], ignore_index=True, sort=False)
    reduced_base = _labelled(reduced_base)
    keep_cols = [c for c in ID_COLS + REDUCED_FEATURES if c in reduced_base.columns]
    missing = sorted(set(REDUCED_FEATURES) - set(reduced_base.columns))
    if missing:
        raise ValueError(f"Missing reduced feature columns: {missing}")
    reduced = reduced_base[keep_cols].dropna(subset=REDUCED_FEATURES).copy()
    reduced, reduced_cap_report = _cap_pelvis_features(reduced)
    reduced_path = out_dir / "reduced_pelvis_l3_features.csv"
    reduced.to_csv(reduced_path, index=False)

    p03_summary = _p03_envelope(p03)
    p03_summary_path = out_dir / "p03_operating_envelope_summary.csv"
    p03_summary.to_csv(p03_summary_path, index=False)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "primary_4imu": {
            "path": str(primary_path),
            "rows": int(len(primary)),
            "participants": sorted(primary["participant_id"].unique().tolist()),
            "pelvis_angle_cap": primary_cap_report,
            "participant_summary": _summarise(primary).to_dict(orient="records"),
        },
        "reduced_pelvis_l3": {
            "path": str(reduced_path),
            "rows": int(len(reduced)),
            "features": REDUCED_FEATURES,
            "participants": sorted(reduced["participant_id"].unique().tolist()),
            "pelvis_angle_cap": reduced_cap_report,
            "participant_summary": _summarise(reduced).to_dict(orient="records"),
        },
        "p03_operating_envelope": {
            "path": str(p03_summary_path),
            "rows": int(len(p03_summary)),
        },
    }
    summary_path = out_dir / "analysis_set_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare final fallback analysis feature sets.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/real/protocol_train_fallback_2session"))
    parser.add_argument("--p03_dir", type=Path, default=Path("data/real/protocol_train_fallback/participant_03/session_001"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/fallback_analysis_sets"))
    args = parser.parse_args()

    print(json.dumps(prepare(args.data_dir, args.p03_dir, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
