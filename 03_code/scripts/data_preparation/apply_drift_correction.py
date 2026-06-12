#!/usr/bin/env python3
"""
Post-hoc Baseline Drift Correction
====================================
FYP 2025/26 | Imperial College London

Applies linear BL1→BL2 drift correction to a processed imu_data.csv file.

The Madgwick AHRS filter (beta=0.033, 100 Hz, no magnetometer) accumulates
orientation drift at roughly 1-3°/min over long sessions. Corrects
it by assuming the participant is upright and still during both baseline windows
(angles should be ~0°), then linearly interpolating the residual drift across
the session and subtracting it from all angle columns.

It also NaN-masks theta_TU columns during any T4 sensor dropout window, so
the pipeline does not treat corrupted values as signal.

Correction logic
----------------
1. BL1 anchor: mean angle during [bl1_start, bl1_end] (skip first bl1_skip_ms
   to avoid Madgwick convergence transient). Should be near 0° post N-pose.
2. BL2 anchor: mean angle during [bl2_start, bl2_end] (separate window for T4
   if it reconnected late). Represents accumulated drift.
3. Drift = BL2_mean - BL1_mean.
4. Correction ramps linearly from 0 at BL1_mid to drift at BL2_mid, then
   holds constant for all samples after BL2_mid (conservative, avoids
   extrapolation error during vigorous post-BL2 movements like FATIGUE_FLEXION).
5. angle_corrected(t) = angle_raw(t) - correction(t)

Usage
-----
    # Auto-detect BL windows from labels.csv:
    python scripts/data_preparation/apply_drift_correction.py \\
        --session_dir data/real/protocol_train_fallback_2session/participant_04/session_001

    # Override BL windows explicitly (all timestamps in ms):
    python scripts/data_preparation/apply_drift_correction.py \\
        --session_dir data/real/protocol_train_fallback_2session/participant_04/session_001 \\
        --bl1_start 10000 --bl1_end 60000 \\
        --bl2_start 1266000 --bl2_end 1386000 \\
        --t4_bl2_start 1380000 \\
        --t4_dropout_start 1140000 --t4_dropout_end 1380000

    # Dry run (print corrections without saving):
    python scripts/data_preparation/apply_drift_correction.py \\
        --session_dir data/real/.../session_001 --dry_run
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Column groups

# All Euler angle columns subject to gyro drift
ANGLE_COLUMNS = [
    "theta_PL_pitch", "theta_PL_roll", "theta_PL_yaw",
    "theta_LT_pitch", "theta_LT_roll", "theta_LT_yaw",
    "theta_TU_pitch", "theta_TU_roll", "theta_TU_yaw",
]

# T4-dependent columns, NaN-masked during sensor dropout
T4_COLUMNS = ["theta_TU_pitch", "theta_TU_roll", "theta_TU_yaw"]

# Columns NOT corrected (raw rates, quaternions, correction acts on derived angles only)
SKIP_CORRECTION = {"angvel_L3_sagittal", "l3_ax_g", "l3_ay_g", "l3_az_g"}


# Core correction function

def apply_bl_drift_correction(
    imu_df: pd.DataFrame,
    bl1_start_ms: float,
    bl1_end_ms: float,
    bl2_start_ms: float,
    bl2_end_ms: float,
    t4_bl2_start_ms: float = None,
    t4_dropout_start_ms: float = None,
    t4_dropout_end_ms: float = None,
    bl1_skip_ms: float = 5000.0,
) -> tuple:
    """
    Apply BL1→BL2 drift correction to angle columns in-place.

    Parameters
    ----------
    imu_df              : processed imu_data.csv DataFrame (not modified in place)
    bl1_start_ms        : BL1 window start timestamp (ms, relative to session start)
    bl1_end_ms          : BL1 window end timestamp (ms)
    bl2_start_ms        : BL2 window start timestamp (ms)
    bl2_end_ms          : BL2 window end timestamp (ms)
    t4_bl2_start_ms     : Start of the valid T4 sub-window within BL2.
                          Use this when T4 reconnected late inside BL2.
                          If None, defaults to bl2_start_ms.
    t4_dropout_start_ms : Start of T4 sensor dropout window (ms). Theta_TU
                          columns are NaN-masked in [dropout_start, dropout_end].
                          Set to None to skip masking.
    t4_dropout_end_ms   : End of T4 sensor dropout window (ms).
    bl1_skip_ms         : Skip this many ms from the start of BL1 to avoid
                          Madgwick filter convergence transient. Default: 5000.

    Returns
    -------
    corrected_df : corrected copy of imu_df
    report       : dict of correction diagnostics (suitable for metadata JSON)
    """
    df = imu_df.copy()
    t = df["timestamp_ms"].to_numpy(dtype=float)

    # Columns present in this file
    non_t4_cols = [c for c in ANGLE_COLUMNS if c not in T4_COLUMNS and c in df.columns]
    t4_angle_cols = [c for c in T4_COLUMNS if c in df.columns]
    all_correctable = non_t4_cols + t4_angle_cols

    if not all_correctable:
        raise ValueError("No correctable angle columns found in imu_data.csv.")

    # Anchor means

    def window_mean(t_start, t_end, cols, skip_ms=0.0):
        eff_start = t_start + skip_ms
        mask = (df["timestamp_ms"] >= eff_start) & (df["timestamp_ms"] <= t_end)
        chunk = df.loc[mask, cols].dropna()
        if len(chunk) == 0:
            raise ValueError(
                f"Empty window [{eff_start:.0f}, {t_end:.0f}] ms for columns {cols}. "
                f"Check that the session contains data in this range."
            )
        return chunk.mean()

    # BL1: skip first bl1_skip_ms for Madgwick convergence
    bl1_means_non_t4 = window_mean(bl1_start_ms, bl1_end_ms, non_t4_cols, skip_ms=bl1_skip_ms)
    bl1_means_t4     = window_mean(bl1_start_ms, bl1_end_ms, t4_angle_cols, skip_ms=bl1_skip_ms)

    # BL2: T4 may have a narrower valid sub-window
    _t4_bl2_start = t4_bl2_start_ms if t4_bl2_start_ms is not None else bl2_start_ms
    bl2_means_non_t4 = window_mean(bl2_start_ms, bl2_end_ms, non_t4_cols)
    bl2_means_t4     = window_mean(_t4_bl2_start, bl2_end_ms, t4_angle_cols)

    # Reference midpoints for the interpolation ramp
    bl1_mid     = (bl1_start_ms + bl1_end_ms) / 2.0
    bl2_mid     = (bl2_start_ms + bl2_end_ms) / 2.0
    t4_bl2_mid  = (_t4_bl2_start + bl2_end_ms) / 2.0

    # Residual drift = BL2_mean - BL1_mean
    # Positive drift → angles have crept positive; we subtract it.
    drift_non_t4 = bl2_means_non_t4 - bl1_means_non_t4
    drift_t4     = bl2_means_t4     - bl1_means_t4

    # Apply correction

    report_corrections = {}

    def build_ramp(drift_val, bl2_ref_mid):
        """Linear ramp from 0 at bl1_mid to drift_val at bl2_ref_mid, then held."""
        span = bl2_ref_mid - bl1_mid
        return np.where(
            t < bl1_mid,   0.0,
            np.where(
                t <= bl2_ref_mid,
                drift_val * (t - bl1_mid) / span,
                drift_val          # hold constant after BL2
            )
        )

    for col in non_t4_cols:
        d = float(drift_non_t4[col])
        df[col] = df[col].to_numpy(dtype=float) - build_ramp(d, bl2_mid)
        report_corrections[col] = {
            "bl1_mean_deg":   round(float(bl1_means_non_t4[col]), 3),
            "bl2_mean_deg":   round(float(bl2_means_non_t4[col]), 3),
            "drift_applied_deg": round(d, 3),
        }

    for col in t4_angle_cols:
        d = float(drift_t4[col])
        df[col] = df[col].to_numpy(dtype=float) - build_ramp(d, t4_bl2_mid)
        report_corrections[col] = {
            "bl1_mean_deg":   round(float(bl1_means_t4[col]), 3),
            "bl2_mean_deg":   round(float(bl2_means_t4[col]), 3),
            "drift_applied_deg": round(d, 3),
            "t4_bl2_window_ms": [_t4_bl2_start, bl2_end_ms],
        }

    # T4 dropout NaN masking

    n_masked = 0
    if t4_dropout_start_ms is not None and t4_dropout_end_ms is not None:
        dropout_mask = (
            (df["timestamp_ms"] >= t4_dropout_start_ms) &
            (df["timestamp_ms"] <= t4_dropout_end_ms)
        )
        n_masked = int(dropout_mask.sum())
        for col in t4_angle_cols:
            df.loc[dropout_mask, col] = np.nan

    # Post-correction verification
    # BL2 angles should now be near zero.

    post_bl2_non_t4 = window_mean(bl2_start_ms, bl2_end_ms, non_t4_cols)
    post_bl2_t4     = window_mean(_t4_bl2_start, bl2_end_ms, t4_angle_cols)
    post_bl2 = pd.concat([post_bl2_non_t4, post_bl2_t4])

    report = {
        "applied_at":           datetime.now().isoformat(),
        "bl1_window_ms":        [bl1_start_ms, bl1_end_ms],
        "bl2_window_ms":        [bl2_start_ms, bl2_end_ms],
        "t4_bl2_window_ms":     [_t4_bl2_start, bl2_end_ms],
        "bl1_skip_ms":          bl1_skip_ms,
        "t4_dropout_window_ms": [t4_dropout_start_ms, t4_dropout_end_ms],
        "t4_samples_nan_masked": n_masked,
        "corrections":          report_corrections,
        "post_correction_bl2_residual_deg": {
            col: round(float(post_bl2[col]), 3) for col in all_correctable
        },
    }

    return df, report


# BL window auto-detection

def detect_baseline_windows(labels_csv: Path):
    """
    Auto-detect BL1 and BL2 from labels.csv.

    Returns (bl1_start_ms, bl1_end_ms, bl2_start_ms, bl2_end_ms).
    BL1 = first BASELINE_STATIC segment, BL2 = last BASELINE_STATIC segment.
    Raises ValueError if fewer than two baselines are found.
    """
    labels = pd.read_csv(labels_csv)
    baselines = labels[labels["label"] == "BASELINE_STATIC"].sort_values("start_ms")

    if len(baselines) == 0:
        raise ValueError(
            f"No BASELINE_STATIC segments found in {labels_csv}. "
            f"Cannot auto-detect BL1/BL2 windows."
        )
    if len(baselines) == 1:
        raise ValueError(
            f"Only one BASELINE_STATIC found in {labels_csv}. "
            f"BL2 is required for drift correction. "
            f"Collect or add an end-of-session static baseline."
        )

    bl1 = baselines.iloc[0]
    bl2 = baselines.iloc[-1]
    return (
        float(bl1["start_ms"]), float(bl1["end_ms"]),
        float(bl2["start_ms"]), float(bl2["end_ms"]),
    )


# Pretty-print summary

def print_summary(report: dict):
    print(f"\n{'='*60}")
    print(f"Drift Correction Summary")
    print(f"{'='*60}")
    print(f"  BL1 window : {report['bl1_window_ms'][0]/1000:.0f}–{report['bl1_window_ms'][1]/1000:.0f}s")
    print(f"  BL2 window : {report['bl2_window_ms'][0]/1000:.0f}–{report['bl2_window_ms'][1]/1000:.0f}s")
    if report["t4_dropout_window_ms"][0] is not None:
        print(f"  T4 dropout : {report['t4_dropout_window_ms'][0]/1000:.0f}–"
              f"{report['t4_dropout_window_ms'][1]/1000:.0f}s "
              f"({report['t4_samples_nan_masked']} samples NaN'd)")
    print()
    print(f"  {'Column':<22} {'BL1 (°)':>8}  {'BL2 (°)':>8}  {'Drift (°)':>10}")
    print(f"  {'-'*22} {'-'*8}  {'-'*8}  {'-'*10}")
    for col, vals in report["corrections"].items():
        flag = ""
        if abs(vals["drift_applied_deg"]) > 20:
            flag = "  ← large"
        print(f"  {col:<22} {vals['bl1_mean_deg']:>+8.2f}  "
              f"{vals['bl2_mean_deg']:>+8.2f}  "
              f"{vals['drift_applied_deg']:>+10.2f}{flag}")
    print()
    print(f"  Post-correction BL2 residuals (should be near 0°):")
    for col, val in report["post_correction_bl2_residual_deg"].items():
        status = "✓" if abs(val) < 2.0 else ("⚠" if abs(val) < 5.0 else "✗")
        print(f"    {status} {col:<22} {val:>+7.2f}°")
    print(f"{'='*60}\n")


# CLI

def main():
    parser = argparse.ArgumentParser(
        description="Apply BL1→BL2 drift correction to a processed imu_data.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--session_dir", required=True,
        help="Path to processed session directory containing imu_data.csv, "
             "labels.csv, and session_metadata.json.",
    )
    # BL window overrides (auto-detected from labels.csv if not provided)
    parser.add_argument("--bl1_start", type=float, default=None,
                        help="BL1 window start (ms). Auto-detected if omitted.")
    parser.add_argument("--bl1_end",   type=float, default=None,
                        help="BL1 window end (ms). Auto-detected if omitted.")
    parser.add_argument("--bl2_start", type=float, default=None,
                        help="BL2 window start (ms). Auto-detected if omitted.")
    parser.add_argument("--bl2_end",   type=float, default=None,
                        help="BL2 window end (ms). Auto-detected if omitted.")
    parser.add_argument(
        "--t4_bl2_start", type=float, default=None,
        help="Start of the valid T4 sub-window within BL2 (ms). "
             "Use when T4 reconnected part-way through BL2. "
             "Defaults to --bl2_start.",
    )
    parser.add_argument(
        "--t4_dropout_start", type=float, default=None,
        help="Start of T4 sensor dropout window (ms). "
             "theta_TU columns are NaN-masked in this range.",
    )
    parser.add_argument(
        "--t4_dropout_end", type=float, default=None,
        help="End of T4 sensor dropout window (ms).",
    )
    parser.add_argument(
        "--bl1_skip_ms", type=float, default=5000.0,
        help="Skip this many ms from BL1 start (Madgwick convergence). Default: 5000.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print correction summary without saving any files.",
    )
    args = parser.parse_args()

    session_dir  = Path(args.session_dir)
    imu_path     = session_dir / "imu_data.csv"
    labels_path  = session_dir / "labels.csv"
    meta_path    = session_dir / "session_metadata.json"
    backup_path  = session_dir / "imu_data_precorrection.csv"

    # Validate paths
    for p in [imu_path, labels_path, meta_path]:
        if not p.exists():
            print(f"[ERROR] Required file not found: {p}", file=sys.stderr)
            sys.exit(1)

    # Check for double-application
    if backup_path.exists() and not args.dry_run:
        print(
            f"[WARNING] Backup file already exists: {backup_path.name}\n"
            f"  Drift correction may have already been applied. "
            f"  Run with --dry_run to inspect, or delete the backup to re-apply.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load
    print(f"\nLoading imu_data.csv ({imu_path})...")
    imu_df = pd.read_csv(imu_path)
    print(f"  {len(imu_df)} rows, duration {imu_df['timestamp_ms'].max()/1000:.1f}s")

    # Resolve BL windows
    if any(v is None for v in [args.bl1_start, args.bl1_end, args.bl2_start, args.bl2_end]):
        print(f"\nAuto-detecting BL windows from {labels_path.name}...")
        bl1_start, bl1_end, bl2_start, bl2_end = detect_baseline_windows(labels_path)
        # Allow CLI overrides to take precedence over auto-detection
        bl1_start = args.bl1_start if args.bl1_start is not None else bl1_start
        bl1_end   = args.bl1_end   if args.bl1_end   is not None else bl1_end
        bl2_start = args.bl2_start if args.bl2_start is not None else bl2_start
        bl2_end   = args.bl2_end   if args.bl2_end   is not None else bl2_end
        print(f"  BL1: {bl1_start/1000:.0f}–{bl1_end/1000:.0f}s")
        print(f"  BL2: {bl2_start/1000:.0f}–{bl2_end/1000:.0f}s")
    else:
        bl1_start = args.bl1_start
        bl1_end   = args.bl1_end
        bl2_start = args.bl2_start
        bl2_end   = args.bl2_end

    # Apply correction
    print("\nApplying drift correction...")
    corrected_df, report = apply_bl_drift_correction(
        imu_df,
        bl1_start_ms        = bl1_start,
        bl1_end_ms          = bl1_end,
        bl2_start_ms        = bl2_start,
        bl2_end_ms          = bl2_end,
        t4_bl2_start_ms     = args.t4_bl2_start,
        t4_dropout_start_ms = args.t4_dropout_start,
        t4_dropout_end_ms   = args.t4_dropout_end,
        bl1_skip_ms         = args.bl1_skip_ms,
    )

    print_summary(report)

    if args.dry_run:
        print("  [DRY RUN] No files written.")
        return

    # Save
    # 1. Back up original
    shutil.copy2(imu_path, backup_path)
    print(f"  Backed up original: {backup_path.name}")

    # 2. Write corrected file
    corrected_df.to_csv(imu_path, index=False)
    print(f"  Saved corrected:    {imu_path.name}")

    # 3. Update session_metadata.json
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["post_hoc_drift_correction"] = {"enabled": True, **report}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  Updated metadata:   {meta_path.name}")

    print(f"\nDone. Run the pipeline on {session_dir} when ready.\n")


if __name__ == "__main__":
    main()
