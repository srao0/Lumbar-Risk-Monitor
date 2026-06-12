#!/usr/bin/env python3
"""
fix_participant_02.py
====================
Two-part post-hoc correction for Participant 02's session_001 data.

Part 1, T12 sensor physical slip correction
    The T12 IMU physically rotated ~21.4° at ~183s during CLEAN_FLEXION rep 8.
    This applies a correction quaternion to all T12 data from 186,000ms onwards
    and recomputes theta_LT and theta_TU angles.
    Output: data/real/protocol_train_fallback/participant_02/session_001_t12corrected/

Part 2, Madgwick drift correction (beta=0.1)
    The default beta=0.033 produces heading drift after ~25 min without a
    magnetometer. Re-running with beta=0.1 damps this.
    Output: data/real/protocol_train_fallback/participant_02/session_001_beta01/

Part 3, Combined correction (T12 shift applied on top of beta=0.1 output)
    Output: data/real/protocol_train_fallback/participant_02/session_001_corrected/

Run from the project root:
    cd "C:\\...\\FYP_Spine_Project"
    python scripts/data_preparation/fix_participant_02.py

Expected runtime: ~60-90 seconds on a modern laptop.
"""

import sys, os, time, json, shutil
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

t_start = time.time()

def elapsed():
    """Wall-clock tag for progress prints — these passes take a minute or so, so timestamps make the slow Madgwick step visible."""
    return f"[{time.time()-t_start:.1f}s]"

# Paths
RAW_IMU    = ROOT / "data/real/raw/participant_02/session_001/imu_arduino.csv"
RAW_LABELS = ROOT / "data/real/raw/participant_02/session_001/labels.csv"
SRC_DIR    = ROOT / "data/real/protocol_train_fallback/participant_02/session_001"
T12_DIR    = ROOT / "data/real/protocol_train_fallback/participant_02/session_001_t12corrected"
B01_DIR    = ROOT / "data/real/protocol_train_fallback/participant_02/session_001_beta01"
OUT_DIR    = ROOT / "data/real/protocol_train_fallback/participant_02/session_001_corrected"
FINAL_DIR  = ROOT / "data/real/protocol_train_fallback/participant_02/session_001_final"

# T12 shift correction constants
SHIFT_START_MS = 186_000   # All T12 data from here gets corrected
# q_corr = conj(q_ref) ⊗ q_shift, where:
# q_ref  = mean T12 quaternion 160-182s (pre-slip)
# q_shift = mean T12 quaternion 192-196s (post-slip)
Q_CORR = np.array([0.9826, -0.0961, -0.1304, -0.0905])  # [w,x,y,z]


# Quaternion math helpers
def qmult_batch(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Vectorised Hamilton product. q1/q2: (N,4) [w,x,y,z]."""
    w1,x1,y1,z1 = q1[:,0],q1[:,1],q1[:,2],q1[:,3]
    w2,x2,y2,z2 = q2[:,0],q2[:,1],q2[:,2],q2[:,3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=1)

def qconj_batch(q: np.ndarray) -> np.ndarray:
    """Conjugate of (N,4) quaternions — negate the vector part; used to form relative orientations (e.g. conj(L3) ⊗ T12)."""
    c = q.copy(); c[:,1:] *= -1; return c

def norm_q(q: np.ndarray) -> np.ndarray:
    """Renormalise quaternions row-wise, guarding the near-zero case so the correction product stays a unit rotation."""
    n = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.where(n < 1e-9, 1.0, n)

def euler_zyx_batch(q: np.ndarray):
    """ZYX Euler angles from (N,4) quaternions. Returns pitch, roll, yaw in degrees."""
    w,x,y,z = q[:,0],q[:,1],q[:,2],q[:,3]
    sp = np.clip(2*(w*y - z*x), -1, 1)
    pitch = np.degrees(np.arcsin(sp))
    roll  = np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    yaw   = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    return pitch, roll, yaw


# PART 1: T12 sensor shift correction (post-hoc on existing imu_data.csv)
def apply_t12_correction(src_imu_csv: Path, dst_dir: Path, src_meta: Path = None):
    """Rotate T12 quaternions from the slip onset (186s) by the fixed correction quaternion, then recompute the LT/TU relative angles for the whole session and write the corrected copy with provenance metadata."""
    print(f"\n{'='*60}")
    print(f"PART 1: T12 sensor shift correction")
    print(f"  Source: {src_imu_csv}")
    print(f"  Output: {dst_dir}")
    print(f"{'='*60}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_csv = dst_dir / "imu_data.csv"

    print(f"{elapsed()} Loading {src_imu_csv.name}...", flush=True)
    df = pd.read_csv(src_imu_csv)
    print(f"{elapsed()} {len(df):,} rows loaded.", flush=True)

    # Handle both column name conventions (session_converter uses 'timestamp_ms',
    # fuse_four_imu_dataframe raw output uses 't_ms')
    ts_col = "timestamp_ms" if "timestamp_ms" in df.columns else "t_ms"

    # Apply correction quaternion to T12 from SHIFT_START_MS onwards
    mask = df[ts_col].values >= SHIFT_START_MS
    n_affected = mask.sum()
    print(f"{elapsed()} Correcting {n_affected:,} T12 rows (>= {SHIFT_START_MS}ms)...", flush=True)

    t12_cols = ["t12_qw","t12_qx","t12_qy","t12_qz"]
    q_t12 = df.loc[mask, t12_cols].values.astype(np.float64)
    q_fix = norm_q(qmult_batch(np.tile(Q_CORR, (n_affected,1)), q_t12))
    df.loc[mask, t12_cols] = q_fix

    # Recompute ALL theta_LT and theta_TU (full dataset, correction only affects post-186s)
    print(f"{elapsed()} Recomputing LT and TU angles...", flush=True)
    l3  = df[["l3_qw","l3_qx","l3_qy","l3_qz"]].values.astype(np.float64)
    t12 = df[["t12_qw","t12_qx","t12_qy","t12_qz"]].values.astype(np.float64)
    t4  = df[["t4_qw","t4_qx","t4_qy","t4_qz"]].values.astype(np.float64)
    q_LT = qmult_batch(qconj_batch(l3), t12)
    q_TU = qmult_batch(qconj_batch(t12), t4)
    p_LT,r_LT,y_LT = euler_zyx_batch(q_LT)
    p_TU,r_TU,y_TU = euler_zyx_batch(q_TU)
    df["theta_LT_pitch"]=p_LT; df["theta_LT_roll"]=r_LT; df["theta_LT_yaw"]=y_LT
    df["theta_TU_pitch"]=p_TU; df["theta_TU_roll"]=r_TU; df["theta_TU_yaw"]=y_TU

    print(f"{elapsed()} Saving corrected imu_data.csv...", flush=True)
    df.to_csv(dst_csv, index=False)

    # Copy labels.csv and update metadata
    labels_src = src_imu_csv.parent / "labels.csv"
    if labels_src.exists():
        shutil.copy2(labels_src, dst_dir / "labels.csv")

    meta_src = src_meta or (src_imu_csv.parent / "session_metadata.json")
    if meta_src.exists():
        with open(meta_src) as f: meta = json.load(f)
        meta["t12_shift_correction"] = {
            "applied": True,
            "shift_start_ms": SHIFT_START_MS,
            "q_corr_wxyz": Q_CORR.tolist(),
            "slip_deg": 21.4,
            "note": "T12 physical slip at ~183s during CLEAN_FLEXION rep 8. "
                    "Correction: q_corr ⊗ q_t12_original for all rows >= 186,000ms."
        }
        with open(dst_dir / "session_metadata.json","w") as f:
            json.dump(meta, f, indent=2)

    print(f"{elapsed()} Part 1 done → {dst_csv}")

    # Sanity check
    pm = df[ts_col] >= SHIFT_START_MS
    orig = pd.read_csv(src_imu_csv)
    orig_ts = "timestamp_ms" if "timestamp_ms" in orig.columns else "t_ms"
    orig_pm = orig[orig_ts] >= SHIFT_START_MS
    orig_lt = orig.loc[orig_pm, "theta_LT_pitch"]
    fixed = df.loc[pm, "theta_LT_pitch"]
    print(f"\n  Sanity — LT_pitch post-186s:")
    print(f"    Original:  mean={orig_lt.mean():.1f}°  p5={orig_lt.quantile(.05):.1f}°  p95={orig_lt.quantile(.95):.1f}°")
    print(f"    Corrected: mean={fixed.mean():.1f}°  p5={fixed.quantile(.05):.1f}°  p95={fixed.quantile(.95):.1f}°")
    pre  = df[(df[ts_col]>=181_000)&(df[ts_col]<186_000)]["theta_LT_pitch"]
    post = df[(df[ts_col]>=186_000)&(df[ts_col]<191_000)]["theta_LT_pitch"]
    jump = abs(post.mean()-pre.mean())
    status = "✓ GOOD" if jump <= 5.0 else "⚠ CHECK — larger than expected (expected during slip recovery window)"
    print(f"    Boundary jump (181–186s → 186–191s): {jump:.1f}°  {status}")
    return df


# PART 2: Madgwick beta=0.1 re-run
def rerun_madgwick_beta01():
    """Re-fuse the raw IMU from scratch with Madgwick beta=0.1 (vs the default 0.033) to damp the magnetometer-free heading drift that builds over this ~30-min session, redoing gyro-bias and N-pose calibration."""
    print(f"\n{'='*60}")
    print(f"PART 2: Madgwick re-run with beta=0.1")
    print(f"  Raw IMU:  {RAW_IMU}")
    print(f"  Output:   {B01_DIR}")
    print(f"{'='*60}")

    from signal_processing.imu.convert import RawConverter, GyroBiasCalibrator, NPoseCalibrator
    from signal_processing.imu.madgwick import fuse_four_imu_dataframe

    B01_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{elapsed()} Loading raw IMU data...", flush=True)
    df_phys = RawConverter.from_csv(RAW_IMU)
    print(f"{elapsed()} {len(df_phys):,} rows. Gyro bias calibration (60s auto)...", flush=True)

    for prefix in ["Pelvis","L3","T12","T4"]:
        offsets = GyroBiasCalibrator.compute_from_csv(
            RAW_IMU, imu_prefix=prefix, start_s=0.0, duration_s=60.0
        )
        df_phys = GyroBiasCalibrator.apply(df_phys, offsets, imu_prefix=prefix)
    print(f"{elapsed()} Calibration done. Running Madgwick (beta=0.1)...", flush=True)

    df_fused = fuse_four_imu_dataframe(df_phys, fs=100.0, beta=0.1)
    print(f"{elapsed()} Madgwick done. N-pose calibration...", flush=True)

    npose_offsets = NPoseCalibrator.compute_offsets(
        df_fused, n_seconds=10.0, skip_seconds=5.0, fs=100.0
    )
    df_out = NPoseCalibrator.apply(df_fused, npose_offsets, recompute_relative=True, fs=100.0)
    print(f"{elapsed()} NPose done. Saving...", flush=True)

    # Load and merge labels
    labels_df = pd.read_csv(RAW_LABELS)
    df_out.to_csv(B01_DIR / "imu_data.csv", index=False)
    shutil.copy2(RAW_LABELS, B01_DIR / "labels.csv")

    # Save metadata
    meta_src = SRC_DIR / "session_metadata.json"
    if meta_src.exists():
        with open(meta_src) as f: meta = json.load(f)
        meta["madgwick_beta"] = 0.1
        meta["madgwick_beta_note"] = "Re-run with beta=0.1 to reduce drift in 30-min session"
        with open(B01_DIR / "session_metadata.json","w") as f:
            json.dump(meta, f, indent=2)

    print(f"{elapsed()} Part 2 done → {B01_DIR / 'imu_data.csv'}")

    # Drift check
    pl = df_out["theta_PL_pitch"].values
    _ts = "timestamp_ms" if "timestamp_ms" in df_out.columns else "t_ms"
    t  = df_out[_ts].values
    late = t >= 25*60*1000
    print(f"\n  Drift check — PL_pitch at 25+ min (resting should be ~0°):")
    print(f"    beta=0.033 original: ±30–40° swing observed")
    pl_late_range = np.max(pl[late]) - np.min(pl[late])
    print(f"    beta=0.1 corrected:  range = {pl_late_range:.1f}° (target < 15°)")
    return df_out


# PART 3: Combined correction (beta=0.1 + T12 shift fix)
def apply_combined():
    """Layer the T12 slip fix on top of the beta=0.1 re-run so both corrections live in one output dir."""
    print(f"\n{'='*60}")
    print(f"PART 3: Combined correction (beta=0.1 + T12 shift)")
    print(f"  Source: {B01_DIR / 'imu_data.csv'}")
    print(f"  Output: {OUT_DIR}")
    print(f"{'='*60}")
    apply_t12_correction(
        src_imu_csv=B01_DIR / "imu_data.csv",
        dst_dir=OUT_DIR,
        src_meta=B01_DIR / "session_metadata.json"
    )
    print(f"{elapsed()} Combined output saved → {OUT_DIR}")


# PART 4: Static window re-zeroing for FATIGUE_FLEXION
# Problem: after ~25 min, Madgwick yaw drift bleeds into PL_pitch via ZYX
# Euler cross-coupling. LT and TU relative angles become chaotic (std 44-95°
# during standing still) and are unrecoverable. PL_pitch has stable drift
# (~12° std) and CAN be corrected by subtracting the offset measured during
# the second BASELINE_STATIC.
# Method: compute mean PL_pitch/roll during second BASELINE_STATIC (1276-1376s,
# skipping first 10s for motion settling). Subtract these offsets from
# FATIGUE_FLEXION rows only. Flag LT/TU FATIGUE_FLEXION data as high-uncertainty.
def apply_fatigue_rezero(src_dir: Path, dst_dir: Path):
    """Re-zero PL_pitch/roll during FATIGUE_FLEXION using the offset measured in the second BASELINE_STATIC, and flag (not correct) the LT/TU angles, which are too noisy by then to recover — see lt_tu_drift_flag."""
    print(f"\n{'='*60}")
    print(f"PART 4: FATIGUE_FLEXION static re-zeroing")
    print(f"  Source: {src_dir}")
    print(f"  Output: {dst_dir}")
    print(f"{'='*60}")

    src_csv  = src_dir / "imu_data.csv"
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Protocol windows (from labels.csv)
    BL2_START       = 1_276_000   # second BASELINE_STATIC, skip first 10s
    BL2_END         = 1_376_000
    FATIGUE_START   = 1_386_000   # FATIGUE_FLEXION begins

    print(f"{elapsed()} Loading {src_csv.name}...", flush=True)
    df = pd.read_csv(src_csv)
    ts = "timestamp_ms" if "timestamp_ms" in df.columns else "t_ms"
    print(f"{elapsed()} {len(df):,} rows, ts='{ts}'", flush=True)

    # Compute drift offsets from second BASELINE_STATIC
    bl2_mask = (df[ts] >= BL2_START) & (df[ts] < BL2_END)
    bl2 = df[bl2_mask]
    n_bl2 = len(bl2)
    if n_bl2 < 500:
        raise ValueError(f"Too few rows in second BASELINE_STATIC ({n_bl2}). "
                         "Check protocol labels alignment.")

    # PL pitch and roll are stable-enough to correct (std ~12°).
    # LT/TU have std 44-95° during standing, not correctable, flag only.
    pl_pitch_offset = bl2["theta_PL_pitch"].mean()
    pl_roll_offset  = bl2["theta_PL_roll"].mean()
    lt_pitch_std    = bl2["theta_LT_pitch"].std()
    tu_pitch_std    = bl2["theta_TU_pitch"].std()

    print(f"\n  Drift offsets from second BASELINE_STATIC ({n_bl2:,} rows):")
    print(f"    theta_PL_pitch: {pl_pitch_offset:+.2f}°  (std={bl2['theta_PL_pitch'].std():.2f}°) → CORRECTING")
    print(f"    theta_PL_roll:  {pl_roll_offset:+.2f}°  (std={bl2['theta_PL_roll'].std():.2f}°) → CORRECTING")
    print(f"    theta_LT_pitch: std={lt_pitch_std:.1f}° → TOO NOISY, not correcting (flagged)")
    print(f"    theta_TU_pitch: std={tu_pitch_std:.1f}° → TOO NOISY, not correcting (flagged)")

    # Apply PL correction to FATIGUE_FLEXION rows
    fat_mask = df[ts] >= FATIGUE_START
    n_fat = fat_mask.sum()
    print(f"\n  Applying to {n_fat:,} FATIGUE_FLEXION rows...", flush=True)

    df.loc[fat_mask, "theta_PL_pitch"] -= pl_pitch_offset
    df.loc[fat_mask, "theta_PL_roll"]  -= pl_roll_offset

    # Add a column flagging which rows have unreliable LT/TU
    # (pipeline can optionally exclude these features for these rows)
    df["lt_tu_drift_flag"] = 0
    df.loc[fat_mask, "lt_tu_drift_flag"] = 1

    # Save
    print(f"{elapsed()} Saving...", flush=True)
    df.to_csv(dst_dir / "imu_data.csv", index=False)
    for fname in ["labels.csv"]:
        src_f = src_dir / fname
        if src_f.exists(): shutil.copy2(src_f, dst_dir / fname)

    meta_src = src_dir / "session_metadata.json"
    if meta_src.exists():
        with open(meta_src) as f: meta = json.load(f)
        meta["fatigue_rezero"] = {
            "applied": True,
            "method": "static_window_offset",
            "reference_window_ms": [BL2_START, BL2_END],
            "pl_pitch_offset_deg": round(pl_pitch_offset, 4),
            "pl_roll_offset_deg": round(pl_roll_offset, 4),
            "lt_tu_status": "unreliable_flagged",
            "lt_pitch_std_deg": round(lt_pitch_std, 2),
            "tu_pitch_std_deg": round(tu_pitch_std, 2),
            "note": (
                "PL_pitch/roll corrected via second BASELINE_STATIC offset. "
                "LT/TU angles during FATIGUE_FLEXION are high-uncertainty "
                "(std >44° during static standing) — use lt_tu_drift_flag==1 "
                "to exclude these features in the pipeline."
            )
        }
        with open(dst_dir / "session_metadata.json","w") as f:
            json.dump(meta, f, indent=2)

    print(f"{elapsed()} Part 4 done → {dst_dir / 'imu_data.csv'}")

    # Result summary
    fat_fixed = df.loc[fat_mask, "theta_PL_pitch"]
    print(f"\n  FATIGUE_FLEXION PL_pitch after correction:")
    print(f"    mean={fat_fixed.mean():.1f}°  p5={fat_fixed.quantile(.05):.1f}°  "
          f"p95={fat_fixed.quantile(.95):.1f}°")
    print(f"    (Expected: resting~0°, forward flexion 30–50°, should be plausible)")
    return df


# MAIN
if __name__ == "__main__":
    print("=" * 60)
    print("Participant 02 — Post-hoc Data Correction")
    print("=" * 60)

    # Part 1: T12 shift correction on beta=0.033 output
    if (T12_DIR / "imu_data.csv").exists():
        print(f"\nPart 1 output already exists: {T12_DIR} — skipping.")
    else:
        apply_t12_correction(src_imu_csv=SRC_DIR / "imu_data.csv", dst_dir=T12_DIR)

    # Part 2: Madgwick beta=0.1 re-run
    if (B01_DIR / "imu_data.csv").exists():
        print(f"\nPart 2 output already exists: {B01_DIR} — skipping.")
    else:
        rerun_madgwick_beta01()

    # Part 3: beta=0.1 + T12 shift combined
    if (OUT_DIR / "imu_data.csv").exists():
        print(f"\nPart 3 output already exists: {OUT_DIR} — skipping.")
    else:
        apply_combined()

    # Part 4: FATIGUE_FLEXION PL re-zeroing (applied on top of Part 3)
    if (FINAL_DIR / "imu_data.csv").exists():
        print(f"\nPart 4 output already exists: {FINAL_DIR} — skipping.")
    else:
        apply_fatigue_rezero(src_dir=OUT_DIR, dst_dir=FINAL_DIR)

    print(f"\n{'='*60}")
    print(f"All corrections complete. {elapsed()} total.")
    print(f"\nOutput directories:")
    print(f"  T12-only fix (beta=0.033):              {T12_DIR.name}/")
    print(f"  Drift-only fix (beta=0.1):              {B01_DIR.name}/")
    print(f"  beta=0.1 + T12 combined:                {OUT_DIR.name}/")
    print(f"  FINAL (use this for pipeline): ✓        {FINAL_DIR.name}/")
    print(f"\n  Note: In FINAL output, lt_tu_drift_flag==1 marks FATIGUE_FLEXION")
    print(f"  rows where LT/TU angles are unreliable. The pipeline should either")
    print(f"  exclude these features or treat them as high-uncertainty.")
    print(f"\nTo run the feature pipeline on the final output:")
    print(f"  python scripts/phase_runners/run_phase2_protocol.py --mode imu_only_fallback")
    print(f"{'='*60}")
