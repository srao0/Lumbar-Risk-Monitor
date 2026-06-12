#!/usr/bin/env python3
"""
Session Converter, Spinal Movement Risk Monitor
=================================================
FYP 2025/26 | Imperial College London

Converts raw hardware recordings into a pipeline-ready session directory:

    data/real/raw/<recording_name>/
        ganglion.csv            ← Ganglion BrainFlow output (EMG + on-board accel)
        imu_arduino.csv         ← Arduino serial CSV (4× MPU-6050 via TCA9548A)
        [still_cal.csv]         ← Optional: 10-30 s still recording for gyro calibration

    ──▶ data/real/processed/<session_id>/
        emg_data.csv            ← Pipeline-ready EMG (timestamp_ms + 4 channels)
        imu_data.csv            ← Pipeline-ready IMU (timestamp_ms + quaternions + angles)
        labels.csv              ← Imported recorded protocol labels in official Phase II

Column mapping
--------------
OpenBCI EMG CSV -> emg_data.csv:
    emg_ch1 → emg_LES_mv   (Left Erector Spinae)
    emg_ch2 → emg_RES_mv   (Right Erector Spinae)
    emg_ch3 → emg_LOBL_mv  (Left Oblique, surface, ~4 cm lateral at L5; NOT multifidus)
    emg_ch4 → emg_ROBL_mv  (Right Oblique, surface; NOT multifidus)
    timestamp_unix → timestamp_ms  (relative to recording start)

Arduino IMU CSV → imu_data.csv:
    RawConverter: raw counts → physical units (g, dps)
    GyroBiasCalibrator: subtract per-unit gyro offset
    MadgwickAHRS (beta=0.033, 100 Hz): accel+gyro → quaternion per segment
    Relative angles: θ_PL (pelvis→L3), θ_LT (L3→T12), θ_TU (T12→T4)

Synchronisation
---------------
Both recordings are aligned to timestamp_ms = 0 at their own start.
They must be triggered within ~1 s of each other (manual start).
A hardware sync signal (shared GPIO pulse captured in both CSVs) would give
sub-ms alignment, out of scope for this prototype.

Usage
-----
    # Minimal: EMG only (no IMU hardware yet)
    python scripts/conversion/session_converter.py \\
        --ganglion  data/real/raw/session_001/ganglion.csv \\
        --out_dir   data/real/processed/session_001

    # Full: EMG + IMU
    python scripts/conversion/session_converter.py \\
        --ganglion  data/real/raw/session_001/ganglion.csv \\
        --imu       data/real/raw/session_001/imu_arduino.csv \\
        --labels    data/real/raw/session_001/labels.csv \\
        --still_cal data/real/raw/session_001/still_cal.csv \\
        --out_dir   data/real/protocol_train/participant_01/session_001

    # Run pipeline on result immediately
    python scripts/conversion/session_converter.py \\
        --ganglion  data/real/raw/session_001/ganglion.csv \\
        --imu       data/real/raw/session_001/imu_arduino.csv \\
        --out_dir   data/real/processed/session_001 \\
        --run_pipeline
"""

# Make the project package importable when this file is run directly.
import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = _Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PKG_ROOT))


import argparse
import json
import sys
from datetime import date
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

# Project imports
# Add project root to sys.path when running as script
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from signal_processing.imu.convert import RawConverter, GyroBiasCalibrator, NPoseCalibrator
from signal_processing.imu.madgwick import fuse_four_imu_dataframe
try:
    from scripts.conversion.generate_protocol_labels import MOVEMENT_CATALOGUE
except ModuleNotFoundError:
    from scripts.conversion.generate_protocol_labels import MOVEMENT_CATALOGUE


# GANGLION → EMG

# Channel mapping: Ganglion channel index → anatomical label
# Electrode placement per the Movement Protocol (v2.0):
# CH1 = Left Erector Spinae (L3 level, 3 cm lateral to spinous process)
# CH2 = Right Erector Spinae (same level)
# CH3 = Left Oblique (surface, L5 level, 4 cm lateral), NOT multifidus
# CH4 = Right Oblique (surface), NOT multifidus
EMG_CHANNEL_MAP = {
    "emg_ch1": "emg_LES_mv",
    "emg_ch2": "emg_RES_mv",
    "emg_ch3": "emg_LOBL_mv",
    "emg_ch4": "emg_ROBL_mv",
}

# BrainFlow returns OpenBCI EXG values in microvolts; convert to mV.
OPENBCI_MICROVOLT_TO_MV = 1e-3
DEFAULT_PIPELINE_EMG_FS = 200.0
SUPPORTED_EMG_BOARDS = {"ganglion", "cyton", "synthetic"}


def _resample_emg_dataframe(emg_df: pd.DataFrame, target_fs: float) -> pd.DataFrame:
    """
    Resample pipeline-format EMG to a uniform target rate using polyphase FIR filtering.

    Is intended for raw, unlabelled OpenBCI data before protocol
    labels are joined. It refuses labelled multi-segment data so movement
    boundaries are not silently flattened during resampling.
    """
    if target_fs <= 0 or len(emg_df) < 2:
        return emg_df

    for col in ["label", "rep", "risk_class"]:
        if col in emg_df.columns and emg_df[col].nunique(dropna=False) > 1:
            raise ValueError(
                f"Refusing to resample EMG with multiple {col!r} values. "
                "Resample raw OpenBCI data before joining protocol labels."
            )

    t_ms = emg_df["timestamp_ms"].to_numpy(dtype=float)
    duration_ms = float(t_ms[-1] - t_ms[0])
    if duration_ms <= 0:
        return emg_df

    source_fs = 1000.0 / float(np.median(np.diff(t_ms)))
    if abs(source_fs - target_fs) / target_fs < 0.01:
        return emg_df

    ratio = Fraction(target_fs / source_fs).limit_denominator(1000)
    up = ratio.numerator
    down = ratio.denominator
    target_step_ms = 1000.0 / target_fs
    expected_len = int(np.ceil(len(emg_df) * up / down))
    target_t_ms = np.arange(expected_len, dtype=float) * target_step_ms
    resampled = pd.DataFrame({"timestamp_ms": np.round(target_t_ms, 1)})

    for col in ["label", "rep", "risk_class"]:
        if col in emg_df.columns:
            resampled[col] = emg_df[col].iloc[0]

    for col in EMG_CHANNEL_MAP.values():
        values = emg_df[col].to_numpy(dtype=float)
        valid = np.isfinite(values)
        if valid.sum() < 2:
            resampled[col] = np.nan
        else:
            if not valid.all():
                values = np.interp(t_ms, t_ms[valid], values[valid])
            resampled[col] = resample_poly(values, up, down)[:expected_len]

    print(
        f"  Resampled EMG: {source_fs:.1f} Hz -> {target_fs:.1f} Hz "
        f"using polyphase FIR ({len(emg_df)} -> {len(resampled)} rows)"
    )
    return resampled


def _resample_imu_dataframe(imu_df: pd.DataFrame, target_fs: float) -> pd.DataFrame:
    """
    Resample pipeline-format IMU onto a uniform target-rate grid via timestamp-based
    linear interpolation.

    The Arduino 4-IMU stream is nominally 100 Hz but arrives with jitter and dropped
    samples (observed ~94 Hz effective, ~20% of intervals >=13 ms on some sessions).
    The downstream pipeline hardcodes ``IMU_FS`` and differentiates with ``dt = 1/IMU_FS``
    (``np.gradient`` in ``signal_processing/imu/sal.py``), which assumes a uniform grid.
    Re-gridding here makes that assumption true, so jerk / LDLJ / SAL are computed on a
    regular grid rather than a jittered one.

    Unlike the EMG path (``resample_poly``, which *requires* a uniformly sampled input),
    the IMU stream is non-uniform, so ``np.interp`` against the real timestamps is the
    correct regulariser. Quaternion columns are renormalised after component-wise
    interpolation. Intended for unlabelled (stub-label) IMU before protocol labels are
    joined: it refuses labelled multi-segment data so movement boundaries are not
    silently flattened.

    NOTE on the skip test: a median-dt check (as used for EMG) would be fooled here, 
    dropped samples leave the *median* interval at 10 ms even when 20% of intervals are
    13 ms. We therefore skip only when the grid is uniform to <0.5 ms across *all*
    intervals.
    """
    if target_fs <= 0 or len(imu_df) < 2:
        return imu_df

    for col in ["label", "rep", "risk_class"]:
        if col in imu_df.columns and imu_df[col].nunique(dropna=False) > 1:
            raise ValueError(
                f"Refusing to resample IMU with multiple {col!r} values. "
                "Resample raw IMU before joining protocol labels."
            )

    t_ms = imu_df["timestamp_ms"].to_numpy(dtype=float)
    duration_ms = float(t_ms[-1] - t_ms[0])
    if duration_ms <= 0:
        return imu_df

    target_step_ms = 1000.0 / target_fs
    dt = np.diff(t_ms)
    if dt.size and float(np.max(np.abs(dt - target_step_ms))) < 0.5:
        return imu_df  # already uniform to <0.5 ms — nothing to do

    source_fs = 1000.0 / float(np.median(dt))
    n_target = int(np.floor(duration_ms / target_step_ms)) + 1
    target_t = t_ms[0] + np.arange(n_target, dtype=float) * target_step_ms

    out = pd.DataFrame({"timestamp_ms": np.round(target_t - t_ms[0], 1)})
    for col in ["label", "rep", "risk_class"]:
        if col in imu_df.columns:
            out[col] = imu_df[col].iloc[0]

    passthrough = {"timestamp_ms", "label", "rep", "risk_class"}
    for col in imu_df.columns:
        if col in passthrough:
            continue
        out[col] = np.interp(target_t, t_ms, imu_df[col].to_numpy(dtype=float))

    # Renormalise quaternions after component-wise interpolation.
    for seg in ["pelvis", "l3", "t12", "t4"]:
        qcols = [f"{seg}_{c}" for c in ("qw", "qx", "qy", "qz")]
        if all(q in out.columns for q in qcols):
            q = out[qcols].to_numpy(dtype=float)
            norm = np.linalg.norm(q, axis=1, keepdims=True)
            norm[norm == 0.0] = 1.0
            out[qcols] = q / norm

    pct_jitter = float(np.mean(np.abs(dt - target_step_ms) > 1.5) * 100.0)
    print(
        f"  Resampled IMU: {source_fs:.1f} Hz -> {target_fs:.1f} Hz "
        f"via timestamp interpolation ({len(imu_df)} -> {len(out)} rows; "
        f"{pct_jitter:.0f}% of source intervals were off-grid)"
    )
    return out


def convert_openbci_to_emg(
    emg_csv: Path,
    board: Optional[str] = None,
    target_fs: float = DEFAULT_PIPELINE_EMG_FS,
    label_defaults: bool = True,
) -> pd.DataFrame:
    """
    Convert a BrainFlow OpenBCI CSV to pipeline-ready emg_data.csv format.

    Parameters
    ----------
    emg_csv : path to ganglion_stream.py or cyton_stream.py output CSV
    board : source board label: ganglion, cyton, or synthetic
    target_fs : output EMG sampling rate. Defaults to 200 Hz so the current
                filtering and windowing assumptions remain valid.
    label_defaults : if True, add stub label/rep/risk_class columns

    Returns
    -------
    emg_df : DataFrame with columns:
        timestamp_ms, label, rep, risk_class,
        emg_LES_mv, emg_RES_mv, emg_LOBL_mv, emg_ROBL_mv
    """
    if board is None:
        raise ValueError("Pass an explicit EMG board label: ganglion, cyton, or synthetic.")
    board = board.lower().strip()
    if board not in SUPPORTED_EMG_BOARDS:
        raise ValueError(f"Unsupported EMG board {board!r}; expected one of {sorted(SUPPORTED_EMG_BOARDS)}.")

    raw = pd.read_csv(emg_csv)
    raw.columns = [c.strip() for c in raw.columns]

    # Timestamp: unix seconds → relative milliseconds
    if "timestamp_unix" not in raw.columns:
        raise ValueError(
            f"Expected 'timestamp_unix' column in {emg_csv}. "
            f"Got columns: {list(raw.columns)}"
        )
    t_unix   = raw["timestamp_unix"].to_numpy(dtype=float)
    t_ms     = (t_unix - t_unix[0]) * 1000.0

    # EMG channels → mV
    emg_df = pd.DataFrame()
    emg_df["timestamp_ms"] = np.round(t_ms, 1)

    if label_defaults:
        emg_df["label"]      = "UNKNOWN"
        emg_df["rep"]        = 0
        emg_df["risk_class"] = -1

    missing_channels = []
    for src_col, dst_col in EMG_CHANNEL_MAP.items():
        if src_col in raw.columns:
            emg_df[dst_col] = raw[src_col].to_numpy(dtype=float) * OPENBCI_MICROVOLT_TO_MV
        else:
            missing_channels.append(src_col)
            emg_df[dst_col] = np.nan

    if missing_channels:
        print(f"  [WARNING] Missing OpenBCI channels: {missing_channels}  "
              f"(filled with NaN)")

    return _resample_emg_dataframe(emg_df, target_fs=target_fs)


def convert_ganglion_to_emg(
    ganglion_csv: Path,
    label_defaults: bool = True,
) -> pd.DataFrame:
    """Backward-compatible wrapper for older scripts and notebooks."""
    return convert_openbci_to_emg(
        ganglion_csv,
        board="ganglion",
        target_fs=DEFAULT_PIPELINE_EMG_FS,
        label_defaults=label_defaults,
    )


# ARDUINO IMU CSV → IMU DATA

def check_emg_imu_sync(
    imu_df: pd.DataFrame,
    emg_df: pd.DataFrame,
    imu_fs: float = 100.0,
    emg_fs: float = 200.0,
    max_lag_ms: float = 200.0,
) -> dict:
    """
    Cross-correlation sync check between IMU angular velocity and EMG envelope.

    During fast bends, the L3 angular velocity peak and the EMG RMS envelope
    peak should be temporally close (<100 ms is excellent, <200 ms acceptable
    for manually-triggered dual recordings).

    If the estimated lag is larger than max_lag_ms, the recordings may have
    been started with a significant delay, consider using align_imu_to_emg().

    Parameters
    ----------
    imu_df      : pipeline-format imu_data.csv DataFrame
    emg_df      : pipeline-format emg_data.csv DataFrame
    imu_fs      : IMU sampling rate in Hz
    emg_fs      : EMG sampling rate in Hz
    max_lag_ms  : warn threshold for estimated lag in milliseconds

    Returns
    -------
    result : dict with keys:
        lag_ms          : estimated temporal offset (IMU leads if positive)
        xcorr_peak      : normalised cross-correlation at lag (0-1)
        sync_ok         : True if |lag_ms| ≤ max_lag_ms
    """
    # Resample both signals to a common 50 Hz grid for cross-correlation
    common_fs  = 50.0
    imu_step   = max(1, int(imu_fs / common_fs))
    emg_step   = max(1, int(emg_fs / common_fs))

    av = imu_df["angvel_L3_sagittal"].to_numpy()
    av_ds = np.abs(av[::imu_step])

    # Use mean RMS across EMG channels as the envelope proxy
    emg_cols = [c for c in ["emg_LES_mv", "emg_RES_mv", "emg_LOBL_mv", "emg_ROBL_mv"]
                if c in emg_df.columns]
    if not emg_cols:
        return {"lag_ms": 0.0, "xcorr_peak": 0.0, "sync_ok": True}

    emg_arr = emg_df[emg_cols].to_numpy(dtype=float)
    rms_envelope = np.sqrt(np.mean(emg_arr ** 2, axis=1))[::emg_step]

    # Trim to same length
    min_len = min(len(av_ds), len(rms_envelope))
    a = av_ds[:min_len]
    b = rms_envelope[:min_len]

    # Normalise
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)

    xcorr    = np.correlate(a, b, mode="full")
    lags     = np.arange(-(min_len - 1), min_len)
    peak_idx = np.argmax(np.abs(xcorr))
    lag_samp = lags[peak_idx]
    lag_ms   = lag_samp * (1000.0 / common_fs)
    xcorr_peak = float(np.abs(xcorr[peak_idx]) / min_len)

    sync_ok = abs(lag_ms) <= max_lag_ms
    status  = "OK" if sync_ok else "WARNING"

    print(f"  Sync check [{status}] — estimated lag = {lag_ms:+.1f} ms  "
          f"(cross-corr peak = {xcorr_peak:.3f}, threshold = ±{max_lag_ms:.0f} ms)")
    if not sync_ok:
        print(f"  [WARNING] Large sync lag detected. Consider running align_imu_to_emg() "
              f"with offset_ms={-lag_ms:.0f} to correct.")

    return {"lag_ms": float(lag_ms), "xcorr_peak": xcorr_peak, "sync_ok": sync_ok}


def convert_arduino_to_imu(
    imu_csv: Path,
    still_cal_csv: Optional[Path] = None,
    npose_seconds: float = 10.0,
    auto_still_cal_seconds: float = 60.0,
    fs: float = 100.0,
    beta: float = 0.033,
    label_defaults: bool = True,
    resample_imu: bool = True,
) -> pd.DataFrame:
    """
    Convert Arduino serial 4-IMU CSV to pipeline-ready imu_data.csv.

    Steps:
        1. RawConverter.from_csv(), raw counts → g / dps
        2. GyroBiasCalibrator, subtract per-axis gyro offset
        3. fuse_four_imu_dataframe(), Madgwick AHRS → quaternions + angles
        4. NPoseCalibrator, anatomical frame alignment using first
                                           npose_seconds of data as the N-pose
                                           reference (participant standing still)

    Parameters
    ----------
    imu_csv       : path to Arduino serial CSV (4-IMU format: t_ms, Pelvis_ax, ...)
    still_cal_csv : optional path to a 10-30 s still-recording CSV for gyro bias
    npose_seconds : duration (s) of the N-pose calibration window at recording
                    start.  Participant must stand still in anatomical position
                    for this long at the very beginning of each session.
                    Set to 0.0 to skip N-pose correction.
    fs            : IMU sampling frequency (Hz, should be 100)
    beta          : Madgwick beta gain (0.033 recommended for trunk kinematics)
    label_defaults: if True, add stub label/rep/risk_class columns

    Returns
    -------
    imu_df : DataFrame matching the synthetic imu_data.csv schema
    """
    # 1. Convert raw counts
    df_phys = RawConverter.from_csv(imu_csv)
    print(f"  IMU: {len(df_phys)} samples loaded from {imu_csv.name}")

    # 2. Gyro bias calibration
    if still_cal_csv is not None:
        print(f"  Calibrating gyro bias from {still_cal_csv.name}...")
        for seg, prefix in [("pelvis", "Pelvis"), ("l3", "L3"),
                             ("t12", "T12"), ("t4", "T4")]:
            offsets = GyroBiasCalibrator.compute_from_csv(
                still_cal_csv, imu_prefix=prefix
            )
            df_phys = GyroBiasCalibrator.apply(df_phys, offsets, imu_prefix=prefix)
    elif auto_still_cal_seconds > 0:
        print(
            f"  Calibrating gyro bias from first {auto_still_cal_seconds:g} s "
            f"of {imu_csv.name}..."
        )
        print("  Assumption: participant is static during the initial baseline.")
        for seg, prefix in [("pelvis", "Pelvis"), ("l3", "L3"),
                             ("t12", "T12"), ("t4", "T4")]:
            offsets = GyroBiasCalibrator.compute_from_csv(
                imu_csv,
                imu_prefix=prefix,
                start_s=0.0,
                duration_s=auto_still_cal_seconds,
            )
            df_phys = GyroBiasCalibrator.apply(df_phys, offsets, imu_prefix=prefix)
    else:
        print("  [WARNING] No still calibration CSV provided. "
              "Gyro bias NOT corrected — angle estimates may drift up to ~6°/s.")
        print("  Recommendation: record 10–30 s still at session start, "
              "then rerun with --still_cal <path>.")

    # 3. Madgwick fusion → quaternions + relative angles
    print(f"  Running Madgwick AHRS (beta={beta}, fs={fs} Hz)...")
    df_fused = fuse_four_imu_dataframe(df_phys, fs=fs, beta=beta)

    # 4. N-pose anatomical frame calibration
    if npose_seconds > 0.0:
        # Skip first 5 s of the N-pose window to avoid Madgwick convergence
        # transient contaminating the reference quaternion.
        npose_skip = min(5.0, npose_seconds * 0.4)
        print(f"  Applying N-pose calibration "
              f"(t={npose_skip:.0f}s–{npose_seconds:.0f}s = anatomical reference, "
              f"first {npose_skip:.0f}s skipped for filter convergence)...")
        try:
            npose_offsets = NPoseCalibrator.compute_offsets(
                df_fused, n_seconds=npose_seconds, skip_seconds=npose_skip, fs=fs
            )
            df_fused = NPoseCalibrator.apply(df_fused, npose_offsets,
                                             recompute_relative=True, fs=fs)
            print(f"  N-pose correction applied. Angles now relative to "
                  f"standing anatomical position.")
        except (ValueError, KeyError) as e:
            print(f"  [WARNING] N-pose calibration failed: {e}")
            print(f"  Proceeding without anatomical frame correction.")
    else:
        print("  N-pose calibration skipped (npose_seconds=0).")

    # 5. Build pipeline-format imu_df
    imu_df = pd.DataFrame()

    # timestamp_ms: zero-reference to recording start.
    # Arduino t_ms is millis() since boot, not since recording start, subtract
    # the first sample's timestamp so t=0 aligns with the first row of the file.
    t_ms_raw = df_fused["t_ms"].to_numpy(dtype=float)
    imu_df["timestamp_ms"] = np.round(t_ms_raw - t_ms_raw[0], 1)

    if label_defaults:
        imu_df["label"]      = "UNKNOWN"
        imu_df["rep"]        = 0
        imu_df["risk_class"] = -1

    # Quaternion columns
    for seg in ["pelvis", "l3", "t12", "t4"]:
        for comp in ["qw", "qx", "qy", "qz"]:
            imu_df[f"{seg}_{comp}"] = df_fused[f"{seg}_{comp}"].to_numpy()

    # Relative angle columns (matching synthetic generator column names)
    for angle_prefix in ["theta_PL", "theta_LT", "theta_TU"]:
        for axis in ["pitch", "roll", "yaw"]:
            imu_df[f"{angle_prefix}_{axis}"] = df_fused[f"{angle_prefix}_{axis}"].to_numpy()

    # Angular velocity
    imu_df["angvel_L3_sagittal"] = df_fused["angvel_L3_sagittal"].to_numpy()

    # Raw L3 accelerometer (g), drift-free trunk tilt source for the pipeline.
    # arctan2(ax, sqrt(ay²+az²)) gives sagittal tilt from vertical without
    # any gyro integration, so it is immune to gyro bias and filter drift.
    for axis in ["ax", "ay", "az"]:
        col = f"L3_{axis}_g"
        if col in df_phys.columns:
            imu_df[f"l3_{axis}_g"] = df_phys[col].to_numpy()

    # 6. Regularise the sampling grid
    # The Arduino IMU arrives non-uniformly (serial jitter + dropped samples). The
    # pipeline differentiates with dt = 1/IMU_FS assuming a uniform grid, so re-grid
    # onto a true `fs` Hz grid before features are computed. No-op if already uniform.
    # Labels are still stubs at this point, so segment boundaries cannot be flattened.
    if resample_imu:
        imu_df = _resample_imu_dataframe(imu_df, target_fs=fs)

    return imu_df


# STUB LABELS

def post_hoc_drift_correct(
    imu_df: pd.DataFrame,
    bl2_start_ms: float,
    bl2_end_ms: float,
) -> tuple[pd.DataFrame, dict]:
    """
    Remove linear post-hoc pitch/roll drift using an end-session still window.

    BL1 gyro bias correction removes a fixed gyro offset estimated at the start.
    On hot or long sessions, residual angle drift can remain. This correction
    assumes the participant is upright during BL2, so relative pitch/roll should
    be near zero there. The measured BL2 residual is linearly interpolated from
    zero at t=0 to the centre of BL2, then subtracted from the full session.
    """
    if bl2_start_ms is None or bl2_end_ms is None:
        raise ValueError("Both bl2_start_ms and bl2_end_ms are required for drift correction.")
    if bl2_start_ms < 0:
        raise ValueError("--bl2_start_ms must be >= 0.")
    if bl2_end_ms <= bl2_start_ms:
        raise ValueError("--bl2_end_ms must be greater than --bl2_start_ms.")

    angle_cols = [
        f"{angle_prefix}_{axis}"
        for angle_prefix in ["theta_PL", "theta_LT", "theta_TU"]
        for axis in ["pitch", "roll"]
    ]
    missing = [col for col in ["timestamp_ms"] + angle_cols if col not in imu_df.columns]
    if missing:
        raise ValueError(f"Cannot apply BL2 drift correction; missing columns: {missing}")

    t = imu_df["timestamp_ms"].to_numpy(dtype=float)
    max_t = float(np.nanmax(t))
    if bl2_end_ms > max_t:
        raise ValueError(
            f"--bl2_end_ms ({bl2_end_ms:g}) exceeds IMU duration ({max_t:g} ms)."
        )

    bl2_mask = (imu_df["timestamp_ms"] >= bl2_start_ms) & (imu_df["timestamp_ms"] <= bl2_end_ms)
    n_bl2_samples = int(bl2_mask.sum())
    if n_bl2_samples < 100:
        raise ValueError(
            f"BL2 window contains only {n_bl2_samples} samples; use at least about 1 s of still data."
        )

    t_ref_ms = float(imu_df.loc[bl2_mask, "timestamp_ms"].mean())
    if t_ref_ms <= 0:
        raise ValueError("BL2 reference time must be greater than zero.")

    corrected = imu_df.copy()
    scale = t / t_ref_ms
    residual_offsets = {}
    for col in angle_cols:
        residual = float(corrected.loc[bl2_mask, col].mean())
        corrected[col] = corrected[col].to_numpy(dtype=float) - residual * scale
        residual_offsets[col] = residual

    max_abs_residual = max(abs(v) for v in residual_offsets.values()) if residual_offsets else 0.0
    metadata = {
        "enabled": True,
        "method": "linear_bl2_zero_reference",
        "bl2_start_ms": float(bl2_start_ms),
        "bl2_end_ms": float(bl2_end_ms),
        "bl2_reference_ms": t_ref_ms,
        "bl2_samples": n_bl2_samples,
        "corrected_columns": angle_cols,
        "residual_offsets_deg": residual_offsets,
        "max_abs_residual_deg": float(max_abs_residual),
    }
    print(
        "  Applied BL2 linear drift correction "
        f"({bl2_start_ms:g}-{bl2_end_ms:g} ms, {n_bl2_samples} samples, "
        f"max residual {max_abs_residual:.2f} deg)."
    )
    return corrected, metadata


def create_stub_labels(
    total_duration_ms: float,
    out_path: Path,
) -> pd.DataFrame:
    """
    Create a development-only single-row labels.csv covering the recording as UNKNOWN.

    The user fills in segment boundaries manually after the session using
    a spreadsheet or the annotation helper (scripts/acquisition/annotate_session.py,
    TBD). The pipeline's majority-vote window labelling will then work
    correctly once real labels are present.

    Columns match labels.csv from synthetic_generator.py:
        start_ms, end_ms, label, rep, risk_class
    """
    labels_df = pd.DataFrame([{
        "start_ms":   0.0,
        "end_ms":     round(total_duration_ms, 1),
        "label":      "UNKNOWN",
        "rep":        0,
        "risk_class": -1,
    }])
    labels_df.to_csv(out_path, index=False)
    return labels_df


REQUIRED_RECORDED_LABEL_COLUMNS = {"start_ms", "end_ms", "label", "rep", "risk_class"}
OPERATING_MODES = {"full_hybrid", "imu_only_fallback"}


def is_official_phase2(phase: str) -> bool:
    """Return True for official Phase II training or held-out testing sessions."""
    normalised = str(phase or "").strip().lower().replace("phase ", "")
    return normalised in {"ii.a", "ii.b", "ii.c", "ii.1", "ii.2"}


def load_recorded_protocol_labels(labels_csv: Path) -> pd.DataFrame:
    """Load and fail-fast validate a recorded protocol labels file."""
    if labels_csv is None or not Path(labels_csv).exists():
        raise ValueError(
            "Official Phase II conversion requires --labels pointing to a recorded protocol labels.csv."
        )
    labels_csv = Path(labels_csv)
    try:
        labels_df = pd.read_csv(labels_csv)
    except Exception as exc:
        raise ValueError(f"Could not read recorded labels file {labels_csv}: {exc}") from exc
    missing = sorted(REQUIRED_RECORDED_LABEL_COLUMNS - set(labels_df.columns))
    if missing:
        raise ValueError(f"Recorded labels file {labels_csv} is missing columns: {missing}")
    if labels_df.empty:
        raise ValueError(f"Recorded labels file {labels_csv} is empty.")

    labels_df = labels_df.copy()
    labels_df["label"] = labels_df["label"].fillna("").astype(str).str.strip()
    if (labels_df["label"] == "").any() or (labels_df["label"].str.upper() == "UNKNOWN").any():
        raise ValueError("Recorded protocol labels must not contain blank or UNKNOWN task labels.")
    unsupported = sorted(set(labels_df["label"]) - set(MOVEMENT_CATALOGUE))
    if unsupported:
        raise ValueError(f"Recorded protocol labels contain unsupported tasks: {unsupported}")
    try:
        start_ms = pd.to_numeric(labels_df["start_ms"], errors="raise")
        end_ms = pd.to_numeric(labels_df["end_ms"], errors="raise")
        risk_class = pd.to_numeric(labels_df["risk_class"], errors="raise").astype(int)
    except Exception as exc:
        raise ValueError("Recorded protocol labels contain non-numeric timing or risk_class values.") from exc
    if (start_ms < 0).any() or (end_ms <= start_ms).any():
        raise ValueError("Recorded protocol labels contain invalid start_ms/end_ms intervals.")
    for label, recorded_risk in zip(labels_df["label"], risk_class):
        expected_risk = MOVEMENT_CATALOGUE[label]["risk_class"]
        if recorded_risk != expected_risk:
            raise ValueError(
                f"Recorded task {label!r} must use risk_class={expected_risk}, found {recorded_risk}."
            )
    return labels_df


def require_official_sensor_source(source_path, sensor_name: str) -> Path:
    """Require an actual non-empty raw sensor source for official Phase II."""
    if source_path is None:
        raise ValueError(f"Official Phase II conversion requires real {sensor_name} input; no stub is permitted.")
    path = Path(source_path)
    if not path.exists():
        raise ValueError(f"Official Phase II conversion requires {sensor_name} input, but file is missing: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Official Phase II conversion requires non-empty {sensor_name} input: {path}")
    return path


def validate_official_processed_sensors(
    emg_df: pd.DataFrame,
    imu_df: pd.DataFrame,
    operating_mode: str = "full_hybrid",
) -> None:
    """Reject empty or recognisable placeholder sensor outputs before saving."""
    if imu_df.empty:
        raise ValueError("Official Phase II conversion produced empty IMU data.")

    imu_signal_columns = [
        "theta_PL_pitch", "theta_PL_roll", "theta_PL_yaw",
        "theta_LT_pitch", "theta_LT_roll", "theta_LT_yaw",
        "theta_TU_pitch", "theta_TU_roll", "theta_TU_yaw",
        "angvel_L3_sagittal",
    ]
    if any(column not in imu_df.columns for column in imu_signal_columns):
        raise ValueError("Official Phase II IMU data are missing required motion channels.")
    imu_signal = imu_df[imu_signal_columns].apply(pd.to_numeric, errors="coerce")
    if imu_signal.isna().all().all() or bool((imu_signal.fillna(0.0) == 0.0).all().all()):
        raise ValueError("Official Phase II conversion rejected IMU stub data (all motion channels are zero/NaN).")

    if operating_mode == "full_hybrid":
        if emg_df.empty:
            raise ValueError("Official full-hybrid Phase II conversion produced empty EMG data.")
        emg_columns = ["emg_LES_mv", "emg_RES_mv", "emg_LOBL_mv", "emg_ROBL_mv"]
        if any(column not in emg_df.columns for column in emg_columns):
            raise ValueError("Official full-hybrid Phase II EMG data are missing required channels.")
        if emg_df[emg_columns].isna().all().all():
            raise ValueError(
                "Official full-hybrid Phase II conversion rejected EMG stub data "
                "(all required channels are NaN)."
            )


# MAIN CONVERTER

def convert_session(
    out_dir: Path,
    emg_csv=None,
    emg_board: Optional[str] = None,
    emg_target_fs: float = DEFAULT_PIPELINE_EMG_FS,
    ganglion_csv=None,
    imu_csv=None,
    still_cal_csv=None,
    labels_csv=None,
    operating_mode: str = "full_hybrid",
    participant_id: Optional[str] = None,
    session_id: Optional[str] = None,
    phase: str = "II.A",
    protocol: str = "standard_phase2",
    npose_seconds: float = 10.0,
    auto_still_cal_seconds: float = 60.0,
    bl2_start_ms: Optional[float] = None,
    bl2_end_ms: Optional[float] = None,
    imu_fs: float = 100.0,
    beta: float = 0.033,
    run_pipeline: bool = False,
) -> dict:
    """
    Convert one session's raw recordings into a pipeline-ready session directory.

    Parameters
    ----------
    out_dir       : output session directory (created if it doesn't exist)
    emg_csv       : optional path to OpenBCI BrainFlow CSV from Ganglion or Cyton.
                    If None, an EMG stub is created (IMU-only mode).
    emg_board     : OpenBCI source board label: ganglion, cyton, or synthetic.
    emg_target_fs : processed EMG sampling rate written to emg_data.csv.
    ganglion_csv  : legacy alias for emg_csv.
    imu_csv       : optional Arduino IMU CSV (4-IMU format).
                    If None, an IMU stub is created.
    still_cal_csv : optional still-recording CSV for gyro bias correction
    labels_csv    : recorded protocol labels CSV. Required for official Phase II.
    operating_mode: full_hybrid (real IMU + EMG) or imu_only_fallback (real IMU only).
    participant_id: Phase II participant identifier, e.g. participant_01
    session_id    : Phase II session identifier, e.g. session_001
    phase         : project phase for metadata (default: II.A)
    protocol      : protocol name for metadata (default: standard_phase2)
    npose_seconds : duration (s) of N-pose anatomical calibration window at
                    session start (participant standing still). Set to 0 to skip.
    bl2_start_ms  : optional start timestamp for end-session still baseline.
    bl2_end_ms    : optional end timestamp for end-session still baseline.
    imu_fs        : IMU sampling frequency (Hz)
    beta          : Madgwick filter gain
    run_pipeline  : if True, immediately run the signal processing pipeline

    Returns
    -------
    dict with keys: 'session_dir', 'emg_path', 'imu_path', 'labels_path'
    """
    official_phase2 = is_official_phase2(phase)
    if operating_mode not in OPERATING_MODES:
        raise ValueError(f"Unsupported operating mode {operating_mode!r}; expected one of {sorted(OPERATING_MODES)}.")
    if (bl2_start_ms is None) != (bl2_end_ms is None):
        raise ValueError("Pass both --bl2_start_ms and --bl2_end_ms, or omit both.")
    drift_correction_metadata = {"enabled": False}
    if emg_csv is not None and ganglion_csv is not None:
        raise ValueError("Pass either --emg or legacy --ganglion, not both.")
    if emg_csv is None:
        emg_csv = ganglion_csv
    if emg_csv is not None and emg_board is None and ganglion_csv is not None:
        emg_board = "ganglion"
    if emg_csv is not None and emg_board is None:
        raise ValueError("Pass --emg_board ganglion, cyton, or synthetic when using --emg.")
    supplied_emg_csv = emg_csv
    recorded_labels = None
    if labels_csv is not None or official_phase2:
        recorded_labels = load_recorded_protocol_labels(Path(labels_csv) if labels_csv else None)
    if official_phase2:
        imu_csv = require_official_sensor_source(imu_csv, "IMU")
        if operating_mode == "full_hybrid":
            emg_csv = require_official_sensor_source(emg_csv, "EMG/OpenBCI")
        else:
            emg_csv = None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    participant_id = participant_id or (out_dir.parent.name if out_dir.parent != out_dir else "unknown")
    session_id = session_id or out_dir.name

    imu_only_mode = emg_csv is None
    emg_only_mode = imu_csv is None

    print(f"\n{'='*60}")
    print(f"Session Converter")
    print(f"  Participant  : {participant_id}")
    print(f"  Session ID   : {session_id}")
    print(f"  Mode         : {operating_mode}")
    if emg_csv:
        print(f"  EMG CSV      : {emg_csv} ({emg_board})")
    else:
        print(f"  EMG CSV      : (none — IMU-only mode)")
    if imu_csv:
        print(f"  IMU CSV      : {imu_csv}")
    else:
        print(f"  IMU CSV      : (none — EMG-only mode)")
    print(f"  Output dir   : {out_dir}")
    print(f"{'='*60}\n")

    # EMG
    emg_path = out_dir / "emg_data.csv"
    if emg_csv is not None:
        print("[1/3] Converting OpenBCI EMG...")
        emg_df = convert_openbci_to_emg(
            Path(emg_csv),
            board=emg_board,
            target_fs=emg_target_fs,
            label_defaults=True,
        )
        if not official_phase2:
            emg_df.to_csv(emg_path, index=False)
        emg_duration_ms = float(emg_df["timestamp_ms"].max())
        print(f"  {'Prepared' if official_phase2 else 'Saved'}: {emg_path.name}  ({len(emg_df)} rows, "
              f"{emg_duration_ms/1000:.1f} s)")
    else:
        print("[1/3] No EMG CSV — EMG stub will be created after IMU conversion.")
        emg_df = None
        emg_duration_ms = None

    # IMU
    imu_path = out_dir / "imu_data.csv"
    if imu_csv is not None:
        imu_csv = Path(imu_csv)
        still_cal = Path(still_cal_csv) if still_cal_csv else None
        print("\n[2/3] Converting Arduino IMU data...")
        imu_df = convert_arduino_to_imu(
            imu_csv, still_cal_csv=still_cal,
            npose_seconds=npose_seconds,
            auto_still_cal_seconds=auto_still_cal_seconds,
            fs=imu_fs, beta=beta, label_defaults=True,
        )
        if bl2_start_ms is not None and bl2_end_ms is not None:
            imu_df, drift_correction_metadata = post_hoc_drift_correct(
                imu_df,
                bl2_start_ms=bl2_start_ms,
                bl2_end_ms=bl2_end_ms,
            )
        if not official_phase2:
            imu_df.to_csv(imu_path, index=False)
        imu_duration_ms = float(imu_df["timestamp_ms"].max())
        print(f"  {'Prepared' if official_phase2 else 'Saved'}: {imu_path.name}  ({len(imu_df)} rows, "
              f"{imu_duration_ms/1000:.1f} s)")

        # If no EMG, create EMG stub now that we know IMU duration
        if emg_df is None:
            print("\n  Creating EMG stub (no OpenBCI data)...")
            _create_emg_stub(imu_df, out_dir)
            emg_df = pd.read_csv(emg_path)
            emg_duration_ms = imu_duration_ms
            print(f"  EMG stub saved: {emg_path.name}")
        else:
            # Warn if EMG and IMU durations differ by more than 2 s
            delta_s = abs(emg_duration_ms - imu_duration_ms) / 1000.0
            if delta_s > 2.0:
                print(f"\n  [WARNING] EMG and IMU durations differ by {delta_s:.1f} s.")
                print(f"  This may indicate the recordings were not started simultaneously.")
                print(f"  Check that both were triggered within ~1 s of each other.")

            # Cross-correlation temporal sync check
            print("\n  Running EMG–IMU sync check...")
            sync_result = check_emg_imu_sync(imu_df, emg_df, imu_fs=imu_fs, emg_fs=emg_target_fs)
    else:
        print("\n[2/3] No IMU CSV provided — creating synthetic IMU placeholder...")
        print("  The pipeline will run in EMG-only mode until real IMU data arrives.")
        if emg_df is None:
            raise ValueError("At least one of --emg/--ganglion or --imu must be provided.")
        _create_imu_stub(emg_df, out_dir)
        imu_duration_ms = emg_duration_ms
        print(f"  Stub saved: {imu_path.name}")

    # Labels
    if official_phase2:
        validate_official_processed_sensors(emg_df, imu_df, operating_mode=operating_mode)
        emg_df.to_csv(emg_path, index=False)
        imu_df.to_csv(imu_path, index=False)
        print("\n  Official Phase II sensor integrity preflight passed.")
        print(f"  Saved: {emg_path.name}, {imu_path.name}")

    total_duration_ms = min(emg_duration_ms, imu_duration_ms)
    labels_path = out_dir / "labels.csv"
    if recorded_labels is not None:
        print("\n[3/3] Importing recorded protocol labels.csv...")
        recorded_labels.to_csv(labels_path, index=False)
        print(f"  Saved: {labels_path.name} ({len(recorded_labels)} labelled segment(s))")
    elif official_phase2:
        raise ValueError("Official Phase II conversion cannot proceed without recorded protocol labels.csv.")
    elif labels_path.exists():
        print(f"\n[3/3] labels.csv already exists - keeping existing annotations.")
    else:
        print("\n[3/3] Creating development-only stub labels.csv...")
        create_stub_labels(total_duration_ms, labels_path)
        print(f"  Saved: {labels_path.name}")
    if recorded_labels is None:
        print(f"\n  ACTION REQUIRED: Open {labels_path} and fill in the correct")
        print(f"  segment boundaries (start_ms, end_ms, label, rep, risk_class)")
        print(f"  before running the ML pipeline. See Movement Protocol v2.0 for")
        print(f"  valid label names and risk_class definitions.")

    metadata_path = out_dir / "session_metadata.json"
    metadata = {
        "participant_id": participant_id,
        "session_id": session_id,
        "phase": phase,
        "protocol": protocol,
        "operating_mode": operating_mode,
        "date": date.today().isoformat(),
        "emg_source": str(emg_csv) if emg_csv else None,
        "emg_source_supplied": str(supplied_emg_csv) if supplied_emg_csv else None,
        "emg_board": emg_board if emg_csv else None,
        "imu_source": str(imu_csv) if imu_csv else None,
        "imu_available": imu_csv is not None,
        "emg_available": operating_mode == "full_hybrid" and emg_csv is not None,
        "emg_used_for_inference": operating_mode == "full_hybrid",
        "still_cal_source": str(still_cal_csv) if still_cal_csv else None,
        "auto_still_cal_seconds": auto_still_cal_seconds if not still_cal_csv else None,
        "imu_fs_hz": imu_fs,
        "emg_fs_hz": emg_target_fs,
        "npose_seconds": npose_seconds,
        "madgwick_beta": beta,
        "post_hoc_drift_correction": drift_correction_metadata,
        "duration_ms": float(total_duration_ms),
        "notes": "Fill participant/session notes before using this session for final thesis claims.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  Metadata saved: {metadata_path.name}")

    # Optional: run pipeline
    if run_pipeline:
        print(f"\n[4/4] Running signal processing pipeline...")
        from signal_processing.pipeline import run_pipeline as _run_pipeline
        feat_df = _run_pipeline(str(out_dir), emg_fs=emg_target_fs)
        print(f"  Feature matrix: {feat_df.shape[0]} windows × {feat_df.shape[1]} features")

    print(f"\n{'='*60}")
    print(f"Done. Session directory: {out_dir}")
    print(f"{'='*60}\n")

    return {
        "session_dir":  str(out_dir),
        "emg_path":     str(emg_path),
        "imu_path":     str(imu_path) if imu_path else None,
        "labels_path":  str(labels_path),
        "metadata_path": str(metadata_path),
    }


# IMU STUB (when no IMU hardware available yet)

def _create_emg_stub(imu_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Create a minimal emg_data.csv with all-NaN EMG channels so the pipeline
    can run in IMU-only mode when no Ganglion data is available.
    """
    emg_df = pd.DataFrame()
    emg_df["timestamp_ms"] = imu_df["timestamp_ms"].values
    emg_df["label"]        = imu_df["label"].values
    emg_df["rep"]          = imu_df["rep"].values
    emg_df["risk_class"]   = imu_df["risk_class"].values
    for col in ["emg_LES_mv", "emg_RES_mv", "emg_LOBL_mv", "emg_ROBL_mv"]:
        emg_df[col] = np.nan
    emg_df.to_csv(out_dir / "emg_data.csv", index=False)


def _create_imu_stub(emg_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Create a minimal imu_data.csv with all zeros so the EMG-only pipeline
    can run without errors.

    The zero quaternions and angles mean the IMU features will be meaningless
    (angles = 0, risk_zone_fraction = 0, etc.), but the pipeline will at
    least complete without crashing. The three-condition evaluation will use
    EMG-only and IMU+sEMG conditions once real IMU data is available.
    """
    imu_df = pd.DataFrame()
    imu_df["timestamp_ms"] = emg_df["timestamp_ms"].values[::2]  # ~100 Hz from 200 Hz EMG
    imu_df["label"]        = emg_df["label"].values[::2]
    imu_df["rep"]          = emg_df["rep"].values[::2]
    imu_df["risk_class"]   = emg_df["risk_class"].values[::2]

    n = len(imu_df)

    # Identity quaternions (no rotation = upright)
    for seg in ["pelvis", "l3", "t12", "t4"]:
        imu_df[f"{seg}_qw"] = 1.0
        for c in ["qx", "qy", "qz"]:
            imu_df[f"{seg}_{c}"] = 0.0

    # Zero angles and velocity
    for col in [
        "theta_PL_pitch", "theta_PL_roll", "theta_PL_yaw",
        "theta_LT_pitch", "theta_LT_roll", "theta_LT_yaw",
        "theta_TU_pitch", "theta_TU_roll", "theta_TU_yaw",
        "angvel_L3_sagittal",
    ]:
        imu_df[col] = 0.0

    imu_df.to_csv(out_dir / "imu_data.csv", index=False)


# TIMESTAMP SYNC UTILITY

def align_imu_to_emg(
    imu_df: pd.DataFrame,
    emg_df: pd.DataFrame,
    offset_ms: float = 0.0,
) -> pd.DataFrame:
    """
    Shift IMU timestamps by a fixed offset to align with EMG timestamps.

    Use this post-hoc if you know the two recordings were started with a
    fixed delay (e.g. EMG started 0.5 s before IMU).

    Parameters
    ----------
    imu_df    : imu_data.csv DataFrame
    emg_df    : emg_data.csv DataFrame (reference timeline)
    offset_ms : ms to ADD to IMU timestamps (positive → IMU started later)

    Returns
    -------
    imu_df copy with adjusted timestamp_ms
    """
    imu_aligned = imu_df.copy()
    imu_aligned["timestamp_ms"] = imu_aligned["timestamp_ms"] + offset_ms
    # Clip to EMG time range
    t_max = float(emg_df["timestamp_ms"].max())
    imu_aligned = imu_aligned[imu_aligned["timestamp_ms"] <= t_max].copy()
    return imu_aligned


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert raw OpenBCI EMG + Arduino recordings to pipeline-ready session directory."
    )
    parser.add_argument(
        "--emg", default=None,
        help="Path to OpenBCI BrainFlow CSV from ganglion_stream.py or cyton_stream.py. "
             "If omitted, an EMG stub is created; do not use that output for formal hybrid collection."
    )
    parser.add_argument(
        "--emg_board", default=None,
        choices=sorted(SUPPORTED_EMG_BOARDS),
        help="OpenBCI board used for --emg. Required with --emg; use ganglion, cyton, or synthetic."
    )
    parser.add_argument(
        "--emg_target_fs", type=float, default=DEFAULT_PIPELINE_EMG_FS,
        help="Processed EMG sampling frequency in Hz. Default keeps the existing pipeline at 200 Hz."
    )
    parser.add_argument(
        "--ganglion", default=None,
        help="Legacy alias for --emg when using Ganglion recordings."
    )
    parser.add_argument(
        "--imu", default=None,
        help="Path to Arduino 4-IMU serial CSV; if omitted, a stub is created and is not formal hybrid data."
    )
    parser.add_argument(
        "--still_cal", default=None,
        help="Path to 10–30 s still-recording CSV for gyro bias calibration"
    )
    parser.add_argument(
        "--labels", default=None,
        help="Path to recorded protocol labels.csv. Required for official Phase II conversion."
    )
    parser.add_argument(
        "--mode", dest="operating_mode", default="full_hybrid",
        choices=sorted(OPERATING_MODES),
        help="Official collection mode: full hybrid or explicitly labelled IMU-only fallback."
    )
    parser.add_argument(
        "--out_dir", required=True,
        help="Output directory (session folder), e.g. data/real/processed/session_001"
    )
    parser.add_argument(
        "--participant_id", default=None,
        help="Phase II participant identifier, e.g. participant_01. "
             "Defaults to the parent folder name."
    )
    parser.add_argument(
        "--session_id", default=None,
        help="Phase II session identifier, e.g. session_001. "
             "Defaults to the output folder name."
    )
    parser.add_argument(
        "--phase", default="II.A",
        help="Project phase stored in session_metadata.json (default: II.A)"
    )
    parser.add_argument(
        "--protocol", default="standard_phase2",
        help="Protocol name stored in session_metadata.json"
    )
    parser.add_argument(
        "--npose_seconds", type=float, default=10.0,
        help="Duration (s) of N-pose anatomical calibration window at session "
             "start. Participant must stand still for this long at the beginning "
             "of each recording. Set to 0 to skip N-pose correction. (default: 10)"
    )
    parser.add_argument(
        "--auto_still_cal_seconds", type=float, default=60.0,
        help="If --still_cal is omitted, use the first N seconds of the IMU recording "
             "for gyro-bias correction. Set to 0 to disable. Default: 60."
    )
    parser.add_argument(
        "--bl2_start_ms", type=float, default=None,
        help="Start timestamp in ms for the end-session still baseline used for linear drift correction."
    )
    parser.add_argument(
        "--bl2_end_ms", type=float, default=None,
        help="End timestamp in ms for the end-session still baseline used for linear drift correction."
    )
    parser.add_argument(
        "--imu_fs", type=float, default=100.0,
        help="IMU sampling frequency in Hz (default: 100)"
    )
    parser.add_argument(
        "--beta", type=float, default=0.033,
        help="Madgwick filter beta gain (default: 0.033)"
    )
    parser.add_argument(
        "--run_pipeline", action="store_true",
        help="Run signal processing pipeline on output immediately"
    )
    args = parser.parse_args()
    if args.ganglion and not args.emg and args.emg_board is None:
        args.emg_board = "ganglion"
    if args.emg and args.emg_board is None:
        parser.error("--emg_board is required when using --emg. Choose ganglion, cyton, or synthetic.")

    convert_session(
        out_dir       = Path(args.out_dir),
        emg_csv       = Path(args.emg) if args.emg else None,
        emg_board     = args.emg_board,
        emg_target_fs = args.emg_target_fs,
        ganglion_csv  = Path(args.ganglion) if args.ganglion else None,
        imu_csv       = Path(args.imu) if args.imu else None,
        still_cal_csv = Path(args.still_cal) if args.still_cal else None,
        labels_csv    = Path(args.labels) if args.labels else None,
        operating_mode= args.operating_mode,
        participant_id= args.participant_id,
        session_id    = args.session_id,
        phase         = args.phase,
        protocol      = args.protocol,
        npose_seconds = args.npose_seconds,
        auto_still_cal_seconds = args.auto_still_cal_seconds,
        bl2_start_ms  = args.bl2_start_ms,
        bl2_end_ms    = args.bl2_end_ms,
        imu_fs        = args.imu_fs,
        beta          = args.beta,
        run_pipeline  = args.run_pipeline,
    )
