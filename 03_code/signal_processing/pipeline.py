"""
Signal Processing Pipeline
===========================
Main feature extraction orchestrator for the lumbar movement risk monitor.

Loads IMU and sEMG data from a synthetic (or real) session, runs both
processing chains in aligned sliding windows, and outputs a single
feature_matrix.csv suitable for ML training.

Project phase use
-----------------
    Phase I synthetic validation uses generated task/protocol labels from
    labels.csv to verify the pipeline on controlled synthetic data. It is
    not evidence of real hardware performance.

    Phase II.A real protocol collection uses predefined movement-protocol
    labels from real participants.

    Phase II.C varied movement testing should evaluate frozen Phase II.A
    models on held-out, independently labelled real movement sessions.

    Phase III uses the trained model/FIS outputs for replay or real-time
    dashboard feedback with scientific and lay explanations.

    Signal-derived risk labels are kept as a diagnostic/explanation channel.
    They should not be used as the main model target when claiming zero
    label-feature circularity.

Pipeline overview
-----------------
    IMU data (100 Hz)                  sEMG data (200 Hz)
           |                                   |
    +------v------------------+       +--------v---------------+
    | Quaternion -> Euler      |       | Notch 50 Hz             |
    | Relative joint angles    |       | Bandpass 20-95 Hz       |
    | (th_PL, th_LT, th_TU)    |       +------------------------+
    | Angular velocity (L3)    |                |
    +--------------------------+       +--------v---------------+
           |                           | Sliding window (2000ms) |
    +------v------------------+       | RMS, MAV, ZCR per ch    |
    | Sliding window (2000ms) |       | Asymmetry index (AI)    |
    | LDLJ smoothness (SAL)   |       | Co-activation index(CAI)|
    | Trunk angle (peak, mean) |       +------------------------+
    | Angular velocity (peak)  |                |
    | Time in risk zone        |       +--------v---------------+
    +--------------------------+       | Timestamp alignment    |
           |                           | (nearest IMU window)   |
           +--------------+------------+
                          |
              +-----------v------------+
              | Baseline z-scores       |
              | z_flex, z_vel, z_sal    |
              | z_rms_r, z_ar           |
              | (from BASELINE_STATIC)  |
              +-----------v------------+
                          |
                 +--------v--------+
                 | feature_matrix  |
                 |   .csv          |
                 +-----------------+

Window design (spec 4.3)
--------------------------
    IMU window : 2000 ms = 200 samples at 100 Hz
    EMG window : 2000 ms = 400 samples at 200 Hz
    Step       : 1000 ms (50% overlap) - feedback update rate 1 Hz
    Alignment  : windows defined by centre timestamp in ms

Baseline calibration (spec 5)
--------------------------------
    BASELINE_STATIC windows (Phase 0) are used to compute per-feature
    mu and sigma for each session. Z-scores are then appended to every window:
        z_flex = (th_L3_peak  - mu_flex) / sigma_flex
        z_vel  = (w_peak     - mu_vel)  / sigma_vel
        z_sal  = (SAL        - mu_sal)  / sigma_sal
        z_rms_r = (RMS_RES  - mu_rms_r) / sigma_rms_r
        z_ar    = (AI_ES    - mu_ar)    / sigma_ar

    Z-scores of 0 = exactly at personal baseline.
    |z| > 2 = two standard deviations from personal norm (elevated risk).

sEMG features (time-domain only - spec 6.2)
----------------------------------------------
    RMS, MAV, ZCR per channel. Asymmetry Index (AI), Co-activation Index (CAI)
    per bilateral pair. MPF/spectral features excluded (200 Hz Nyquist constraint).

References
----------
    Flash & Hogan (1985). J Neuroscience, 5(7), 1688-1703.
    Marras et al. (1993). Spine, 18(5), 617-628.
    NIOSH (2007). DHHS Publication No. 97-141.
"""

import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from scripts.datasets.dataset_manifest import MANIFEST_FILENAME, write_dataset_manifest

from signal_processing.imu.sal import (
    ldlj,
    ldlj_multi_axis,
    compute_jerk,
    jerk_rms,
    jerk_peak,
    STATIC_THRESHOLD_DEG_S,
)
from signal_processing.emg.filtering import filter_emg_array
from signal_processing.emg.features import (
    extract_window_features,
    DEFAULT_CHANNEL_NAMES,
    DEFAULT_BILATERAL_PAIRS,
)


# --- Pipeline constants (spec 4.3) -------------------------------------------

IMU_FS                = 100.0    # Hz - IMU sampling rate
EMG_FS                = 200.0    # Hz - Ganglion hardware limit (spec 2.2)
WINDOW_MS             = 2000     # ms - 2 s window captures full movement cycles
STEP_MS               = 1000     # ms - 50% overlap, 1 Hz feedback update rate

# Derived sample counts
IMU_WINDOW_SAMPLES    = int(WINDOW_MS * IMU_FS / 1000)   # 200
EMG_WINDOW_SAMPLES    = int(WINDOW_MS * EMG_FS / 1000)   # 400
IMU_STEP_SAMPLES      = int(STEP_MS * IMU_FS / 1000)     # 100
EMG_STEP_SAMPLES      = int(STEP_MS * EMG_FS / 1000)     # 200

# Risk thresholds (spec 7.1)
RISK_ANGLE_DEG        = 45.0     # degrees - NIOSH threshold for lumbar flexion
RISK_VELOCITY_DEG_S   = 40.0    # deg/s  - Marras et al. threshold

# Minimum baseline windows needed for reliable mu/sigma
MIN_BASELINE_WINDOWS  = 5

# IMU column groups
QUAT_SEGMENTS = ["pelvis", "l3", "t12", "t4"]
QUAT_COLS = {
    seg: [f"{seg}_q{c}" for c in ["w", "x", "y", "z"]]
    for seg in QUAT_SEGMENTS
}
REL_ANGLE_COLS = [
    "theta_PL_pitch", "theta_PL_roll", "theta_PL_yaw",
    "theta_LT_pitch", "theta_LT_roll", "theta_LT_yaw",
    "theta_TU_pitch", "theta_TU_roll", "theta_TU_yaw",
    "angvel_L3_sagittal",
]
EMG_CHANNEL_COLS = ["LES", "RES", "LOBL", "ROBL"]

# Channels 3-4 are surface obliques (~4 cm lateral, L5 level), NOT multifidus.
# Older code mapped emg_LOBL_mv/emg_ROBL_mv to "LMF"/"RMF" (multifidus) - a
# mislabel, since the electrodes never recorded multifidus. The legacy "LMF"/
# "RMF" keys are retained below as backward-compatible aliases so older inputs
# still parse, but they now resolve to the correct LOBL/ROBL channel names.
EMG_COL_ALIASES = {
    "emg_LES_mv":  "LES",
    "emg_RES_mv":  "RES",
    "emg_LOBL_mv": "LOBL",
    "emg_ROBL_mv": "ROBL",
    "LES": "LES",
    "RES": "RES",
    "LOBL": "LOBL",
    "ROBL": "ROBL",
    # legacy aliases (pre-fix mislabel) -> correct oblique channels
    "LMF": "LOBL",
    "RMF": "ROBL",
}

# Features used for baseline z-score computation (spec 6.1, 6.2)
BASELINE_LABEL = "BASELINE_STATIC"
Z_SCORE_FEATURES = {
    # z_name: (raw_feature_col, output_col)
    "z_flex":  ("imu_trunk_angle_peak", "imu_z_flex"),
    "z_vel":   ("imu_angvel_peak",      "imu_z_vel"),
    "z_ldlj":   ("imu_ldlj",              "imu_z_ldlj"),
    "z_rms_r": ("emg_rms_RES",         "emg_z_rms_r"),
    "z_ar":    ("emg_ai_ES",            "emg_z_ar"),
}



# --- Signal-derived risk labelling (spec 6.3) --------------------------------
# Labels are computed from the extracted window features using published
# biomechanical thresholds rather than from generator class membership.
# Each criterion maps to a specific risk mechanism and literature citation.
# Multiple criteria may fire simultaneously; all are stored for downstream use.

RISK_REASON_TEMPLATES = {
    "postural": {
        "clinical": (
            "Lumbar flexion exceeded 45 deg threshold (NIOSH, 2007); "
            "sustained loading of intervertebral disc and posterior ligaments"
        ),
        "layman": "You bent your back too far forward",
    },
    "dynamic": {
        "clinical": (
            "High trunk jerk during flexion; excessive spinal loading rate "
            "associated with disc and muscle injury (Marras et al., 1993)"
        ),
        "layman": "You moved too quickly -- sudden bending increases spinal strain",
    },
    "pattern": {
        "clinical": (
            "Thoracic compensation detected; upper spine driving motion without "
            "adequate lumbo-pelvic rhythm (McGill, 2007); "
            "increased shear load at L3-L4"
        ),
        "layman": "Your upper back is doing the bending instead of your hips",
    },
    "combined": {
        "clinical": (
            "Combined lateral and sagittal loading detected; "
            "tri-planar co-loading increases compressive and shear stress "
            "on lumbar discs (Marras et al., 1993)"
        ),
        "layman": "You bent forward and sideways at the same time",
    },
}

# Thresholds for signal-derived labelling
_THR_TIME_IN_RISK   = 0.05      # fraction of window above 45 deg [NIOSH 2007]
_THR_JERK_RMS       = 120_000   # deg/s^3 -- separates FAST_BEND from safe classes
_THR_COMPENSATION   = 0.60      # thoracic compensation index [McGill 2007]
_THR_LP_RATIO_LOW   = 0.45      # lumbopelv ratio below this = pelvis-dominant with lat
_THR_LAT_COMBINED   = 5.0       # deg -- minimum lateral angle for combined criterion


def apply_protocol_labels(
    window_start_ms: float,
    window_end_ms: float,
    labels_df: pd.DataFrame,
    min_overlap_fraction: float = 0.5,
) -> dict:
    """
    Assign a protocol label to a window using researcher-defined time segments.

    A window inherits the label of the segment with which it has the greatest
    temporal overlap, provided that overlap is at least ``min_overlap_fraction``
    of the window duration (default 50%).  If no segment meets the threshold,
    the window is marked UNKNOWN with risk_class = -1.

    This is the Phase 2 labelling strategy for real participant data.
    Labels are assigned from the experimental protocol (e.g. "participant
    is performing CLEAN_FLEXION between t=30 s and t=60 s") and have ZERO
    label-feature circularity - no IMU or EMG feature values are read during
    label assignment.

    Parameters
    ----------
    window_start_ms       : window start time in milliseconds
    window_end_ms         : window end time in milliseconds
    labels_df             : DataFrame loaded from labels.csv, columns:
                            start_ms, end_ms, label, rep, risk_class
    min_overlap_fraction  : minimum fraction of window duration that must be
                            covered by a single protocol segment to inherit
                            its label (default 0.5 - majority-vote equivalent)

    Returns
    -------
    dict with keys:
        label       : str  movement label from labels.csv (or "UNKNOWN")
        risk_class  : int  0 = safe, 1 = risky, -1 = unknown/ambiguous
        rep         : int  repetition number (0 if unknown)
        overlap_frac: float fraction of window covered by the best segment
    """
    if labels_df is None or len(labels_df) == 0:
        return {"label": "UNKNOWN", "risk_class": -1, "rep": 0, "overlap_frac": 0.0}

    window_dur = window_end_ms - window_start_ms
    if window_dur <= 0:
        return {"label": "UNKNOWN", "risk_class": -1, "rep": 0, "overlap_frac": 0.0}

    best_label   = "UNKNOWN"
    best_risk    = -1
    best_rep     = 0
    best_overlap = 0.0

    for _, seg in labels_df.iterrows():
        seg_start = float(seg["start_ms"])
        seg_end   = float(seg["end_ms"])

        overlap_ms = max(0.0, min(window_end_ms, seg_end) - max(window_start_ms, seg_start))
        if overlap_ms > best_overlap:
            best_overlap = overlap_ms
            best_label   = str(seg["label"])
            best_risk    = int(seg["risk_class"]) if pd.notna(seg.get("risk_class", -1)) else -1
            best_rep     = int(seg["rep"])        if pd.notna(seg.get("rep",        0))  else 0

    overlap_frac = best_overlap / window_dur
    if overlap_frac < min_overlap_fraction:
        return {"label": "UNKNOWN", "risk_class": -1, "rep": 0, "overlap_frac": overlap_frac}

    return {
        "label":        best_label,
        "risk_class":   best_risk,
        "rep":          best_rep,
        "overlap_frac": overlap_frac,
    }


def label_window_from_signal(feats: dict) -> dict:
    """
    Derive risk_class and human-readable explanations from window features.

    Uses four mechanistically distinct criteria with published thresholds:
      1. Postural  -- time in risk zone > 5% of window (NIOSH 2007)
      2. Dynamic   -- jerk RMS > 120,000 deg/s^3 (Marras et al. 1993)
      3. Pattern   -- compensation index > 0.60 (McGill 2007)
      4. Combined  -- low lumbopelv ratio + lateral angle (Marras et al. 1993)

    Parameters
    ----------
    feats : dict of extracted IMU window features

    Returns
    -------
    dict with keys:
        risk_class      : int  (1 = risky, 0 = safe)
        risk_criteria   : str  (comma-separated fired criterion names, or "none")
        risk_clinical   : str  (semicolon-joined clinical explanations)
        risk_layman     : str  (natural-language explanation for non-clinicians)
    """
    fired = []

    # 1. Postural: time fraction above 45-deg threshold
    if feats.get("imu_time_in_risk_zone", 0.0) > _THR_TIME_IN_RISK:
        fired.append("postural")

    # 2. Dynamic: high jerk RMS (abrupt movement regardless of peak angle)
    if feats.get("imu_jerk_rms", 0.0) > _THR_JERK_RMS:
        fired.append("dynamic")

    # 3. Pattern: thoracic compensation (shoulder-driven flexion)
    if feats.get("imu_compensation_index", 0.0) > _THR_COMPENSATION:
        fired.append("pattern")

    # 4. Combined loading: reduced lumbo-pelvic ratio co-occurring with lateral bend
    #    Distinguishes asymmetric multi-planar loading from pure lateral bends
    lp   = feats.get("imu_lumbopelv_ratio",  0.5)
    lat  = feats.get("imu_lat_angle_peak",   0.0)
    if lp < _THR_LP_RATIO_LOW and lat > _THR_LAT_COMBINED:
        fired.append("combined")

    if not fired:
        return {
            "risk_class":    0,
            "risk_criteria": "none",
            "risk_clinical": "",
            "risk_layman":   "",
        }

    clinical_parts = [RISK_REASON_TEMPLATES[c]["clinical"] for c in fired]
    layman_parts   = [RISK_REASON_TEMPLATES[c]["layman"]   for c in fired]

    return {
        "risk_class":    1,
        "risk_criteria": ",".join(fired),
        "risk_clinical": "; ".join(clinical_parts),
        "risk_layman":   " and ".join(layman_parts),
    }


# --- IMU window feature extraction -------------------------------------------

def extract_imu_window_features(imu_window: pd.DataFrame) -> Dict[str, float]:
    """
    Extract kinematic features from one 2000 ms IMU window.

    Features
    --------
    imu_trunk_angle_peak    : max (|th_PL| + |th_LT|) in window (deg) - total lumbar
                              flexion from pelvis; aligns with NIOSH 45 deg criterion
                              (absolute trunk angle from vertical approximation)
    imu_trunk_angle_mean    : mean (|th_PL| + |th_LT|) in window (deg)
    imu_angvel_peak         : max |w_L3| (deg/s)
    imu_angvel_mean         : mean |w_L3| (deg/s)
    imu_time_in_risk_zone   : fraction of window where (|th_PL|+|th_LT|) > 45 deg [0-1]
    imu_time_high_velocity  : fraction where |w| > 40 deg/s [0-1]
    imu_ldlj                 : LDLJ smoothness (log jerk^2 integral; lower = jerky)
    imu_jerk_rms            : RMS jerk (deg/s^3)
    imu_jerk_peak           : peak jerk (deg/s^3)
    imu_ldlj_multiaxis       : LDLJ over 3-axis resultant velocity
    imu_compensation_index  : th_TU / (|th_PL| + |th_TU|) - shoulder compensation
    imu_lumbopelv_ratio     : |th_PL| / (|th_PL| + |th_LT|) - lumbar dominance
    """
    feats = {}

    theta_PL = imu_window["theta_PL_pitch"].to_numpy()
    theta_LT = imu_window["theta_LT_pitch"].to_numpy()
    theta_TU = imu_window["theta_TU_pitch"].to_numpy()
    angvel   = imu_window["angvel_L3_sagittal"].to_numpy()

    # 5-sample moving-average pre-smoothing on angular velocity (spec sec 4.2).
    # Reduces high-frequency noise before jerk differentiation without
    # distorting the movement envelope at 100 Hz (5 samples = 50 ms cutoff).
    _MA = 5
    angvel = np.convolve(angvel, np.ones(_MA) / _MA, mode="same")

    # Total lumbar flexion = pelvis-L3 joint angle + L3-T12 joint angle.
    # This approximates trunk flexion from the global vertical and is the
    # correct measure for the NIOSH 45 deg threshold (Marras et al. 1993).
    # Using th_PL alone underestimates total flexion by the thoracolumbar
    # contribution (typically 5-15 deg) and would never breach 45 deg for
    # movements parameterised with realistic inter-segment distributions.
    trunk_flex = np.abs(theta_PL) + np.abs(theta_LT)

    feats["imu_trunk_angle_peak"] = float(np.max(trunk_flex))
    feats["imu_trunk_angle_mean"] = float(np.mean(trunk_flex))

    feats["imu_angvel_peak"] = float(np.max(np.abs(angvel)))
    feats["imu_angvel_mean"] = float(np.mean(np.abs(angvel)))

    n = len(theta_PL)
    feats["imu_time_in_risk_zone"] = float(
        np.sum(trunk_flex > RISK_ANGLE_DEG) / n
    )
    feats["imu_time_high_velocity"] = float(
        np.sum(np.abs(angvel) > RISK_VELOCITY_DEG_S) / n
    )

    feats["imu_ldlj"]       = ldlj(angvel, IMU_FS)
    feats["imu_jerk_rms"]  = jerk_rms(angvel, IMU_FS)
    feats["imu_jerk_peak"] = jerk_peak(angvel, IMU_FS)

    if all(c in imu_window.columns for c in
           ["theta_PL_pitch", "theta_PL_roll", "theta_PL_yaw"]):
        omega_p = np.gradient(theta_PL, 1.0 / IMU_FS)
        omega_r = np.gradient(imu_window["theta_PL_roll"].to_numpy(), 1.0 / IMU_FS)
        omega_y = np.gradient(imu_window["theta_PL_yaw"].to_numpy(), 1.0 / IMU_FS)
        feats["imu_ldlj_multiaxis"] = ldlj_multi_axis(omega_p, omega_r, omega_y, IMU_FS)
    else:
        feats["imu_ldlj_multiaxis"] = np.nan

    denom_comp = np.abs(theta_PL) + np.abs(theta_TU)
    with np.errstate(divide="ignore", invalid="ignore"):
        comp_vals = np.where(denom_comp > 0.5, np.abs(theta_TU) / denom_comp, np.nan)
    feats["imu_compensation_index"] = float(np.nanmean(comp_vals))

    denom_lp = np.abs(theta_PL) + np.abs(theta_LT)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp_vals = np.where(denom_lp > 0.5, np.abs(theta_PL) / denom_lp, np.nan)
    feats["imu_lumbopelv_ratio"] = float(np.nanmean(lp_vals))

    # Explicit pelvic tilt (sagittal) -- separates pelvis contribution from total trunk flex.
    # Complements imu_lumbopelv_ratio with an absolute angle value.
    feats["imu_pelvis_angle_peak"] = float(np.max(np.abs(theta_PL)))
    feats["imu_pelvis_angle_mean"] = float(np.mean(np.abs(theta_PL)))

    # Lateral flexion (roll component of PL joint angle).
    # Captures coronal-plane asymmetry not visible in sagittal features.
    if "theta_PL_roll" in imu_window.columns:
        theta_lat = imu_window["theta_PL_roll"].to_numpy()
        feats["imu_lat_angle_peak"] = float(np.max(np.abs(theta_lat)))
        feats["imu_lat_angle_mean"] = float(np.mean(np.abs(theta_lat)))
    else:
        feats["imu_lat_angle_peak"] = np.nan
        feats["imu_lat_angle_mean"] = np.nan

    # -- Accelerometer-based trunk tilt (L3) ----------------------------------
    # Compute trunk inclination directly from the L3 gravity vector.
    # arctan2(ax, sqrt(ay^2+az^2)) gives pitch from vertical - no integration,
    # no gyro drift.  Valid for quasi-static movements (<1 g lateral accel).
    # For fast bends the signal is contaminated by linear acceleration, but
    # velocity features already capture those windows; the accel tilt is most
    # informative for slow/deep flexion where the angle threshold matters most.
    l3_ax_col = next((c for c in imu_window.columns
                      if c.lower() in ("l3_ax_g", "l3_ax")), None)
    l3_ay_col = next((c for c in imu_window.columns
                      if c.lower() in ("l3_ay_g", "l3_ay")), None)
    l3_az_col = next((c for c in imu_window.columns
                      if c.lower() in ("l3_az_g", "l3_az")), None)
    if l3_ax_col and l3_ay_col and l3_az_col:
        ax = imu_window[l3_ax_col].to_numpy(dtype=float)
        ay = imu_window[l3_ay_col].to_numpy(dtype=float)
        az = imu_window[l3_az_col].to_numpy(dtype=float)
        tilt = np.degrees(np.arctan2(ax, np.sqrt(ay**2 + az**2)))
        feats["imu_l3_accel_tilt_peak"] = float(np.max(np.abs(tilt)))
        feats["imu_l3_accel_tilt_mean"] = float(np.mean(np.abs(tilt)))
        feats["imu_l3_accel_tilt_range"] = float(np.ptp(tilt))
    else:
        feats["imu_l3_accel_tilt_peak"]  = np.nan
        feats["imu_l3_accel_tilt_mean"]  = np.nan
        feats["imu_l3_accel_tilt_range"] = np.nan

    return feats


# --- Window alignment --------------------------------------------------------

def build_window_index(
    timestamps_ms: np.ndarray,
    window_ms: int = WINDOW_MS,
    step_ms: int = STEP_MS,
) -> List[Dict[str, float]]:
    """
    Build window descriptors centred on equally-spaced timestamps.

    Parameters
    ----------
    timestamps_ms : (N,) sorted timestamp array in milliseconds
    window_ms     : window duration in ms
    step_ms       : step size in ms

    Returns
    -------
    List of dicts: {'centre_ms', 'start_ms', 'end_ms'}
    """
    half = window_ms / 2.0
    t_start = float(timestamps_ms[0]) + half
    t_end   = float(timestamps_ms[-1]) - half

    windows = []
    t = t_start
    while t <= t_end:
        windows.append({"centre_ms": t, "start_ms": t - half, "end_ms": t + half})
        t += step_ms
    return windows


def slice_by_time(
    df: pd.DataFrame,
    start_ms: float,
    end_ms: float,
    time_col: str = "timestamp_ms",
) -> pd.DataFrame:
    """Return rows where time_col is in [start_ms, end_ms)."""
    return df[(df[time_col] >= start_ms) & (df[time_col] < end_ms)]


# --- Baseline z-score computation (spec 5) -----------------------------------

def compute_baseline_stats(
    feature_df: pd.DataFrame,
    baseline_label: str = BASELINE_LABEL,
    z_feature_map: dict = Z_SCORE_FEATURES,
    min_windows: int = MIN_BASELINE_WINDOWS,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute per-feature (mu, sigma) from BASELINE_STATIC windows.

    Parameters
    ----------
    feature_df     : DataFrame with all windows including movement_label column
    baseline_label : label string identifying calibration windows
    z_feature_map  : dict mapping z-name -> (raw_col, output_col)
    min_windows    : minimum windows required; returns NaN stats if below

    Returns
    -------
    stats : dict mapping raw_col -> (mean, std)
    """
    baseline_rows = feature_df[feature_df["movement_label"] == baseline_label]
    stats = {}

    for z_name, (raw_col, _) in z_feature_map.items():
        if raw_col not in baseline_rows.columns:
            stats[raw_col] = (np.nan, np.nan)
            continue
        vals = baseline_rows[raw_col].dropna().values
        if len(vals) < min_windows:
            stats[raw_col] = (np.nan, np.nan)
        else:
            mu  = float(np.mean(vals))
            sig = float(np.std(vals))
            stats[raw_col] = (mu, max(sig, 1e-9))   # avoid 0/0

    return stats


def add_z_scores(
    feature_df: pd.DataFrame,
    baseline_stats: Dict[str, Tuple[float, float]],
    z_feature_map: dict = Z_SCORE_FEATURES,
) -> pd.DataFrame:
    """
    Append z-score columns to feature_df using pre-computed baseline stats.

    For each feature in z_feature_map:
        z = (x - mu_baseline) / sigma_baseline

    Z-scores are clipped to +-5 to limit the influence of outliers.

    Parameters
    ----------
    feature_df      : DataFrame with raw feature columns
    baseline_stats  : dict mapping raw_col -> (mean, std)
    z_feature_map   : dict mapping z-name -> (raw_col, output_col)

    Returns
    -------
    feature_df with additional z-score columns appended in-place copy
    """
    df = feature_df.copy()
    for z_name, (raw_col, out_col) in z_feature_map.items():
        if raw_col not in df.columns:
            df[out_col] = np.nan
            continue
        mu, sig = baseline_stats.get(raw_col, (np.nan, np.nan))
        if np.isnan(mu) or np.isnan(sig):
            df[out_col] = np.nan
        else:
            z = (df[raw_col] - mu) / sig
            df[out_col] = z.clip(-5.0, 5.0)
    return df


# --- sEMG amplitude normalisation (resting-baseline ratio) -------------------

EMG_AMPLITUDE_NORM_METHODS = ("none", "resting_baseline_ratio")


def normalize_emg_amplitude_to_baseline(
    feature_df: pd.DataFrame,
    baseline_label: str = BASELINE_LABEL,
    min_windows: int = MIN_BASELINE_WINDOWS,
    eps: float = 1e-9,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Normalise per-channel sEMG amplitude features to the session's resting
    baseline (resting-baseline ratio):

        x_norm = x / mean(x over BASELINE_STATIC windows)

    Applied IN PLACE to ``emg_rms_*`` and ``emg_mav_*`` columns only. ZCR is
    amplitude-independent and AI/CAI are within-window L/R ratios, so both are
    already amplitude-invariant and are left untouched. After normalisation a
    resting window reads ~1.0, making amplitude comparable across participants,
    sessions and electrode applications without an MVC reference (none is
    available in this protocol).

    Computed per session, since electrodes are re-applied each session. The
    existing baseline z-scores (Z_SCORE_FEATURES) are unaffected: a z-score is
    invariant to dividing the raw column by a per-session constant.

    Parameters
    ----------
    feature_df     : single-session feature matrix (must contain movement_label)
    baseline_label : label identifying resting windows
    min_windows    : minimum resting windows required; below this the channel is
                     left un-normalised and flagged (reference = NaN)
    eps            : guard against division by a ~zero resting mean

    Returns
    -------
    (df, report) : normalised copy and {column: reference_mean_used or NaN}
    """
    df = feature_df.copy()
    if "movement_label" not in df.columns:
        return df, {}
    amp_cols = [
        c for c in df.columns
        if c.startswith("emg_rms_") or c.startswith("emg_mav_")
    ]
    baseline_rows = df[df["movement_label"] == baseline_label]
    report: Dict[str, float] = {}
    for col in amp_cols:
        vals = baseline_rows[col].dropna().values if col in baseline_rows else []
        if len(vals) < min_windows:
            report[col] = float("nan")          # insufficient baseline -> leave raw
            continue
        ref = float(np.mean(vals))
        if not np.isfinite(ref) or ref < eps:
            report[col] = float("nan")          # degenerate reference -> leave raw
            continue
        df[col] = df[col] / ref
        report[col] = ref
    return df, report


# --- NaN fill helper ---------------------------------------------------------

def _emg_nan_row(channel_names: List[str]) -> dict:
    """Return a dict of NaN EMG features for windows with insufficient EMG data."""
    row = {}
    for name in channel_names:
        row[f"emg_rms_{name}"] = np.nan
        row[f"emg_mav_{name}"] = np.nan
        row[f"emg_zcr_{name}"] = np.nan
    row["emg_ai_ES"]   = np.nan
    row["emg_ai_OBL"]  = np.nan
    row["emg_cai_ES"]  = np.nan
    row["emg_cai_OBL"] = np.nan
    return row


# --- Main pipeline -----------------------------------------------------------

def run_pipeline(
    session_dir: str,
    output_dir: Optional[str] = None,
    apply_notch: bool = True,
    window_ms: int = WINDOW_MS,
    step_ms: int = STEP_MS,
    imu_fs: float = IMU_FS,
    emg_fs: float = EMG_FS,
    label_source: str = "protocol",
    emg_amplitude_norm: str = "none",
) -> pd.DataFrame:
    """
    Run the full signal processing pipeline on one session directory.

    Expects:
        <session_dir>/imu_data.csv
        <session_dir>/emg_data.csv
        <session_dir>/labels.csv

    Parameters
    ----------
    session_dir  : path to session folder
    output_dir   : where to save feature_matrix.csv (defaults to session_dir)
    apply_notch  : apply 50 Hz notch to EMG
    window_ms    : window duration in milliseconds (spec: 2000)
    step_ms      : window step in milliseconds (spec: 1000)
    label_source : how to assign the primary ``risk_class`` column:

        ``"protocol"`` (default; Phase I synthetic and Phase II real data)
            Derives risk_class from task/time segments in labels.csv. For
            synthetic sessions these labels are generated by
            synthetic_generator.py; for real sessions they should come from
            the movement protocol, label_logger.py, generate_protocol_labels.py,
            or manual annotation. This is the preferred model target because
            it avoids using extracted features to create labels.

        ``"signal"``   (diagnostic / threshold-derived labels)
            Derives risk_class from IMU features using the four
            biomechanical thresholds in label_window_from_signal(). Fast and
            automatic, but circular if used as the model target because the
            label is computed from features that may also be model inputs.

    Both ``risk_class_signal`` and ``risk_class_protocol`` are always
    written to the output CSV regardless of ``label_source``, so the
    two labelling strategies can be compared on the same data.

    emg_amplitude_norm : "none" (default; raw mV) or "resting_baseline_ratio"
        (divide emg_rms_*/emg_mav_* by the per-session BASELINE_STATIC mean).

    Returns
    -------
    feature_df : DataFrame, one row per window
    """
    session_dir = Path(session_dir)
    if output_dir is None:
        output_dir = session_dir

    imu_window_min = int(window_ms * imu_fs / 1000) // 2
    emg_window_min = int(window_ms * emg_fs / 1000) // 2

    # -- Load data ---------------------------------------------------------
    imu_df    = pd.read_csv(session_dir / "imu_data.csv")
    emg_df    = pd.read_csv(session_dir / "emg_data.csv")
    labels_df = pd.read_csv(session_dir / "labels.csv")

    # -- Remap EMG column names --------------------------------------------
    emg_df = emg_df.rename(columns={
        alias: canon for alias, canon in EMG_COL_ALIASES.items()
        if alias in emg_df.columns
    })
    present_emg_cols = [c for c in EMG_CHANNEL_COLS if c in emg_df.columns]

    # -- sEMG filtering (20-95 Hz bandpass + 50 Hz notch at 200 Hz) -------
    emg_channels_raw = emg_df[present_emg_cols].to_numpy(dtype=float)
    emg_filtered_arr = filter_emg_array(
        emg_channels_raw, fs=emg_fs, apply_notch=apply_notch
    )
    emg_df_filt = emg_df.copy()
    for i, col in enumerate(present_emg_cols):
        emg_df_filt[col + "_filt"] = emg_filtered_arr[:, i]
    filt_cols = [c + "_filt" for c in present_emg_cols]

    # -- Window index ------------------------------------------------------
    imu_times = imu_df["timestamp_ms"].to_numpy()
    windows   = build_window_index(imu_times, window_ms=window_ms, step_ms=step_ms)

    rows = []
    for win in windows:
        s_ms, e_ms = win["start_ms"], win["end_ms"]

        # -- IMU window ----------------------------------------------------
        imu_win = slice_by_time(imu_df, s_ms, e_ms)
        if len(imu_win) < imu_window_min:
            continue

        # -- EMG window ----------------------------------------------------
        emg_win = slice_by_time(emg_df_filt, s_ms, e_ms)
        has_emg = len(emg_win) >= emg_window_min

        # -- Protocol label: time-segment lookup from labels.csv ----------
        # Zero circularity - no features are read during assignment.
        proto = apply_protocol_labels(s_ms, e_ms, labels_df)

        # -- Archetype label: from imu_data.csv columns (synthetic only) --
        # For real data these will all be UNKNOWN/-1; for synthetic data
        # the generator writes per-row labels into imu_data.csv directly.
        if "risk_class" in imu_win.columns:
            rc_counts       = imu_win["risk_class"].value_counts()
            risk_class_arch = int(rc_counts.idxmax()) if len(rc_counts) else -1
        else:
            risk_class_arch = -1

        # Movement label: prefer protocol segment label, fall back to
        # per-row imu_data label (synthetic), then UNKNOWN.
        if proto["label"] != "UNKNOWN":
            movement_label = proto["label"]
        elif "label" in imu_win.columns:
            lbl_counts     = imu_win["label"].value_counts()
            movement_label = str(lbl_counts.idxmax()) if len(lbl_counts) else "UNKNOWN"
        else:
            movement_label = "UNKNOWN"

        # -- Feature extraction --------------------------------------------
        row = {
            "window_centre_ms":    win["centre_ms"],
            "movement_label":      movement_label,
            "risk_class_archetype": risk_class_arch,  # generator-assigned (synthetic only)
        }

        row.update(extract_imu_window_features(imu_win))

        # -- Signal-derived labelling (spec 6.3) - always computed --------
        # Written to risk_class_signal. Has label-feature circularity on
        # synthetic data (IMU features used both as criteria and as inputs).
        # Use only for synthetic evaluation; not for real participant data.
        sig_label = label_window_from_signal(row)
        row["risk_class_signal"] = sig_label["risk_class"]
        row["risk_criteria"]     = sig_label["risk_criteria"]
        row["risk_clinical"]     = sig_label["risk_clinical"]
        row["risk_layman"]       = sig_label["risk_layman"]

        # -- Protocol labelling - always computed --------------------------
        # Written to risk_class_protocol. Zero circularity: derived entirely
        # from researcher-defined time segments in labels.csv.
        row["risk_class_protocol"] = proto["risk_class"]

        # -- Active risk_class: controlled by label_source -----------------
        if label_source == "protocol":
            row["risk_class"] = proto["risk_class"]
        else:
            # "signal" (default) - backwards compatible with synthetic pipeline
            row["risk_class"] = sig_label["risk_class"]

        if has_emg:
            emg_arr = emg_win[filt_cols].to_numpy(dtype=float)
            row.update(extract_window_features(
                emg_arr,
                fs=emg_fs,
                channel_names=present_emg_cols,
                bilateral_pairs=DEFAULT_BILATERAL_PAIRS,
            ))
        else:
            row.update(_emg_nan_row(present_emg_cols))

        rows.append(row)

    feature_df = pd.DataFrame(rows)

    # -- sEMG amplitude normalisation (resting-baseline ratio) -------------
    # Default "none" preserves the original raw-mV amplitude behaviour and the
    # frozen IMU-only/fallback lineage. When enabled, emg_rms_*/emg_mav_* are
    # divided by their per-session resting-baseline mean (see
    # normalize_emg_amplitude_to_baseline). Applied before z-scoring; z-scores
    # are invariant to this per-session constant rescale.
    if emg_amplitude_norm not in EMG_AMPLITUDE_NORM_METHODS:
        raise ValueError(
            f"emg_amplitude_norm must be one of {EMG_AMPLITUDE_NORM_METHODS}; "
            f"got {emg_amplitude_norm!r}"
        )
    if emg_amplitude_norm == "resting_baseline_ratio":
        feature_df, _emg_norm_report = normalize_emg_amplitude_to_baseline(feature_df)
        _missing = [c for c, r in _emg_norm_report.items() if not np.isfinite(r)]
        if _missing:
            print(f"  [emg-norm] insufficient/degenerate baseline; left raw: {_missing}")
        else:
            print(f"  [emg-norm] resting-baseline ratio applied to "
                  f"{len(_emg_norm_report)} amplitude features")

    # -- Baseline z-scores (spec 5) ----------------------------------------
    baseline_stats = compute_baseline_stats(feature_df)
    feature_df     = add_z_scores(feature_df, baseline_stats)

    # -- Save --------------------------------------------------------------
    out_path = Path(output_dir) / "feature_matrix.csv"
    feature_df.to_csv(out_path, index=False)

    return feature_df


# --- Batch runner ------------------------------------------------------------

def _discover_session_dirs(data_dir: Path) -> list[Path]:
    """
    Find complete session directories under a dataset root.

    Phase I synthetic data usually stores sessions directly as
    ``data/synthetic/session_0001``. Phase II real data is expected to use a
    participant/session hierarchy such as
    ``data/real/protocol_train/participant_01/session_001``.
    """
    session_dirs = [
        d for d in data_dir.rglob("*")
        if d.is_dir()
        and (d / "imu_data.csv").exists()
        and (d / "emg_data.csv").exists()
        and (d / "labels.csv").exists()
    ]
    return sorted(session_dirs, key=lambda p: str(p.relative_to(data_dir)).lower())


def _session_identity(data_dir: Path, session_dir: Path) -> tuple[str, str]:
    """
    Return a stable session_id and participant_id for combined feature tables.

    ``session_id`` is the relative path with path separators normalised so that
    repeated names such as ``session_001`` remain unique across participants.
    ``participant_id`` is inferred from the parent folder in Phase II layouts.
    """
    rel = session_dir.relative_to(data_dir)
    session_id = "__".join(rel.parts)
    participant_id = rel.parts[-2] if len(rel.parts) >= 2 else session_dir.name
    return session_id, participant_id


def _infer_phase(data_dir: Path) -> str:
    """Infer a manifest phase from the standard dataset directory layout."""
    normalised = str(data_dir).replace("\\", "/").lower()
    if "varied_test" in normalised:
        return "Phase II.C"
    if "protocol_train" in normalised:
        return "Phase II.A"
    return "Phase I"


def run_pipeline_batch(
    data_dir: str,
    output_dir: Optional[str] = None,
    label_source: str = "protocol",
    phase: Optional[str] = None,
    operating_mode: str = "full_hybrid",
    force: bool = False,
    command_used: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Run the pipeline on all session directories under data_dir.
    Combines all session feature matrices into combined_features.csv.

    Parameters
    ----------
    data_dir     : root directory containing complete session directories.
                   Direct ``session_*`` folders and nested
                   ``participant_*/session_*`` folders are both supported.
    output_dir   : where to save combined_features.csv (defaults to data_dir)
    label_source : ``"protocol"`` (default for Phase I/II task labels) or
                   ``"signal"`` (diagnostic threshold labels). Passed to
                   run_pipeline().
                   See run_pipeline() docstring for full explanation.
    phase        : study phase recorded in dataset_manifest.json.
    operating_mode : declared system mode recorded in dataset_manifest.json.
    force        : allow replacement of existing official Phase II outputs.
    command_used : command recorded in dataset_manifest.json.
    **kwargs     : additional keyword args passed to run_pipeline
                   (e.g. emg_amplitude_norm)

    Returns
    -------
    combined_df : DataFrame with all windows, session_id and participant_id
                  prepended
    """
    data_dir = Path(data_dir)
    if output_dir is None:
        output_dir = data_dir
    output_dir = Path(output_dir)
    phase = phase or _infer_phase(data_dir)
    official_phase2 = phase in {"Phase II.A", "Phase II.B", "Phase II.C", "Phase II.1", "Phase II.2"}

    session_dirs = _discover_session_dirs(data_dir)

    if not session_dirs:
        raise FileNotFoundError(f"No complete session directories found in {data_dir}")

    protected_outputs = [output_dir / "combined_features.csv", output_dir / MANIFEST_FILENAME]
    protected_outputs.extend(sd / "feature_matrix.csv" for sd in session_dirs)
    existing_outputs = [path for path in protected_outputs if path.exists()]
    if official_phase2 and existing_outputs and not force:
        listed = "\n  - ".join(str(path) for path in existing_outputs)
        raise FileExistsError(
            "Official Phase II feature outputs already exist. Refusing to overwrite "
            f"without --force:\n  - {listed}"
        )

    all_dfs = []
    for sd in session_dirs:
        session_id, participant_id = _session_identity(data_dir, sd)
        print(f"  Processing {session_id}...")
        df = run_pipeline(sd, output_dir=sd, label_source=label_source, **kwargs)
        df.insert(0, "participant_id", participant_id)
        df.insert(0, "session_id", session_id)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = output_dir / "combined_features.csv"
    combined.to_csv(out_path, index=False)
    manifest_path = write_dataset_manifest(
        source_dir=data_dir,
        session_dirs=session_dirs,
        label_source=label_source,
        phase=phase,
        operating_mode=operating_mode,
        feature_file=out_path,
        command_used=command_used or " ".join(sys.argv),
    )
    print(f"  Manifest: {manifest_path}")
    print(f"\nCombined feature matrix saved -> {out_path}")
    print(f"  Shape   : {combined.shape}")
    print(f"  Classes : {combined['risk_class'].value_counts().to_dict()}")
    return combined


# --- CLI entry point ---------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the lumbar movement risk signal processing pipeline."
    )
    parser.add_argument("--data_dir",  default=None,
                        help="Root data directory containing session folders "
                             "(direct session_* or nested participant_*/session_*). "
                             "Mutually exclusive with --session_dir.")
    parser.add_argument("--session_dir", default=None,
                        help="Single session directory to process (e.g. "
                             "data/real/protocol_train/participant_01/session_001). Mutually exclusive "
                             "with --data_dir.")
    parser.add_argument("--output_dir", default=None,
                        help="Output dir for feature_matrix / combined_features.csv "
                             "(default: session_dir or data_dir)")
    parser.add_argument("--no_notch", action="store_true",
                        help="Skip 50 Hz notch filter")
    parser.add_argument(
        "--label_source", default="protocol", choices=["signal", "protocol"],
        help=(
            "How to assign the primary risk_class label for each window.\n"
            "  protocol (default) - derive from task/time segments in labels.csv.\n"
            "            Use for Phase I synthetic validation and all real data.\n"
            "  signal   - derive from IMU features using biomechanical\n"
            "            thresholds (label_window_from_signal). Fast and automatic\n"
            "            but circular if used as a model target. Use only as a\n"
            "            diagnostic/explanation channel."
        ),
    )
    parser.add_argument(
        "--phase",
        default=None,
        choices=["Phase I", "Phase II.A", "Phase II.B", "Phase II.C", "Phase II.1", "Phase II.2"],
        help="Declared study phase written to dataset_manifest.json in batch mode.",
    )
    parser.add_argument(
        "--mode",
        dest="operating_mode",
        default="full_hybrid",
        choices=["full_hybrid", "imu_only_fallback"],
        help="Declared system mode written to dataset_manifest.json in batch mode.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacement of existing official Phase II derived feature outputs.",
    )
    args = parser.parse_args()

    if args.session_dir and args.data_dir:
        parser.error("--session_dir and --data_dir are mutually exclusive.")

    if args.session_dir:
        # Single-session mode
        feat_df = run_pipeline(
            session_dir  = args.session_dir,
            output_dir   = args.output_dir or args.session_dir,
            apply_notch  = not args.no_notch,
            label_source = args.label_source,
        )
        print(f"\nfeature_matrix.csv saved -> {args.output_dir or args.session_dir}")
        print(f"  Shape   : {feat_df.shape}")
        print(f"  Classes : {feat_df['risk_class'].value_counts().to_dict()}")
    else:
        data_dir = args.data_dir or "data/real/protocol_train"
        run_pipeline_batch(
            data_dir=data_dir,
            output_dir=args.output_dir or data_dir,
            apply_notch=not args.no_notch,
            label_source=args.label_source,
            phase=args.phase,
            operating_mode=args.operating_mode,
            force=args.force,
            command_used=" ".join(sys.argv),
        )
