"""
SAL: Smoothness of Angular Loading
====================================
Implements the log-dimensionless jerk (LDLJ) metric for trunk movement
smoothness, adapted from Balasubramanian et al. (2012, 2015) for angular
kinematics.

Definition
----------
    LDLJ = -log( T^5 / (2 * v_peak^2) * integral(j^2 dt) )

    where j = d²v/dt³ = third derivative of angular position = JERK.

Sign convention (important):
    Smooth (min-jerk) movement  → small jerk integral → LDLJ closer to 0
    Jerky / fast movement       → large jerk integral → LDLJ << 0 (more negative)

    So: MORE NEGATIVE = jerkier.  LESS NEGATIVE = smoother.

Windowed vs. full-movement use
--------------------------------
    LDLJ was designed for complete discrete movements.  When applied to
    short sliding windows (e.g. 250 ms), the metric captures instantaneous
    jerk content but loses its dimensionless reference interpretation because
    the window may sample only a phase of a movement (e.g. constant-velocity
    region).  For windowed classification, prefer jerk_rms() and jerk_peak().
    For offline evaluation on complete labelled repetitions, use
    sal_from_segment_boundaries().

Reference value
---------------
    For a minimum-jerk trajectory the dimensionless value depends on movement
    duration (the formula is not strictly scale-invariant with respect to T
    when computed on the real-valued velocity profile).  As a practical guide:
    values around -5 to -7 correspond to clean minimum-jerk-like movements at
    typical trunk movement speeds (3-4 s duration, 20-50°/s peak).  Values
    substantially more negative indicate jerky or rapid loading.

    LDLJ_MIN_JERK_REFERENCE = -np.log(30.0) ≈ -3.40 is kept for API
    compatibility but should NOT be used as an absolute threshold in this
    implementation.

References:
    Balasubramanian et al. (2012) "Is smoothness in movement the result of
    optimal motor planning?" J Neurophysiology.

    Balasubramanian et al. (2015) "On the analysis of movement smoothness."
    J NeuroEngineering and Rehabilitation.

    Flash & Hogan (1985) "The coordination of arm movements."
    J Neuroscience, 5(7), 1688-1703.
"""

import numpy as np
from typing import Union


# Constants

# Theoretical LDLJ for a minimum-jerk trajectory (dimensionless jerk = 30).
# Clean, healthy movements should approach this value.
LDLJ_MIN_JERK_REFERENCE = -np.log(30.0)   # ≈ -3.40

# Angular velocity threshold below which a window is considered quasi-static.
# SAL is undefined for static postures.
STATIC_THRESHOLD_DEG_S = 2.0


# Core computation

def compute_jerk(angular_velocity: np.ndarray, fs: float) -> np.ndarray:
    """
    Estimate jerk (deg/s³) from angular velocity (deg/s) using two
    successive central finite differences (np.gradient applied twice).

    Jerk = d³(position)/dt³ = d²(velocity)/dt²

    A single np.gradient on velocity gives acceleration (deg/s²).
    A second np.gradient on acceleration gives jerk (deg/s³).

    Parameters
    ----------
    angular_velocity : (N,) array in deg/s
    fs               : sampling frequency in Hz

    Returns
    -------
    jerk : (N,) array in deg/s³
    """
    dt = 1.0 / fs
    accel = np.gradient(angular_velocity, dt)   # deg/s²
    jerk  = np.gradient(accel, dt)              # deg/s³
    return jerk


def compute_acceleration(angular_velocity: np.ndarray, fs: float) -> np.ndarray:
    """
    Estimate angular acceleration (deg/s²) from angular velocity.
    Single first-order finite difference.  This is NOT jerk.
    """
    return np.gradient(angular_velocity, 1.0 / fs)


def jerk_rms(angular_velocity: np.ndarray, fs: float) -> float:
    """
    Root-mean-square jerk of a velocity window.

    Appropriate for windowed (sliding-window) feature extraction where
    the window may capture only a phase of a complete movement.  Unlike
    LDLJ, jerk RMS does not require the window to span a full discrete
    movement.

    Parameters
    ----------
    angular_velocity : (N,) array in deg/s
    fs               : sampling frequency in Hz

    Returns
    -------
    rms_jerk : float in deg/s³
    """
    jk = compute_jerk(angular_velocity, fs)
    return float(np.sqrt(np.mean(jk ** 2)))


def jerk_peak(angular_velocity: np.ndarray, fs: float) -> float:
    """
    Peak absolute jerk of a velocity window (deg/s³).

    Captures brief high-jerk loading events that LDLJ may average away.
    """
    jk = compute_jerk(angular_velocity, fs)
    return float(np.max(np.abs(jk)))


def ldlj(angular_velocity: np.ndarray, fs: float) -> float:
    """
    Log-dimensionless jerk for a single movement segment or window.

    Parameters
    ----------
    angular_velocity : (N,) array in deg/s
    fs               : sampling frequency in Hz

    Returns
    -------
    score : float  (NaN if window is quasi-static)
    """
    v_peak = np.max(np.abs(angular_velocity))
    if v_peak < STATIC_THRESHOLD_DEG_S:
        return np.nan

    T  = len(angular_velocity) / fs          # duration in seconds
    jk = compute_jerk(angular_velocity, fs)
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    jerk_integral = trapezoid(jk ** 2) / fs   # integral of squared jerk

    dimensionless = (T ** 5) / (2.0 * v_peak ** 2) * jerk_integral
    if dimensionless <= 0:
        return np.nan

    return -np.log(dimensionless)


def ldlj_multi_axis(
    omega_pitch: np.ndarray,
    omega_roll:  np.ndarray,
    omega_yaw:   np.ndarray,
    fs: float,
) -> float:
    """
    LDLJ computed on the resultant angular velocity magnitude.
    Captures combined multi-planar movement smoothness rather than
    individual axes, which is more appropriate for asymmetric tasks
    (e.g. PICKUP_ASYM).

    Parameters
    ----------
    omega_pitch, omega_roll, omega_yaw : (N,) arrays in deg/s
    fs                                 : sampling frequency in Hz
    """
    magnitude = np.sqrt(omega_pitch**2 + omega_roll**2 + omega_yaw**2)
    return ldlj(magnitude, fs)


# Windowed computation

def windowed_ldlj(
    angular_velocity: np.ndarray,
    fs: float,
    window_samples: int,
    step_samples:   int,
) -> np.ndarray:
    """
    Compute LDLJ for each sliding window.

    Parameters
    ----------
    angular_velocity : (N,) array in deg/s
    fs               : sampling frequency
    window_samples   : number of samples per window (e.g. 25 at 100 Hz = 250 ms)
    step_samples     : stride between windows (e.g. 13 for ~50% overlap)

    Returns
    -------
    scores : (M,) array of LDLJ values, one per window
    """
    n = len(angular_velocity)
    scores = []
    starts = range(0, n - window_samples + 1, step_samples)
    for s in starts:
        w = angular_velocity[s : s + window_samples]
        scores.append(ldlj(w, fs))
    return np.array(scores)


def sal_from_segment_boundaries(
    imu_df,
    labels_df,
    fs: float = 100.0,
    velocity_col: str = "angvel_L3_sagittal",
) -> dict:
    """
    Compute LDLJ for each labelled movement repetition (not windowed).
    Useful for offline feature analysis and report figures.

    Parameters
    ----------
    imu_df    : DataFrame with timestamp_ms, label, rep, angvel_L3_sagittal
    labels_df : DataFrame with label, rep, start_ms, end_ms, risk_class
    fs        : IMU sampling frequency
    velocity_col : column name for angular velocity

    Returns
    -------
    dict mapping (label, rep) → LDLJ score
    """
    results = {}
    for _, row in labels_df.iterrows():
        mask = (
            (imu_df["timestamp_ms"] >= row["start_ms"]) &
            (imu_df["timestamp_ms"] <  row["end_ms"])
        )
        segment = imu_df.loc[mask, velocity_col].values
        if len(segment) < 5:
            continue
        score = ldlj(segment, fs)
        results[(row["label"], int(row["rep"]))] = score
    return results
