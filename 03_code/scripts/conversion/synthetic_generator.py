#!/usr/bin/env python3
"""
Synthetic IMU + sEMG Data Generator
Spinal Movement Risk Monitor, FYP 2025/26

Generates labelled synthetic datasets for all movement classes
defined in the Movement Protocol (v2.0, April 2026).

Each simulated session models one participant running the full protocol:
  Phase 0  : Baseline (static + self-paced + MVC)
  Section 5: 9 movement classes (clean, compensation, functional, fatigue)

Output per session, data/synthetic/session_XXXX/
  imu_data.csv      100 Hz | 4 × quaternion + derived joint angles + angular velocity
  emg_data.csv      200 Hz | 4 × sEMG channels (L-ES, R-ES, L-OBL, R-OBL) in mV
  labels.csv               | segment boundaries with risk class labels

Usage:
  python scripts/conversion/synthetic_generator.py --n_sessions 5 --seed 42

Dependencies: numpy, pandas, scipy
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from pathlib import Path
import argparse
import json

# GLOBAL CONFIGURATION

CONFIG = {
    'imu_fs':            100,    # Hz  — IMU sampling rate
    'emg_fs':            200,    # Hz  — sEMG sampling rate (Ganglion 200 Hz limit)
    'imu_noise_std':     0.5,    # deg — IMU white noise (typical MEMS gyro RMS)
    'imu_drift_rate':    0.02,   # deg/s — gyro integration drift
    'emg_base_noise':    0.005,  # mV  — baseline EMG electrode/amplifier noise
    'emg_motion_art':    0.03,   # mV  — peak motion artefact amplitude
    'inter_rep_jitter':  0.10,   # s   — timing variability between reps
    'participant_var':   0.15,   # fraction — inter-participant ROM variability (±15%)
}

# QUATERNION MATH

def euler_to_quat(pitch_deg, roll_deg, yaw_deg):
    """
    Convert Euler angles → quaternion [w, x, y, z].
    Convention: ZYX extrinsic (yaw applied first, then pitch, then roll).
    Matches typical IMU AHRS firmware output convention.

    pitch (+) = forward trunk flexion  (sagittal)
    roll  (+) = right lateral lean     (coronal)
    yaw   (+) = right axial rotation   (transverse)
    """
    p = np.radians(pitch_deg) / 2
    r = np.radians(roll_deg)  / 2
    y = np.radians(yaw_deg)   / 2

    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    cy, sy = np.cos(y), np.sin(y)

    w  =  cy*cp*cr + sy*sp*sr
    x  =  cy*cp*sr - sy*sp*cr
    y_ =  sy*cp*sr + cy*sp*cr
    z  =  sy*cp*cr - cy*sp*sr

    return np.array([w, x, y_, z])


def quat_mul(q1, q2):
    """Hamilton product of two quaternion arrays [..., 4] = [w, x, y, z]."""
    w1, x1, y1, z1 = q1[...,0], q1[...,1], q1[...,2], q1[...,3]
    w2, x2, y2, z2 = q2[...,0], q2[...,1], q2[...,2], q2[...,3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def quat_inv(q):
    """Quaternion inverse (= conjugate for unit quaternions)."""
    inv = q.copy()
    inv[..., 1:] *= -1
    return inv


def relative_quat(q_distal, q_proximal):
    """Relative rotation: q_rel = q_distal ⊗ q_proximal⁻¹"""
    return quat_mul(q_distal, quat_inv(q_proximal))


def quat_to_euler(q):
    """
    Quaternion [w, x, y, z] → Euler angles [pitch, roll, yaw] in degrees.
    ZYX convention. Returns array of shape (..., 3).
    """
    w, x, y, z = q[...,0], q[...,1], q[...,2], q[...,3]

    sinp = np.clip(2*(w*y - z*x), -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sinp))

    sinr_cosp = 2*(w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = np.degrees(np.arctan2(sinr_cosp, cosr_cosp))

    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))

    return np.stack([pitch, roll, yaw], axis=-1)


def normalise_quat(q):
    """Normalise quaternion array to unit length along last axis."""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(norm, 1e-10, None)

# TRAJECTORY PROFILES

def min_jerk_traj(t_arr, T, D):
    """
    Minimum-jerk position trajectory (Flash & Hogan, 1985).
    Moves from 0 → D in time T. Minimises the integral of squared jerk.
    Used for clean, controlled voluntary movements.
    """
    tau = np.clip(t_arr / T, 0.0, 1.0)
    return D * (10*tau**3 - 15*tau**4 + 6*tau**5)


def trapezoidal_traj(t_arr, T, D, accel_frac=0.20):
    """
    Trapezoidal velocity profile, higher peak velocity, more jerk.
    Used for fast / fatigued movements.
    accel_frac: fraction of T spent accelerating (and decelerating).
    """
    t_a = T * accel_frac           # acceleration phase ends
    t_d = T * (1.0 - accel_frac)  # deceleration phase starts
    v_peak = D / (T - t_a)

    out = np.zeros_like(t_arr, dtype=float)
    for i, t in enumerate(t_arr):
        if t <= 0:
            out[i] = 0
        elif t <= t_a:
            out[i] = 0.5 * v_peak * t**2 / t_a
        elif t <= t_d:
            out[i] = 0.5 * v_peak * t_a + v_peak * (t - t_a)
        elif t <= T:
            dt = t - t_d
            out[i] = D - 0.5 * v_peak * (T - t)**2 / t_a
        else:
            out[i] = D
    return np.clip(out, 0, D)


def generate_rep_angles(profile_angles, rep_dur_s, hold_dur_s,
                         fs, traj_type='min_jerk', rng=None):
    """
    Generate one repetition of Euler-angle time series for all 4 IMUs.

    Returns:
        angle_series : ndarray (N, 4, 3)  [sample, imu, axis] in degrees
                       axes = [pitch, roll, yaw]
        t_arr        : ndarray (N,)  time in seconds from rep start
    """
    if rng is None:
        rng = np.random.default_rng()

    total_dur = rep_dur_s + hold_dur_s + rep_dur_s
    N = int(round(total_dur * fs))
    t = np.arange(N) / fs

    result = np.zeros((N, 4, 3))
    imu_keys = ['pelvis', 'l3', 't12', 't4']

    for imu_idx, key in enumerate(imu_keys):
        peaks = profile_angles[key]   # [pitch, roll, yaw]

        for ax_idx, peak in enumerate(peaks):
            if abs(peak) < 0.01:
                continue

            # Descent phase 0 → peak
            t_desc = np.clip(t, 0, rep_dur_s)
            if traj_type == 'min_jerk':
                desc = min_jerk_traj(t_desc, rep_dur_s, peak)
            else:
                desc = trapezoidal_traj(t_desc, rep_dur_s, peak)

            # Ascent phase peak → 0
            t_asc = np.clip(t - rep_dur_s - hold_dur_s, 0, rep_dur_s)
            if traj_type == 'min_jerk':
                asc = peak - min_jerk_traj(t_asc, rep_dur_s, peak)
            else:
                asc = peak - trapezoidal_traj(t_asc, rep_dur_s, peak)

            # Combine segments
            in_hold = (t >= rep_dur_s) & (t < rep_dur_s + hold_dur_s)
            in_asc  = t >= (rep_dur_s + hold_dur_s)

            angle = desc.copy()
            angle[in_hold] = peak
            angle[in_asc]  = asc[in_asc]

            # Small within-rep smoothness noise (larger for trapezoidal)
            noise_scale = abs(peak) * (0.03 if traj_type == 'trapezoidal' else 0.01)
            angle += rng.normal(0, noise_scale, N)

            result[:, imu_idx, ax_idx] = angle

    return result, t


def angles_to_absolute_quats(angle_series):
    """
    Convert (N, 4, 3) Euler angle series → (N, 4, 4) absolute quaternions.
    Each sample, each IMU gets its own unit quaternion.
    """
    N, n_imus, _ = angle_series.shape
    quats = np.zeros((N, n_imus, 4))
    for i in range(N):
        for imu in range(n_imus):
            p, r, y = angle_series[i, imu]
            quats[i, imu] = euler_to_quat(p, r, y)
    return quats

# EMG GENERATION

def _bandpass(signal, lowcut, highcut, fs, order=4):
    nyq = fs / 2.0
    lo  = max(lowcut,  1.0) / nyq
    hi  = min(highcut, nyq - 1.0) / nyq
    if lo >= hi:
        hi = lo + 0.05
    b, a = butter(order, [lo, hi], btype='band')
    pad  = min(len(signal) - 1, 3 * order)
    return filtfilt(b, a, signal, padlen=pad)


def generate_emg_channel(envelope, fs, mvc_fraction=0.30,
                          center_freq=100.0, bandwidth_hz=200.0, rng=None):
    """
    Realistic sEMG from an amplitude envelope using amplitude-modulated
    band-limited white noise (standard simulation method).

    envelope     : (N,) normalised activation 0-1
    mvc_fraction : peak activation as fraction of MVC
    center_freq  : spectral centroid in Hz (decreases with fatigue)
    bandwidth_hz : spectral bandwidth in Hz

    MVC = 2 mV RMS (typical surface EMG RMS at full activation).
    """
    if rng is None:
        rng = np.random.default_rng()

    N    = len(envelope)
    lo   = max(20.0, center_freq - bandwidth_hz / 2)
    hi   = min(fs / 2 - 10.0, center_freq + bandwidth_hz / 2)

    noise = rng.normal(0, 1, N)
    filt  = _bandpass(noise, lo, hi, fs)

    rms_filt = np.sqrt(np.mean(filt**2)) or 1.0
    filt     = filt / rms_filt                       # unit-RMS noise

    amplitude_mv = mvc_fraction * 2.0               # 2 mV = 100% MVC
    return envelope * filt * amplitude_mv


def get_activation_envelope(n_samples, rep_dur_s, hold_dur_s, fs):
    """
    Half-cosine activation envelope for one rep.
    Rises during descent, plateaus during hold, falls during ascent.
    """
    n_desc = int(rep_dur_s  * fs)
    n_hold = int(hold_dur_s * fs)
    n_asc  = int(rep_dur_s  * fs)
    total  = n_desc + n_hold + n_asc

    t_d = np.linspace(0,    np.pi, n_desc)
    t_a = np.linspace(np.pi, 0,    n_asc)
    env = np.concatenate([
        (1 - np.cos(t_d)) / 2,
        np.ones(n_hold),
        (1 - np.cos(t_a)) / 2,
    ])
    return env[:total]


def add_motion_artefact(N, fs, rng):
    """Low-frequency motion artefact burst at movement onset."""
    art = np.zeros(N)
    onset = int(0.05 * N)
    dur   = int(0.08 * fs)
    if onset + dur < N:
        t_art = np.arange(dur) / fs
        burst = CONFIG['emg_motion_art'] * np.sin(2*np.pi*5*t_art) * np.exp(-t_art*25)
        art[onset:onset+dur] += burst
    return art

# IMU NOISE MODEL

def add_imu_noise(angle_series, rng):
    """
    Add realistic IMU noise to (N, 4, 3) angle array:
    - White noise  : MEMS accelerometer + gyro thermal noise
    - Random walk  : gyro integration drift
    """
    N, n_imus, n_ax = angle_series.shape
    noisy = angle_series.copy()

    for imu in range(n_imus):
        for ax in range(n_ax):
            white = rng.normal(0, CONFIG['imu_noise_std'], N)
            drift = np.cumsum(rng.normal(0, CONFIG['imu_drift_rate'] / np.sqrt(N), N))
            noisy[:, imu, ax] += white + drift

    return noisy

# MOVEMENT PROFILES
# Angle peaks  : [pitch_deg, roll_deg, yaw_deg] per IMU (absolute orientation)
# ABSOLUTE convention: each IMU reports its angle from the global vertical.
# Relative joint angles (theta_PL, theta_LT, theta_TU) computed post-hoc.
# Profile sources:
# Clean flexion pelvis-lumbar coupling ratio ~0.65  → pelvis ≈ 35°, L3 ≈ 25°  [McGill 2007]
# Lumbar dominant: θ_PL > 35°, pelvis < 10°                                    [Adams & Hutton 1983]
# Shoulder driven: θ_TU >> θ_PL, T4 dominant                                   [McGill 2007]
# Fast bend angular velocity >40°/s at L3                                       [Marras et al. 1993]
# Lateral bend 20-30° trunk tilt                                                [Marras et al. 1993]
# Rotation T4 yaw dominant, attenuated caudally                                 [Granata & Marras 1993]

PROFILES = {

    'BASELINE_STATIC': {
        'reps': 1, 'rep_duration_s': 30.0, 'hold_duration_s': 0.0,
        'rest_s': 0.0, 'traj_type': 'min_jerk', 'risk_class': 0,
        'angles': {
            'pelvis': [ 0.0,  0.0,  0.0],
            'l3':     [ 0.0,  0.0,  0.0],
            't12':    [ 0.0,  0.0,  0.0],
            't4':     [ 0.0,  0.0,  0.0],
        },
        'emg': {
            'LES':  {'mvc_fraction': 0.04, 'center_freq':  90, 'bandwidth_hz': 150},
            'RES':  {'mvc_fraction': 0.04, 'center_freq':  90, 'bandwidth_hz': 150},
            'LOBL': {'mvc_fraction': 0.02, 'center_freq':  80, 'bandwidth_hz': 120},
            'ROBL': {'mvc_fraction': 0.02, 'center_freq':  80, 'bandwidth_hz': 120},
        },
    },

    'CLEAN_FLEXION': {
        'reps': 8, 'rep_duration_s': 3.0, 'hold_duration_s': 1.0,
        'rest_s': 1.0, 'traj_type': 'min_jerk', 'risk_class': 0,
        'angles': {
            'pelvis': [35.0,  0.0,  0.0],   # dominant pelvic tilt (hip-dominant)
            'l3':     [25.0,  0.0,  0.0],   # moderate lumbar (θ_PL = L3-pelvis ≈ -10°)
            't12':    [30.0,  0.0,  0.0],
            't4':     [33.0,  0.0,  0.0],
        },
        'emg': {
            'LES':  {'mvc_fraction': 0.30, 'center_freq': 100, 'bandwidth_hz': 220},
            'RES':  {'mvc_fraction': 0.28, 'center_freq': 100, 'bandwidth_hz': 220},
            'LOBL': {'mvc_fraction': 0.08, 'center_freq':  85, 'bandwidth_hz': 155},
            'ROBL': {'mvc_fraction': 0.08, 'center_freq':  85, 'bandwidth_hz': 155},
        },
    },

    'LUMBAR_DOMINANT': {
        'reps': 6, 'rep_duration_s': 3.0, 'hold_duration_s': 1.0,
        'rest_s': 1.5, 'traj_type': 'min_jerk', 'risk_class': 1,
        'angles': {
            'pelvis': [ 8.0,  0.0,  0.0],   # minimal pelvic tilt
            'l3':     [54.0,  0.0,  0.0],   # θ_PL ≈ 46° (l3 − pelvis)
            't12':    [62.0,  0.0,  0.0],   # θ_LT ≈ 8°; total trunk flex ≈ 54°
            't4':     [66.0,  0.0,  0.0],   # at ±15% ROM scale: min ≈ 45.9°
        },
        'emg': {
            'LES':  {'mvc_fraction': 0.48, 'center_freq': 105, 'bandwidth_hz': 220},
            'RES':  {'mvc_fraction': 0.44, 'center_freq': 105, 'bandwidth_hz': 220},
            'LOBL': {'mvc_fraction': 0.10, 'center_freq':  88, 'bandwidth_hz': 155},
            'ROBL': {'mvc_fraction': 0.10, 'center_freq':  88, 'bandwidth_hz': 155},
        },
    },

    'FAST_BEND': {
        'reps': 6, 'rep_duration_s': 0.7, 'hold_duration_s': 0.2,
        'rest_s': 1.8, 'traj_type': 'trapezoidal', 'risk_class': 1,
        'angles': {
            'pelvis': [32.0,  0.0,  0.0],
            'l3':     [24.0,  0.0,  0.0],
            't12':    [28.0,  0.0,  0.0],
            't4':     [30.0,  0.0,  0.0],
        },
        'emg': {
            'LES':  {'mvc_fraction': 0.58, 'center_freq': 112, 'bandwidth_hz': 230},
            'RES':  {'mvc_fraction': 0.55, 'center_freq': 112, 'bandwidth_hz': 230},
            'LOBL': {'mvc_fraction': 0.14, 'center_freq':  92, 'bandwidth_hz': 160},
            'ROBL': {'mvc_fraction': 0.14, 'center_freq':  92, 'bandwidth_hz': 160},
        },
    },

    'SHOULDER_DRIVEN': {
        'reps': 5, 'rep_duration_s': 4.0, 'hold_duration_s': 1.0,
        'rest_s': 1.5, 'traj_type': 'min_jerk', 'risk_class': 1,
        'angles': {
            'pelvis': [ 5.0,  0.0,  0.0],   # almost no pelvic movement
            'l3':     [10.0,  0.0,  0.0],   # small lumbar contribution
            't12':    [22.0,  0.0,  0.0],   # large thoracolumbar
            't4':     [43.0,  0.0,  0.0],   # dominant upper thoracic (θ_TU ≈ 21°)
        },
        'emg': {
            'LES':  {'mvc_fraction': 0.15, 'center_freq':  90, 'bandwidth_hz': 180},
            'RES':  {'mvc_fraction': 0.15, 'center_freq':  90, 'bandwidth_hz': 180},
            'LOBL': {'mvc_fraction': 0.06, 'center_freq':  80, 'bandwidth_hz': 135},
            'ROBL': {'mvc_fraction': 0.06, 'center_freq':  80, 'bandwidth_hz': 135},
        },
    },

    'CLEAN_LATERAL_L': {
        'reps': 6, 'rep_duration_s': 2.0, 'hold_duration_s': 0.5,
        'rest_s': 1.0, 'traj_type': 'min_jerk', 'risk_class': 0,
        'angles': {
            'pelvis': [0.0,  4.0,  0.0],
            'l3':     [0.0, 18.0,  0.0],
            't12':    [0.0, 24.0,  0.0],
            't4':     [0.0, 28.0,  0.0],
        },
        'emg': {
            'LES':  {'mvc_fraction': 0.35, 'center_freq':  96, 'bandwidth_hz': 200},
            'RES':  {'mvc_fraction': 0.12, 'center_freq':  90, 'bandwidth_hz': 180},
            'LOBL': {'mvc_fraction': 0.28, 'center_freq':  90, 'bandwidth_hz': 175},
            'ROBL': {'mvc_fraction': 0.08, 'center_freq':  82, 'bandwidth_hz': 145},
        },
    },

    'CLEAN_LATERAL_R': {
        'reps': 6, 'rep_duration_s': 2.0, 'hold_duration_s': 0.5,
        'rest_s': 1.0, 'traj_type': 'min_jerk', 'risk_class': 0,
        'angles': {
            'pelvis': [0.0,  -4.0,  0.0],
            'l3':     [0.0, -18.0,  0.0],
            't12':    [0.0, -24.0,  0.0],
            't4':     [0.0, -28.0,  0.0],
        },
        'emg': {
            'LES':  {'mvc_fraction': 0.12, 'center_freq':  90, 'bandwidth_hz': 180},
            'RES':  {'mvc_fraction': 0.35, 'center_freq':  96, 'bandwidth_hz': 200},
            'LOBL': {'mvc_fraction': 0.08, 'center_freq':  82, 'bandwidth_hz': 145},
            'ROBL': {'mvc_fraction': 0.28, 'center_freq':  90, 'bandwidth_hz': 175},
        },
    },

    'CLEAN_ROTATION_L': {
        'reps': 6, 'rep_duration_s': 2.0, 'hold_duration_s': 0.5,
        'rest_s': 1.0, 'traj_type': 'min_jerk', 'risk_class': 0,
        'angles': {
            'pelvis': [0.0,  0.0,  4.0],
            'l3':     [0.0,  0.0, 12.0],
            't12':    [0.0,  0.0, 22.0],
            't4':     [0.0,  0.0, 38.0],   # yaw dominant at upper thoracic
        },
        'emg': {
            'LES':  {'mvc_fraction': 0.20, 'center_freq':  92, 'bandwidth_hz': 185},
            'RES':  {'mvc_fraction': 0.28, 'center_freq':  92, 'bandwidth_hz': 185},
            'LOBL': {'mvc_fraction': 0.30, 'center_freq':  88, 'bandwidth_hz': 170},
            'ROBL': {'mvc_fraction': 0.18, 'center_freq':  86, 'bandwidth_hz': 160},
        },
    },

    'CLEAN_ROTATION_R': {
        'reps': 6, 'rep_duration_s': 2.0, 'hold_duration_s': 0.5,
        'rest_s': 1.0, 'traj_type': 'min_jerk', 'risk_class': 0,
        'angles': {
            'pelvis': [0.0,  0.0,  -4.0],
            'l3':     [0.0,  0.0, -12.0],
            't12':    [0.0,  0.0, -22.0],
            't4':     [0.0,  0.0, -38.0],
        },
        'emg': {
            'LES':  {'mvc_fraction': 0.28, 'center_freq':  92, 'bandwidth_hz': 185},
            'RES':  {'mvc_fraction': 0.20, 'center_freq':  92, 'bandwidth_hz': 185},
            'LOBL': {'mvc_fraction': 0.18, 'center_freq':  86, 'bandwidth_hz': 160},
            'ROBL': {'mvc_fraction': 0.30, 'center_freq':  88, 'bandwidth_hz': 170},
        },
    },

    'PICKUP_SYM': {
        'reps': 5, 'rep_duration_s': 2.5, 'hold_duration_s': 0.5,
        'rest_s': 1.5, 'traj_type': 'min_jerk', 'risk_class': 0,
        'angles': {
            'pelvis': [28.0,  2.0,  2.0],   # sagittal dominant, slight asymmetry
            'l3':     [22.0,  2.0,  2.0],
            't12':    [26.0,  3.0,  2.0],
            't4':     [28.0,  3.0,  2.0],
        },
        'emg': {
            'LES':  {'mvc_fraction': 0.32, 'center_freq':  98, 'bandwidth_hz': 205},
            'RES':  {'mvc_fraction': 0.30, 'center_freq':  98, 'bandwidth_hz': 205},
            'LOBL': {'mvc_fraction': 0.10, 'center_freq':  86, 'bandwidth_hz': 158},
            'ROBL': {'mvc_fraction': 0.10, 'center_freq':  86, 'bandwidth_hz': 158},
        },
    },

    'PICKUP_ASYM': {
        'reps': 5, 'rep_duration_s': 2.5, 'hold_duration_s': 0.5,
        'rest_s': 1.5, 'traj_type': 'min_jerk', 'risk_class': 1,
        'angles': {
            'pelvis': [22.0,  8.0, 12.0],   # combined tri-planar loading
            'l3':     [18.0, 14.0, 10.0],
            't12':    [22.0, 18.0, 14.0],
            't4':     [24.0, 20.0, 18.0],
        },
        'emg': {
            'LES':  {'mvc_fraction': 0.44, 'center_freq': 100, 'bandwidth_hz': 210},
            'RES':  {'mvc_fraction': 0.26, 'center_freq':  96, 'bandwidth_hz': 192},
            'LOBL': {'mvc_fraction': 0.22, 'center_freq':  88, 'bandwidth_hz': 165},
            'ROBL': {'mvc_fraction': 0.15, 'center_freq':  84, 'bandwidth_hz': 155},
        },
    },

    'SIT_TO_STAND_NORMAL': {
        'reps': 5, 'rep_duration_s': 2.0, 'hold_duration_s': 0.5,
        'rest_s': 1.0, 'traj_type': 'min_jerk', 'risk_class': 0,
        'angles': {
            'pelvis': [20.0,  0.0,  0.0],
            'l3':     [15.0,  0.0,  0.0],
            't12':    [18.0,  0.0,  0.0],
            't4':     [20.0,  0.0,  0.0],
        },
        'emg': {
            'LES':  {'mvc_fraction': 0.40, 'center_freq': 100, 'bandwidth_hz': 215},
            'RES':  {'mvc_fraction': 0.38, 'center_freq': 100, 'bandwidth_hz': 215},
            'LOBL': {'mvc_fraction': 0.12, 'center_freq':  88, 'bandwidth_hz': 162},
            'ROBL': {'mvc_fraction': 0.12, 'center_freq':  88, 'bandwidth_hz': 162},
        },
    },

    'SIT_TO_STAND_FAST': {
        'reps': 3, 'rep_duration_s': 0.9, 'hold_duration_s': 0.2,
        'rest_s': 1.5, 'traj_type': 'trapezoidal', 'risk_class': -1,  # ambiguous
        'angles': {
            'pelvis': [20.0,  0.0,  0.0],
            'l3':     [15.0,  0.0,  0.0],
            't12':    [18.0,  0.0,  0.0],
            't4':     [20.0,  0.0,  0.0],
        },
        'emg': {
            'LES':  {'mvc_fraction': 0.55, 'center_freq': 108, 'bandwidth_hz': 225},
            'RES':  {'mvc_fraction': 0.53, 'center_freq': 108, 'bandwidth_hz': 225},
            'LOBL': {'mvc_fraction': 0.15, 'center_freq':  92, 'bandwidth_hz': 168},
            'ROBL': {'mvc_fraction': 0.15, 'center_freq':  92, 'bandwidth_hz': 168},
        },
    },
}

# FATIGUE BLOCK PARAMETERS
# 20 consecutive reps. EMG fatigue modelled per De Luca (1993):
# RMS amplitude  : increases +22% over block (more motor units recruited)
# Center freq    : decreases, 20% over block (spectral compression)
# Trajectory degrades from min_jerk → trapezoidal after rep 10.

FATIGUE_BASE = {
    'rep_dur_s': 2.5, 'hold_dur_s': 0.5,
    'angles': {
        'pelvis': [32.0, 0.0, 0.0],
        'l3':     [24.0, 0.0, 0.0],
        't12':    [28.0, 0.0, 0.0],
        't4':     [30.0, 0.0, 0.0],
    },
    'base_mvc':   {'LES': 0.28, 'RES': 0.26, 'LOBL': 0.08, 'ROBL': 0.08},
    'base_cf':    {'LES': 100,  'RES': 100,   'LOBL': 88,   'ROBL': 88},
}

# BLOCK + SESSION GENERATION

def _imu_row(ts_ms, label, rep, risk, quats, euler_PL, euler_LT, euler_TU, angvel_L3, i):
    return {
        'timestamp_ms': round(ts_ms, 1),
        'label': label, 'rep': rep, 'risk_class': risk,
        # Absolute quaternions
        'pelvis_qw': quats[i,0,0], 'pelvis_qx': quats[i,0,1],
        'pelvis_qy': quats[i,0,2], 'pelvis_qz': quats[i,0,3],
        'l3_qw':     quats[i,1,0], 'l3_qx':     quats[i,1,1],
        'l3_qy':     quats[i,1,2], 'l3_qz':     quats[i,1,3],
        't12_qw':    quats[i,2,0], 't12_qx':    quats[i,2,1],
        't12_qy':    quats[i,2,2], 't12_qz':    quats[i,2,3],
        't4_qw':     quats[i,3,0], 't4_qx':     quats[i,3,1],
        't4_qy':     quats[i,3,2], 't4_qz':     quats[i,3,3],
        # Relative joint angles (derived from quaternion differences)
        'theta_PL_pitch': round(euler_PL[i,0], 3),
        'theta_PL_roll':  round(euler_PL[i,1], 3),
        'theta_PL_yaw':   round(euler_PL[i,2], 3),
        'theta_LT_pitch': round(euler_LT[i,0], 3),
        'theta_LT_roll':  round(euler_LT[i,1], 3),
        'theta_LT_yaw':   round(euler_LT[i,2], 3),
        'theta_TU_pitch': round(euler_TU[i,0], 3),
        'theta_TU_roll':  round(euler_TU[i,1], 3),
        'theta_TU_yaw':   round(euler_TU[i,2], 3),
        # Angular velocity at L3 in sagittal plane (deg/s)
        'angvel_L3_sagittal': round(angvel_L3[i], 3),
    }


def _process_angles(angle_series, fs_imu):
    """Compute quaternions, relative angles, angular velocity from angle series."""
    quats   = normalise_quat(angles_to_absolute_quats(angle_series))
    q_PL    = relative_quat(quats[:,1], quats[:,0])
    q_LT    = relative_quat(quats[:,2], quats[:,1])
    q_TU    = relative_quat(quats[:,3], quats[:,2])
    euler_PL = quat_to_euler(q_PL)
    euler_LT = quat_to_euler(q_LT)
    euler_TU = quat_to_euler(q_TU)
    angvel_L3 = np.gradient(euler_PL[:,0], 1.0/fs_imu)
    return quats, euler_PL, euler_LT, euler_TU, angvel_L3


def generate_block(label, profile, rng, t_offset_ms=0.0, p_scale=1.0):
    """
    Generate all reps for one movement label block.
    Returns (imu_rows, emg_rows, label_records, total_duration_ms).
    """
    fs_imu  = CONFIG['imu_fs']
    fs_emg  = CONFIG['emg_fs']
    ups     = fs_emg // fs_imu

    imu_rows, emg_rows, label_recs = [], [], []
    t_ms = t_offset_ms

    for rep in range(profile['reps']):

        # Scale angles for inter-participant + trial variability
        scale = p_scale * (1.0 + rng.normal(0, CONFIG['participant_var'] / 5))
        angs  = {k: [a * scale for a in v] for k, v in profile['angles'].items()}

        # Trajectory
        ang_series, _ = generate_rep_angles(
            angs,
            profile['rep_duration_s'], profile['hold_duration_s'],
            fs_imu, traj_type=profile.get('traj_type', 'min_jerk'), rng=rng
        )
        ang_series = add_imu_noise(ang_series, rng)
        quats, ePL, eLT, eTU, av = _process_angles(ang_series, fs_imu)

        N_imu = len(ang_series)
        N_emg = N_imu * ups
        rep_dur_ms = N_imu * 1000.0 / fs_imu

        ts_imu = t_ms + np.arange(N_imu) * 1000.0 / fs_imu
        ts_emg = t_ms + np.arange(N_emg) * 1000.0 / fs_emg

        risk = profile['risk_class']

        # IMU rows
        for i in range(N_imu):
            imu_rows.append(_imu_row(ts_imu[i], label, rep+1, risk,
                                     quats, ePL, eLT, eTU, av, i))

        # EMG rows
        envelope = get_activation_envelope(N_imu, profile['rep_duration_s'],
                                            profile['hold_duration_s'], fs_imu)
        env_up = np.interp(np.linspace(0, 1, N_emg), np.linspace(0, 1, N_imu), envelope)

        ch_signals = {}
        for ch, cfg in profile['emg'].items():
            sig = generate_emg_channel(env_up, fs_emg,
                                        mvc_fraction=cfg['mvc_fraction'] * scale,
                                        center_freq=cfg['center_freq'],
                                        bandwidth_hz=cfg['bandwidth_hz'], rng=rng)
            sig += rng.normal(0, CONFIG['emg_base_noise'], N_emg)
            sig += add_motion_artefact(N_emg, fs_emg, rng)
            ch_signals[ch] = sig

        for i in range(N_emg):
            emg_rows.append({
                'timestamp_ms': round(ts_emg[i], 2),
                'label': label, 'rep': rep+1, 'risk_class': risk,
                'emg_LES_mv':  round(ch_signals['LES'][i],  6),
                'emg_RES_mv':  round(ch_signals['RES'][i],  6),
                'emg_LOBL_mv': round(ch_signals['LOBL'][i], 6),
                'emg_ROBL_mv': round(ch_signals['ROBL'][i], 6),
            })

        label_recs.append({
            'label': label, 'rep': rep+1,
            'start_ms': round(t_ms, 1), 'end_ms': round(t_ms + rep_dur_ms, 1),
            'risk_class': risk,
        })

        t_ms += rep_dur_ms
        jitter_ms = rng.uniform(-1, 1) * CONFIG['inter_rep_jitter'] * 1000
        t_ms += max(0, profile['rest_s'] * 1000 + jitter_ms)

    return imu_rows, emg_rows, label_recs, t_ms - t_offset_ms


def generate_fatigue_block(rng, t_offset_ms=0.0, p_scale=1.0):
    """
    20-rep fatigue block with progressive EMG drift per De Luca (1993).
    EMG RMS +22% and center_freq -20% over the block.
    Trajectory degrades from min_jerk to trapezoidal after rep 10.
    Risk class: 0 (reps 1-6), -1 transitional (7-11), 1 (reps 12-20).
    """
    fs_imu = CONFIG['imu_fs']
    fs_emg = CONFIG['emg_fs']
    ups    = fs_emg // fs_imu
    prof   = FATIGUE_BASE
    n_reps = 20

    imu_rows, emg_rows, label_recs = [], [], []
    t_ms = t_offset_ms

    for rep in range(n_reps):
        frac      = rep / (n_reps - 1)        # 0 → 1 over block
        mvc_scale = 1.0 + 0.22 * frac
        cf_scale  = 1.0 - 0.20 * frac
        traj_type = 'min_jerk' if rep < 10 else 'trapezoidal'
        scale     = p_scale * (1.0 + rng.normal(0, 0.02))

        angs = {k: [a * scale for a in v] for k, v in prof['angles'].items()}

        ang_series, _ = generate_rep_angles(
            angs, prof['rep_dur_s'], prof['hold_dur_s'],
            fs_imu, traj_type=traj_type, rng=rng
        )
        ang_series = add_imu_noise(ang_series, rng)
        quats, ePL, eLT, eTU, av = _process_angles(ang_series, fs_imu)

        N_imu      = len(ang_series)
        N_emg      = N_imu * ups
        rep_dur_ms = N_imu * 1000.0 / fs_imu
        ts_imu     = t_ms + np.arange(N_imu) * 1000.0 / fs_imu
        ts_emg     = t_ms + np.arange(N_emg) * 1000.0 / fs_emg

        risk = 0 if rep < 6 else (-1 if rep < 12 else 1)

        for i in range(N_imu):
            imu_rows.append(_imu_row(ts_imu[i], 'FATIGUE_FLEXION', rep+1, risk,
                                     quats, ePL, eLT, eTU, av, i))

        envelope = get_activation_envelope(N_imu, prof['rep_dur_s'],
                                            prof['hold_dur_s'], fs_imu)
        env_up = np.interp(np.linspace(0, 1, N_emg), np.linspace(0, 1, N_imu), envelope)

        ch_signals = {}
        for ch in ['LES', 'RES', 'LOBL', 'ROBL']:
            sig = generate_emg_channel(env_up, fs_emg,
                                        mvc_fraction=prof['base_mvc'][ch] * mvc_scale * scale,
                                        center_freq=prof['base_cf'][ch] * cf_scale,
                                        bandwidth_hz=200.0, rng=rng)
            sig += rng.normal(0, CONFIG['emg_base_noise'], N_emg)
            ch_signals[ch] = sig

        for i in range(N_emg):
            emg_rows.append({
                'timestamp_ms': round(ts_emg[i], 2),
                'label': 'FATIGUE_FLEXION', 'rep': rep+1, 'risk_class': risk,
                'emg_LES_mv':  round(ch_signals['LES'][i],  6),
                'emg_RES_mv':  round(ch_signals['RES'][i],  6),
                'emg_LOBL_mv': round(ch_signals['LOBL'][i], 6),
                'emg_ROBL_mv': round(ch_signals['ROBL'][i], 6),
            })

        label_recs.append({
            'label': 'FATIGUE_FLEXION', 'rep': rep+1,
            'start_ms': round(t_ms, 1), 'end_ms': round(t_ms + rep_dur_ms, 1),
            'risk_class': risk, 'fatigue_fraction': round(frac, 3),
        })

        t_ms += rep_dur_ms

    return imu_rows, emg_rows, label_recs, t_ms - t_offset_ms


SESSION_ORDER = [
    'BASELINE_STATIC',
    'CLEAN_FLEXION', 'LUMBAR_DOMINANT',
    'CLEAN_LATERAL_L', 'CLEAN_LATERAL_R',
    'CLEAN_ROTATION_L', 'CLEAN_ROTATION_R',
    'FAST_BEND', 'SHOULDER_DRIVEN',
    'PICKUP_SYM', 'PICKUP_ASYM',
    'SIT_TO_STAND_NORMAL', 'SIT_TO_STAND_FAST',
    # FATIGUE_FLEXION appended separately
]

INTER_BLOCK_REST_MS = 3000.0  # 3 s between blocks


def generate_session(session_id, output_dir, seed=42):
    """
    Generate a complete labelled session with all protocol blocks.
    Saves imu_data.csv, emg_data.csv, labels.csv, session_config.json.
    """
    rng       = np.random.default_rng(seed)
    p_scale   = 1.0 + rng.uniform(-CONFIG['participant_var'], CONFIG['participant_var'])

    all_imu, all_emg, all_labels = [], [], []
    t_cursor = 0.0

    for label in SESSION_ORDER:
        imu_r, emg_r, lbl_r, dur = generate_block(
            label, PROFILES[label], rng, t_offset_ms=t_cursor, p_scale=p_scale
        )
        all_imu.extend(imu_r)
        all_emg.extend(emg_r)
        all_labels.extend(lbl_r)
        t_cursor += dur + INTER_BLOCK_REST_MS

    # Fatigue block
    imu_r, emg_r, lbl_r, dur = generate_fatigue_block(
        rng, t_offset_ms=t_cursor, p_scale=p_scale
    )
    all_imu.extend(imu_r)
    all_emg.extend(emg_r)
    all_labels.extend(lbl_r)

    # Write output
    out = Path(output_dir) / f'session_{session_id:04d}'
    out.mkdir(parents=True, exist_ok=True)

    imu_df    = pd.DataFrame(all_imu)
    emg_df    = pd.DataFrame(all_emg)
    labels_df = pd.DataFrame(all_labels)

    imu_df.to_csv(out / 'imu_data.csv', index=False)
    emg_df.to_csv(out / 'emg_data.csv', index=False)
    labels_df.to_csv(out / 'labels.csv', index=False)

    cfg_out = {
        'session_id': session_id, 'seed': seed,
        'participant_scale': round(float(p_scale), 4),
        'imu_samples': len(imu_df), 'emg_samples': len(emg_df),
        'duration_s': round(imu_df['timestamp_ms'].max() / 1000, 1),
        'config': CONFIG,
    }
    with open(out / 'session_config.json', 'w') as f:
        json.dump(cfg_out, f, indent=2)

    # Summary
    label_counts = labels_df['label'].value_counts().to_dict()
    class_dist   = labels_df.query('risk_class >= 0')['risk_class'].value_counts().to_dict()

    print(f"  Session {session_id:04d}  |  "
          f"IMU: {len(imu_df):,} samples ({len(imu_df)/CONFIG['imu_fs']:.0f} s)  |  "
          f"sEMG: {len(emg_df):,} samples  |  "
          f"Labels: {len(labels_df)} segments  |  "
          f"Risk 0: {class_dist.get(0,'?')}  Risk 1: {class_dist.get(1,'?')}")

    return imu_df, emg_df, labels_df

# ENTRY POINT

def main():
    parser = argparse.ArgumentParser(
        description='Synthetic IMU + sEMG generator — Spinal Movement Risk Monitor'
    )
    parser.add_argument('--n_sessions', type=int, default=5,
                        help='Number of synthetic participant sessions (default: 5)')
    parser.add_argument('--seed',       type=int, default=42,
                        help='Base random seed; session i uses seed+i (default: 42)')
    parser.add_argument('--output_dir', type=str, default='data/synthetic',
                        help='Output dir relative to project root (default: data/synthetic)')
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parents[2] / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'\nSpinal Movement Risk Monitor — Synthetic Data Generator')
    print(f'{"=" * 60}')
    print(f'Sessions  : {args.n_sessions}')
    print(f'Base seed : {args.seed}')
    print(f'Output    : {output_dir}')
    print(f'IMU fs    : {CONFIG["imu_fs"]} Hz    |   sEMG fs : {CONFIG["emg_fs"]} Hz')
    print(f'{"=" * 60}\n')

    for i in range(args.n_sessions):
        generate_session(
            session_id=i + 1,
            output_dir=output_dir,
            seed=args.seed + i,
        )

    print(f'\n[OK] Done - {args.n_sessions} session(s) written to {output_dir}\n')


if __name__ == '__main__':
    main()
