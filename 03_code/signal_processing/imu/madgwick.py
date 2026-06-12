#!/usr/bin/env python3
"""
Madgwick AHRS Filter
====================
Spinal Movement Risk Monitor — FYP 2025/26

Implements the Madgwick gradient-descent AHRS algorithm for fusing
accelerometer and gyroscope readings into a quaternion orientation estimate.

Reference
---------
    Madgwick, S. O. H., Harrison, A. J. L., & Vaidyanathan, R. (2011).
    Estimation of IMU and MARG orientation using a gradient descent algorithm.
    IEEE International Conference on Rehabilitation Robotics, Zurich.

The algorithm corrects gyro integration drift using the gravitational
reference vector measured by the accelerometer. The beta parameter trades off
between gyro trust (low beta → responsive but drifty) and accel trust
(high beta → stable but noisy during dynamic motion).

For lumbar spine kinematics at 100 Hz:
    beta = 0.033  →  ~2° steady-state error, good for 1–3 Hz trunk movements
    beta = 0.1    →  ~5° error, appropriate if sensor is very noisy

Usage
-----
    from signal_processing.imu.madgwick import MadgwickAHRS, fuse_imu_dataframe
    import numpy as np

    # Single-sensor fusion
    ahrs = MadgwickAHRS(beta=0.033, sample_freq=100.0)
    for i in range(len(accel_g)):
        q = ahrs.update(
            accel_g[i],          # [ax, ay, az] in g
            gyro_dps[i],         # [gx, gy, gz] in dps (converted to rad/s internally)
        )

    # Batch fusion on a DataFrame (output of RawConverter + GyroBiasCalibrator)
    imu_q_df = fuse_imu_dataframe(df_phys, imu_prefix="L3", fs=100.0, beta=0.033)
"""

import numpy as np
import pandas as pd
from typing import Optional, Union


# ─────────────────────────────────────────────────────────────────────────────
# QUATERNION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def quat_multiply(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Hamilton product of two quaternions [w, x, y, z].

    q ⊗ r  (non-commutative — order matters for rotation composition)
    """
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1,
    ], dtype=float)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Return conjugate (inverse for unit quaternion) of q = [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length. Returns identity if near-zero."""
    n = np.linalg.norm(q)
    if n < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_to_euler_zyx(q: np.ndarray) -> tuple:
    """
    Convert unit quaternion [w, x, y, z] to Euler angles (ZYX convention).

    Returns
    -------
    (pitch, roll, yaw) in degrees
        pitch : rotation about Y axis (sagittal — flexion/extension)
        roll  : rotation about X axis (frontal — lateral bend)
        yaw   : rotation about Z axis (transverse — rotation)

    Note: ZYX (yaw-pitch-roll) is the standard aerospace convention and
    matches how trunk angles are reported clinically (flexion first).
    Gimbal lock occurs near ±90° pitch; not a concern for trunk movements.
    """
    w, x, y, z = q

    # Pitch (Y) — flexion/extension
    sinp = 2.0 * (w*y - z*x)
    sinp = np.clip(sinp, -1.0, 1.0)   # clamp for numerical stability
    pitch = np.degrees(np.arcsin(sinp))

    # Roll (X) — lateral flexion
    sinr_cosp = 2.0 * (w*x + y*z)
    cosr_cosp = 1.0 - 2.0 * (x*x + y*y)
    roll = np.degrees(np.arctan2(sinr_cosp, cosr_cosp))

    # Yaw (Z) — axial rotation
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))

    return pitch, roll, yaw


def relative_quaternion(q_child: np.ndarray, q_parent: np.ndarray) -> np.ndarray:
    """
    Compute quaternion of child segment relative to parent.

    q_rel = q_parent^* ⊗ q_child

    This gives the rotation needed to go from parent's frame to child's frame,
    i.e. the inter-segment angle (e.g. lumbar angle relative to pelvis).
    """
    return quat_multiply(quat_conjugate(q_parent), q_child)


# ─────────────────────────────────────────────────────────────────────────────
# MADGWICK AHRS
# ─────────────────────────────────────────────────────────────────────────────

class MadgwickAHRS:
    """
    Madgwick gradient-descent orientation filter for a single IMU.

    The filter maintains a quaternion estimate of the sensor orientation
    in the world frame. It fuses gyroscope (integration) and accelerometer
    (gravity reference correction) at each time step.

    Parameters
    ----------
    beta : float
        Filter gain. Controls how strongly the accelerometer corrects
        gyro drift. Typical range 0.01–0.1 for MEMS IMUs at 100 Hz.
        - 0.033 : recommended for slow lumbar movements (< 1 Hz)
        - 0.05  : recommended for faster movements / more accel noise
    sample_freq : float
        Sampling frequency in Hz. Determines integration step size.
    initial_q : array-like, optional
        Starting quaternion [w, x, y, z]. Defaults to identity (upright, no rotation).
    """

    def __init__(
        self,
        beta: float = 0.033,
        sample_freq: float = 100.0,
        initial_q: Optional[np.ndarray] = None,
    ):
        self.beta        = float(beta)
        self.sample_freq = float(sample_freq)
        self.dt          = 1.0 / self.sample_freq

        if initial_q is not None:
            self.q = quat_normalize(np.array(initial_q, dtype=float))
        else:
            self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def reset(self, q: Optional[np.ndarray] = None) -> None:
        """Reset filter state to identity or a given quaternion."""
        self.q = quat_normalize(np.array(q, dtype=float)) if q is not None \
                 else np.array([1.0, 0.0, 0.0, 0.0])

    def update(
        self,
        accel_g: np.ndarray,
        gyro_dps: np.ndarray,
    ) -> np.ndarray:
        """
        Process one sample and return the updated orientation quaternion.

        Parameters
        ----------
        accel_g  : (3,) array — accelerometer reading in g [ax, ay, az]
        gyro_dps : (3,) array — gyroscope reading in degrees/second [gx, gy, gz]

        Returns
        -------
        q : (4,) array — updated quaternion [w, x, y, z] (unit norm)
        """
        q = self.q
        w, x, y, z = q

        # Convert gyro from dps to rad/s
        gx, gy, gz = np.radians(gyro_dps)

        # Normalise accelerometer measurement; skip update if degenerate
        a = np.array(accel_g, dtype=float)
        norm_a = np.linalg.norm(a)
        if norm_a < 1e-6:
            # No accel signal — pure gyro integration this step
            q_dot = 0.5 * quat_multiply(q, np.array([0.0, gx, gy, gz]))
            self.q = quat_normalize(q + q_dot * self.dt)
            return self.q.copy()
        a = a / norm_a

        ax, ay, az = a

        # ── Gradient descent: objective function f and Jacobian J ────────────
        # The objective function measures the error between measured gravity
        # (in sensor frame) and expected gravity (world +z = [0, 0, 1]).
        # f = J^T(q) · [ q ⊗ [0,0,0,1] ⊗ q^* - a_measured ]

        # f(q, a) — gradient of objective wrt sensor gravity reference
        f = np.array([
            2.0*(x*z - w*y)     - ax,
            2.0*(w*x + y*z)     - ay,
            2.0*(0.5 - x*x - y*y) - az,
        ])

        # Jacobian J(q)  — 3×4 matrix (partial derivatives of f wrt q)
        J = np.array([
            [-2.0*y,  2.0*z, -2.0*w, 2.0*x],
            [ 2.0*x,  2.0*w,  2.0*z, 2.0*y],
            [ 0.0,   -4.0*x, -4.0*y,  0.0 ],
        ])

        # Gradient step: ∇F = J^T · f
        grad = J.T @ f
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm

        # ── Gyro integration ─────────────────────────────────────────────────
        q_dot_gyro = 0.5 * quat_multiply(q, np.array([0.0, gx, gy, gz]))

        # ── Combined update ──────────────────────────────────────────────────
        q_dot = q_dot_gyro - self.beta * grad
        self.q = quat_normalize(q + q_dot * self.dt)

        return self.q.copy()

    def update_batch(
        self,
        accel_g: np.ndarray,
        gyro_dps: np.ndarray,
    ) -> np.ndarray:
        """
        Process a batch of samples.

        Parameters
        ----------
        accel_g  : (N, 3) array — accelerometer readings in g
        gyro_dps : (N, 3) array — gyroscope readings in degrees/second

        Returns
        -------
        quaternions : (N, 4) array — orientation quaternion [w, x, y, z] per sample
        """
        n = len(accel_g)
        quats = np.zeros((n, 4))
        for i in range(n):
            quats[i] = self.update(accel_g[i], gyro_dps[i])
        return quats


# ─────────────────────────────────────────────────────────────────────────────
# BATCH FUSION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# PLTU segment order — matches TCA9548A channel assignment and pipeline columns
SEGMENT_ORDER = ["pelvis", "l3", "t12", "t4"]
ARDUINO_PREFIX_MAP = {
    "pelvis": "Pelvis",
    "l3":     "L3",
    "t12":    "T12",
    "t4":     "T4",
}


def fuse_imu_dataframe(
    df_phys: pd.DataFrame,
    imu_prefix: str = "imu",
    fs: float = 100.0,
    beta: float = 0.033,
    reset_q: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Run Madgwick fusion on a single-IMU converted DataFrame.

    Parameters
    ----------
    df_phys : output of RawConverter.from_csv() for a single IMU — must contain
              columns {prefix}_{ax/ay/az}_g and {prefix}_{gx/gy/gz}_dps
    imu_prefix : column prefix used in df_phys (e.g. "imu", "L3")
    fs    : IMU sampling frequency (Hz)
    beta  : Madgwick beta gain
    reset_q : initial quaternion; defaults to identity

    Returns
    -------
    df_phys copy with appended columns:
        {prefix}_qw, {prefix}_qx, {prefix}_qy, {prefix}_qz
        {prefix}_pitch_deg, {prefix}_roll_deg, {prefix}_yaw_deg
    """
    p = imu_prefix
    accel = df_phys[[f"{p}_ax_g", f"{p}_ay_g", f"{p}_az_g"]].to_numpy()
    gyro  = df_phys[[f"{p}_gx_dps", f"{p}_gy_dps", f"{p}_gz_dps"]].to_numpy()

    ahrs = MadgwickAHRS(beta=beta, sample_freq=fs,
                        initial_q=reset_q)
    quats = ahrs.update_batch(accel, gyro)

    df_out = df_phys.copy()
    df_out[f"{p}_qw"] = quats[:, 0]
    df_out[f"{p}_qx"] = quats[:, 1]
    df_out[f"{p}_qy"] = quats[:, 2]
    df_out[f"{p}_qz"] = quats[:, 3]

    # Euler angles for convenience / sanity checking
    euler = np.array([quat_to_euler_zyx(quats[i]) for i in range(len(quats))])
    df_out[f"{p}_pitch_deg"] = euler[:, 0]
    df_out[f"{p}_roll_deg"]  = euler[:, 1]
    df_out[f"{p}_yaw_deg"]   = euler[:, 2]

    return df_out


def fuse_four_imu_dataframe(
    df_phys: pd.DataFrame,
    fs: float = 100.0,
    beta: float = 0.033,
) -> pd.DataFrame:
    """
    Run Madgwick fusion on a four-IMU converted DataFrame (output of
    RawConverter._convert_four_imu) and compute PLTU relative joint angles.

    Parameters
    ----------
    df_phys : DataFrame with columns Pelvis_ax_g ... T4_gz_dps (output of RawConverter)
    fs      : IMU sampling frequency (Hz)
    beta    : Madgwick beta gain

    Returns
    -------
    DataFrame with columns:
        t_ms (preserved)
        {seg}_qw, {seg}_qx, {seg}_qy, {seg}_qz   — world-frame quaternion per segment
        {seg}_pitch_deg, {seg}_roll_deg, {seg}_yaw_deg
        theta_PL_pitch/roll/yaw   — Pelvis→L3 relative angles (deg)
        theta_LT_pitch/roll/yaw   — L3→T12 relative angles (deg)
        theta_TU_pitch/roll/yaw   — T12→T4 relative angles (deg)
        angvel_L3_sagittal        — L3 angular velocity in sagittal plane (deg/s)
    """
    df_out = pd.DataFrame()
    if "t_ms" in df_phys.columns:
        df_out["t_ms"] = df_phys["t_ms"].values

    quats_by_seg = {}

    for seg in SEGMENT_ORDER:
        prefix = ARDUINO_PREFIX_MAP[seg]
        accel  = df_phys[[f"{prefix}_ax_g", f"{prefix}_ay_g", f"{prefix}_az_g"]].to_numpy()
        gyro   = df_phys[[f"{prefix}_gx_dps", f"{prefix}_gy_dps", f"{prefix}_gz_dps"]].to_numpy()

        # Initialise from first 0.5 s of accel (gravity direction) so the
        # filter does not need several seconds to converge from identity.
        n_init = max(1, min(50, len(accel)))
        q_init = init_quaternion_from_accel(accel[:n_init].mean(axis=0))
        ahrs = MadgwickAHRS(beta=beta, sample_freq=fs, initial_q=q_init)
        quats = ahrs.update_batch(accel, gyro)
        quats_by_seg[seg] = quats

        df_out[f"{seg}_qw"] = quats[:, 0]
        df_out[f"{seg}_qx"] = quats[:, 1]
        df_out[f"{seg}_qy"] = quats[:, 2]
        df_out[f"{seg}_qz"] = quats[:, 3]

    # ── Relative joint angles ────────────────────────────────────────────────
    pairs = [
        ("theta_PL", "pelvis", "l3"),    # Pelvis → L3  (lumbar flexion)
        ("theta_LT", "l3",   "t12"),     # L3 → T12     (thoracolumbar junction)
        ("theta_TU", "t12",  "t4"),      # T12 → T4     (upper thoracic)
    ]

    for angle_prefix, parent_seg, child_seg in pairs:
        q_parent = quats_by_seg[parent_seg]
        q_child  = quats_by_seg[child_seg]
        n = len(q_parent)
        pitches = np.zeros(n)
        rolls   = np.zeros(n)
        yaws    = np.zeros(n)

        for i in range(n):
            q_rel         = relative_quaternion(q_child[i], q_parent[i])
            pitch, roll, yaw = quat_to_euler_zyx(q_rel)
            pitches[i]    = pitch
            rolls[i]      = roll
            yaws[i]       = yaw

        df_out[f"{angle_prefix}_pitch"] = pitches
        df_out[f"{angle_prefix}_roll"]  = rolls
        df_out[f"{angle_prefix}_yaw"]   = yaws

    # ── L3 sagittal angular velocity (deg/s) ─────────────────────────────────
    # Derived from Pelvis→L3 pitch angle using central difference
    theta_PL_pitch = df_out["theta_PL_pitch"].to_numpy()
    dt             = 1.0 / fs
    angvel         = np.gradient(theta_PL_pitch, dt)
    df_out["angvel_L3_sagittal"] = angvel

    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# INITIALISATION FROM STATIC RECORDING
# ─────────────────────────────────────────────────────────────────────────────

def init_quaternion_from_accel(
    accel_mean_g: np.ndarray,
) -> np.ndarray:
    """
    Compute an initial quaternion aligning sensor frame gravity axis with
    world down [0, 0, 1] (or whichever axis is dominant).

    This gives the Madgwick filter a better starting point than identity
    when the sensor is not mounted upright, reducing initial transient error.

    Parameters
    ----------
    accel_mean_g : (3,) array — mean accelerometer reading over a short still
                                recording, in g  [ax, ay, az]

    Returns
    -------
    q_init : (4,) unit quaternion  [w, x, y, z]
    """
    a = np.array(accel_mean_g, dtype=float)
    a_norm = np.linalg.norm(a)
    if a_norm < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    a = a / a_norm

    # Target: world z-axis points down (0, 0, 1) — same convention as Madgwick
    # Rotation from a to [0, 0, 1]
    target = np.array([0.0, 0.0, 1.0])
    cross  = np.cross(a, target)
    dot    = np.dot(a, target)

    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-6:
        # Vectors are already aligned (or anti-aligned)
        if dot > 0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        else:
            # Anti-aligned — rotate 180° about any perpendicular axis
            return np.array([0.0, 1.0, 0.0, 0.0])

    axis  = cross / cross_norm
    angle = np.arctan2(cross_norm, dot)

    w = np.cos(angle / 2.0)
    xyz = np.sin(angle / 2.0) * axis
    return quat_normalize(np.array([w, xyz[0], xyz[1], xyz[2]]))


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

def _selftest() -> None:
    """
    Smoke-test the Madgwick filter.

    Scenario A (still convergence):
        Sensor upright — gravity on +Z (az = 1g, standard mounting with
        sensor Z pointing downward). Feed 5 s of still data at 100 Hz.
        Expected: pitch, roll, yaw all converge to ~0° (identity orientation).

    Scenario B (forward flexion):
        Inject gy = +20 dps (rotation around sensor Y-axis = sagittal pitch).
        Expected: pitch increases by ~60° over 3 s (20 dps × 3 s = 60°).

    The two scenarios run sequentially on the same AHRS instance to verify
    that the filter tracks dynamic motion after convergence.
    """
    rng     = np.random.default_rng(0)
    fs      = 100.0
    n_still = int(5.0 * fs)       # 5 s still
    n_flex  = int(3.0 * fs)       # 3 s flexion

    # Scenario A: gravity on Z (az = +1g). The gravity objective function
    # f[2] = 2*(0.5 - x² - y²) - az = 0 at identity, so filter starts settled.
    accel_still        = np.zeros((n_still, 3))
    accel_still[:, 2]  = 1.0    # az = 1 g
    accel_still       += rng.normal(0, 0.01, accel_still.shape)
    gyro_still         = rng.normal(0, 0.5, (n_still, 3))  # noise only

    # Scenario B: forward flexion — gy = +20 dps (rotation about sensor Y)
    # Gravity stays approximately on Z during a slow forward lean.
    accel_flex        = np.zeros((n_flex, 3))
    accel_flex[:, 2]  = 1.0
    accel_flex       += rng.normal(0, 0.01, accel_flex.shape)
    gyro_flex         = rng.normal(0, 0.5, (n_flex, 3))
    gyro_flex[:, 1]  += 20.0   # gy = 20 dps sagittal rotation

    ahrs = MadgwickAHRS(beta=0.033, sample_freq=fs)

    # Still phase
    q_still     = ahrs.update_batch(accel_still, gyro_still)
    pitch_still = np.array([quat_to_euler_zyx(q)[0] for q in q_still])

    # Flexion phase (continues from final still orientation)
    q_flex      = ahrs.update_batch(accel_flex, gyro_flex)
    pitch_flex  = np.array([quat_to_euler_zyx(q)[0] for q in q_flex])

    pitch_end_still = float(np.mean(pitch_still[-50:]))
    pitch_end_flex  = float(pitch_flex[-1])

    print("Madgwick self-test:")
    print(f"  Pitch after 5 s still  : {pitch_end_still:+.2f}° (expected ~0°)")
    print(f"  Pitch after 3 s flexion: {pitch_end_flex:+.2f}° (expected >20°)")
    assert abs(pitch_end_still) < 10.0, \
        f"Still convergence failed — pitch too large: {pitch_end_still:.2f}°"
    assert pitch_end_flex > 20.0, \
        f"Flexion not tracked — pitch too small: {pitch_end_flex:.2f}°"
    print("  PASS")


if __name__ == "__main__":
    _selftest()
