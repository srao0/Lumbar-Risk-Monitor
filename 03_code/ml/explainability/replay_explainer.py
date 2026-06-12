#!/usr/bin/env python3
"""
replay_explainer.py
===================
Spinal Movement Risk Monitor, FYP 2025/26 | Imperial College London

Generates two human-readable explanations for every replay prediction:

    engineering_reason : technical, feature-grounded rationale citing lumbar
                         angle, angular velocity, smoothness abnormality, time
                         in the risk zone, EMG asymmetry, baseline deviation,
                         and the component scores R_IMU / R_EMG / R_total.

    layman_reason      : plain-English statement a non-technical user can read,
                         explaining why the movement was Safe, Cautious, or
                         Risky within this system.

Design constraints (intentional and load-bearing)
-------------------------------------------------
  * This is NOT a medical diagnosis. The wording deliberately uses hedged,
    system-scoped language: "flagged", "suggests", "appeared", "within this
    system". No clinical claims (no "injury", "damage", "pathology") are made.
  * For the IMU-only fallback the explanation states explicitly that the
    decision used movement features only and that no muscle (EMG) signal
    contributed.
  * For full_hybrid the EMG contribution is only described when valid EMG
    features are actually available for the window; otherwise EMG is omitted
    rather than fabricated.

The thresholds below mirror the flags already used in demo_risk_monitor.py and
the membership-function breakpoints in ml/fuzzy/mamdani_fis.py so the prose is
consistent with the fuzzy decision layer.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


# Feature thresholds (kept consistent with demo_risk_monitor + Mamdani MFs)

ANGLE_RISK_DEG       = 45.0     # trunk flexion risk zone boundary
ANGLE_MODERATE_DEG   = 30.0
ANGVEL_FAST_DEG_S    = 40.0     # fast bending
ANGVEL_MODERATE_DEG_S = 25.0
TIME_IN_RISK_HIGH    = 0.50     # sustained exposure (matches FIS "High")
TIME_IN_RISK_MOD     = 0.15
Z_SAL_POOR           = -1.0     # z_ldlj below baseline → jerkier than usual
Z_SAL_REDUCED        = -0.5
Z_DEV_LARGE          = 1.5      # mean |z| → large departure from baseline
Z_DEV_MODERATE       = 0.8
AR_HIGH              = 1.5      # asymmetry ratio (|AI_ES|*2) → high
AR_MODERATE          = 1.0


# Small helpers

def _get(row, key: str, default: float = float("nan")) -> float:
    try:
        val = row.get(key, default)
    except AttributeError:
        val = row[key] if key in row else default
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    return f


def _finite(x: float) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _asymmetry_ratio(row) -> float:
    """Reconstruct the AR fed to the FIS: |emg_ai_ES| * 2, clipped to [0, 3]."""
    ai = _get(row, "emg_ai_ES", float("nan"))
    if not _finite(ai):
        return float("nan")
    return float(np.clip(abs(ai) * 2.0, 0.0, 3.0))


def _baseline_deviation(row) -> float:
    """Mean absolute IMU z-score across flex / velocity / smoothness."""
    vals = []
    for c in ("imu_z_flex", "imu_z_vel", "imu_z_ldlj"):
        v = _get(row, c, float("nan"))
        if _finite(v):
            vals.append(abs(v))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def emg_features_valid(row, emg_cols: Optional[list] = None) -> bool:
    """
    Return True when at least one EMG feature on this row is present and finite.

    Used to decide whether a full_hybrid explanation may legitimately talk about
    muscle activity for this specific window.
    """
    emg_cols = emg_cols or [
        "emg_rms_LES", "emg_rms_RES", "emg_rms_LOBL", "emg_rms_ROBL",
        "emg_ai_ES", "emg_ai_OBL",
    ]
    for c in emg_cols:
        v = _get(row, c, float("nan"))
        if _finite(v) and abs(v) > 0.0:
            return True
    return False


# Engineering explanation

def _engineering_reason(
    row,
    result: dict,
    operating_mode: str,
    emg_available: bool,
) -> str:
    """Technical, feature-grounded rationale string."""
    risk_level = result.get("risk_level") or _risk_level_from_label(result.get("label"))
    R_total = result.get("p_risk", float("nan"))
    R_IMU   = result.get("R_IMU", float("nan"))
    R_EMG   = result.get("R_EMG", float("nan"))

    angle  = _get(row, "imu_trunk_angle_peak")
    vel    = _get(row, "imu_angvel_peak")
    tirz   = _get(row, "imu_time_in_risk_zone")
    z_sal  = _get(row, "imu_z_ldlj")
    z_dev  = _baseline_deviation(row)
    ar     = _asymmetry_ratio(row)

    drivers: list[str] = []

    # Kinematic drivers (always available, IMU is the baseline system)
    if _finite(angle):
        if angle >= ANGLE_RISK_DEG:
            drivers.append(
                f"peak trunk flexion {angle:.1f}deg exceeded the {ANGLE_RISK_DEG:.0f}deg risk-zone boundary"
            )
        elif angle >= ANGLE_MODERATE_DEG:
            drivers.append(f"moderate peak trunk flexion {angle:.1f}deg")

    if _finite(vel):
        if vel >= ANGVEL_FAST_DEG_S:
            drivers.append(f"high peak angular velocity {vel:.1f}deg/s suggested fast bending")
        elif vel >= ANGVEL_MODERATE_DEG_S:
            drivers.append(f"moderate peak angular velocity {vel:.1f}deg/s")

    if _finite(tirz) and tirz >= TIME_IN_RISK_MOD:
        pct = tirz * 100.0
        descriptor = "sustained" if tirz >= TIME_IN_RISK_HIGH else "appreciable"
        drivers.append(f"{descriptor} time in the high-flexion zone ({pct:.0f}% of the window)")

    if _finite(z_sal) and z_sal <= Z_SAL_REDUCED:
        quality = "poor" if z_sal <= Z_SAL_POOR else "reduced"
        drivers.append(
            f"{quality} movement smoothness (LDLJ z = {z_sal:+.2f} vs personal baseline)"
        )

    if _finite(z_dev) and z_dev >= Z_DEV_MODERATE:
        size = "large" if z_dev >= Z_DEV_LARGE else "moderate"
        drivers.append(f"{size} departure from the personal IMU baseline (mean |z| = {z_dev:.2f})")

    # EMG drivers, only when this is full_hybrid AND EMG is genuinely present
    emg_clause = ""
    if operating_mode == "full_hybrid" and emg_available:
        if _finite(ar) and ar >= AR_MODERATE:
            level = "high" if ar >= AR_HIGH else "moderate"
            drivers.append(f"{level} bilateral EMG asymmetry (AR = {ar:.2f})")
        if _finite(R_EMG):
            emg_clause = f", R_EMG = {R_EMG:.2f}"

    # Compose the component-score tail
    parts = []
    if _finite(R_IMU):
        parts.append(f"R_IMU = {R_IMU:.2f}")
    if operating_mode == "full_hybrid" and emg_available and _finite(R_EMG):
        parts.append(f"R_EMG = {R_EMG:.2f}")
    if _finite(R_total):
        parts.append(f"R_total = {R_total:.2f}")
    score_tail = "; ".join(parts)

    # Mode preamble
    if operating_mode == "imu_only_fallback":
        mode_note = (
            "IMU-only fallback: the decision was based on movement features only; "
            "no EMG (muscle) signal contributed to this score"
        )
    elif operating_mode == "full_hybrid" and not emg_available:
        mode_note = (
            "Full-hybrid mode but no valid EMG features for this window; "
            "the score reflects movement features only for this window"
        )
    else:
        mode_note = "Full-hybrid: movement (IMU) and muscle (EMG) features both contributed"

    fis_reason = str(result.get("fis_reason") or "").strip()

    if drivers:
        driver_text = "; ".join(drivers)
        body = f"Flagged as {risk_level} because {driver_text}"
    else:
        body = (
            f"Classified as {risk_level}; movement features stayed within "
            f"normal ranges for this window"
        )

    segments = [mode_note, body]
    if fis_reason and fis_reason.lower() not in ("", "nan", "no rule fired"):
        segments.append(f"dominant FIS rule: {fis_reason}")
    if score_tail:
        segments.append(score_tail)

    return ". ".join(segments) + "."


# Layman explanation

def _layman_reason(
    row,
    result: dict,
    operating_mode: str,
    emg_available: bool,
) -> str:
    """Plain-English statement for a non-technical reader."""
    risk_level = (result.get("risk_level") or _risk_level_from_label(result.get("label")) or "Cautious")

    angle  = _get(row, "imu_trunk_angle_peak")
    vel    = _get(row, "imu_angvel_peak")
    tirz   = _get(row, "imu_time_in_risk_zone")
    z_sal  = _get(row, "imu_z_ldlj")
    ar     = _asymmetry_ratio(row)

    reasons: list[str] = []
    if _finite(angle) and angle >= ANGLE_RISK_DEG:
        reasons.append("you bent forward quite far")
    elif _finite(angle) and angle >= ANGLE_MODERATE_DEG:
        reasons.append("you bent forward a moderate amount")

    if _finite(vel) and vel >= ANGVEL_FAST_DEG_S:
        reasons.append("the movement was fast")

    if _finite(tirz) and tirz >= TIME_IN_RISK_HIGH:
        reasons.append("you stayed bent over for much of the movement")

    if _finite(z_sal) and z_sal <= Z_SAL_POOR:
        reasons.append("the movement looked less smooth than your usual pattern")

    if operating_mode == "full_hybrid" and emg_available and _finite(ar) and ar >= AR_HIGH:
        reasons.append("your back muscles worked unevenly on the two sides")

    # Build the sentence by risk level
    if risk_level == "Safe":
        lead = "This movement appeared safe within this system"
        if reasons:
            why = "even though " + _join(reasons)
            sentence = f"{lead} — {why}."
        else:
            sentence = f"{lead}: your bending stayed gentle, controlled and within your normal range."
    elif risk_level == "Risky":
        lead = "This movement was flagged as risky within this system"
        why = _join(reasons) if reasons else "the overall movement pattern departed from your safer baseline"
        sentence = f"{lead} because {why}."
    else:  # Cautious / Amber
        lead = "This movement was flagged for caution within this system"
        why = _join(reasons) if reasons else "the movement pattern was slightly outside your safer baseline"
        sentence = f"{lead} because {why}."

    # Mode-specific honesty note
    if operating_mode == "imu_only_fallback":
        sentence += (
            " This judgement used only how you moved (motion sensors); "
            "muscle sensors were not used."
        )
    elif operating_mode == "full_hybrid" and not emg_available:
        sentence += " Muscle-sensor data was unavailable for this moment, so only movement was used."

    sentence += " This is movement feedback, not a medical diagnosis."
    return sentence


def _join(items: list[str]) -> str:
    items = [s for s in items if s]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _risk_level_from_label(label: Optional[str]) -> str:
    return {"GREEN": "Safe", "AMBER": "Cautious", "RED": "Risky"}.get(
        str(label or "").upper(), "Cautious"
    )


# Public interface

def explain_window(
    feature_row,
    result: dict,
    operating_mode: str = "full_hybrid",
    emg_available: Optional[bool] = None,
) -> dict:
    """
    Produce engineering and layman explanations for one classified window.

    Parameters
    ----------
    feature_row    : pandas Series (one feature_matrix.csv row)
    result         : dict returned by demo_risk_monitor.classify_window
                     (keys: p_risk, label, risk_level, predicted, R_IMU, R_EMG,
                     fis_reason, operating_mode)
    operating_mode : "full_hybrid" or "imu_only_fallback"
    emg_available  : override for whether EMG features are valid on this row.
                     If None, inferred from the row. Forced False for fallback.

    Returns
    -------
    dict with keys: engineering_reason, layman_reason
    """
    if operating_mode == "imu_only_fallback":
        emg_ok = False
    elif emg_available is None:
        emg_ok = emg_features_valid(feature_row)
    else:
        emg_ok = bool(emg_available)

    return {
        "engineering_reason": _engineering_reason(feature_row, result, operating_mode, emg_ok),
        "layman_reason": _layman_reason(feature_row, result, operating_mode, emg_ok),
    }


if __name__ == "__main__":
    # Minimal self-check with a synthetic risky window.
    demo_row = pd.Series({
        "imu_trunk_angle_peak": 62.0,
        "imu_angvel_peak": 55.0,
        "imu_time_in_risk_zone": 0.7,
        "imu_z_ldlj": -1.8,
        "imu_z_flex": 1.9,
        "imu_z_vel": 1.2,
        "emg_ai_ES": 0.9,
        "emg_rms_LES": 0.05,
    })
    demo_result = {
        "p_risk": 0.82, "label": "RED", "risk_level": "Risky", "predicted": 1,
        "R_IMU": 0.79, "R_EMG": 0.61,
        "fis_reason": "Excessive spinal load with bilateral muscle overactivation",
    }
    for mode in ("full_hybrid", "imu_only_fallback"):
        out = explain_window(demo_row, demo_result, operating_mode=mode)
        print(f"\n[{mode}]")
        print("  ENG:", out["engineering_reason"])
        print("  LAY:", out["layman_reason"])
