"""
Mamdani Fuzzy Inference System, Spinal Movement Risk Monitor
FYP 2025/26 | Imperial College London

Implements the 13-rule Mamdani FIS specified in the Pipeline Architecture
document (spec §7.4-7.5). Pure NumPy implementation, no skfuzzy dependency.

Architecture:
  Inputs (6):
    R_IMU: RF predict_proba on IMU features      [0, 1]
    R_EMG: LR predict_proba on EMG features      [0, 1]
    z_sal: SAL baseline deviation (clipped ±3)   [-3, 3]
    time_in_risk_zone: proportion of window above risk angle [0, 1]
    ar: EMG bilateral asymmetry ratio          [0,  3]
    z_imu_mean: mean absolute z-score of IMU features [0,  3]

  Output (1):
    R_total: defuzzified total risk score           [0, 1]
     colour: Green / Amber / Red
     risk_level: Safe / Cautious / Risky
    reason: highest-activation rule's antecedent text

Membership functions:
  Inputs use triangular MFs with 50% overlap (spec §7.4).
  Output uses 5 triangular labels: Very Low, Low, Medium, High, Very High.
  Defuzzification: centroid (centre of area).

Rule base: original biomechanical fusion rules plus fixed abnormality
escalation rules for time in risk zone and baseline deviation.

Usage
-----
    from ml.fuzzy.mamdani_fis import MamdaniFIS

    fis = MamdaniFIS()

    # Single window
    result = fis.infer(
        R_IMU=0.72, R_EMG=0.58, z_sal=1.8,
        time_in_risk_zone=0.6, ar=1.7, z_imu_mean=2.1,
    )
    print(result)
    # {'R_total': 0.81, 'colour': 'RED', 'reason': '...', 'rule_strengths': {...}}

    # Batch (DataFrame)
    output_df = fis.infer_batch(feature_df)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# MEMBERSHIP FUNCTION PRIMITIVES

def trimf(x: float, a: float, b: float, c: float) -> float:
    """
    Triangular membership function.

      μ(x) = 0        if x ≤ a
           = (x-a)/(b-a)  if a < x ≤ b
           = (c-x)/(c-b)  if b < x ≤ c
           = 0        if x > c

    Parameters
    ----------
    x : input value
    a : left foot
    b : peak
    c : right foot
    """
    if b == a:
        left = 1.0 if x <= a else 0.0
    else:
        left = np.clip((x - a) / (b - a), 0.0, 1.0)
    if c == b:
        right = 1.0 if x >= c else 0.0
    else:
        right = np.clip((c - x) / (c - b), 0.0, 1.0)
    return float(min(left, right))


def trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """
    Trapezoidal membership function.

      μ(x) = 0            if x ≤ a or x ≥ d
           = (x-a)/(b-a)  if a < x < b  (rising slope)
           = 1            if b ≤ x ≤ c  (flat top)
           = (d-x)/(d-c)  if c < x < d  (falling slope)
    """
    rise = np.clip((x - a) / (b - a), 0.0, 1.0) if b > a else float(x >= a)
    fall = np.clip((d - x) / (d - c), 0.0, 1.0) if d > c else float(x <= d)
    flat = 1.0 if b <= x <= c else 0.0
    return float(max(flat, min(rise, fall)))


# INPUT MEMBERSHIP FUNCTIONS (spec §7.4)

def _risk_mfs(x: float) -> Dict[str, float]:
    """
    Triangular MFs for R_IMU or R_EMG in [0, 1].
    Three labels: Low / Medium / High, 50% overlap.
      Low:    peak 0.2, range [0.0, 0.4]
      Medium: peak 0.5, range [0.2, 0.8]
      High:   peak 0.8, range [0.55, 1.0]
    """
    return {
        "Low":    trimf(x,  0.00, 0.20, 0.40),
        "Medium": trimf(x,  0.20, 0.50, 0.80),
        "High":   trimf(x,  0.55, 0.80, 1.00),
    }


def _smoothness_mfs(x: float) -> Dict[str, float]:
    """
    Trapezoidal MFs for z_sal (clipped to [-3, 3]).
    Positive z_sal → SAL is HIGHER than baseline (smoother) → Normal.
    Negative z_sal → SAL is LOWER (jerker) → Poor.
      Normal  : z > 0 (movements at or above baseline quality)
      Reduced : -1.5 < z < 0.5  (mildly below baseline)
      Poor    : z < -1.0  (significantly jerky)
    """
    return {
        "Normal":  trapmf(x, -0.5,  0.0,  3.0,  3.0),
        "Reduced": trapmf(x, -2.0, -1.5,  0.0,  0.5),
        "Poor":    trapmf(x, -3.0, -3.0, -1.5, -0.5),
    }


def _time_in_risk_zone_mfs(x: float) -> Dict[str, float]:
    """
    Memberships for the proportion of a window above the risk-angle zone.
      Low      : predominantly outside the risk zone
      Moderate : appreciable time in the risk zone
      High     : sustained time in the risk zone
    """
    return {
        "Low":      trimf(x, 0.00, 0.00, 0.35),
        "Moderate": trimf(x, 0.15, 0.45, 0.70),
        "High":     trapmf(x, 0.50, 0.70, 1.00, 1.00),
    }


def _asymmetry_mfs(x: float) -> Dict[str, float]:
    """
    Trapezoidal MFs for asymmetry ratio AR = |emg_ai_ES| × 2 (rescaled to [0, 3]).
    Note: AI ∈ [-1,1] → AR = |AI| * 2 maps to [0, 2]; values above 1.5 = risky.
      Balanced : AR < 1.2
      Moderate : 0.8 < AR < 1.8
      High     : AR > 1.5
    """
    return {
        "Balanced": trapmf(x, 0.0,  0.0,  0.8,  1.2),
        "Moderate": trapmf(x, 0.8,  1.0,  1.4,  1.8),
        "High":     trapmf(x, 1.5,  1.8,  3.0,  3.0),
    }


def _deviation_mfs(x: float) -> Dict[str, float]:
    """
    Triangular MFs for mean IMU z-score (clipped [0, 3]).
      Small    : peak 0.0, range [0.0, 0.8]
      Moderate : peak 1.0, range [0.5, 2.0]
      Large    : peak 2.0, range [1.5, 3.0]
    """
    return {
        "Small":    trimf(x, 0.0, 0.0, 0.8),
        "Moderate": trimf(x, 0.5, 1.0, 2.0),
        "Large":    trimf(x, 1.5, 2.0, 3.0),
    }


# OUTPUT MEMBERSHIP FUNCTIONS (spec §7.4)

# Five output labels with triangular MFs on [0, 1]
OUTPUT_MFS = {
    "Very Low":  (0.0,  0.1, 0.25),
    "Low":       (0.1,  0.3, 0.5),
    "Medium":    (0.35, 0.5, 0.65),
    "High":      (0.5,  0.7, 0.9),
    "Very High": (0.75, 0.9, 1.0),
}

# Universe of discourse for defuzzification
OUTPUT_UNIVERSE = np.linspace(0.0, 1.0, 501)

# Risk colour thresholds (spec §7.4)
COLOUR_THRESHOLDS = {
    "Green": (0.00, 0.35),
    "Amber": (0.35, 0.65),
    "Red":   (0.65, 1.00),
}
RISKY_THRESHOLD = COLOUR_THRESHOLDS["Red"][0]


def risk_level_from_score(R_total: float) -> str:
    """Map the defuzzified risk score to the deployed three-level output."""
    if R_total < COLOUR_THRESHOLDS["Amber"][0]:
        return "Safe"
    if R_total < RISKY_THRESHOLD:
        return "Cautious"
    return "Risky"


# RULE BASE (spec §7.5)
# Each rule is a dict:
# antecedent : dict of (input_name, label) conditions (None = don't care)
# consequent : output label string
# reason     : human-readable antecedent text (displayed to user)

RULES = [
    {   # R1
        "antecedent": {"R_IMU": "High", "R_EMG": "High"},
        "consequent": "Very High",
        "reason": "Excessive spinal load with bilateral muscle overactivation",
    },
    {   # R2
        "antecedent": {"R_IMU": "High", "R_EMG": "Medium"},
        "consequent": "High",
        "reason": "Excessive lumbar flexion with elevated muscle response",
    },
    {   # R3
        "antecedent": {"R_IMU": "High", "R_EMG": "Low", "smoothness": "Poor"},
        "consequent": "High",
        "reason": "High flexion combined with loss of movement smoothness",
    },
    {   # R4
        "antecedent": {"R_IMU": "High", "asymmetry": "High"},
        "consequent": "High",
        "reason": "Excessive flexion combined with bilateral muscle imbalance",
    },
    {   # R5
        "antecedent": {"R_IMU": "Medium", "R_EMG": "High", "asymmetry": "High"},
        "consequent": "High",
        "reason": "Moderate kinematics but asymmetric overload — compensation risk",
    },
    {   # R6
        "antecedent": {"R_IMU": "Medium", "R_EMG": "High"},
        "consequent": "Medium",
        "reason": "Moderate flexion with elevated muscle activation",
    },
    {   # R7
        "antecedent": {"R_IMU": "Medium", "R_EMG": "Medium", "smoothness": "Poor"},
        "consequent": "Medium",
        "reason": "Moderate load with poor movement smoothness",
    },
    {   # R8
        "antecedent": {"R_IMU": "Medium", "R_EMG": "Low"},
        "consequent": "Medium",
        "reason": "Moderate trunk movement with muscle within normal range",
    },
    {   # R9
        "antecedent": {"R_IMU": "Low", "R_EMG": "High", "asymmetry": "High"},
        "consequent": "Medium",
        "reason": "Low kinematic risk but significant muscle asymmetry detected",
    },
    {   # R10
        "antecedent": {"R_IMU": "Low", "R_EMG": "Medium"},
        "consequent": "Medium",
        "reason": "Low movement risk with mild muscle activation — monitor",
    },
    {   # R11
        "antecedent": {"R_IMU": "Low", "R_EMG": "Low", "smoothness": "Normal"},
        "consequent": "Very Low",
        "reason": "Movement within personal baseline — safe",
    },
    {   # R12
        "antecedent": {"R_IMU": "Medium", "R_EMG": "Medium"},
        "consequent": "Medium",
        "reason": "Movement and muscle activation at moderate levels",
    },
    {   # R13
        "antecedent": {
            "R_IMU": "High", "R_EMG": "Medium",
            "smoothness": "Poor", "asymmetry": "High",
        },
        "consequent": "Very High",
        "reason": "Maximal risk: flexion combined with poor control and muscle asymmetry",
    },
    {   # R14
        "antecedent": {"time_in_risk_zone": "High"},
        "consequent": "High",
        "reason": "Sustained time in the high-flexion risk zone",
    },
    {   # R15
        "antecedent": {"deviation": "Large", "smoothness": "Poor"},
        "consequent": "High",
        "reason": "Large departure from baseline combined with poor movement smoothness",
    },
    {   # R16
        "antecedent": {"deviation": "Large", "R_IMU": "Medium"},
        "consequent": "High",
        "reason": "Moderate kinematic risk with a large baseline deviation",
    },
    {   # R17
        "antecedent": {"deviation": "Large", "R_EMG": "High"},
        "consequent": "High",
        "reason": "High muscle risk with a large baseline deviation",
    },
]


# MAMDANI FIS CLASS

FALLBACK_IMU_RULES = [
    {
        "antecedent": {"R_IMU": "High"},
        "consequent": "High",
        "reason": "High IMU classifier risk from sagittal movement features",
    },
    {
        "antecedent": {"R_IMU": "High", "time_in_risk_zone": "High"},
        "consequent": "Very High",
        "reason": "High classifier risk with sustained high-flexion exposure",
    },
    {
        "antecedent": {"R_IMU": "High", "smoothness": "Poor"},
        "consequent": "Very High",
        "reason": "High classifier risk combined with poor movement smoothness",
    },
    {
        "antecedent": {"R_IMU": "Medium", "time_in_risk_zone": "High"},
        "consequent": "High",
        "reason": "Moderate classifier risk with sustained high-flexion exposure",
    },
    {
        "antecedent": {"R_IMU": "Medium", "smoothness": "Poor"},
        "consequent": "High",
        "reason": "Moderate classifier risk with poor movement smoothness",
    },
    {
        "antecedent": {"R_IMU": "Medium", "deviation": "Large"},
        "consequent": "High",
        "reason": "Moderate classifier risk with a large IMU baseline deviation",
    },
    {
        "antecedent": {"deviation": "Large", "smoothness": "Poor"},
        "consequent": "High",
        "reason": "Large IMU baseline deviation combined with poor smoothness",
    },
    {
        "antecedent": {"time_in_risk_zone": "High"},
        "consequent": "High",
        "reason": "Sustained time in the high-flexion risk zone",
    },
    {
        "antecedent": {"R_IMU": "Medium"},
        "consequent": "Medium",
        "reason": "Moderate IMU classifier risk",
    },
    {
        "antecedent": {"R_IMU": "Low", "deviation": "Large"},
        "consequent": "Medium",
        "reason": "Low classifier risk but large departure from personal IMU baseline",
    },
    {
        "antecedent": {"R_IMU": "Low", "smoothness": "Reduced"},
        "consequent": "Low",
        "reason": "Low classifier risk with mildly reduced smoothness",
    },
    {
        "antecedent": {
            "R_IMU": "Low",
            "smoothness": "Normal",
            "time_in_risk_zone": "Low",
            "deviation": "Small",
        },
        "consequent": "Very Low",
        "reason": "IMU movement remains within the safe personal baseline",
    },
]


class MamdaniFIS:
    """
    Mamdani Fuzzy Inference System for spinal movement risk scoring.

    Parameters
    ----------
    rules      : list of rule dicts (defaults to fixed deployed rule base)
    resolution : number of points in defuzzification universe (default 501)

    Example
    -------
    >>> fis = MamdaniFIS()
    >>> result = fis.infer(
    ...     R_IMU=0.72, R_EMG=0.58, z_sal=1.8,
    ...     time_in_risk_zone=0.6, ar=1.7, z_imu_mean=2.1,
    ... )
    >>> result['R_total'], result['risk_level'], result['reason']
    (0.81, 'Risky', 'Excessive spinal load with bilateral muscle overactivation')
    """

    def __init__(self, rules: Optional[List[dict]] = None, resolution: int = 501):
        self.rules      = rules or RULES
        self.universe   = np.linspace(0.0, 1.0, resolution)

    # Fuzzification

    def _fuzzify(
        self,
        R_IMU: float,
        R_EMG: float,
        z_sal: float,
        time_in_risk_zone: float,
        ar: float,
        z_imu_mean: float,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute membership degrees for all input variables.

        Parameters
        ----------
        R_IMU      : RF probability for kinematic risk [0, 1]
        R_EMG      : LR probability for muscle risk    [0, 1]
        z_sal      : SAL z-score vs baseline           (clipped ±3)
        time_in_risk_zone : fraction of window in risk zone [0, 1]
        ar         : asymmetry ratio |AI_ES| * 2       [0, 3]
        z_imu_mean : mean absolute z-score of IMU feats [0, 3]

        Returns
        -------
        Dict of input_name → {label → degree}
        """
        # Clip inputs to their defined universes
        R_IMU      = float(np.clip(R_IMU,     0.0, 1.0))
        R_EMG      = float(np.clip(R_EMG,     0.0, 1.0))
        z_sal      = float(np.clip(z_sal,    -3.0, 3.0))
        time_in_risk_zone = float(np.clip(time_in_risk_zone, 0.0, 1.0))
        ar         = float(np.clip(ar,        0.0, 3.0))
        z_imu_mean = float(np.clip(z_imu_mean, 0.0, 3.0))

        return {
            "R_IMU":      _risk_mfs(R_IMU),
            "R_EMG":      _risk_mfs(R_EMG),
            "smoothness": _smoothness_mfs(z_sal),
            "time_in_risk_zone": _time_in_risk_zone_mfs(time_in_risk_zone),
            "asymmetry":  _asymmetry_mfs(ar),
            "deviation":  _deviation_mfs(z_imu_mean),
        }

    # Inference (min-based AND, max-based aggregation)

    def _fire_rules(
        self, memberships: Dict[str, Dict[str, float]]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Evaluate all rules and aggregate clipped output MFs.

        Returns
        -------
        aggregated : (resolution,) output MF after max-aggregation
        strengths  : dict of rule_index → firing strength
        """
        aggregated = np.zeros(len(self.universe))
        strengths  = {}

        for i, rule in enumerate(self.rules):
            # Firing strength = min of all antecedent memberships
            firing = 1.0
            for inp, label in rule["antecedent"].items():
                deg = memberships.get(inp, {}).get(label, 0.0)
                firing = min(firing, deg)

            strengths[i] = firing
            if firing < 1e-6:
                continue

            # Clip the consequent MF at the firing strength
            a, b, c = OUTPUT_MFS[rule["consequent"]]
            clipped_mf = np.array([
                min(firing, trimf(float(x), a, b, c))
                for x in self.universe
            ])
            aggregated = np.maximum(aggregated, clipped_mf)

        return aggregated, strengths

    # Defuzzification (centroid)

    def _defuzzify(self, aggregated: np.ndarray) -> float:
        """
        Centroid defuzzification: x* = Σ(x · μ(x)) / Σ(μ(x))

        Falls back to 0.5 if the aggregated MF is zero everywhere.
        """
        total = np.sum(aggregated)
        if total < 1e-9:
            return 0.5  # uncertain — return mid-range
        return float(np.sum(self.universe * aggregated) / total)

    # Colour mapping

    @staticmethod
    def _colour(R_total: float) -> str:
        if R_total < COLOUR_THRESHOLDS["Amber"][0]:
            return "Green"
        elif R_total < COLOUR_THRESHOLDS["Red"][0]:
            return "Amber"
        else:
            return "Red"

    # Reason string

    def _dominant_reason(self, strengths: Dict[int, float]) -> str:
        if not strengths:
            return "No rule fired"
        best_idx = max(
            strengths,
            key=lambda i: (strengths[i], OUTPUT_MFS[self.rules[i]["consequent"]][1]),
        )
        return self.rules[best_idx]["reason"]

    # Public interface

    def infer(
        self,
        R_IMU:      float,
        R_EMG:      float,
        z_sal:      float = 0.0,
        time_in_risk_zone: float = 0.0,
        ar:         float = 0.0,
        z_imu_mean: float = 0.0,
    ) -> dict:
        """
        Run the Mamdani FIS for a single window.

        Parameters
        ----------
        R_IMU      : RF predict_proba for IMU-only model [0, 1]
        R_EMG      : LR predict_proba for EMG-only model [0, 1]
        z_sal      : LDLJ smoothness z-score (baseline deviation; clipped ±3)
        time_in_risk_zone : fraction of window above angle-risk zone [0, 1]
        ar         : |emg_ai_ES| * 2, asymmetry ratio [0, 3]
        z_imu_mean : mean(|z_flex|, |z_vel|, |z_sal|), overall IMU deviation [0, 3]

        Returns
        -------
        dict with keys:
            R_IMU: input kinematic risk score
            R_EMG: input muscle risk score
            R_total: defuzzified total risk [0, 1]
            colour: "Green" / "Amber" / "Red"
            risk_level: "Safe" / "Cautious" / "Risky"
            reason: antecedent text of highest-firing rule
            rule_strengths: dict of rule index → firing strength
        """
        memberships       = self._fuzzify(
            R_IMU, R_EMG, z_sal, time_in_risk_zone, ar, z_imu_mean
        )
        aggregated, strengths = self._fire_rules(memberships)
        R_total           = self._defuzzify(aggregated)
        colour            = self._colour(R_total)
        risk_level        = risk_level_from_score(R_total)
        reason            = self._dominant_reason(strengths)

        return {
            "R_IMU":         float(R_IMU),
            "R_EMG":         float(R_EMG),
            "R_total":       round(R_total, 4),
            "colour":        colour,
            "risk_level":    risk_level,
            "reason":        reason,
            "rule_strengths": {str(i): round(s, 4) for i, s in strengths.items()},
        }

    def infer_batch(self, feature_df: "pd.DataFrame") -> "pd.DataFrame":
        """
        Run the FIS on every row of a feature DataFrame.

        Expects columns: R_IMU, R_EMG
        Optional columns: imu_z_ldlj, imu_time_in_risk_zone, emg_ai_ES,
        imu_z_flex, imu_z_vel

        Returns
        -------
        DataFrame with columns: R_total, colour, risk_level, reason
        """
        import pandas as pd

        def _ai_to_ar(ai_val):
            """Convert asymmetry index AI ∈ [-1,1] → AR ∈ [0,3]."""
            if pd.isna(ai_val):
                return 0.0
            return float(np.clip(abs(ai_val) * 2, 0.0, 3.0))

        def _mean_z(row):
            z_cols = ["imu_z_flex", "imu_z_vel", "imu_z_ldlj"]
            vals = [abs(row[c]) for c in z_cols if c in row.index and not pd.isna(row[c])]
            return float(np.mean(vals)) if vals else 0.0

        rows = []
        for _, row in feature_df.iterrows():
            R_IMU = float(row.get("R_IMU", 0.5))
            R_EMG = float(row.get("R_EMG", 0.5))
            z_sal = float(row["imu_z_ldlj"]) if "imu_z_ldlj" in row.index and not pd.isna(row["imu_z_ldlj"]) else 0.0
            time_in_risk_zone = float(row.get("imu_time_in_risk_zone", 0.0))
            ar    = _ai_to_ar(row.get("emg_ai_ES", 0.0))
            z_imu = _mean_z(row)

            result = self.infer(
                R_IMU, R_EMG, z_sal, time_in_risk_zone, ar, z_imu
            )
            rows.append({
                "R_total": result["R_total"],
                "colour":  result["colour"],
                "risk_level": result["risk_level"],
                "reason":  result["reason"],
            })
        return pd.DataFrame(rows, index=feature_df.index)


class IMUFallbackFIS(MamdaniFIS):
    """
    Mamdani FIS for the IMU-only fallback route.

    Route:
        IMU features -> RF -> R_IMU
        R_IMU + IMU smoothness/exposure/deviation features -> FIS risk output

    EMG and asymmetry are neutralised so missing EMG is not treated as
    measured evidence.
    """

    def __init__(self, resolution: int = 501):
        super().__init__(rules=FALLBACK_IMU_RULES, resolution=resolution)

    def infer(
        self,
        R_IMU: float,
        z_sal: float = 0.0,
        time_in_risk_zone: float = 0.0,
        z_imu_mean: float = 0.0,
    ) -> dict:
        result = super().infer(
            R_IMU=R_IMU,
            R_EMG=0.2,
            z_sal=z_sal,
            time_in_risk_zone=time_in_risk_zone,
            ar=0.0,
            z_imu_mean=z_imu_mean,
        )
        result["R_EMG"] = float("nan")
        result["route"] = "imu_only_fallback_rf_imu_mamdani_fis"
        return result


# UNIT TESTS, run via: python ml/fuzzy/mamdani_fis.py

def _run_unit_tests():
    """
    Verify the FIS produces expected output for canonical input combinations.
    Tests derived from the spec §7.5 rule base.
    """
    fis = MamdaniFIS()

    tests = [
        # (R_IMU, R_EMG, z_sal, time_in_risk_zone, ar, z_imu_mean, expected_colour, description)
        (0.85, 0.85,  0.5, 0.1, 0.3, 0.2,  "Red",   "R1: both HIGH -> Very High"),
        (0.85, 0.55,  0.5, 0.1, 0.3, 0.2,  "Red",   "R2: IMU HIGH + EMG MEDIUM -> High"),
        (0.85, 0.15, -2.5, 0.1, 0.3, 0.2,  "Red",   "R3: IMU HIGH + EMG LOW + Poor smooth"),
        (0.85, 0.30,  0.5, 0.1, 2.0, 0.2,  "Red",   "R4: IMU HIGH + high asymmetry"),
        (0.55, 0.85,  0.5, 0.1, 2.0, 0.2,  "Amber", "R5: IMU MED + EMG HIGH + asymmetry -> Amber/Red boundary"),
        (0.20, 0.20,  0.5, 0.1, 0.3, 0.2,  "Green", "R11: both LOW + Normal smooth -> safe"),
        (0.20, 0.55,  0.5, 0.1, 0.3, 0.2,  "Amber", "R10: IMU LOW + EMG MEDIUM -> monitor (Amber)"),
        (0.20, 0.20,  0.5, 0.9, 0.3, 0.2,  "Amber", "R14: sustained risk-zone exposure escalates output"),
        (0.55, 0.20, -2.5, 0.1, 0.3, 2.4,  "Amber", "R15/R16: large baseline deviation escalates output"),
    ]

    print("Running MamdaniFIS unit tests...\n")
    passed = 0
    for i, (rimu, remg, zsal, trisk, ar, zdev, exp_colour, desc) in enumerate(tests):
        result = fis.infer(rimu, remg, zsal, trisk, ar, zdev)
        ok = result["colour"] == exp_colour
        status = "PASS" if ok else "FAIL"
        print(
            f"  {status} Test {i+1}: {desc}\n"
            f"    R_total={result['R_total']:.3f}  colour={result['colour']}  "
            f"(expected {exp_colour})\n"
            f"    reason: {result['reason']}\n"
        )
        if ok:
            passed += 1

    baseline = fis.infer(0.55, 0.20, -2.5, 0.1, 0.3, 0.2)["R_total"]
    deviated = fis.infer(0.55, 0.20, -2.5, 0.1, 0.3, 2.4)["R_total"]
    influence_ok = deviated > baseline
    status = "PASS" if influence_ok else "FAIL"
    print(
        f"  {status} Test {len(tests) + 1}: baseline deviation influences R_total\n"
        f"    normal={baseline:.3f}  large_deviation={deviated:.3f}\n"
    )
    if influence_ok:
        passed += 1

    total = len(tests) + 1
    print(f"Results: {passed}/{total} passed")
    return passed == total


if __name__ == "__main__":
    success = _run_unit_tests()
    raise SystemExit(0 if success else 1)
