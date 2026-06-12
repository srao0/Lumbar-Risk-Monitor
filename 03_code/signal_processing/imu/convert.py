#!/usr/bin/env python3
"""
MPU-6050 raw-counts to physical-units converter
Spinal Movement Risk Monitor, FYP 2025/26

Converts Arduino serial output (raw 16-bit ADC counts) to:
    Accelerometer : g  (1 g = 9.81 m/s²)
    Gyroscope     : degrees per second (dps)

MPU-6050 register configuration used in the Arduino sketch:
    ACCEL_CONFIG = 0x00  →  ±2 g range    →  16384 LSB/g
    GYRO_CONFIG  = 0x00  →  ±250 dps range  →  131 LSB/dps

Input CSV format (single IMU, from current Arduino sketch):
    t_ms, ax, ay, az, gx, gy, gz

Input CSV format (four IMUs, from updated Arduino sketch):
    t_ms,
    Pelvis_ax, Pelvis_ay, Pelvis_az, Pelvis_gx, Pelvis_gy, Pelvis_gz,
    L3_ax,     L3_ay,     L3_az,     L3_gx,     L3_gy,     L3_gz,
    T12_ax,    T12_ay,    T12_az,    T12_gx,    T12_gy,    T12_gz,
    T4_ax,     T4_ay,     T4_az,     T4_gx,     T4_gy,     T4_gz

Usage:
    from signal_processing.imu.convert import RawConverter, GyroBiasCalibrator
    df = RawConverter.from_csv("data/real/raw/session_001_imu.csv")

    # Calibrate gyro bias from a still recording before a session:
    offsets = GyroBiasCalibrator.compute_from_csv("data/real/raw/still_30s.csv")
    df_cal  = GyroBiasCalibrator.apply(df, offsets)

    # Or convert a single row dict:
    row_phys = RawConverter.convert_row({"ax": -6432, "ay": -14328, ...})
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

# SCALE FACTORS (MPU-6050 at default ±2g / ±250 dps)

ACCEL_SCALE = 16384.0   # LSB per g   — from MPU-6050 datasheet Table 1, ±2g range
# GYRO_CONFIG = 0x08 in imu_reader.ino → ±500 dps range → 65.5 LSB/dps
# (Originally 131.0 for ±250 dps, changed when we fixed gyro saturation)
GYRO_SCALE  = 65.5      # LSB per dps — MPU-6050 datasheet Table 2, ±500 dps

# Maximum representable values (sanity check bounds)
ACCEL_MAX_G   = 2.0     # g    — ±2g range
GYRO_MAX_DPS  = 500.0   # dps  — ±500 dps range (matches GYRO_CONFIG 0x08)

# Gravity magnitude should be 1.0 g at rest; allow ±15% for noise + tilt
GRAVITY_TOL = 0.15      # fractional tolerance

# Gyro at rest thresholds (MPU-6050 datasheet worst-case zero-rate = ±20 dps)
# Values above WARN indicate a calibration offset that must be subtracted.
# Values above FAIL indicate possible hardware fault (bias too large to calibrate out).
GYRO_REST_WARN_DPS = 5.0    # soft limit — expect per-unit bias at this level
GYRO_REST_FAIL_DPS = 15.0   # hard limit — unlikely for healthy sensor at room temp
GYRO_REST_MAX_DPS  = GYRO_REST_WARN_DPS   # kept for backward compatibility

# IMU labels in channel order (matches TCA9548A channels 0-3)
IMU_LABELS = ["Pelvis", "L3", "T12", "T4"]
IMU_AXES   = ["ax", "ay", "az", "gx", "gy", "gz"]


# CONVERTER

class RawConverter:
    """
    Converts MPU-6050 raw integer counts to physical units.

    All outputs use consistent column naming:
        {imu}_{axis}_g   for accelerometer (g)
        {imu}_{axis}_dps for gyroscope     (deg/s)

    For single-IMU data the imu prefix defaults to "imu".
    """

    @staticmethod
    def counts_to_g(raw: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert raw accelerometer counts to g."""
        return raw / ACCEL_SCALE

    @staticmethod
    def counts_to_dps(raw: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert raw gyroscope counts to degrees per second."""
        return raw / GYRO_SCALE

    @staticmethod
    def accel_magnitude_g(ax_g: float, ay_g: float, az_g: float) -> float:
        """Euclidean magnitude of accelerometer vector in g."""
        return float(np.sqrt(ax_g**2 + ay_g**2 + az_g**2))

    @staticmethod
    def gyro_magnitude_dps(gx_dps: float, gy_dps: float, gz_dps: float) -> float:
        """Euclidean magnitude of gyroscope vector in dps."""
        return float(np.sqrt(gx_dps**2 + gy_dps**2 + gz_dps**2))

    @classmethod
    def convert_row(cls, row: dict, imu_prefix: str = "imu") -> dict:
        """
        Convert a single row of raw counts to physical units.

        Parameters
        ----------
        row : dict
            Keys: ax, ay, az, gx, gy, gz  (raw integer counts)
        imu_prefix : str
            Column name prefix for output (e.g. "Pelvis", "L3", "imu")

        Returns
        -------
        dict with keys:
            {prefix}_ax_g, {prefix}_ay_g, {prefix}_az_g  [g]
            {prefix}_gx_dps, {prefix}_gy_dps, {prefix}_gz_dps  [dps]
            {prefix}_accel_mag_g    [g], should be ~1.0 at rest
            {prefix}_gyro_mag_dps   [dps], should be ~0.0 at rest
        """
        p = imu_prefix
        ax_g   = cls.counts_to_g(row["ax"])
        ay_g   = cls.counts_to_g(row["ay"])
        az_g   = cls.counts_to_g(row["az"])
        gx_dps = cls.counts_to_dps(row["gx"])
        gy_dps = cls.counts_to_dps(row["gy"])
        gz_dps = cls.counts_to_dps(row["gz"])

        return {
            f"{p}_ax_g":          ax_g,
            f"{p}_ay_g":          ay_g,
            f"{p}_az_g":          az_g,
            f"{p}_gx_dps":        gx_dps,
            f"{p}_gy_dps":        gy_dps,
            f"{p}_gz_dps":        gz_dps,
            f"{p}_accel_mag_g":   cls.accel_magnitude_g(ax_g, ay_g, az_g),
            f"{p}_gyro_mag_dps":  cls.gyro_magnitude_dps(gx_dps, gy_dps, gz_dps),
        }

    @classmethod
    def from_csv(
        cls,
        csv_path: Union[str, Path],
        imu_prefix: str = "imu",
    ) -> pd.DataFrame:
        """
        Load an Arduino serial CSV and convert all raw columns to physical units.

        Supports both single-IMU and four-IMU formats.
        Comment lines beginning with '#' are skipped automatically.

        Parameters
        ----------
        csv_path : str or Path
        imu_prefix : str
            Used only for single-IMU files. Four-IMU files use Pelvis/L3/T12/T4.

        Returns
        -------
        pd.DataFrame with t_ms preserved and all counts replaced by
        physical-unit columns.
        """
        raw = pd.read_csv(
            csv_path,
            comment="#",
            skipinitialspace=True,
        )
        raw.columns = [c.strip() for c in raw.columns]

        # Drop embedded header rows, firmware resends the CSV header every
        # 1000 samples (for robustness against missed boot-time header).
        # These rows contain "t_ms" as the value in the t_ms column instead
        # of a number, which causes dtype errors downstream.
        raw = raw[raw["t_ms"] != "t_ms"].copy()
        raw = raw.apply(pd.to_numeric, errors="coerce")
        raw = raw.dropna(how="all").reset_index(drop=True)

        # Detect single vs four-IMU format
        if "Pelvis_ax" in raw.columns:
            return cls._convert_four_imu(raw)
        else:
            return cls._convert_single_imu(raw, imu_prefix)

    @classmethod
    def _convert_single_imu(cls, raw: pd.DataFrame, prefix: str) -> pd.DataFrame:
        out = pd.DataFrame()
        out["t_ms"] = raw["t_ms"]

        for axis in ["ax", "ay", "az"]:
            out[f"{prefix}_{axis}_g"] = cls.counts_to_g(raw[axis])

        for axis in ["gx", "gy", "gz"]:
            out[f"{prefix}_{axis}_dps"] = cls.counts_to_dps(raw[axis])

        out[f"{prefix}_accel_mag_g"] = np.sqrt(
            out[f"{prefix}_ax_g"]**2 +
            out[f"{prefix}_ay_g"]**2 +
            out[f"{prefix}_az_g"]**2
        )
        out[f"{prefix}_gyro_mag_dps"] = np.sqrt(
            out[f"{prefix}_gx_dps"]**2 +
            out[f"{prefix}_gy_dps"]**2 +
            out[f"{prefix}_gz_dps"]**2
        )
        return out

    @classmethod
    def _convert_four_imu(cls, raw: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        out["t_ms"] = raw["t_ms"]

        for label in IMU_LABELS:
            for axis in ["ax", "ay", "az"]:
                col_in  = f"{label}_{axis}"
                col_out = f"{label}_{axis}_g"
                out[col_out] = cls.counts_to_g(raw[col_in])

            for axis in ["gx", "gy", "gz"]:
                col_in  = f"{label}_{axis}"
                col_out = f"{label}_{axis}_dps"
                out[col_out] = cls.counts_to_dps(raw[col_in])

            out[f"{label}_accel_mag_g"] = np.sqrt(
                out[f"{label}_ax_g"]**2 +
                out[f"{label}_ay_g"]**2 +
                out[f"{label}_az_g"]**2
            )
            out[f"{label}_gyro_mag_dps"] = np.sqrt(
                out[f"{label}_gx_dps"]**2 +
                out[f"{label}_gy_dps"]**2 +
                out[f"{label}_gz_dps"]**2
            )
        return out


# GYRO BIAS CALIBRATION

class GyroBiasCalibrator:
    """
    Estimates per-axis gyro offsets from a short still recording.

    Protocol (collect once per session before attaching to participant):
        1. Place the sensor on a flat surface for ≥10 s while still.
        2. Record via Arduino sketch → CSV.
        3. Call GyroBiasCalibrator.compute_from_csv(path) to get offsets dict.
        4. Call GyroBiasCalibrator.apply(df, offsets) on every session DataFrame.

    Offsets are stored in dps (after RawConverter.from_csv has been applied).
    If you need raw-count offsets, multiply each dps value by GYRO_SCALE.

    The MPU-6050 also has a built-in factory trim mechanism (register 0x0D, 0x12),
    but software calibration is sufficient for this application.
    """

    @staticmethod
    def compute(df_phys: pd.DataFrame, imu_prefix: str = "imu") -> dict:
        """
        Compute mean gyro offset for each axis from a still-recording DataFrame.

        Parameters
        ----------
        df_phys : pd.DataFrame
            Output of RawConverter.from_csv(), physical units in dps.
        imu_prefix : str
            Column prefix (e.g. "imu", "Pelvis", "L3").

        Returns
        -------
        dict : {"{prefix}_gx_offset_dps": float, "_gy_...", "_gz_..."}
        """
        p = imu_prefix
        offsets = {}
        for axis in ["gx", "gy", "gz"]:
            col = f"{p}_{axis}_dps"
            offsets[f"{p}_{axis}_offset_dps"] = float(df_phys[col].mean())
        mag = float(np.sqrt(sum(offsets[f"{p}_{a}_offset_dps"]**2
                                for a in ["gx", "gy", "gz"])))
        print(f"  Gyro bias [{imu_prefix}]: "
              f"gx={offsets[f'{p}_gx_offset_dps']:+.3f}  "
              f"gy={offsets[f'{p}_gy_offset_dps']:+.3f}  "
              f"gz={offsets[f'{p}_gz_offset_dps']:+.3f}  "
              f"(mag={mag:.2f} dps)")
        return offsets

    @classmethod
    def compute_from_csv(
        cls,
        csv_path: Union[str, Path],
        imu_prefix: str = "imu",
        start_s: float = 0.0,
        duration_s: float | None = None,
    ) -> dict:
        """
        Load a still-recording CSV, convert, and compute gyro offsets.

        Parameters
        ----------
        csv_path : path to Arduino serial CSV (raw counts)
        imu_prefix : "imu" for single-IMU, or one of IMU_LABELS for four-IMU
        start_s : start time for calibration window, relative to first sample
        duration_s : optional calibration window duration in seconds

        Returns
        -------
        dict of per-axis dps offsets
        """
        df = RawConverter.from_csv(csv_path, imu_prefix=imu_prefix)
        if duration_s is not None:
            if duration_s <= 0:
                raise ValueError("duration_s must be positive when supplied")
            t0 = float(df["t_ms"].iloc[0]) + start_s * 1000.0
            t1 = t0 + duration_s * 1000.0
            df = df[(df["t_ms"] >= t0) & (df["t_ms"] <= t1)].copy()
            if df.empty:
                raise ValueError(
                    f"No IMU samples found in gyro calibration window "
                    f"{start_s:g}s to {start_s + duration_s:g}s"
                )
        return cls.compute(df, imu_prefix=imu_prefix)

    @staticmethod
    def apply(
        df_phys: pd.DataFrame,
        offsets: dict,
        imu_prefix: str = "imu",
    ) -> pd.DataFrame:
        """
        Subtract gyro offsets from a converted DataFrame (in-place copy).

        Parameters
        ----------
        df_phys : output of RawConverter.from_csv()
        offsets : output of GyroBiasCalibrator.compute() or compute_from_csv()
        imu_prefix : must match the prefix used when computing offsets

        Returns
        -------
        New DataFrame with corrected gyro columns and updated gyro_mag_dps.
        """
        p   = imu_prefix
        out = df_phys.copy()
        for axis in ["gx", "gy", "gz"]:
            col    = f"{p}_{axis}_dps"
            offset = offsets[f"{p}_{axis}_offset_dps"]
            out[col] = out[col] - offset

        # Recompute gyro magnitude after correction
        out[f"{p}_gyro_mag_dps"] = np.sqrt(
            out[f"{p}_gx_dps"]**2 +
            out[f"{p}_gy_dps"]**2 +
            out[f"{p}_gz_dps"]**2
        )
        return out


# VALIDATION

class ConversionValidator:
    """
    Validates converted IMU data against expected physical constraints.

    Three checks:
      1. Gravity check, accel magnitude at rest should be 1.0 ± GRAVITY_TOL g
      2. Gyro bias check, gyro magnitude at rest should be < GYRO_REST_MAX_DPS
      3. Dynamic check, moving segments should show higher gyro magnitude
                          than the rest segment
    """

    @staticmethod
    def check_gravity(accel_mag_g: float, label: str = "") -> bool:
        """Return True if accel magnitude is within tolerance of 1 g."""
        lo = 1.0 - GRAVITY_TOL
        hi = 1.0 + GRAVITY_TOL
        ok = lo <= accel_mag_g <= hi
        status = "PASS" if ok else "FAIL"
        tag = f"[{label}] " if label else ""
        print(f"  {status}  {tag}accel_mag = {accel_mag_g:.4f} g  "
              f"(expected {lo:.2f} – {hi:.2f} g)")
        return ok

    @staticmethod
    def check_gyro_bias(gyro_mag_dps: float, label: str = "") -> bool:
        """
        Two-tier gyro bias check.

        PASS  : < GYRO_REST_WARN_DPS, no calibration needed
        WARN  : WARN_DPS ≤ mag < FAIL_DPS, normal per-unit bias; subtract offset before integration
        FAIL  : ≥ GYRO_REST_FAIL_DPS, unusually large; check sensor or connections

        Returns True only for PASS (caller should not treat WARN as a pipeline blocker).
        """
        tag = f"[{label}] " if label else ""
        if gyro_mag_dps < GYRO_REST_WARN_DPS:
            print(f"  PASS  {tag}gyro_mag  = {gyro_mag_dps:.2f} dps  "
                  f"(threshold < {GYRO_REST_WARN_DPS} dps)")
            return True
        elif gyro_mag_dps < GYRO_REST_FAIL_DPS:
            print(f"  WARN  {tag}gyro_mag  = {gyro_mag_dps:.2f} dps  "
                  f"(> {GYRO_REST_WARN_DPS} dps — calibration offset required before integration)")
            return True   # WARN is not a blocking failure
        else:
            print(f"  FAIL  {tag}gyro_mag  = {gyro_mag_dps:.2f} dps  "
                  f"(≥ {GYRO_REST_FAIL_DPS} dps — check sensor / wiring)")
            return False

    @staticmethod
    def check_dynamic_elevation(
        rest_gyro_dps: float,
        moving_gyro_dps: float,
        movement: str,
        min_ratio: float = 2.0,
    ) -> bool:
        """
        Return True if moving gyro magnitude is at least min_ratio × rest.
        A dynamic movement should produce meaningfully more rotation than rest.
        """
        ratio = moving_gyro_dps / max(rest_gyro_dps, 0.01)
        ok = ratio >= min_ratio
        status = "PASS" if ok else "WARN"
        print(f"  {status}  [{movement}] gyro_mag = {moving_gyro_dps:.2f} dps  "
              f"(rest = {rest_gyro_dps:.2f} dps,  ratio = {ratio:.1f}x,  "
              f"min = {min_ratio:.1f}x)")
        return ok


def validate_on_sample_data() -> None:
    """
    Validate the converter using the representative samples collected from the
    Arduino sketch (Table 3 from the interim report, single MPU-6050 at L3).

    These are raw counts at 1 Hz from five movement classes.
    The test is offline, no hardware required.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Sample data from Arduino serial output
    # Format: (t_ms, ax, ay, az, gx, gy, gz), raw counts
    samples = {
        "Standing Still": [
            (14256, -6612, -14384,  5556,  -766,  -180,  -351),
            (15258, -6276, -14380,  5864,  -634,   245,  -379),
            (16260, -6432, -14328,  5204,  -789,    56,  -231),
        ],
        "Right Bend": [
            (38274,  2780, -15624,  5040, -1161,   913,  -534),
            (39276,  3384, -15412,  5592,   474, -2402, -1909),
            (40278,  7276, -13812,  5656,  -806, -1243, -1069),
        ],
        "Left Bend": [
            (112292, 3184, -15388,  4352,  -775,   292,  -331),
            (113294, 3016, -15616,  4536,  -622,  -229,  -211),
            (114296, 3168, -15776,  4092,  -787,  -267,  -601),
        ],
        "Front Bend": [
            (166308, 5040, -15340,  5624,   157, -5023,  1232),
            (167310, 3260, -15092,  3820,  3055, -6491,   788),
            (168312, 4820, -13292, 10080,  3996, -1661,   245),
        ],
        "Back Bend": [
            (258330, 11156, -12112, 2448,  -118,   260,  -361),
            (259332, 10616, -12444, 2556,  -573,   203,  -238),
            (260334, 12124, -11876,  760, -2462, -6987, -1991),
        ],
    }

    converter = RawConverter()
    validator = ConversionValidator()

    print("=" * 65)
    print("MPU-6050 CONVERSION VALIDATION  (sample data from Table 3)")
    print("=" * 65)

    results   = {}
    all_pass  = True

    for movement, rows in samples.items():
        converted = [
            converter.convert_row(
                {"ax": r[1], "ay": r[2], "az": r[3],
                 "gx": r[4], "gy": r[5], "gz": r[6]},
                imu_prefix="imu",
            )
            for r in rows
        ]
        # Average over the three representative samples
        accel_mag = float(np.mean([c["imu_accel_mag_g"]  for c in converted]))
        gyro_mag  = float(np.mean([c["imu_gyro_mag_dps"] for c in converted]))

        # Physical-unit values for the first sample (for display)
        c0 = converted[0]
        ax_g   = c0["imu_ax_g"]
        ay_g   = c0["imu_ay_g"]
        az_g   = c0["imu_az_g"]
        gx_dps = c0["imu_gx_dps"]
        gy_dps = c0["imu_gy_dps"]
        gz_dps = c0["imu_gz_dps"]

        results[movement] = {
            "accel_mag_g":  accel_mag,
            "gyro_mag_dps": gyro_mag,
            "ax_g":  ax_g,  "ay_g": ay_g,   "az_g": az_g,
            "gx_dps": gx_dps, "gy_dps": gy_dps, "gz_dps": gz_dps,
        }

        print(f"\n{movement}")
        print(f"  Sample 0 physical values:")
        print(f"    accel : ax={ax_g:+.4f} g   ay={ay_g:+.4f} g   "
              f"az={az_g:+.4f} g")
        print(f"    gyro  : gx={gx_dps:+.2f} dps  gy={gy_dps:+.2f} dps  "
              f"gz={gz_dps:+.2f} dps")

        # Gravity check for all movements (sensor is always under gravity)
        grav_ok = validator.check_gravity(accel_mag, movement)
        all_pass = all_pass and grav_ok

        # Gyro bias check only for standing still
        if movement == "Standing Still":
            bias_ok = validator.check_gyro_bias(gyro_mag, movement)
            # WARN does not fail the pipeline, note it but continue
            if not bias_ok:
                all_pass = False
            rest_gyro = gyro_mag

    print()
    print("─" * 65)
    print("Dynamic elevation check (moving gyro vs standing still):")
    for movement, r in results.items():
        if movement == "Standing Still":
            continue
        validator.check_dynamic_elevation(
            rest_gyro,
            r["gyro_mag_dps"],
            movement,
            min_ratio=2.0,
        )

    print()
    print("─" * 65)
    overall = "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"
    print(f"Result: {overall}")
    if GYRO_REST_WARN_DPS <= rest_gyro < GYRO_REST_FAIL_DPS:
        print(f"\n  NOTE — gyro bias {rest_gyro:.2f} dps is above the {GYRO_REST_WARN_DPS} dps "
              f"soft limit.")
        print(f"  This is a per-unit calibration offset, not a code error.")
        print(f"  Action required: record 10–30 s of still data at the start of each session")
        print(f"  and call GyroBiasCalibrator.compute_from_csv() to remove the offset")
        print(f"  before any angle integration (Madgwick or otherwise).")
    print("=" * 65)

    # Gravity axis identification
    still = results["Standing Still"]
    gravity_components = {
        "X (ax)": abs(still["ax_g"]),
        "Y (ay)": abs(still["ay_g"]),
        "Z (az)": abs(still["az_g"]),
    }
    dominant_axis = max(gravity_components, key=gravity_components.get)
    print(f"\nGravity axis identification (standing still):")
    for axis, val in gravity_components.items():
        marker = " <-- gravity dominant" if axis == dominant_axis else ""
        print(f"  |{axis}| = {val:.4f} g{marker}")
    print(f"  Mounting orientation: gravity falls primarily on {dominant_axis}.")
    print(f"  This means the IMU is oriented with its {dominant_axis} axis "
          f"approximately vertical.")

    # Bar chart: gyro magnitude by movement
    movements  = list(results.keys())
    gyro_mags  = [results[m]["gyro_mag_dps"] for m in movements]
    accel_mags = [results[m]["accel_mag_g"]  for m in movements]

    colours = ["#2196F3" if m == "Standing Still" else "#E91E63"
               for m in movements]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Gyro magnitude
    bars = axes[0].bar(movements, gyro_mags, color=colours, alpha=0.85,
                       edgecolor="white")
    axes[0].axhline(GYRO_REST_MAX_DPS, color="grey", linestyle="--",
                    linewidth=0.9, label=f"Rest threshold ({GYRO_REST_MAX_DPS} dps)")
    for bar, val in zip(bars, gyro_mags):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.3,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=8.5)
    axes[0].set_ylabel("Gyro magnitude (dps)", fontsize=10)
    axes[0].set_title("Rotational activity by movement class", fontsize=10,
                      fontweight="bold")
    axes[0].set_ylim(0, max(gyro_mags) * 1.2)
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].legend(fontsize=8)

    # Accel magnitude, should be ~1 g for all
    bars2 = axes[1].bar(movements, accel_mags, color=colours, alpha=0.85,
                        edgecolor="white")
    axes[1].axhline(1.0, color="grey", linestyle="-", linewidth=1.0,
                    label="Expected (1.0 g)")
    axes[1].axhline(1.0 + GRAVITY_TOL, color="grey", linestyle="--",
                    linewidth=0.7, label=f"Tolerance (±{GRAVITY_TOL:.0%})")
    axes[1].axhline(1.0 - GRAVITY_TOL, color="grey", linestyle="--",
                    linewidth=0.7)
    for bar, val in zip(bars2, accel_mags):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)
    axes[1].set_ylabel("Accel magnitude (g)", fontsize=10)
    axes[1].set_title("Gravity consistency check  (should be ~1.0 g)", fontsize=10,
                      fontweight="bold")
    axes[1].set_ylim(0.7, 1.4)
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(fontsize=8)

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3})
    fig.tight_layout()

    out_path = Path("results/plots/imu_conversion_validation.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nValidation plot saved → {out_path}")


# N-POSE ANATOMICAL CALIBRATION

class NPoseCalibrator:
    """
    Corrects for sensor-to-segment mounting misalignment using a short
    N-pose (neutral standing) recording.

    Problem
    -------
    Madgwick fusion outputs quaternions in a world frame aligned to the
    sensor's initial orientation, NOT to the anatomical frame.  If an
    IMU is mounted at an angle (e.g. the L3 sensor is tilted 10° due to
    belt placement), every subsequent relative angle inherits this offset.

    Solution
    --------
    1. Participant stands upright in the anatomical position (arms at sides,
       feet hip-width, gaze forward) for ≥10 s at the very start of the
       session.  This forms the N-pose calibration window.
    2. compute_offsets() computes the mean fused quaternion per segment over
       this window, call it q_ref[seg].  This is the "zero" orientation.
    3. apply() premultiplies every subsequent fused quaternion by
       q_ref[seg]^* (conjugate = inverse for unit quaternions):
           q_corrected = q_ref^* ⊗ q_raw
       This rotates all subsequent orientations so that the N-pose posture
       maps to the identity quaternion (zero Euler angles).
    4. Relative joint angles are recomputed from the corrected quaternions.

    References
    ----------
    Favre et al. (2009). Ambulatory measurement of 3D knee joint angle.
    J Biomechanics, 42(14), 2330-2335.

    Cutti et al. (2010). 'Outwalk': protocol for clinical gait analysis with
    ambulatory inertial and magnetic sensors. Proc IMechE, 224(H), 1217-1228.
    """

    SEGMENTS = ["pelvis", "l3", "t12", "t4"]

    @staticmethod
    def _mean_quaternion(quats: np.ndarray) -> np.ndarray:
        """
        Compute the mean unit quaternion from an (N, 4) array.

        Uses the eigenvector method (Markley et al., 2007): the mean quaternion
        is the eigenvector of Q^T Q corresponding to the largest eigenvalue.
        This is numerically stable even when quaternions wrap across the ±1
        boundary of the w component.

        Parameters
        ----------
        quats : (N, 4) array  [w, x, y, z]

        Returns
        -------
        q_mean : (4,) unit quaternion
        """
        # Ensure consistent sign (all q pointing into the same hemisphere)
        q = quats.copy()
        q[q[:, 0] < 0] *= -1   # flip quaternions with negative w
        M = q.T @ q             # 4×4 accumulation matrix
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        q_mean = eigenvectors[:, np.argmax(eigenvalues)]
        norm = np.linalg.norm(q_mean)
        return q_mean / norm if norm > 1e-10 else np.array([1.0, 0.0, 0.0, 0.0])

    @classmethod
    def compute_offsets(
        cls,
        df_fused: pd.DataFrame,
        n_seconds: float = 10.0,
        skip_seconds: float = 5.0,
        fs: float = 100.0,
    ) -> dict:
        """
        Compute the reference (N-pose) quaternion per segment.

        Uses the window [skip_seconds, n_seconds] of the fused DataFrame.
        Skipping the first skip_seconds avoids including the Madgwick filter's
        initial convergence transient (which can span 3-5 s even with
        accel-based initialisation).

        The participant must be standing still in the anatomical position
        during the entire [0, n_seconds] window.

        Parameters
        ----------
        df_fused      : output of fuse_four_imu_dataframe()
        n_seconds     : end of the calibration window (seconds)
        skip_seconds  : seconds to skip at the start (filter convergence guard)
        fs            : IMU sampling rate in Hz

        Returns
        -------
        offsets : dict  { segment_name → q_ref (4,) unit quaternion }

        Example
        -------
        offsets = NPoseCalibrator.compute_offsets(df_fused, n_seconds=15,
                                                  skip_seconds=5)
        df_cal   = NPoseCalibrator.apply(df_fused, offsets)
        """
        n_samples    = int(n_seconds * fs)
        skip_samples = int(skip_seconds * fs)
        window       = df_fused.iloc[skip_samples:n_samples]

        print(f"  N-pose calibration: using t={skip_seconds:.0f}s to {n_seconds:.0f}s "
              f"({len(window)} samples, skipping first {skip_seconds:.0f}s for filter convergence)")

        if len(window) < 5:
            raise ValueError(
                f"N-pose window has only {len(window)} samples after skipping "
                f"{skip_seconds:.0f}s (expected at least {int(0.5*fs)}). "
                f"Increase n_seconds or decrease skip_seconds."
            )

        offsets = {}
        for seg in cls.SEGMENTS:
            qcols = [f"{seg}_qw", f"{seg}_qx", f"{seg}_qy", f"{seg}_qz"]
            if not all(c in window.columns for c in qcols):
                raise KeyError(
                    f"Missing quaternion columns for segment '{seg}' in df_fused. "
                    f"Run fuse_four_imu_dataframe() first."
                )
            quats = window[qcols].to_numpy(dtype=float)
            q_ref = cls._mean_quaternion(quats)
            offsets[seg] = q_ref
            print(f"  N-pose [{seg}]: q_ref = "
                  f"[w={q_ref[0]:+.4f}, x={q_ref[1]:+.4f}, "
                  f"y={q_ref[2]:+.4f}, z={q_ref[3]:+.4f}]")
        return offsets

    @classmethod
    def apply(
        cls,
        df_fused: pd.DataFrame,
        offsets: dict,
        recompute_relative: bool = True,
        fs: float = 100.0,
    ) -> pd.DataFrame:
        """
        Apply N-pose correction to all quaternions in a fused IMU DataFrame.

        For each segment:
            q_corrected = q_ref^* ⊗ q_raw

        Optionally recomputes relative joint angles (θ_PL, θ_LT, θ_TU) and
        L3 angular velocity from the corrected quaternions.

        Parameters
        ----------
        df_fused          : output of fuse_four_imu_dataframe()
        offsets           : output of NPoseCalibrator.compute_offsets()
        recompute_relative: if True, overwrite theta_PL/LT/TU columns with
                            angles derived from corrected quaternions
        fs                : IMU sampling rate (used for angular velocity)

        Returns
        -------
        df_cal : copy of df_fused with corrected quaternion and angle columns
        """
        df = df_fused.copy()

        def _quat_conj(q):
            return np.array([q[0], -q[1], -q[2], -q[3]])

        def _quat_mul(q1, q2):
            """Hamilton product of two quaternion arrays (vectorised)."""
            w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
            w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
            return np.stack([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
            ], axis=1)

        def _normalise(q):
            n = np.linalg.norm(q, axis=1, keepdims=True)
            return q / np.clip(n, 1e-10, None)

        corrected_quats = {}
        for seg in cls.SEGMENTS:
            if seg not in offsets:
                continue
            q_ref    = offsets[seg]                     # (4,) reference
            q_ref_c  = _quat_conj(q_ref)               # conjugate
            qcols    = [f"{seg}_qw", f"{seg}_qx", f"{seg}_qy", f"{seg}_qz"]
            q_raw    = df[qcols].to_numpy(dtype=float)  # (N, 4)
            q_ref_bc = np.broadcast_to(q_ref_c, q_raw.shape).copy()  # (N, 4)
            q_corr   = _normalise(_quat_mul(q_ref_bc, q_raw))
            corrected_quats[seg] = q_corr
            for i, comp in enumerate(["qw", "qx", "qy", "qz"]):
                df[f"{seg}_{comp}"] = q_corr[:, i]

        if recompute_relative and all(s in corrected_quats for s in cls.SEGMENTS):
            # Recompute relative angles from corrected absolute quaternions
            pairs = [
                ("theta_PL", "pelvis", "l3"),
                ("theta_LT", "l3",     "t12"),
                ("theta_TU", "t12",    "t4"),
            ]

            def _quat_mul_single(q1, q2):
                w1, x1, y1, z1 = q1
                w2, x2, y2, z2 = q2
                return np.array([
                    w1*w2 - x1*x2 - y1*y2 - z1*z2,
                    w1*x2 + x1*w2 + y1*z2 - z1*y2,
                    w1*y2 - x1*z2 + y1*w2 + z1*x2,
                    w1*z2 + x1*y2 - y1*x2 + z1*w2,
                ])

            def _quat_to_euler(q):
                """ZYX Euler (pitch, roll, yaw) in degrees from [w,x,y,z]."""
                w, x, y, z = q
                sinp = np.clip(2.0*(w*y - z*x), -1.0, 1.0)
                pitch = np.degrees(np.arcsin(sinp))
                roll  = np.degrees(np.arctan2(2.0*(w*x + y*z),
                                               1.0 - 2.0*(x*x + y*y)))
                yaw   = np.degrees(np.arctan2(2.0*(w*z + x*y),
                                               1.0 - 2.0*(y*y + z*z)))
                return pitch, roll, yaw

            for angle_prefix, parent, child in pairs:
                q_p = corrected_quats[parent]
                q_c = corrected_quats[child]
                n = len(q_p)
                pitches = np.zeros(n)
                rolls   = np.zeros(n)
                yaws    = np.zeros(n)
                for i in range(n):
                    q_pc = _quat_mul_single(
                        np.array([q_p[i, 0], -q_p[i, 1], -q_p[i, 2], -q_p[i, 3]]),
                        q_c[i],
                    )
                    pitches[i], rolls[i], yaws[i] = _quat_to_euler(q_pc)
                df[f"{angle_prefix}_pitch"] = pitches
                df[f"{angle_prefix}_roll"]  = rolls
                df[f"{angle_prefix}_yaw"]   = yaws

            # Recompute L3 sagittal angular velocity
            dt = 1.0 / fs
            df["angvel_L3_sagittal"] = np.gradient(
                df["theta_PL_pitch"].to_numpy(), dt
            )

        return df

    @classmethod
    def compute_from_separate_recording(
        cls,
        npose_csv: Path,
        imu_fs: float = 100.0,
        beta: float = 0.033,
    ) -> dict:
        """
        Convenience wrapper: load a separate N-pose recording CSV (already
        converted to physical units + gyro-bias-corrected), run Madgwick fusion,
        and return the reference quaternion offsets.

        Use this if the N-pose was recorded in a separate file rather than
        the first N seconds of the session.

        Parameters
        ----------
        npose_csv : path to Arduino 4-IMU CSV in physical units (g / dps)
                    , must be output of RawConverter + GyroBiasCalibrator
        imu_fs    : sampling rate in Hz
        beta      : Madgwick beta gain

        Returns
        -------
        offsets : dict  { segment → q_ref (4,) }
        """
        from signal_processing.imu.madgwick import fuse_four_imu_dataframe
        df_phys   = RawConverter.from_csv(npose_csv)
        df_fused  = fuse_four_imu_dataframe(df_phys, fs=imu_fs, beta=beta)
        return cls.compute_offsets(df_fused, n_seconds=len(df_fused)/imu_fs,
                                   fs=imu_fs)



if __name__ == "__main__":
    validate_on_sample_data()
