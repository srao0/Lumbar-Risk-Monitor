"""
Window Size Validation
======================
Empirically compares candidate window sizes on synthetic data using
Fisher's discriminant ratio (FDR) and system latency.

Is an auxiliary design check for the current deployed feature
pipeline. The main pipeline uses a 2000 ms window, 1000 ms step, 100 Hz IMU,
and 200 Hz Ganglion EMG. It does not compute MPF/MDF spectral EMG features.

Window candidates
-----------------
    500 ms   --  50 IMU samples, 100 EMG samples. Low latency but short for
                trunk-movement smoothness and full bend/return cycles.
    1000 ms  --  100 IMU samples, 200 EMG samples. Faster feedback but may
                capture only part of slower functional movements.
    2000 ms  --  200 IMU samples, 400 EMG samples. Current pipeline setting:
                captures complete movement cycles with 1 Hz feedback updates.
    3000 ms  --  300 IMU samples, 600 EMG samples. More context but higher
                latency and less responsive feedback.

Evaluation metric: Fisher's Discriminant Ratio (FDR)
------------------------------------------------------
    FDR = (μ₁ - μ₀)² / (σ₁² + σ₀²)

    Higher FDR → better class separability for a given feature.
    Computed per feature and averaged across features for each window size.

    This is a univariate proxy for separability, not a final classifier
    metric.  Its purpose is to verify that the chosen window size preserves
    class-discriminating information.

References
----------
    Nazmi, N. et al. (2016) "A review of classification techniques of EMG
    signals during isotonic and isometric contractions." Sensors, 16(8), 1304.

    Phinyomark, A. et al. (2012) "EMG feature evaluation for movement control
    of upper limb prostheses." Expert Systems with Applications, 39(12).
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from typing import Optional

# Add project root so relative imports work when run as a script
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from signal_processing.pipeline import run_pipeline


# Configuration

WINDOW_SIZES_MS = [500, 1000, 2000, 3000]
STEP_RATIO      = 0.5         # 50% overlap for all sizes
IMU_FS          = 100.0
EMG_FS          = 200.0

# Features to evaluate (subset that are meaningful across all window sizes)
EVAL_FEATURES = [
    "imu_trunk_angle_peak",
    "imu_angvel_peak",
    "imu_time_in_risk_zone",
    "imu_ldlj",
    "emg_rms_LES",
    "emg_rms_RES",
    "emg_ai_ES",
]


# Fisher's Discriminant Ratio

def fisher_discriminant_ratio(
    values: np.ndarray,
    labels: np.ndarray,
    class_0: int = 0,
    class_1: int = 1,
) -> float:
    """
    Compute the Fisher discriminant ratio between two classes.

        FDR = (μ₁ - μ₀)² / (σ₁² + σ₀²)

    Parameters
    ----------
    values  : (N,) feature values
    labels  : (N,) class labels
    class_0 : label for negative class
    class_1 : label for positive class

    Returns
    -------
    fdr : float (NaN if a class has zero variance or fewer than 2 samples)
    """
    mask_0 = labels == class_0
    mask_1 = labels == class_1
    v0 = values[mask_0 & ~np.isnan(values)]
    v1 = values[mask_1 & ~np.isnan(values)]

    if len(v0) < 2 or len(v1) < 2:
        return np.nan

    mu0, mu1 = np.mean(v0), np.mean(v1)
    s0,  s1  = np.var(v0),  np.var(v1)

    denom = s0 + s1
    if denom < 1e-12:
        return np.nan

    return float((mu1 - mu0) ** 2 / denom)


# Per-window-size evaluation

def evaluate_window_size(
    session_dir: str,
    window_ms: int,
    features: list = EVAL_FEATURES,
) -> dict:
    """
    Run the pipeline with the given window size and compute FDR for each feature.

    Parameters
    ----------
    session_dir : path to one complete session folder
    window_ms   : window duration to test
    features    : list of feature column names to evaluate

    Returns
    -------
    dict with:
        window_ms   : int
        step_ms     : int
        n_windows   : number of windows produced
        latency_ms  : window_ms (maximum latency for real-time output)
        fdr_<feat>  : per-feature FDR
        mean_fdr    : average FDR across evaluated features
    """
    step_ms = max(1, int(window_ms * STEP_RATIO))

    feature_df = run_pipeline(
        session_dir=session_dir,
        output_dir=None,
        apply_notch=True,
        window_ms=window_ms,
        step_ms=step_ms,
        imu_fs=IMU_FS,
        emg_fs=EMG_FS,
    )

    # Only evaluate windows with binary labels (0 or 1)
    binary_mask = feature_df["risk_class"].isin([0, 1])
    df_binary = feature_df[binary_mask].copy()

    result = {
        "window_ms":  window_ms,
        "step_ms":    step_ms,
        "n_windows":  len(feature_df),
        "n_labelled": int(binary_mask.sum()),
        "latency_ms": window_ms,
    }

    fdr_values = []
    for feat in features:
        if feat not in df_binary.columns:
            result[f"fdr_{feat}"] = np.nan
            continue
        fdr = fisher_discriminant_ratio(
            df_binary[feat].to_numpy(dtype=float),
            df_binary["risk_class"].to_numpy(dtype=int),
        )
        result[f"fdr_{feat}"] = round(fdr, 4) if not np.isnan(fdr) else np.nan
        if not np.isnan(fdr):
            fdr_values.append(fdr)

    result["mean_fdr"] = round(float(np.mean(fdr_values)), 4) if fdr_values else np.nan
    return result


# Validation runner

def run_window_validation(
    synthetic_data_dir: str,
    output_path: Optional[str] = None,
    n_sessions: int = 3,
) -> pd.DataFrame:
    """
    Run window validation across multiple sessions and all candidate sizes.

    Results are aggregated (mean ± std) across sessions.

    Parameters
    ----------
    synthetic_data_dir : root directory with session subdirectories
    output_path        : where to save validation_results.csv
    n_sessions         : maximum number of sessions to use

    Returns
    -------
    summary_df : DataFrame with one row per window size
    """
    data_dir = Path(synthetic_data_dir)
    session_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir()
            and (d / "imu_data.csv").exists()
            and (d / "emg_data.csv").exists()
    ])[:n_sessions]

    if not session_dirs:
        raise FileNotFoundError(f"No complete session directories in {data_dir}")

    print(f"Window validation — {len(session_dirs)} session(s), "
          f"{len(WINDOW_SIZES_MS)} window sizes\n")

    all_results = []
    for sd in session_dirs:
        for wms in WINDOW_SIZES_MS:
            print(f"  {sd.name}  window={wms} ms ...", end="  ", flush=True)
            res = evaluate_window_size(str(sd), wms)
            res["session"] = sd.name
            all_results.append(res)
            print(f"mean_fdr={res['mean_fdr']:.3f}  n_windows={res['n_windows']}")

    results_df = pd.DataFrame(all_results)

    # Aggregate across sessions
    numeric_cols = [c for c in results_df.columns
                    if c not in ("session", "window_ms", "step_ms")]
    agg = results_df.groupby("window_ms")[numeric_cols].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()

    # Print summary table
    print("\n" + "=" * 60)
    print("WINDOW VALIDATION SUMMARY")
    print("=" * 60)
    print(f"{'Window':>10}  {'Latency':>10}  {'Mean FDR':>10}  {'±std':>8}")
    print("-" * 45)
    for _, row in agg.iterrows():
        wms  = int(row["window_ms"])
        lat  = int(wms)
        mfdr = row["mean_fdr_mean"]
        sfdr = row["mean_fdr_std"]
        flag = "  <-- CURRENT PIPELINE" if wms == 2000 else ""
        print(f"{wms:>8} ms  {lat:>8} ms  {mfdr:>10.3f}  ±{sfdr:.3f}{flag}")
    print("=" * 60)

    # Frequency resolution check
    print("\nEffective EMG bin width if spectral checks are done offline:")
    for wms in WINDOW_SIZES_MS:
        n_emg = int(wms * EMG_FS / 1000)
        df_emg = EMG_FS / n_emg
        print(f"  {wms:4d} ms  EMG df = {df_emg:.1f} Hz at {EMG_FS:.0f} Hz")

    # Save
    if output_path is None:
        output_path = str(data_dir / "window_validation_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nFull results saved → {output_path}")

    return agg


# Recommendation summary

def print_recommendation(summary_df: pd.DataFrame) -> None:
    """
    Print a brief, report-ready justification for the recommended window size.
    """
    best_row = summary_df.loc[summary_df["mean_fdr_mean"].idxmax()]
    rec_wms  = int(best_row["window_ms"])

    print("\n" + "─" * 60)
    print("DESIGN RECOMMENDATION")
    print("─" * 60)
    print(f"Highest mean Fisher discriminant ratio: {rec_wms} ms window")
    print(f"  Mean FDR  : {best_row['mean_fdr_mean']:.3f}")
    print(f"  Latency   : {rec_wms} ms")
    print()
    print("Justification:")
    print(f"  A {rec_wms} ms window provides sufficient temporal resolution")
    print(f"  for trunk kinematics at 100 Hz IMU sampling "
          f"({int(rec_wms * IMU_FS / 1000)} samples)")
    print( "  and uses only time-domain EMG features, matching the 200 Hz")
    print( "  Ganglion acquisition constraint.")
    print( "  With 50% overlap, the feature refresh rate is")
    print(f"  {1000 / (rec_wms * 0.5):.1f} Hz — sufficient for real-time feedback.")
    print()
    if rec_wms != 2000:
        print(f"  NOTE: The deployed pipeline currently uses 2000 ms windows.")
        print(f"  Treat this empirical result ({rec_wms} ms) as a design check,")
        print(f"  not as a report claim until verified on real participant data.")
    print("─" * 60)


# CLI entry point

if __name__ == "__main__":
    import argparse
    from typing import Optional

    parser = argparse.ArgumentParser(
        description="Validate window size choice for the lumbar movement risk pipeline."
    )
    parser.add_argument(
        "--data_dir",
        default="data/synthetic",
        help="Root directory of synthetic session data.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save window_validation_results.csv.",
    )
    parser.add_argument(
        "--n_sessions",
        type=int,
        default=3,
        help="Number of sessions to use (default: 3).",
    )
    args = parser.parse_args()

    summary = run_window_validation(
        synthetic_data_dir=args.data_dir,
        output_path=args.output,
        n_sessions=args.n_sessions,
    )
    print_recommendation(summary)
