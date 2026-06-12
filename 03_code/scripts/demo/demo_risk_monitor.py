#!/usr/bin/env python3
"""
demo_risk_monitor.py
====================
Spinal Movement Risk Monitor, FYP 2025/26

Real-time risk classification demo. Replays a session at 1 Hz (one window per
second) and displays a traffic-light risk output in the terminal and as a live
matplotlib figure.

Two modes
---------
    Replay (default)
        Loads a processed session directory (or runs the pipeline on a raw
        session), then steps through feature windows at 1 Hz. Use this for
        the FYP demo before hardware is available.

            py scripts/demo/demo_risk_monitor.py
            py scripts/demo/demo_risk_monitor.py --session data/synthetic/session_0001
            py scripts/demo/demo_risk_monitor.py --session data/synthetic/session_0001 --no_plot

    Live (future, requires hardware)
        Pass --live to enable the live pipeline mode. This will be wired to
        session_converter.py output once the Ganglion + ESP32 are connected.

            py scripts/demo/demo_risk_monitor.py --live --session data/real/processed/live_session

Output
------
    Terminal: coloured risk banner (Green / Amber / Red) + key feature values
    Plot:     scrolling timeline of risk probability and trunk angle

Requirements
------------
    pip install matplotlib joblib scikit-learn --break-system-packages
"""

import sys
import time
import argparse
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Project root on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# CONSTANTS

# Feature columns used by each condition (must match train_classifier.py)
IMU_FEATURES = [
    "imu_trunk_angle_peak", "imu_trunk_angle_mean",
    "imu_angvel_peak",      "imu_angvel_mean",
    "imu_time_in_risk_zone","imu_time_high_velocity",
    "imu_ldlj",              "imu_jerk_rms",
    "imu_jerk_peak",        "imu_ldlj_multiaxis",
    "imu_compensation_index","imu_lumbopelv_ratio",
    # Phase 3b additions, pelvis and lateral angle features
    "imu_pelvis_angle_peak","imu_pelvis_angle_mean",
    "imu_lat_angle_peak",   "imu_lat_angle_mean",
    "imu_z_flex",           "imu_z_vel",           "imu_z_ldlj",
]

FALLBACK_REDUCED_IMU_FEATURES = [
    "imu_angvel_peak",
    "imu_angvel_mean",
    "imu_time_high_velocity",
    "imu_ldlj",
    "imu_jerk_rms",
    "imu_jerk_peak",
    "imu_pelvis_angle_peak",
    "imu_pelvis_angle_mean",
    "imu_l3_accel_tilt_peak",
    "imu_l3_accel_tilt_mean",
    "imu_l3_accel_tilt_range",
    "imu_z_vel",
    "imu_z_ldlj",
]

EMG_FEATURES = [
    "emg_rms_LES", "emg_rms_RES", "emg_rms_LOBL", "emg_rms_ROBL",
    "emg_mav_LES", "emg_mav_RES", "emg_mav_LOBL", "emg_mav_ROBL",
    "emg_zcr_LES", "emg_zcr_RES", "emg_zcr_LOBL", "emg_zcr_ROBL",
    "emg_ai_ES",   "emg_ai_OBL",
    "emg_cai_ES",  "emg_cai_OBL",
    "emg_z_rms_r", "emg_z_ar",
]

IMU_EMG_FEATURES = IMU_FEATURES + EMG_FEATURES

# Risk probability thresholds for traffic light bands
THRESH_GREEN  = 0.35    # p_risk < 0.35  → Green  (safe)
THRESH_AMBER  = 0.65    # p_risk < 0.65  → Amber  (monitor)
                        # p_risk ≥ 0.65  → Red    (risky)

# ANSI colour codes for terminal output
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[42m\033[30m"   # black text on green background
_AMBER  = "\033[43m\033[30m"   # black text on amber background
_RED    = "\033[41m\033[97m"   # white text on red background
_GREY   = "\033[100m\033[37m"  # grey background


# MODEL LOADER

def load_models(models_dir: Path, operating_mode: str = "full_hybrid"):
    """
    Load the IMU-only RF and EMG-only LR models for FIS-based classification.

    The Mamdani FIS is the final decision layer:
        IMU-only RF  -> R_IMU  --.
                                  +--> Mamdani FIS --> R_total --> traffic light
        EMG-only LR  -> R_EMG  --'

    Returns
    -------
    imu_model    : fitted sklearn Pipeline (RF, IMU condition)
    imu_features : list of IMU feature column names
    emg_model    : fitted sklearn Pipeline (LR, EMG condition), or None
    emg_features : list of EMG feature column names
    """
    from joblib import load as joblib_load

    imu_model = None
    emg_model = None
    imu_features = IMU_FEATURES

    if operating_mode == "imu_only_fallback":
        metadata_path = models_dir / "fallback_model_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            imu_features = metadata.get("feature_columns", {}).get(
                "reduced_pelvis_l3", FALLBACK_REDUCED_IMU_FEATURES
            )
        for name in ["rf_reduced_pelvis_l3.joblib", "rf_primary_4imu.joblib"]:
            p = models_dir / name
            if p.exists():
                imu_model = joblib_load(p)
                print(f"  Loaded: {p.name}")
                break

    if imu_model is None:
        for fold in range(1, 6):
            p = models_dir / f"RF_IMU_fold{fold}.joblib"
            if p.exists():
                imu_model = joblib_load(p)
                print(f"  Loaded: {p.name}")
                break

    if operating_mode == "full_hybrid":
        for fold in range(1, 6):
            p = models_dir / f"LR_EMG_fold{fold}.joblib"
            if p.exists():
                emg_model = joblib_load(p)
                print(f"  Loaded: {p.name}")
                break

    if imu_model is None:
        raise FileNotFoundError(
            "No IMU RF model found. Run train_fallback_analysis_models.py for "
            "fallback mode, or train_classifier.py for full-hybrid mode."
        )

    if operating_mode == "full_hybrid" and emg_model is None:
        raise FileNotFoundError(
            "Full-hybrid demo requires LR_EMG models. Select --mode imu_only_fallback "
            "only when explicitly reporting fallback output."
        )

    return imu_model, imu_features, emg_model, EMG_FEATURES


def load_or_build_features(session_dir: Path) -> pd.DataFrame:
    """
    Return the feature matrix for a session directory.

    If feature_matrix.csv already exists, load it directly.
    Otherwise run the pipeline to generate it.
    """
    feat_path = session_dir / "feature_matrix.csv"
    if feat_path.exists():
        df = pd.read_csv(feat_path)
        print(f"  Feature matrix loaded: {feat_path.name}  ({len(df)} windows)")
        return df

    print(f"  No feature_matrix.csv found — running pipeline on {session_dir.name}...")
    from signal_processing.pipeline import run_pipeline
    df = run_pipeline(str(session_dir), output_dir=str(session_dir))
    print(f"  Pipeline complete: {len(df)} windows")
    return df


# RISK CLASSIFICATION

def classify_window(
    imu_model,
    imu_features: list,
    emg_model,
    emg_features: list,
    feature_row: pd.Series,
    operating_mode: str = "full_hybrid",
) -> dict:
    """
    Classify one window through the full FIS pipeline:

        IMU features -> IMU-only RF  -> R_IMU --.
                                                 +--> Mamdani FIS -> R_total -> label
        EMG features -> EMG-only LR  -> R_EMG --'
        (+ smoothness, time-in-risk-zone, asymmetry and baseline-deviation inputs)

    Returns
    -------
    dict with keys:
        p_risk      : R_total [0, 1]
        label       : "GREEN" | "AMBER" | "RED"
        predicted   : 0 (safe) or 1 (risky)
        R_IMU       : IMU-only RF probability
        R_EMG       : EMG-only LR probability (0.5 if no EMG model)
        fis_reason  : dominant rule text from FIS
    """
    import sys as _sys
    from pathlib import Path as _Path
    _repo = str(_Path(__file__).resolve().parents[2])
    if _repo not in _sys.path:
        _sys.path.insert(0, _repo)
    from ml.fuzzy.mamdani_fis import IMUFallbackFIS, MamdaniFIS, RISKY_THRESHOLD

    # R_IMU: IMU-only RF
    X_imu = feature_row[imu_features].to_frame().T.fillna(0.0)
    try:
        R_IMU = float(imu_model.predict_proba(X_imu)[0, 1])
    except Exception:
        R_IMU = float(imu_model.predict(X_imu)[0])

    z_sal      = float(np.clip(feature_row.get("imu_z_ldlj", 0.0), -3.0, 3.0))
    time_in_risk_zone = float(np.clip(feature_row.get("imu_time_in_risk_zone", 0.0), 0.0, 1.0))
    z_candidates = [
        abs(float(feature_row.get("imu_z_flex", 0.0))),
        abs(float(feature_row.get("imu_z_vel", 0.0))),
        abs(float(feature_row.get("imu_z_ldlj", 0.0))),
    ]
    z_imu_mean = float(np.clip(np.mean(z_candidates), 0.0, 3.0))

    if operating_mode == "imu_only_fallback":
        fis = IMUFallbackFIS()
        result = fis.infer(
            R_IMU=R_IMU,
            z_sal=z_sal,
            time_in_risk_zone=time_in_risk_zone,
            z_imu_mean=z_imu_mean,
        )
        R_total    = result["R_total"]
        fis_colour = result["colour"]
        risk_level = result["risk_level"]
        label_map  = {"Green": "GREEN", "Amber": "AMBER", "Red": "RED"}
        label      = label_map.get(fis_colour, "AMBER")
        return {
            "p_risk": R_total,
            "label": label,
            "risk_level": risk_level,
            "predicted": 1 if R_total >= RISKY_THRESHOLD else 0,
            "R_IMU": R_IMU,
            "R_EMG": float("nan"),
            "fis_reason": result.get("reason", ""),
            "operating_mode": "imu_only_fallback",
        }

    # R_EMG: EMG-only LR
    if emg_model is None:
        raise ValueError(
            "Full-hybrid classification requires an LR_EMG model. "
            "Use --mode imu_only_fallback only when explicitly reporting fallback output."
        )
    if emg_model is not None:
        X_emg = feature_row[emg_features].to_frame().T.fillna(0.0)
        try:
            R_EMG = float(emg_model.predict_proba(X_emg)[0, 1])
        except Exception:
            R_EMG = float(emg_model.predict(X_emg)[0])

    # Auxiliary FIS inputs
    ai_es      = float(feature_row.get("emg_ai_ES", 0.0))
    ar         = float(np.clip(abs(ai_es) * 2.0, 0.0, 3.0))

    # Mamdani FIS
    fis    = MamdaniFIS()
    result = fis.infer(
        R_IMU=R_IMU,
        R_EMG=R_EMG,
        z_sal=z_sal,
        time_in_risk_zone=time_in_risk_zone,
        ar=ar,
        z_imu_mean=z_imu_mean,
    )

    R_total    = result["R_total"]
    fis_colour = result["colour"]     # "Green" / "Amber" / "Red"
    risk_level = result["risk_level"]  # "Safe" / "Cautious" / "Risky"
    fis_reason = result.get("reason", "")

    # Normalise FIS colour to uppercase traffic-light label
    label_map  = {"Green": "GREEN", "Amber": "AMBER", "Red": "RED"}
    label      = label_map.get(fis_colour, "AMBER")

    predicted  = 1 if R_total >= RISKY_THRESHOLD else 0

    return {
        "p_risk":     R_total,
        "label":      label,
        "risk_level": risk_level,
        "predicted":  predicted,
        "R_IMU":      R_IMU,
        "R_EMG":      R_EMG,
        "fis_reason": fis_reason,
        "operating_mode": "full_hybrid",
    }


# TERMINAL DISPLAY

_BANNER = {
    "GREEN": f"{_GREEN}  ●  SAFE   {_RESET}",
    "AMBER": f"{_AMBER}  ●  MONITOR{_RESET}",
    "RED":   f"{_RED}  ●  RISKY  {_RESET}",
}

_TRAFFIC_LIGHT = {
    "GREEN": """
  ┌─────────┐
  │  ( )    │  ← safe
  │  [●]    │  green
  │  ( )    │
  └─────────┘""",
    "AMBER": """
  ┌─────────┐
  │  ( )    │
  │  [●]    │  amber
  │  ( )    │  ← monitor
  └─────────┘""",
    "RED": """
  ┌─────────┐
  │  [●]    │  ← RISKY
  │  ( )    │  red
  │  ( )    │
  └─────────┘""",
}


def _print_status(
    result: dict,
    window_idx: int,
    total_windows: int,
    feature_row: pd.Series,
    true_label: str = None,
) -> None:
    """Print one window result to terminal, including risk explanations."""
    import textwrap as _tw

    colour_map = {"GREEN": _GREEN, "AMBER": _AMBER, "RED": _RED}
    col = colour_map[result["label"]]

    # Clear and redraw
    print("\033[2J\033[H", end="")

    mode_str = "  REPLAY MODE  " if true_label is not None else "  LIVE MODE  "
    system_str = (
        "  IMU-ONLY FALLBACK  "
        if result.get("operating_mode") == "imu_only_fallback"
        else "  FULL HYBRID  "
    )
    print(
        f"\n  Spinal Movement Risk Monitor   "
        f"[Window {window_idx+1}/{total_windows}]  {mode_str}{system_str}\n"
    )

    # Traffic light symbol
    print(f"{col}{_BOLD}")
    for line in _TRAFFIC_LIGHT[result["label"]].splitlines():
        print(f"  {line}")
    print(f"{_RESET}")

    risk_label = result["label"]
    p_risk = result["p_risk"]
    print(f"\n  Risk level : {col}{_BOLD}  {risk_label}  {_RESET}   (p = {p_risk:.2f})\n")

    # Key features
    angle  = feature_row.get("imu_trunk_angle_peak",  float("nan"))
    vel    = feature_row.get("imu_angvel_peak",        float("nan"))
    ldlj   = feature_row.get("imu_ldlj",               float("nan"))
    jerk   = feature_row.get("imu_jerk_rms",           float("nan"))
    comp   = feature_row.get("imu_compensation_index", float("nan"))
    lp     = feature_row.get("imu_lumbopelv_ratio",    float("nan"))
    rms_l  = feature_row.get("emg_rms_LES",            float("nan"))
    rms_r  = feature_row.get("emg_rms_RES",            float("nan"))
    asym   = feature_row.get("emg_asymmetry",          float("nan"))

    angle_flag = "[!] >45 deg"    if angle > 45       else ""
    vel_flag   = "[!] >40 deg/s"  if vel   > 40       else ""
    jerk_flag  = "[!] high"       if jerk  > 120000   else ""
    lp_flag    = "[!] <0.45"      if lp    < 0.45     else ""
    comp_flag  = "[!] >0.60"      if comp  > 0.60     else ""
    asym_flag  = "[!] >0.20"      if asym  > 0.20     else ""

    print(f"  ---- Key features ----")
    print(f"  Trunk flexion   : {angle:>6.1f} deg    {angle_flag}")
    print(f"  Angular vel     : {vel:>6.1f} deg/s   {vel_flag}")
    print(f"  Movement jerk   : {jerk:>10.0f}    {jerk_flag}")
    print(f"  Smoothness LDLJ : {ldlj:>7.2f}")
    print(f"  Lumbo-pelv ratio: {lp:>5.2f}    {lp_flag}")
    print(f"  Compensation    : {comp:>5.2f}    {comp_flag}")
    if not np.isnan(rms_l):
        print(f"  EMG L-ES RMS    : {rms_l*1000:>6.3f} mV")
        print(f"  EMG R-ES RMS    : {rms_r*1000:>6.3f} mV")
    if not np.isnan(asym):
        print(f"  EMG asymmetry   : {asym:>5.2f}    {asym_flag}")

    # Risk explanation block (AMBER or RED only)
    if result["label"] in ("AMBER", "RED"):
        criteria_raw = str(feature_row.get("risk_criteria", ""))
        layman_raw   = str(feature_row.get("risk_layman",   ""))
        clinical_raw = str(feature_row.get("risk_clinical", ""))

        _CRITERION_LABELS = {
            "postural": "Excessive trunk flexion",
            "dynamic":  "High movement speed / jerk",
            "pattern":  "Poor lumbo-pelvic rhythm",
            "combined": "Combined lateral + sagittal loading",
        }

        if criteria_raw and criteria_raw not in ("", "none", "nan"):
            triggered  = [c.strip() for c in criteria_raw.split(",") if c.strip()]
            nice_names = [_CRITERION_LABELS.get(c, c) for c in triggered]

            print(f"\n  ---- Why it is risky ----")
            triggers_str = " + ".join(nice_names)
            print(f"  Triggers    : {triggers_str}")

            if layman_raw and layman_raw not in ("", "nan"):
                print(f"\n  Plain English:")
                for seg in layman_raw.split(" and "):
                    seg = seg.strip().capitalize()
                    if seg:
                        print(f"    - {seg}")

            if clinical_raw and clinical_raw not in ("", "nan"):
                print(f"\n  Clinical:")
                wrapped = _tw.fill(
                    clinical_raw, width=68,
                    initial_indent="    ", subsequent_indent="    ",
                )
                print(wrapped)

        # FIS dominant rule (from fuzzy inference)
        fis_reason = result.get('fis_reason', '')
        if fis_reason and fis_reason not in ('', 'nan'):
            print(f'\n  FIS rule   : {fis_reason}')
        # Component probabilities
        R_IMU = result.get('R_IMU', float('nan'))
        R_EMG = result.get('R_EMG', float('nan'))
        if result.get("operating_mode") == "imu_only_fallback":
            print(f'  IMU fallback: R_IMU={R_IMU:.3f}  -> FIS R_total={result["p_risk"]:.3f}')
        elif not np.isnan(R_IMU):
            print(f'  R_IMU={R_IMU:.3f}  R_EMG={R_EMG:.3f}  -> FIS R_total={result["p_risk"]:.3f}')

    if true_label is not None:
        rc    = feature_row.get("risk_class", -1)
        match = "OK" if (int(rc) == result["predicted"]) else "WRONG"
        mvt   = feature_row.get("movement_label", "?")
        print(f"\n  True label  : {mvt}  (risk={int(rc)})   Prediction: {match}")

    print(f"\n  ------------------------------------------")
    print(f"  Press Ctrl+C to stop\n")


def _setup_plot(window_size: int = 60):
    """Initialise the scrolling matplotlib figure. Returns (fig, axes, data_buf)."""
    import matplotlib
    matplotlib.use("TkAgg" if _tk_available() else "Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, (ax_risk, ax_angle) = plt.subplots(
        2, 1, figsize=(10, 5), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.5]},
    )
    fig.suptitle("Spinal Movement Risk Monitor", fontweight="bold", fontsize=13)
    fig.patch.set_facecolor("#1a1a2e")
    for ax in [ax_risk, ax_angle]:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")

    # Risk probability panel
    ax_risk.set_ylim(-0.05, 1.05)
    ax_risk.set_ylabel("Risk probability", color="white", fontsize=9)
    ax_risk.axhline(THRESH_GREEN, color="#4CAF50", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_risk.axhline(THRESH_AMBER, color="#FF9800", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_risk.fill_between([], [], alpha=0.15, color="#4CAF50")

    # Trunk angle panel
    ax_angle.axhline(45, color="#F44336", linewidth=0.8, linestyle="--", alpha=0.7,
                     label="45° threshold")
    ax_angle.set_ylabel("Trunk flexion (°)", color="white", fontsize=9)
    ax_angle.set_xlabel("Time (windows)", color="white", fontsize=9)

    # Legend patches
    patches = [
        mpatches.Patch(color="#4CAF50", label="Safe"),
        mpatches.Patch(color="#FF9800", label="Monitor"),
        mpatches.Patch(color="#F44336", label="Risky"),
    ]
    ax_risk.legend(handles=patches, loc="upper right", fontsize=7,
                   facecolor="#16213e", labelcolor="white")

    buf = {"p_risk": [], "angle": [], "colour": []}
    plt.tight_layout()
    plt.ion()
    plt.show()

    return fig, (ax_risk, ax_angle), buf


def _tk_available() -> bool:
    try:
        import tkinter  # noqa: F401
        return True
    except ImportError:
        return False


def _update_plot(fig, axes, buf: dict, result: dict, angle: float) -> None:
    """Append one window and redraw."""
    try:
        import matplotlib.pyplot as plt

        ax_risk, ax_angle = axes
        colour_map = {"GREEN": "#4CAF50", "AMBER": "#FF9800", "RED": "#F44336"}

        buf["p_risk"].append(result["p_risk"])
        buf["angle"].append(max(angle, 0))
        buf["colour"].append(colour_map[result["label"]])

        x = list(range(len(buf["p_risk"])))

        # Risk panel
        ax_risk.clear()
        ax_risk.set_ylim(-0.05, 1.05)
        ax_risk.set_ylabel("Risk probability", color="white", fontsize=9)
        ax_risk.axhline(THRESH_GREEN, color="#4CAF50", linewidth=0.8, linestyle="--", alpha=0.7)
        ax_risk.axhline(THRESH_AMBER, color="#FF9800", linewidth=0.8, linestyle="--", alpha=0.7)
        for ax in axes:
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#444")

        # Filled risk bars coloured by band
        for i, (xi, pi, ci) in enumerate(zip(x, buf["p_risk"], buf["colour"])):
            ax_risk.bar(xi, pi, color=ci, alpha=0.75, width=0.8)

        # Angle panel
        ax_angle.clear()
        ax_angle.set_facecolor("#16213e")
        ax_angle.tick_params(colors="white")
        ax_angle.spines[:].set_color("#444")
        ax_angle.set_ylabel("Trunk flexion (°)", color="white", fontsize=9)
        ax_angle.set_xlabel("Window", color="white", fontsize=9)
        ax_angle.axhline(45, color="#F44336", linewidth=0.8, linestyle="--", alpha=0.7)
        ax_angle.fill_between(x, 0, buf["angle"], alpha=0.4, color="#42a5f5")
        ax_angle.plot(x, buf["angle"], color="#90caf9", linewidth=1.2)

        fig.canvas.draw_idle()
        plt.pause(0.001)
    except Exception:
        pass   # plotting is best-effort; never crash the demo over a plot error


# REPLAY DEMO

def run_replay(
    session_dir: Path,
    models_dir: Path,
    condition: str = "IMU_EMG",
    classifier: str = "RF",
    replay_speed: float = 1.0,
    show_plot: bool = True,
    loop: bool = False,
    operating_mode: str = "full_hybrid",
) -> None:
    """
    Replay a processed session at real-time speed (1 Hz window rate).

    Parameters
    ----------
    session_dir   : path to session directory (with imu_data.csv, emg_data.csv)
    models_dir    : path to ml/models/
    condition     : feature condition to classify with ("IMU", "EMG", "IMU_EMG")
    classifier    : model type ("RF", "SVM", "LDA", "LR")
    replay_speed  : multiplier for replay speed (2.0 = 2× faster)
    show_plot     : display scrolling matplotlib window
    loop          : if True, loop the session continuously
    """
    print(f"\n{'='*55}")
    print(f"Risk Monitor — Replay Mode")
    print(f"  Session    : {session_dir}")
    print(f"  Condition  : {condition}  ({classifier})")
    print(f"  Speed      : {replay_speed}×")
    print(f"  System mode: {operating_mode}")
    print(f"{'='*55}\n")

    # Load models (IMU RF + EMG LR) for FIS pipeline
    print("Loading models...")
    imu_model, imu_features, emg_model, emg_features = load_models(models_dir, operating_mode)

    # Load or build feature matrix
    print("Loading feature matrix...")
    feat_df = load_or_build_features(session_dir)
    available_emg_features = [column for column in emg_features if column in feat_df.columns]
    emg_available = bool(available_emg_features) and not feat_df[available_emg_features].isna().all().all()
    if operating_mode == "full_hybrid" and not emg_available:
        raise ValueError(
            "EMG features are unavailable for a full-hybrid demo. "
            "Re-run explicitly with --mode imu_only_fallback to report contingency output."
        )
    if operating_mode == "imu_only_fallback":
        print("  [MODE] IMU-only fallback enabled: RF_IMU + IMU Mamdani FIS.")

    # Drop unlabelled windows (risk_class == -1)
    labelled = feat_df[feat_df["risk_class"] != -1].reset_index(drop=True)
    if len(labelled) == 0:
        print("[WARNING] No labelled windows found. Running on all windows.")
        labelled = feat_df.reset_index(drop=True)

    # Fill NaN features with 0
    all_feat_cols = list(dict.fromkeys(imu_features + emg_features))
    for col in all_feat_cols:
        if col in labelled.columns:
            labelled[col] = labelled[col].fillna(0.0)

    print(f"  {len(labelled)} windows ready.\n")

    # Optional plot setup
    fig = ax_pair = buf = None
    if show_plot:
        try:
            fig, ax_pair, buf = _setup_plot()
        except Exception as e:
            print(f"  [WARNING] Could not initialise plot: {e}")
            print(f"  Run with --no_plot to suppress this warning.")
            show_plot = False

    # Replay loop
    delay_s = 1.0 / replay_speed

    try:
        while True:
            for idx, row in labelled.iterrows():
                t0 = time.time()

                result = classify_window(
                    imu_model, imu_features, emg_model, emg_features, row, operating_mode
                )
                angle  = float(row.get("imu_trunk_angle_peak", 0.0))

                _print_status(result, idx, len(labelled), row,
                               true_label=str(row.get("movement_label", "?")))

                if show_plot and fig is not None:
                    _update_plot(fig, ax_pair, buf, result, angle)

                # Pace to match real-time window rate
                elapsed = time.time() - t0
                sleep_s = max(0.0, delay_s - elapsed)
                time.sleep(sleep_s)

            if not loop:
                break

            print("\n  [Loop] Replaying from start...\n")
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n\n  Demo stopped.\n")

    # Summary
    if len(labelled) > 0:
        _print_summary(labelled, imu_model, imu_features, emg_model, emg_features, operating_mode)


def _print_summary(
    feat_df: pd.DataFrame,
    imu_model,
    imu_features: list,
    emg_model,
    emg_features: list,
    operating_mode: str = "full_hybrid",
) -> None:
    """Print final Mamdani FIS risk-output summary after replay."""
    from sklearn.metrics import classification_report

    y_true = feat_df["risk_class"].astype(int)
    y_pred = np.array([
        classify_window(imu_model, imu_features, emg_model, emg_features, row, operating_mode)["predicted"]
        for _, row in feat_df.iterrows()
    ])

    valid = y_true != -1
    if valid.sum() == 0:
        return

    title = "Final hybrid FIS risk summary" if operating_mode == "full_hybrid" else "IMU-only fallback risk summary"
    print(f"\n  --- {title} ---")
    print(classification_report(
        y_true[valid], y_pred[valid],
        target_names=["Safe (0)", "Risky (1)"],
        digits=3,
    ))


# LIVE MODE STUB (hardware not yet connected)

def run_live(session_dir: Path, models_dir: Path, **kwargs) -> None:
    """
    Live risk classification from a continuously-updated session directory.

    Polls the feature_matrix.csv in session_dir for new rows
    (written by the pipeline as data arrives) and classifies each new window.

    Status: STUB, requires the following to be wired up first:
        1. ganglion_stream.py running in parallel → EMG CSV
        2. record_imu_serial.py running in parallel → IMU CSV
        3. A streaming version of session_converter.py + pipeline (TBD)

    For now, prints instructions and falls back to replay mode if a
    feature_matrix.csv already exists in session_dir.
    """
    print("\n  [LIVE MODE]")
    print("  Live classification requires the Ganglion + ESP32 to be connected")
    print("  and session_converter.py to be running in streaming mode (not yet built).")
    print()
    print("  To run a live demo:")
    print("    Terminal 1: py scripts/acquisition/ganglion_stream.py --port COM_EMG --duration 120 \\")
    print("                    --out data/real/raw/live/ganglion.csv")
    print("    Terminal 2: py scripts/acquisition/record_imu_serial.py --port COM_IMU --duration 120 \\")
    print("                    --out data/real/raw/live/imu_arduino.csv")
    print("    Terminal 3: py scripts/conversion/session_converter.py --ganglion data/real/raw/live/ganglion.csv \\")
    print("                    --imu data/real/raw/live/imu_arduino.csv --out data/real/processed/live")
    print("    Terminal 4: py scripts/demo/demo_risk_monitor.py --live --session data/real/processed/live")
    print()
    print("  For now, falling back to replay on the default synthetic session...")
    time.sleep(2)

    fallback = session_dir if (session_dir / "imu_data.csv").exists() \
               else _REPO_ROOT / "data" / "synthetic" / "session_0001"
    run_replay(fallback, models_dir, **kwargs)


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spinal Movement Risk Monitor — real-time demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--session", type=str,
        default=str(_REPO_ROOT / "data" / "synthetic" / "session_0001"),
        help="Session directory to replay (or live session directory with --live)",
    )
    parser.add_argument(
        "--models_dir", type=str,
        default=str(_REPO_ROOT / "ml" / "models"),
        help="Directory containing trained .joblib models",
    )
    parser.add_argument(
        "--condition", type=str, default="IMU_EMG",
        choices=["IMU", "EMG", "IMU_EMG"],
        help="Feature condition: IMU-only, EMG-only, or fused",
    )
    parser.add_argument(
        "--mode", choices=["full_hybrid", "imu_only_fallback"], default="full_hybrid",
        help="Run the primary full-hybrid system or explicitly reported IMU-only fallback.",
    )
    parser.add_argument(
        "--classifier", type=str, default="RF",
        choices=["RF", "SVM", "LDA", "LR"],
        help="Classifier to use",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Replay speed multiplier (e.g. 2.0 = 2× faster than real-time)",
    )
    parser.add_argument(
        "--no_plot", action="store_true",
        help="Disable matplotlib scrolling window",
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Loop the session continuously",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Live mode (requires hardware — see instructions in script)",
    )
    args = parser.parse_args()

    session_dir = Path(args.session)
    models_dir  = Path(args.models_dir)

    if args.live:
        run_live(
            session_dir  = session_dir,
            models_dir   = models_dir,
            condition    = args.condition,
            classifier   = args.classifier,
            replay_speed = args.speed,
            show_plot    = not args.no_plot,
            loop         = args.loop,
            operating_mode= args.mode,
        )
    else:
        run_replay(
            session_dir  = session_dir,
            models_dir   = models_dir,
            condition    = args.condition,
            classifier   = args.classifier,
            replay_speed = args.speed,
            show_plot    = not args.no_plot,
            loop         = args.loop,
            operating_mode= args.mode,
        )
