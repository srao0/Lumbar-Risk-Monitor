#!/usr/bin/env python3
"""
replay_recorded_session.py
==========================
Spinal Movement Risk Monitor — FYP 2025/26 | Imperial College London

End-to-end *offline replay* of a recorded session, producing the artefacts the
Streamlit replay dashboard (scripts/replay_dashboard.py) renders for the final
demonstration.

Pipeline (all stages reuse existing project code — nothing is re-implemented):

    raw session  ──(session_converter.convert_session, run_pipeline=True)──▶
        imu_data.csv / emg_data.csv / labels.csv / session_metadata.json
        + feature_matrix.csv
    feature_matrix.csv ──(demo_risk_monitor.classify_window + Mamdani FIS)──▶
        per-window risk classification (R_IMU, R_EMG, R_total, traffic light)
    each window ──(ml.explainability.replay_explainer.explain_window)──▶
        engineering_reason + layman_reason

Outputs written into --out_session:
    replay_predictions.csv   one row per window (scores, labels, explanations)
    replay_summary.json      session-level summary + preflight + provenance
    replay_timeline.png      risk timeline figure (unless --no_plot)

Modes
-----
    full_hybrid        IMU + EMG. Raw EMG is REQUIRED. Default models: ml/models
    imu_only_fallback  IMU only, first-class mode (never silently pretends to be
                       hybrid). Default models: ml/models/fallback_final

Usage
-----
    py scripts/replay_recorded_session.py \
        --raw_session data/real/raw/participant_11/session_001 \
        --out_session data/real/replay_full_hybrid/participant_11/session_001 \
        --mode full_hybrid --emg_board cyton
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project root on sys.path ────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse existing model-loading, classification and explanation code.
# NOTE: scripts.conversion.session_converter (which imports SciPy) is
# intentionally NOT imported here. It is imported lazily inside
# ensure_processed_session() only on the branch that actually performs
# conversion, so cached replay (where feature_matrix.csv already exists) runs
# without SciPy installed.
from scripts.demo.demo_risk_monitor import (  # noqa: E402
    load_models,
    load_or_build_features,
    classify_window,
    IMU_FEATURES,
    EMG_FEATURES,
)
from ml.explainability.replay_explainer import explain_window, emg_features_valid  # noqa: E402

try:
    from signal_processing.pipeline import WINDOW_MS as _WINDOW_MS
except Exception:  # pragma: no cover
    _WINDOW_MS = 2000


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODELS = {
    "full_hybrid": _REPO_ROOT / "ml" / "models",
    "imu_only_fallback": _REPO_ROOT / "ml" / "models" / "fallback_final",
}

RAW_IMU_NAME = "imu_arduino.csv"
RAW_EMG_NAMES = {"cyton": "cyton.csv", "ganglion": "ganglion.csv"}
RAW_LABELS_NAME = "labels.csv"
RAW_SYNC_NAME = "session_sync_metadata.json"

# Key feature columns surfaced in replay_predictions.csv
KEY_IMU_COLS = [
    "imu_trunk_angle_peak", "imu_angvel_peak", "imu_time_in_risk_zone",
    "imu_ldlj", "imu_z_ldlj", "imu_jerk_rms",
    "imu_pelvis_angle_peak", "imu_compensation_index", "imu_lumbopelv_ratio",
    "imu_z_flex", "imu_z_vel",
]
KEY_EMG_COLS = [
    "emg_rms_LES", "emg_rms_RES", "emg_rms_LOBL", "emg_rms_ROBL",
    "emg_ai_ES", "emg_ai_OBL",
]

RISK_LEVELS = ["Safe", "Cautious", "Risky"]


# ─────────────────────────────────────────────────────────────────────────────
# PREFLIGHT
# ─────────────────────────────────────────────────────────────────────────────

def preflight(raw_session: Path, mode: str, emg_board: str) -> dict:
    """
    Inspect a raw session directory and verify the inputs each mode requires.

    Returns a dict:
        status        : "ok" | "error"
        emg_board     : resolved board ("cyton"/"ganglion"/None)
        raw_imu       : path or None
        raw_emg       : path or None
        raw_labels    : path or None
        raw_sync      : path or None
        warnings      : list[str]
        errors        : list[str]
    """
    warnings: list[str] = []
    errors: list[str] = []

    raw_session = Path(raw_session)
    if not raw_session.is_dir():
        return {
            "status": "error", "emg_board": None, "raw_imu": None, "raw_emg": None,
            "raw_labels": None, "raw_sync": None, "warnings": warnings,
            "errors": [f"Raw session directory not found: {raw_session}"],
        }

    # IMU is mandatory in every mode (IMU is the baseline system).
    imu_path = raw_session / RAW_IMU_NAME
    if not imu_path.exists():
        errors.append(f"Required raw IMU file missing: {RAW_IMU_NAME}")
        imu_path = None

    # Resolve the EMG board / file.
    cyton = raw_session / RAW_EMG_NAMES["cyton"]
    ganglion = raw_session / RAW_EMG_NAMES["ganglion"]
    resolved_board = None
    emg_path = None
    if emg_board == "auto":
        if cyton.exists() and ganglion.exists():
            resolved_board, emg_path = "cyton", cyton
            warnings.append("Both cyton.csv and ganglion.csv present; auto-selected cyton.csv.")
        elif cyton.exists():
            resolved_board, emg_path = "cyton", cyton
        elif ganglion.exists():
            resolved_board, emg_path = "ganglion", ganglion
    else:
        candidate = raw_session / RAW_EMG_NAMES[emg_board]
        if candidate.exists():
            resolved_board, emg_path = emg_board, candidate
        else:
            # Tolerate a board mismatch: if the other board's file exists, note it.
            other = "ganglion" if emg_board == "cyton" else "cyton"
            other_path = raw_session / RAW_EMG_NAMES[other]
            if other_path.exists():
                warnings.append(
                    f"--emg_board {emg_board} requested but {RAW_EMG_NAMES[emg_board]} "
                    f"not found; {RAW_EMG_NAMES[other]} is present. Using --emg_board "
                    f"{emg_board} will fail — re-run with --emg_board {other} or auto."
                )

    # Mode-specific EMG requirement.
    if mode == "full_hybrid":
        if emg_path is None:
            errors.append(
                "full_hybrid requires raw EMG (cyton.csv or ganglion.csv), but none was "
                "found/selected. Re-run with --mode imu_only_fallback to report contingency "
                "output, or supply the correct --emg_board."
            )
    else:  # imu_only_fallback
        if emg_path is not None:
            warnings.append(
                "imu_only_fallback mode: raw EMG is present but will be ignored for "
                "inference (decision uses movement features only)."
            )
        # In fallback mode EMG is intentionally not used.
        resolved_board = resolved_board if emg_path else None

    # Labels + sync metadata are recommended, not strictly required.
    labels_path = raw_session / RAW_LABELS_NAME
    if not labels_path.exists():
        warnings.append(
            f"{RAW_LABELS_NAME} not found — converter will create stub labels and the "
            f"session will be treated as non-official (no true_risk_class)."
        )
        labels_path = None
    sync_path = raw_session / RAW_SYNC_NAME
    if not sync_path.exists():
        warnings.append(f"{RAW_SYNC_NAME} not found — continuing without sync metadata.")
        sync_path = None

    status = "error" if errors else "ok"
    return {
        "status": status,
        "emg_board": resolved_board,
        "raw_imu": str(imu_path) if imu_path else None,
        "raw_emg": str(emg_path) if emg_path else None,
        "raw_labels": str(labels_path) if labels_path else None,
        "raw_sync": str(sync_path) if sync_path else None,
        "warnings": warnings,
        "errors": errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONVERSION + FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def ensure_processed_session(
    raw_session: Path,
    out_session: Path,
    mode: str,
    pf: dict,
    force: bool,
) -> Path:
    """
    Run the existing converter (with run_pipeline=True) unless a feature matrix
    already exists and --force was not given. Returns out_session.
    """
    out_session = Path(out_session)
    feat_path = out_session / "feature_matrix.csv"

    if feat_path.exists() and not force:
        print(f"  feature_matrix.csv already present in {out_session} — skipping conversion "
              f"(use --force to rebuild).")
        return out_session

    out_session.mkdir(parents=True, exist_ok=True)

    # Lazy import: only needed when conversion actually runs. Keeping this out of
    # the module-level imports means cached replay does not require SciPy.
    from scripts.conversion.session_converter import convert_session, is_official_phase2

    participant_id = raw_session.parent.name or "unknown"
    session_id = raw_session.name or "session"

    # If labels are present, keep the official Phase II path (true labels + sensor
    # integrity validation). Without labels, downgrade phase so the converter can
    # create stub labels instead of aborting.
    phase = "II.A" if pf["raw_labels"] else "II.A-replay-nolabels"
    if not is_official_phase2(phase):
        print("  [preflight] No labels.csv — running converter in non-official mode "
              "with stub labels.")

    emg_csv = None
    emg_board = None
    if mode == "full_hybrid":
        emg_csv = Path(pf["raw_emg"]) if pf["raw_emg"] else None
        emg_board = pf["emg_board"]

    print(f"\n  Converting raw session -> {out_session}")
    convert_session(
        out_dir=out_session,
        emg_csv=emg_csv,
        emg_board=emg_board,
        imu_csv=Path(pf["raw_imu"]) if pf["raw_imu"] else None,
        labels_csv=Path(pf["raw_labels"]) if pf["raw_labels"] else None,
        operating_mode=mode,
        participant_id=participant_id,
        session_id=session_id,
        phase=phase,
        run_pipeline=True,
    )

    if not feat_path.exists():
        raise RuntimeError(
            f"Conversion completed but {feat_path} was not created. "
            f"Check the pipeline output above."
        )
    return out_session


# ─────────────────────────────────────────────────────────────────────────────
# REPLAY PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def run_predictions(
    feat_df: pd.DataFrame,
    models_dir: Path,
    mode: str,
    speed: float = 0.0,
    loaded_models: tuple | None = None,
) -> tuple[pd.DataFrame, bool]:
    """
    Classify every window and attach explanations.

    Returns (predictions_df, emg_available_session).
    """
    if loaded_models is None:
        print("\n  Loading models...")
        loaded_models = load_models(models_dir, mode)
    imu_model, imu_features, emg_model, emg_features = loaded_models

    missing = missing_model_features(feat_df, imu_features, emg_features, mode)
    if missing:
        preview = ", ".join(missing[:12])
        more = f", ... (+{len(missing) - 12} more)" if len(missing) > 12 else ""
        raise ValueError(
            "feature_matrix.csv is missing model-required columns: "
            f"{preview}{more}. Rebuild the processed session with --force."
        )

    # Session-level EMG availability (does the matrix carry any usable EMG?).
    available_emg_cols = [c for c in emg_features if c in feat_df.columns]
    emg_available_session = bool(available_emg_cols) and not (
        feat_df[available_emg_cols].isna().all().all()
    )
    if mode == "full_hybrid" and not emg_available_session:
        raise ValueError(
            "full_hybrid replay requested but the feature matrix has no usable EMG "
            "features. Re-run with --mode imu_only_fallback to report contingency output."
        )

    # Fill NaNs in the feature columns the models consume.
    model_cols = list(dict.fromkeys(imu_features + emg_features))
    work = feat_df.copy()
    for c in model_cols:
        if c in work.columns:
            work[c] = work[c].fillna(0.0)

    half_win = _WINDOW_MS / 2.0
    delay = (1.0 / speed) if speed and speed > 0 else 0.0

    records: list[dict] = []
    for idx, row in work.reset_index(drop=True).iterrows():
        result = classify_window(imu_model, imu_features, emg_model, emg_features, row, mode)

        # Per-window EMG validity (full_hybrid only — fallback forces False).
        row_emg_ok = (mode == "full_hybrid") and emg_features_valid(row)
        expl = explain_window(row, result, operating_mode=mode, emg_available=row_emg_ok)

        centre = float(row.get("window_centre_ms", float("nan")))
        true_rc = row.get("risk_class", None)
        try:
            true_rc = int(true_rc)
            true_rc = true_rc if true_rc != -1 else None
        except (TypeError, ValueError):
            true_rc = None

        win_start = centre - half_win if np.isfinite(centre) else float("nan")
        win_end = centre + half_win if np.isfinite(centre) else float("nan")

        rec = {
            "window_idx": int(idx),
            "window_centre_ms": centre,
            # Contract columns (preferred) + explicit _ms aliases for clarity.
            "window_start": win_start,
            "window_end": win_end,
            "window_start_ms": win_start,
            "window_end_ms": win_end,
            "movement_label": row.get("movement_label", "UNKNOWN"),
            "true_risk_class": true_rc,
            "predicted": result.get("risk_level"),
            "predicted_binary": result.get("predicted"),
            "traffic_light": result.get("label"),
            "R_IMU": result.get("R_IMU", float("nan")),
            "R_EMG": result.get("R_EMG", float("nan")),
            "R_total": result.get("p_risk", float("nan")),
            "fis_reason": result.get("fis_reason", ""),
            "engineering_reason": expl["engineering_reason"],
            "layman_reason": expl["layman_reason"],
        }
        for c in KEY_IMU_COLS:
            if c in row.index:
                rec[c] = row.get(c)
        if mode == "full_hybrid":
            for c in KEY_EMG_COLS:
                if c in row.index:
                    rec[c] = row.get(c)

        records.append(rec)
        if delay:
            time.sleep(delay)

    pred_df = pd.DataFrame.from_records(records)
    # In fallback mode EMG is deliberately not used for inference; report it as
    # unavailable so the summary never implies hybrid behaviour.
    emg_reported = emg_available_session if mode == "full_hybrid" else False
    return pred_df, emg_reported


def missing_model_features(
    feat_df: pd.DataFrame,
    imu_features: list[str],
    emg_features: list[str],
    mode: str,
) -> list[str]:
    """Return model input columns absent from the current feature matrix."""
    required = list(imu_features)
    if mode == "full_hybrid":
        required += list(emg_features)
    return [c for c in dict.fromkeys(required) if c not in feat_df.columns]


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY + PLOT
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(
    pred_df: pd.DataFrame,
    raw_session: Path,
    out_session: Path,
    mode: str,
    models_dir: Path,
    emg_available: bool,
    pf: dict,
    generated_files: list[str],
) -> dict:
    n = len(pred_df)
    counts = pred_df["predicted"].value_counts().to_dict()
    counts = {lvl: int(counts.get(lvl, 0)) for lvl in RISK_LEVELS}
    pct = {lvl: round(100.0 * counts[lvl] / n, 1) if n else 0.0 for lvl in RISK_LEVELS}

    duration_s = float("nan")
    if "window_end_ms" in pred_df.columns and n:
        end = pred_df["window_end_ms"].max()
        start = pred_df["window_start_ms"].min()
        if np.isfinite(end) and np.isfinite(start):
            duration_s = round((end - start) / 1000.0, 1)

    highest = None
    if n:
        top = pred_df.loc[pred_df["R_total"].idxmax()]
        highest = {
            "window_idx": int(top["window_idx"]),
            "R_total": round(float(top["R_total"]), 4),
            "traffic_light": top["traffic_light"],
            "predicted": top["predicted"],
            "movement_label": top.get("movement_label", "UNKNOWN"),
            "window_start_ms": float(top.get("window_start_ms", float("nan"))),
        }

    # Optional accuracy vs recorded protocol labels.
    accuracy = None
    labelled = pred_df[pred_df["true_risk_class"].notna()]
    if len(labelled):
        correct = (labelled["true_risk_class"].astype(int) == labelled["predicted_binary"].astype(int)).sum()
        accuracy = {
            "n_labelled_windows": int(len(labelled)),
            "binary_accuracy": round(float(correct) / len(labelled), 4),
        }

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "raw_session": str(raw_session),
        "out_session": str(out_session),
        "operating_mode": mode,
        "model_directory_used": str(models_dir),
        "emg_available": bool(emg_available),
        "total_windows": int(n),
        "duration_s": duration_s,
        "risk_counts": counts,
        "risk_percentages": pct,
        "highest_risk_window": highest,
        "accuracy_vs_protocol_labels": accuracy,
        "preflight": {
            "status": pf["status"],
            "emg_board": pf["emg_board"],
            "raw_imu": pf["raw_imu"],
            "raw_emg": pf["raw_emg"],
            "raw_labels": pf["raw_labels"],
            "raw_sync": pf["raw_sync"],
            "warnings": pf["warnings"],
            "errors": pf["errors"],
        },
        "generated_files": generated_files,
        "disclaimer": (
            "Movement-risk feedback within this system only. Not a medical diagnosis. "
            "Risk reflects biomechanical thresholds and deviation from personal baseline."
        ),
    }


def save_timeline_plot(pred_df: pd.DataFrame, out_path: Path, mode: str) -> bool:
    """Render the risk timeline figure. Returns True on success."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = pred_df["window_idx"].to_numpy()
        fig, (ax_cls, ax_score) = plt.subplots(
            2, 1, figsize=(12, 6), sharex=True,
            gridspec_kw={"height_ratios": [1, 1.4]},
        )
        fig.suptitle(
            f"Spinal Movement Risk Replay — {mode}", fontweight="bold", fontsize=13
        )

        # Risk class over time (coloured scatter/step)
        colour_map = {"GREEN": "#4CAF50", "AMBER": "#FF9800", "RED": "#F44336"}
        level_y = {"Safe": 0, "Cautious": 1, "Risky": 2}
        ys = pred_df["predicted"].map(level_y).to_numpy()
        cs = pred_df["traffic_light"].map(colour_map).fillna("#888").to_numpy()
        ax_cls.scatter(x, ys, c=cs, s=12)
        ax_cls.set_yticks([0, 1, 2])
        ax_cls.set_yticklabels(["Safe", "Cautious", "Risky"])
        ax_cls.set_ylabel("Risk class")
        ax_cls.grid(alpha=0.2)

        # Component scores over time
        ax_score.plot(x, pred_df["R_total"], color="#222", linewidth=1.4, label="R_total")
        if "R_IMU" in pred_df.columns:
            ax_score.plot(x, pred_df["R_IMU"], color="#42a5f5", linewidth=1.0,
                          alpha=0.8, label="R_IMU")
        if mode == "full_hybrid" and pred_df["R_EMG"].notna().any():
            ax_score.plot(x, pred_df["R_EMG"], color="#ab47bc", linewidth=1.0,
                          alpha=0.8, label="R_EMG")
        ax_score.axhline(0.35, color="#4CAF50", ls="--", lw=0.8, alpha=0.7)
        ax_score.axhline(0.65, color="#F44336", ls="--", lw=0.8, alpha=0.7)
        ax_score.set_ylim(-0.02, 1.02)
        ax_score.set_ylabel("Risk score")
        ax_score.set_xlabel("Window index")
        ax_score.legend(loc="upper right", fontsize=8)
        ax_score.grid(alpha=0.2)

        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return True
    except Exception as e:  # pragma: no cover
        print(f"  [WARNING] Could not render timeline plot: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline replay of a recorded session -> predictions, summary, "
                    "explanations and a timeline figure for the replay dashboard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw_session", required=True,
                        help="Raw recorded session dir (imu_arduino.csv, cyton/ganglion.csv, labels.csv...).")
    parser.add_argument("--out_session", required=True,
                        help="Output processed/replay session directory.")
    parser.add_argument("--mode", choices=["full_hybrid", "imu_only_fallback"],
                        default="full_hybrid",
                        help="Primary full-hybrid system, or first-class IMU-only fallback.")
    parser.add_argument("--emg_board", choices=["cyton", "ganglion", "auto"], default="auto",
                        help="OpenBCI board for the raw EMG file (auto detects cyton/ganglion).")
    parser.add_argument("--models_dir", default=None,
                        help="Override model directory. Defaults by mode: "
                             "full_hybrid->ml/models, imu_only_fallback->ml/models/fallback_final.")
    parser.add_argument("--speed", type=float, default=0.0,
                        help="Optional per-window pacing (windows/sec). 0 = as fast as possible.")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip writing replay_timeline.png.")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild the processed session even if feature_matrix.csv exists.")
    args = parser.parse_args(argv)

    raw_session = Path(args.raw_session)
    out_session = Path(args.out_session)
    mode = args.mode

    models_dir = Path(args.models_dir) if args.models_dir else DEFAULT_MODELS[mode]

    print("=" * 64)
    print("Spinal Movement Risk — Recorded Session Replay")
    print(f"  Raw session : {raw_session}")
    print(f"  Out session : {out_session}")
    print(f"  Mode        : {mode}")
    print(f"  EMG board   : {args.emg_board}")
    print(f"  Models dir  : {models_dir}")
    print("=" * 64)

    # ── Preflight ───────────────────────────────────────────────────────────
    print("\n[1/5] Preflight checks...")
    pf = preflight(raw_session, mode, args.emg_board)
    for w in pf["warnings"]:
        print(f"  [warn] {w}")
    for e in pf["errors"]:
        print(f"  [ERROR] {e}")
    if pf["status"] != "ok":
        print("\nPreflight failed. Aborting.")
        return 2
    print(f"  Preflight OK (EMG board: {pf['emg_board']}).")

    if not models_dir.exists():
        print(f"\n[ERROR] Model directory does not exist: {models_dir}")
        return 2

    # ── Convert + extract features ────────────────────────────────────────────
    print("\n[2/5] Convert + extract features (reusing session_converter + pipeline)...")
    try:
        out_session = ensure_processed_session(raw_session, out_session, mode, pf, args.force)
    except Exception as e:
        print(f"\n[ERROR] Conversion/feature extraction failed: {e}")
        traceback.print_exc()
        return 3

    feat_df = load_or_build_features(out_session)

    # ── Replay prediction + explanations ──────────────────────────────────────
    print("\n[3/5] Replay prediction + explanations...")
    try:
        print("\n  Loading models...")
        loaded_models = load_models(models_dir, mode)
        _, imu_features, _, emg_features = loaded_models

        missing = missing_model_features(feat_df, imu_features, emg_features, mode)
        if missing and not args.force:
            preview = ", ".join(missing[:8])
            more = f", ... (+{len(missing) - 8} more)" if len(missing) > 8 else ""
            print(
                "  Existing feature_matrix.csv is stale for the selected model "
                f"(missing: {preview}{more}). Rebuilding once..."
            )
            out_session = ensure_processed_session(raw_session, out_session, mode, pf, force=True)
            feat_df = load_or_build_features(out_session)
            missing = missing_model_features(feat_df, imu_features, emg_features, mode)

        if missing:
            preview = ", ".join(missing[:12])
            more = f", ... (+{len(missing) - 12} more)" if len(missing) > 12 else ""
            raise ValueError(
                "feature_matrix.csv is still missing model-required columns after conversion: "
                f"{preview}{more}"
            )

        pred_df, emg_available = run_predictions(
            feat_df, models_dir, mode, speed=args.speed, loaded_models=loaded_models
        )
    except Exception as e:
        print(f"\n[ERROR] Replay prediction failed: {e}")
        traceback.print_exc()
        return 4

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\n[4/5] Saving outputs...")
    generated_files: list[str] = []

    pred_path = out_session / "replay_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    generated_files.append(pred_path.name)
    print(f"  Saved: {pred_path.name}  ({len(pred_df)} windows)")

    plot_ok = False
    if not args.no_plot:
        plot_path = out_session / "replay_timeline.png"
        plot_ok = save_timeline_plot(pred_df, plot_path, mode)
        if plot_ok:
            generated_files.append(plot_path.name)
            print(f"  Saved: {plot_path.name}")

    summary = build_summary(
        pred_df, raw_session, out_session, mode, models_dir,
        emg_available, pf, generated_files,
    )
    summary_path = out_session / "replay_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # The summary lists itself among the generated files for completeness.
    summary["generated_files"].append(summary_path.name)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Saved: {summary_path.name}")

    # ── Report ──────────────────────────────────────────────────────────────
    print("\n[5/5] Done.")
    print("=" * 64)
    print(f"  MODEL DIRECTORY USED : {models_dir}")
    print(f"  Operating mode       : {mode}")
    print(f"  EMG available        : {emg_available}")
    print(f"  Total windows        : {summary['total_windows']}")
    print(f"  Duration (s)         : {summary['duration_s']}")
    print(f"  Safe/Cautious/Risky  : "
          f"{summary['risk_percentages']['Safe']}% / "
          f"{summary['risk_percentages']['Cautious']}% / "
          f"{summary['risk_percentages']['Risky']}%")
    if summary["highest_risk_window"]:
        h = summary["highest_risk_window"]
        print(f"  Highest-risk window  : #{h['window_idx']} "
              f"(R_total={h['R_total']}, {h['traffic_light']}, {h['movement_label']})")
    if summary["accuracy_vs_protocol_labels"]:
        a = summary["accuracy_vs_protocol_labels"]
        print(f"  Binary accuracy      : {a['binary_accuracy']} "
              f"(n={a['n_labelled_windows']} labelled windows)")
    print("=" * 64)
    print(f"\n  View the dashboard with:")
    print(f"    streamlit run scripts/replay_dashboard.py -- --session \"{out_session}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
