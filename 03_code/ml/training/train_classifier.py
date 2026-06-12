#!/usr/bin/env python3
"""
ML Classifier Training — Spinal Movement Risk Monitor
FYP 2025/26 | Imperial College London

Trains and evaluates classifiers using Leave-One-Session-Out (LOSO)
cross-validation on three feature conditions (spec §11):

  Condition    Features                   Models
  ─────────────────────────────────────────────────
  IMU-only     12 kinematic + 3 z-scores  RF, SVM, LDA
  sEMG-only    12 muscle + 2 z-scores     RF, LR, LDA
  IMU+sEMG     All 24+ features           RF, SVM, LDA

The sEMG-only condition isolates the contribution of muscle activation.
The Logistic Regression (LR) model is the spec-specified classifier for
the sEMG pathway (spec §7.3) — interpretable coefficients, appropriate for
moderate-size datasets with well-conditioned features.

Binary classification target:
  Class 0 → safe movement
  Class 1 → risky movement

Outputs (ml/evaluation/ and ml/models/):
  loso_results.csv          — per-fold metrics for all conditions and models
  summary_results.csv       — mean ± std across LOSO folds
  feature_importance_RF.csv — RF Gini importances averaged over folds
  delta_imu_vs_emg.csv      — IMU+sEMG lift over IMU-only
  *.joblib                  — fitted model files

Usage:
  python ml/training/train_classifier.py [--data_dir data/synthetic] [--seed 42]
"""

import argparse
import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Mamdani FIS (spec §7.4 — optional; skipped gracefully if not found) ──────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
from scripts.datasets.dataset_manifest import validate_dataset_manifest

try:
    from ml.fuzzy.mamdani_fis import MamdaniFIS, RISKY_THRESHOLD
    FIS_AVAILABLE = True
except ImportError:
    FIS_AVAILABLE = False
    RISKY_THRESHOLD = 0.65
    print("[WARNING] ml/fuzzy/mamdani_fis.py not importable — FIS evaluation skipped")

# R_total threshold for the deployed Red/Risky output.
FIS_RISK_THRESHOLD = RISKY_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SETS (spec §6.1, §6.2)
# ─────────────────────────────────────────────────────────────────────────────

# IMU kinematic features (16) + personalised z-scores (3)
IMU_FEATURES = [
    "imu_trunk_angle_peak",
    "imu_trunk_angle_mean",
    "imu_angvel_peak",
    "imu_angvel_mean",
    "imu_time_in_risk_zone",
    "imu_time_high_velocity",
    "imu_ldlj",
    "imu_jerk_rms",
    "imu_jerk_peak",
    "imu_ldlj_multiaxis",
    "imu_compensation_index",
    "imu_lumbopelv_ratio",
    "imu_pelvis_angle_peak",
    "imu_pelvis_angle_mean",
    "imu_lat_angle_peak",
    "imu_lat_angle_mean",
    # Personalised baseline z-scores (spec §6.1)
    "imu_z_flex",
    "imu_z_vel",
    "imu_z_ldlj",
]

# sEMG time-domain features (4 channels x 3 features) + AI/CAI (4) + z-scores (2)
# ALL time-domain — MPF excluded per 200 Hz hardware constraint (spec §6.2)
# Channels: LES/RES = erector spinae (bilateral); LOBL/ROBL = obliques (bilateral).
# (Channels 3-4 are surface obliques, NOT multifidus — earlier "LMF/RMF" was a mislabel.)
EMG_FEATURES = [
    "emg_rms_LES", "emg_rms_RES", "emg_rms_LOBL", "emg_rms_ROBL",
    "emg_mav_LES", "emg_mav_RES", "emg_mav_LOBL", "emg_mav_ROBL",
    "emg_zcr_LES", "emg_zcr_RES", "emg_zcr_LOBL", "emg_zcr_ROBL",
    "emg_ai_ES",  "emg_ai_OBL",
    "emg_cai_ES", "emg_cai_OBL",
    # Personalised baseline z-scores (spec §6.2)
    "emg_z_rms_r",
    "emg_z_ar",
]

IMU_EMG_FEATURES = IMU_FEATURES + EMG_FEATURES

TARGET_COL  = "risk_class"
SESSION_COL = "session_id"
PARTICIPANT_COL = "participant_id"
UNLABELLED  = -1

# Threshold tuning: minimum sensitivity we want to guarantee on the training fold.
# The tuned threshold is then applied to the held-out test fold.  A fall-back of
# 0.5 is used whenever the ROC curve cannot achieve the target (e.g. degenerate
# training folds with very few positive examples).
TARGET_SENSITIVITY = 0.85
PROVENANCE_FILENAME = "model_provenance.json"


def _looks_like_real_data_path(data_dir: Path) -> bool:
    """Return True when the input path is inside a real-data folder."""
    return any(part.lower() == "real" for part in data_dir.parts)


def _validate_real_data_target(df: pd.DataFrame, data_dir: Path, target_col: str) -> None:
    """
    Guard against training Phase II/III models on circular signal-derived labels.

    Synthetic Phase I runs may still use signal labels as a diagnostic stress test,
    but real-data claims should use protocol/task labels as the supervised target.
    """
    if not _looks_like_real_data_path(data_dir):
        return

    if target_col not in df.columns:
        return

    if target_col == "risk_class_signal":
        raise ValueError(
            "Refusing to train a real-data model with --label_source signal. "
            "Signal-derived labels are threshold outputs from the same features "
            "and would make Phase II/III evaluation circular. Use "
            "--label_source protocol for real participant data."
        )

    if target_col == TARGET_COL and "risk_class_protocol" in df.columns:
        labelled = df["risk_class_protocol"] != UNLABELLED
        if labelled.any():
            default_target = df.loc[labelled, TARGET_COL].to_numpy()
            protocol_target = df.loc[labelled, "risk_class_protocol"].to_numpy()
            if (default_target != protocol_target).any():
                raise ValueError(
                    "Real-data combined_features.csv contains both risk_class and "
                    "risk_class_protocol, but they do not match. Re-run the signal "
                    "processing pipeline with --label_source protocol, or train with "
                    "--label_source protocol explicitly."
                )


def _resolve_cv_group(df: pd.DataFrame, data_dir: Path, cv_group: str) -> str:
    """
    Choose the cross-validation grouping column.

    Phase I synthetic validation uses session-level folds. Phase II real-data
    evaluation should prefer participant-level folds when participant_id is
    available, because holding out only one session can leak participant-specific
    sensor placement and movement style into the training set.
    """
    if cv_group == "participant":
        if PARTICIPANT_COL not in df.columns:
            raise ValueError(
                "Participant-level CV requested but combined_features.csv has no "
                "participant_id column. Re-run the pipeline on a "
                "participant_XX/session_YY layout."
            )
        return PARTICIPANT_COL

    if cv_group == "session":
        return SESSION_COL

    if _looks_like_real_data_path(data_dir) and PARTICIPANT_COL in df.columns:
        return PARTICIPANT_COL
    return SESSION_COL


def _git_commit_hash() -> str | None:
    """Return the current git commit hash when this workspace is versioned."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(_PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "not-version-controlled; integrity via FROZEN_MANIFEST.json (SHA-256)"
    return completed.stdout.strip() or "not-version-controlled; integrity via FROZEN_MANIFEST.json (SHA-256)"


def _write_phase2_provenance(
    models_dir: Path,
    data_dir: Path,
    feature_file: Path,
    feature_columns: dict[str, list[str]],
    seed: int,
    command: str,
    operating_mode: str = "full_hybrid",
    cv_group: str = "participant",
) -> Path:
    """Write auditable provenance for frozen Phase II.A model outputs."""
    model_types = sorted(
        {
            path.stem.rsplit("_fold", 1)[0]
            for path in models_dir.glob("*_fold*.joblib")
        }
    )
    metadata = {
        "phase": "Phase II.A",
        "training_data_dir": str(data_dir.resolve()),
        "feature_file": str(feature_file.resolve()),
        "label_source": "protocol",
        "cv_group": cv_group,
        "operating_mode": operating_mode,
        "system_route": (
            "full_hybrid_rf_imu_lr_emg_mamdani_fis"
            if operating_mode == "full_hybrid"
            else "imu_only_fallback_rf_imu_mamdani_fis"
        ),
        "contingency_only": operating_mode == "imu_only_fallback",
        "model_types_trained": model_types,
        "feature_columns": feature_columns,
        "random_seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": _git_commit_hash(),
        "command_used": command,
    }
    path = models_dir / PROVENANCE_FILENAME
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f'Phase II.A model provenance saved to {path}')
    return path


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIER DEFINITIONS (spec §7.2, §7.3)
# ─────────────────────────────────────────────────────────────────────────────

def build_imu_classifiers(seed: int) -> dict:
    """
    Classifiers for the IMU-only and IMU+sEMG conditions.
    RF is the spec-specified primary model (spec §7.2).
    SVM (linear, calibrated) and LDA added for comparative evaluation.
    """
    lsvc = CalibratedClassifierCV(
        LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed),
        cv=3,
    )
    return {
        "RF": Pipeline([
            ("sc", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=150, max_depth=None, min_samples_leaf=5,
                class_weight="balanced", random_state=seed, n_jobs=-1,
            )),
        ]),
        "SVM": Pipeline([
            ("sc", StandardScaler()),
            ("clf", lsvc),
        ]),
        "LDA": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="svd")),
        ]),
    }


def build_emg_classifiers(seed: int) -> dict:
    """
    Classifiers for the sEMG-only condition.
    LR is the spec-specified primary model (spec §7.3):
      interpretable coefficients, calibrated probabilities, appropriate
      for moderate-size datasets. RF and LDA added for comparison.
    """
    return {
        "LR": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0, class_weight="balanced", solver="lbfgs",
                max_iter=1000, random_state=seed,
            )),
        ]),
        "RF": Pipeline([
            ("sc", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=150, min_samples_leaf=5,
                class_weight="balanced", random_state=seed, n_jobs=-1,
            )),
        ]),
        "LDA": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="svd")),
        ]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD TUNING  (spec §11 — sensitivity ≥ 0.85 design target)
# ─────────────────────────────────────────────────────────────────────────────

def tune_threshold(
    y_true_train: np.ndarray,
    y_prob_train: np.ndarray,
    target_sensitivity: float = TARGET_SENSITIVITY,
) -> float:
    """
    Find the decision threshold on *training* data that achieves
    sensitivity ≥ target_sensitivity while maximising specificity.

    Strategy:
      1. Compute the full ROC curve on the training fold.
      2. Collect all candidate thresholds where TPR ≥ target_sensitivity.
      3. Among candidates, choose the one with the highest TNR (specificity)
         — i.e. the Youden-optimal point within the sensitivity constraint.
      4. Fall back to 0.5 if the target cannot be met (rare with balanced
         class weights but possible on very small synthetic folds).

    The returned threshold is applied to test-fold probabilities to produce
    `y_pred_tuned`.  This avoids post-hoc threshold shopping on test data.

    Parameters
    ----------
    y_true_train     : ground-truth labels for the training fold
    y_prob_train     : predicted probabilities (class-1) for the training fold
    target_sensitivity: minimum acceptable sensitivity (default 0.85)

    Returns
    -------
    threshold : float  (in [0, 1])
    """
    if len(np.unique(y_true_train)) < 2:
        return 0.5  # degenerate fold — no ROC curve possible

    fpr, tpr, thresholds = roc_curve(y_true_train, y_prob_train, pos_label=1)
    # thresholds from sklearn are in descending order; match fpr/tpr indices
    tnr = 1.0 - fpr  # specificity at each point

    # Mask to points that meet the sensitivity constraint
    meets = tpr >= target_sensitivity
    if not meets.any():
        # Cannot hit target — use the threshold that maximises sensitivity
        best_idx = int(np.argmax(tpr))
        return float(thresholds[best_idx])

    # Among qualifying points, maximise specificity (minimise FPR)
    tnr_filtered = np.where(meets, tnr, -np.inf)
    best_idx = int(np.argmax(tnr_filtered))
    return float(thresholds[best_idx])


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """
    Binary classification metrics for a risk monitor.
    Sensitivity (recall for class 1) is the primary metric — missing a risky
    movement is more consequential than a false alarm (spec §11).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "accuracy":       accuracy_score(y_true, y_pred),
        "f1_macro":       f1_score(y_true, y_pred, average="macro",  zero_division=0),
        "f1_risk":        f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0),
        "precision_risk": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "sensitivity":    sensitivity,
        "specificity":    specificity,
        "auc":            roc_auc_score(y_true, y_prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_test":         int(len(y_true)),
        "n_risk_true":    int((y_true == 1).sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# LOSO CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def loso_cv(
    datasets: dict,
    sessions: list,
    seed: int,
    models_dir: Path,
    group_col: str = SESSION_COL,
) -> tuple:
    """
    Leave-one-group-out CV across all three conditions.

    Parameters
    ----------
    datasets  : dict mapping condition_name → (DataFrame, feature_cols, classifier_fn)
    sessions  : list of held-out group IDs
    seed      : random seed
    models_dir: where to save fitted model files
    group_col : column used for the held-out split

    Returns
    -------
    results    : list of per-fold metric dicts
    fi_accum   : accumulated RF feature importances per condition
    fold_probs : dict {test_session → {'R_IMU': pd.Series, 'R_EMG': pd.Series}}
                 Stores test-fold classifier probabilities needed to run the FIS
                 (IMU-RF for R_IMU, EMG-LR for R_EMG).
    """
    results = []
    fi_accum = {
        cond: {f: [] for f in feat_cols}
        for cond, (_, feat_cols, _) in datasets.items()
    }
    # Per-fold probabilities used for Mamdani FIS evaluation (spec §7.4)
    fold_probs: dict = {}

    for fold_idx, test_session in enumerate(sessions):
        train_sessions = [s for s in sessions if s != test_session]
        print(f"\n  Fold {fold_idx + 1}/{len(sessions)}  [test {group_col}: {test_session}]")

        for cond, (df, feat_cols, clf_fn) in datasets.items():
            tr = df[df[group_col].isin(train_sessions)]
            te = df[df[group_col] == test_session]
            X_tr = tr[feat_cols].values
            y_tr = tr[TARGET_COL].values
            X_te = te[feat_cols].values
            y_te = te[TARGET_COL].values

            if len(X_tr) == 0 or len(np.unique(y_tr)) < 2:
                print(f"    [{cond}] skipped -- insufficient training classes")
                continue

            if len(X_te) == 0 or len(np.unique(y_te)) < 2:
                print(f"    [{cond}] skipped — insufficient test data")
                continue

            clfs = clf_fn(seed)
            for clf_name, pipe in clfs.items():
                import time; t0 = time.time()
                pipe.fit(X_tr, y_tr)

                # ── Default threshold (0.5) metrics ──────────────────────
                y_pred = pipe.predict(X_te)
                y_prob = pipe.predict_proba(X_te)[:, 1]
                m = compute_metrics(y_te, y_pred, y_prob)

                # ── Tuned threshold — derived from training fold only ─────
                y_prob_tr  = pipe.predict_proba(X_tr)[:, 1]
                threshold  = tune_threshold(y_tr, y_prob_tr, TARGET_SENSITIVITY)
                y_pred_tun = (y_prob >= threshold).astype(int)
                m_tun      = compute_metrics(y_te, y_pred_tun, y_prob)

                m.update({
                    "fold":              fold_idx + 1,
                    "test_session":      test_session,
                    "cv_group":          group_col,
                    "classifier":        clf_name,
                    "condition":         cond,
                    # Tuned-threshold columns
                    "threshold_tuned":   round(threshold, 4),
                    "sensitivity_tuned": round(m_tun["sensitivity"], 4),
                    "specificity_tuned": round(m_tun["specificity"], 4),
                    "f1_risk_tuned":     round(m_tun["f1_risk"],     4),
                    "accuracy_tuned":    round(m_tun["accuracy"],     4),
                    "tp_tuned":          m_tun["tp"],
                    "fp_tuned":          m_tun["fp"],
                    "tn_tuned":          m_tun["tn"],
                    "fn_tuned":          m_tun["fn"],
                })
                results.append(m)

                print(
                    f"    [{cond:<8}] {clf_name:<4} "
                    f"AUC={m['auc']:.3f}  "
                    f"sens={m['sensitivity']:.3f}→{m_tun['sensitivity']:.3f}  "
                    f"spec={m['specificity']:.3f}→{m_tun['specificity']:.3f}  "
                    f"F1={m['f1_risk']:.3f}→{m_tun['f1_risk']:.3f}  "
                    f"thr={threshold:.3f}  ({time.time()-t0:.1f}s)"
                )

                model_path = models_dir / f"{clf_name}_{cond}_fold{fold_idx+1}.joblib"
                dump(pipe, model_path)

                if clf_name == "RF" and hasattr(pipe.named_steps["clf"], "feature_importances_"):
                    for feat, imp in zip(feat_cols, pipe.named_steps["clf"].feature_importances_):
                        fi_accum[cond][feat].append(imp)

                # ── Capture probabilities for Mamdani FIS (spec §7.4) ─────────
                # IMU-RF  → R_IMU;  EMG-LR → R_EMG
                if cond == "IMU" and clf_name == "RF":
                    fold_probs.setdefault(test_session, {})
                    fold_probs[test_session]["R_IMU"] = pd.Series(
                        y_prob, index=te.index, name="R_IMU"
                    )
                if cond == "EMG" and clf_name == "LR":
                    fold_probs.setdefault(test_session, {})
                    fold_probs[test_session]["R_EMG"] = pd.Series(
                        y_prob, index=te.index, name="R_EMG"
                    )

    return results, fi_accum, fold_probs


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    base_metrics = [
        "accuracy", "f1_macro", "f1_risk",
        "precision_risk", "sensitivity", "specificity", "auc",
    ]
    tuned_metrics = [
        "threshold_tuned",
        "sensitivity_tuned", "specificity_tuned",
        "f1_risk_tuned", "accuracy_tuned",
    ]
    # Only include tuned columns that actually exist (guard for older result files)
    available_tuned = [c for c in tuned_metrics if c in results_df.columns]
    metric_cols = base_metrics + available_tuned

    summary = (
        results_df
        .groupby(["classifier", "condition"])[metric_cols]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    return summary.reset_index()


def build_feature_importance_df(fi_accum: dict) -> pd.DataFrame:
    rows = []
    for cond, feat_dict in fi_accum.items():
        for feat_name, vals in feat_dict.items():
            if vals:
                rows.append({
                    "condition":        cond,
                    "feature":          feat_name,
                    "importance_mean":  np.mean(vals),
                    "importance_std":   np.std(vals),
                })
    df = pd.DataFrame(rows)
    return df.sort_values(["condition", "importance_mean"], ascending=[True, False])


def print_summary(summary_df: pd.DataFrame):
    print("\n" + "=" * 100)
    print("LOSO SUMMARY  (mean ± std across folds)")
    print("  Tuned = sensitivity-constrained threshold (≥{:.0f}%) derived from training fold".format(
        TARGET_SENSITIVITY * 100))
    print("=" * 100)
    cols = [
        "classifier", "condition",
        "auc_mean", "auc_std",
        # Default-threshold metrics
        "sensitivity_mean", "sensitivity_std",
        "specificity_mean", "specificity_std",
        "f1_risk_mean", "f1_risk_std",
        "accuracy_mean",
        # Tuned-threshold metrics
        "threshold_tuned_mean",
        "sensitivity_tuned_mean", "sensitivity_tuned_std",
        "specificity_tuned_mean",
        "f1_risk_tuned_mean",
    ]
    avail = [c for c in cols if c in summary_df.columns]
    print(summary_df[avail].to_string(index=False, float_format="{:.3f}".format))


def print_lift_table(summary_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("IMU+sEMG LIFT over IMU-only (RF only)")
    print("=" * 70)
    rf_imu    = summary_df.loc[(summary_df.classifier == "RF") & (summary_df.condition == "IMU")]
    rf_fusion = summary_df.loc[(summary_df.classifier == "RF") & (summary_df.condition == "IMU_EMG")]
    if rf_imu.empty or rf_fusion.empty:
        return
    for m in ["auc", "sensitivity", "specificity", "f1_risk", "accuracy"]:
        col = f"{m}_mean"
        v_imu    = rf_imu[col].values[0]
        v_fusion = rf_fusion[col].values[0]
        delta    = v_fusion - v_imu
        sign     = "▲" if delta > 0 else "▼"
        print(f"  {m:<14}  IMU={v_imu:.3f}  IMU+sEMG={v_fusion:.3f}  {sign}{abs(delta):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAMDANI FIS LOSO EVALUATION  (spec §7.4)
# ─────────────────────────────────────────────────────────────────────────────

def run_fis_loso(
    df_fuse: pd.DataFrame,
    fold_probs: dict,
    sessions: list,
    fold_lookup: dict,
    group_col: str = SESSION_COL,
) -> list:
    """
    Run the Mamdani FIS on each LOSO test fold and compute metrics.

    The FIS receives:
      R_IMU  — IMU-RF class-1 probability (from fold_probs)
      R_EMG  — EMG-LR class-1 probability (from fold_probs)
      + interpretable abnormality inputs already in df_fuse:
        imu_z_ldlj, imu_time_in_risk_zone, emg_ai_ES and baseline z-scores.

    Binary prediction: R_total >= FIS_RISK_THRESHOLD → class 1 (risky)
    AUC computed using R_total as the soft probability.

    Parameters
    ----------
    df_fuse     : combined IMU+sEMG feature DataFrame (all sessions)
    fold_probs  : {test_session: {'R_IMU': Series, 'R_EMG': Series}}
    sessions    : ordered list of session IDs (determines fold index)
    fold_lookup : {test_session: fold_index}  (1-based)

    Returns
    -------
    list of per-fold metric dicts (same schema as loso_cv results)
    """
    if not FIS_AVAILABLE:
        print("[FIS] Skipped — MamdaniFIS not available")
        return []

    fis = MamdaniFIS()
    results = []
    print("\n  Running Mamdani FIS evaluation...")

    for test_session in sessions:
        probs = fold_probs.get(test_session, {})
        if "R_IMU" not in probs or "R_EMG" not in probs:
            print(f"    [FIS] fold {test_session}: missing R_IMU or R_EMG — skipped")
            continue

        # Build test fold DataFrame with injected classifier probabilities
        te = df_fuse[df_fuse[group_col] == test_session].copy()
        if len(te) == 0 or len(np.unique(te[TARGET_COL].values)) < 2:
            print(f"    [FIS] fold {test_session}: insufficient test data -- skipped")
            continue

        # Align probabilities to test fold index (inner join handles any dropna mismatches)
        te = te.join(probs["R_IMU"], how="inner")
        te = te.join(probs["R_EMG"], how="inner")

        # Run FIS inference
        fis_out = fis.infer_batch(te)   # returns DataFrame: R_total, colour, reason
        r_total  = fis_out["R_total"].values
        y_pred   = (r_total >= FIS_RISK_THRESHOLD).astype(int)
        y_true   = te[TARGET_COL].values

        m = compute_metrics(y_true, y_pred, r_total)
        fold_idx = fold_lookup.get(test_session, 0)
        m.update({
            "fold":              fold_idx,
            "test_session":      test_session,
            "cv_group":          group_col,
            "classifier":        "FIS",
            "condition":         "FIS",
            "threshold_tuned":   FIS_RISK_THRESHOLD,
            "sensitivity_tuned": round(m["sensitivity"], 4),
            "specificity_tuned": round(m["specificity"], 4),
            "f1_risk_tuned":     round(m["f1_risk"], 4),
            "accuracy_tuned":    round(m["accuracy"], 4),
            "tp_tuned": m["tp"], "fp_tuned": m["fp"],
            "tn_tuned": m["tn"], "fn_tuned": m["fn"],
        })
        results.append(m)

        print(
            f"    [FIS] fold {fold_idx} [{test_session}]  "
            f"AUC={m['auc']:.4f}  "
            f"sens={m['sensitivity']:.4f}  "
            f"spec={m['specificity']:.4f}  "
            f"F1={m['f1_risk']:.4f}"
        )

    return results


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main(
    data_dir: str,
    seed: int,
    label_source: str = "default",
    cv_group: str = "auto",
    models_dir: str = "ml/models",
    eval_dir: str = "ml/evaluation",
    write_phase2_provenance: bool = False,
    command_used: str | None = None,
    force: bool = False,
    operating_mode: str = "full_hybrid",
    fallback_rf_imu_only: bool = False,
):
    """
    Parameters
    ----------
    data_dir     : root data directory containing combined_features.csv
    seed         : random seed for reproducibility
    label_source : which risk_class column to use as the classification target:
        "default"  — use ``risk_class`` column as-is (backwards compatible;
                     works with both old synthetic CSVs and new real-data CSVs)
        "signal"   — use ``risk_class_signal`` (signal-derived, may have circularity)
        "protocol" — use ``risk_class_protocol`` (protocol-derived, no circularity;
                     requires Phase 2 real data processed with --label_source protocol)
    """
    # Resolve which column to use as the classification target
    _label_col_map = {
        "default":  "risk_class",
        "signal":   "risk_class_signal",
        "protocol": "risk_class_protocol",
    }
    target_col = _label_col_map.get(label_source, "risk_class")

    data_dir   = Path(data_dir)
    eval_dir   = Path(eval_dir)
    models_dir = Path(models_dir)
    official_phase2_training = _looks_like_real_data_path(data_dir) and label_source == "protocol"

    csv_path = data_dir / 'combined_features.csv'
    if not csv_path.exists():
        raise FileNotFoundError(
            f'Feature matrix not found at {csv_path}.\n'
            f'  Synthetic: python -m signal_processing.pipeline --data_dir data/synthetic\n'
            f'  Real data: python -m signal_processing.pipeline --data_dir data/real/protocol_train '
            f'--label_source protocol'
        )

    if official_phase2_training or write_phase2_provenance:
        _, manifest_errors = validate_dataset_manifest(
            data_dir,
            expected_phase="Phase II.A",
            expected_label_source="protocol",
            expected_operating_mode=operating_mode,
        )
        if manifest_errors:
            raise ValueError(
                "Phase II.A dataset manifest validation failed before training:\n  - "
                + "\n  - ".join(manifest_errors)
            )
        protected = (
            list(models_dir.glob("*_fold*.joblib"))
            + [models_dir / PROVENANCE_FILENAME]
            + [
                eval_dir / "loso_results.csv",
                eval_dir / "summary_results.csv",
                eval_dir / "feature_importance_RF.csv",
                eval_dir / "delta_conditions.csv",
            ]
        )
        existing = [path for path in protected if path.exists()]
        if existing and not force:
            raise FileExistsError(
                "Official Phase II.A model/evaluation outputs already exist. "
                "Refusing to overwrite without --force:\n  - "
                + "\n  - ".join(str(path) for path in existing)
            )

    eval_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Validate target column exists
    if target_col not in df.columns:
        available = [c for c in df.columns if 'risk_class' in c]
        raise ValueError(
            f"Target column '{target_col}' not found in {csv_path}.\n"
            f"  Available risk_class columns: {available}\n"
            f"  Tip: run the pipeline with --label_source {label_source} first "
            f"to generate this column."
        )

    _validate_real_data_target(df, data_dir, target_col)

    df = df[df[target_col] != UNLABELLED].copy()
    print(f'Loaded {len(df)} labelled windows from {csv_path}')
    print(f'Label source : {label_source!r}  →  column: {target_col!r}')
    print(f'Class balance: {df[target_col].value_counts().to_dict()}')

    group_col = _resolve_cv_group(df, data_dir, cv_group)
    allowed_provenance_groups = {PARTICIPANT_COL}
    if operating_mode == "imu_only_fallback":
        allowed_provenance_groups.add(SESSION_COL)
    if write_phase2_provenance and (
        label_source != "protocol" or group_col not in allowed_provenance_groups
    ):
        raise ValueError(
            "Phase II.A provenance can only be written for protocol labels "
            "with approved cross-validation grouping."
        )
    sessions = sorted(df[group_col].unique().tolist())
    if len(sessions) < 2:
        raise ValueError(
            f"Need at least two {group_col} values for leave-one-group-out CV; "
            f"found {sessions}."
        )
    print(f'CV grouping column: {group_col!r}')
    print(f'Folds: {sessions}')

    def available(cols):
        return [c for c in cols if c in df.columns]

    imu_cols     = available(IMU_FEATURES)
    emg_cols     = available(EMG_FEATURES)
    imu_emg_cols = available(IMU_EMG_FEATURES)

    # ── Feature-availability guard ────────────────────────────────────────────
    # Fail loudly if an EXPECTED feature column is missing, instead of silently
    # training on a smaller feature set. This catches the failure mode where a
    # producer-side rename in signal_processing/ drops a feature: available()
    # would quietly omit it and every downstream number would shift with no
    # error. IMU features are always produced by extract_imu_window_features();
    # EMG features are legitimately absent only on the declared IMU-only
    # fallback route (--fallback_rf_imu_only).
    def _require_features(expected, group_name, hint):
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(
                f"{group_name} feature column(s) missing from {csv_path}: {missing}\n  {hint}"
            )

    _require_features(
        IMU_FEATURES, "IMU",
        "IMU features are always produced by signal_processing/pipeline.py — a missing "
        "one usually means a producer/consumer rename. Compare the columns emitted by "
        "extract_imu_window_features() against IMU_FEATURES.",
    )
    if not fallback_rf_imu_only:
        _require_features(
            EMG_FEATURES, "EMG",
            "EMG features are missing but this is not the IMU-only fallback route. If this "
            "dataset is genuinely IMU-only, re-run with --fallback_rf_imu_only "
            "--operating_mode imu_only_fallback; otherwise a producer/consumer rename has "
            "dropped these columns.",
        )
    # ──────────────────────────────────────────────────────────────────────────

    # Rename target_col → TARGET_COL so the rest of the pipeline (loso_cv,
    # run_fis_loso) can use the global TARGET_COL constant without changes.
    # Drop the original TARGET_COL first to avoid duplicate column names when
    # renaming (e.g. risk_class_protocol → risk_class while risk_class exists).
    if target_col != TARGET_COL:
        if TARGET_COL in df.columns:
            df = df.drop(columns=[TARGET_COL])
        df = df.rename(columns={target_col: TARGET_COL})

    id_cols = [SESSION_COL]
    if group_col != SESSION_COL:
        id_cols.append(group_col)

    df_imu  = df[id_cols + [TARGET_COL] + imu_cols].dropna(subset=imu_cols)
    df_emg  = df[id_cols + [TARGET_COL] + emg_cols].dropna(subset=emg_cols)
    df_fuse = df[id_cols + [TARGET_COL] + imu_emg_cols].dropna(subset=imu_emg_cols)

    print(f'IMU-only dataset:   {len(df_imu)} windows  ({len(imu_cols)} features)')
    print(f'sEMG-only dataset:  {len(df_emg)} windows  ({len(emg_cols)} features)')
    print(f'IMU+sEMG dataset:   {len(df_fuse)} windows  ({len(imu_emg_cols)} features)')

    if fallback_rf_imu_only:
        if operating_mode != "imu_only_fallback":
            raise ValueError("--fallback_rf_imu_only requires --operating_mode imu_only_fallback")

        def _build_rf_only(seed_value: int) -> dict:
            return {"RF": build_imu_classifiers(seed_value)["RF"]}

        datasets = {
            'IMU': (df_imu, imu_cols, _build_rf_only),
        }
        print("IMU-only fallback route: training RF_IMU only; EMG/LR/FIS/comparator models are not trained.")
    else:
        datasets = {
            'IMU':     (df_imu,  imu_cols,     build_imu_classifiers),
            'EMG':     (df_emg,  emg_cols,     build_emg_classifiers),
            'IMU_EMG': (df_fuse, imu_emg_cols, build_imu_classifiers),
        }

    print(f'Starting leave-one-group-out CV  |  {len(sessions)} folds  |  seed={seed}')
    results, fi_accum, fold_probs = loso_cv(
        datasets, sessions, seed, models_dir, group_col=group_col
    )

    if not fallback_rf_imu_only:
        # Mamdani FIS evaluation (spec sec 7.4)
        fold_lookup = {s: i + 1 for i, s in enumerate(sessions)}
        fis_results = run_fis_loso(
            df_fuse, fold_probs, sessions, fold_lookup, group_col=group_col
        )
        if fis_results:
            results.extend(fis_results)
            print(f"  FIS evaluation complete -- {len(fis_results)} folds appended")

    results_df = pd.DataFrame(results)
    results_df.to_csv(eval_dir / 'loso_results.csv', index=False)

    summary_df = aggregate_results(results_df)
    summary_df.to_csv(eval_dir / 'summary_results.csv', index=False)

    fi_df = build_feature_importance_df(fi_accum)
    fi_df.to_csv(eval_dir / 'feature_importance_RF.csv', index=False)

    print_summary(summary_df)
    print_lift_table(summary_df)

    delta_rows = []
    for clf in sorted(summary_df['classifier'].unique()):
        for cond_a, cond_b in [('IMU', 'IMU_EMG'), ('EMG', 'IMU_EMG')]:
            row_a = summary_df[(summary_df.classifier == clf) & (summary_df.condition == cond_a)]
            row_b = summary_df[(summary_df.classifier == clf) & (summary_df.condition == cond_b)]
            if row_a.empty or row_b.empty:
                continue
            row = {'classifier': clf, 'comparison': f'{cond_b}_vs_{cond_a}'}
            for m in ['auc', 'sensitivity', 'specificity', 'f1_risk', 'accuracy']:
                row[f'delta_{m}'] = round(
                    float(row_b[f'{m}_mean'].values[0]) - float(row_a[f'{m}_mean'].values[0]), 4
                )
            delta_rows.append(row)

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(eval_dir / 'delta_conditions.csv', index=False)

    if write_phase2_provenance:
        _write_phase2_provenance(
            models_dir=models_dir,
            data_dir=data_dir,
            feature_file=csv_path,
            feature_columns={
                "IMU": imu_cols,
                **({} if fallback_rf_imu_only else {"EMG": emg_cols, "IMU_EMG": imu_emg_cols}),
            },
            seed=seed,
            command=command_used or " ".join(sys.argv),
            operating_mode=operating_mode,
            cv_group=group_col,
        )

    print(f'All results saved to {eval_dir}/')
    print('Done.')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Train and evaluate ML classifiers for lumbar movement risk.\n\n'
            'Phase I (synthetic pipeline validation, generated protocol labels):\n'
            '  python ml/training/train_classifier.py --data_dir data/synthetic\n\n'
            'Phase II.A (real protocol data collection and model training/fine-tuning):\n'
            '  python ml/training/train_classifier.py --data_dir data/real/protocol_train'
            ' --label_source protocol --cv_group participant\n\n'
            'Phase II.C (held-out varied-movement evaluation):\n'
            '  evaluate frozen Phase II.A models on data/real/varied_test without retraining.\n\n'
            'Use --label_source signal only for diagnostic threshold-derived labels;\n'
            'it is circular if used as the main model target.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--data_dir',
        default='data/synthetic',
        help='Directory containing combined_features.csv (default: data/synthetic)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)',
    )
    parser.add_argument(
        '--label_source',
        default='default',
        choices=['default', 'signal', 'protocol'],
        help=(
            'Which label column to use as the classification target. '
            'default=risk_class (backwards-compat), '
            'signal=risk_class_signal (IMU thresholds, has circularity), '
            'protocol=risk_class_protocol (time-segments, zero circularity)'
        ),
    )
    parser.add_argument(
        '--cv_group',
        default='auto',
        choices=['auto', 'session', 'participant'],
        help=(
            'Cross-validation grouping. auto uses participant folds for real '
            'data when participant_id exists; session is suitable for Phase I.'
        ),
    )
    parser.add_argument(
        '--models_dir',
        default='ml/models',
        help='Directory for trained model files (default: ml/models)',
    )
    parser.add_argument(
        '--eval_dir',
        default='ml/evaluation',
        help='Directory for evaluation output files (default: ml/evaluation)',
    )
    parser.add_argument(
        '--write_phase2_provenance',
        action='store_true',
        help='Write Phase II.A frozen-model metadata; requires protocol labels and participant CV.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Allow replacement of existing official Phase II.A model and evaluation outputs.',
    )
    parser.add_argument(
        '--operating_mode',
        choices=['full_hybrid', 'imu_only_fallback'],
        default='full_hybrid',
        help='Mode expected in the dataset manifest and written to Phase II.A provenance.',
    )
    parser.add_argument(
        '--fallback_rf_imu_only',
        action='store_true',
        help='Contingency route: train only RF_IMU for imu_only_fallback data.',
    )
    args = parser.parse_args()
    main(
        args.data_dir,
        args.seed,
        args.label_source,
        args.cv_group,
        args.models_dir,
        args.eval_dir,
        args.write_phase2_provenance,
        " ".join([sys.executable, *sys.argv]),
        args.force,
        args.operating_mode,
        args.fallback_rf_imu_only,
    )
