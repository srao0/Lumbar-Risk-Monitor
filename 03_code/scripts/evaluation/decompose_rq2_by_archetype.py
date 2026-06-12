#!/usr/bin/env python3
"""
decompose_rq2_by_archetype.py
================================================================================
RQ2 follow-up for the P14 full-hybrid (rest-corrected) re-freeze.

The training run reports only an AGGREGATE sEMG lift (RF: IMU 0.808 ->
IMU+sEMG 0.830, +0.022 AUC). The RQ2 question is *where* that lift lives:
core lumbar-flexion risk, or compensation / fatigue behaviours?

This script answers that WITHOUT retraining. It:
  1. Reloads the corrected full-hybrid feature table.
  2. Rebuilds the IMU and IMU+sEMG matrices with the exact same feature
     columns, target rename, and NaN-row policy used in train_classifier.py
     (constants imported directly so feature ORDER is identical).
  3. For each LOSO fold, loads the frozen RF model for that fold and scores
     its held-out session -> proper out-of-fold (OOF) probabilities. This is
     a faithful reconstruction of the predictions the aggregate AUC came from.
  4. Tags every window with its movement archetype + risk_criteria.
  5. Computes a ONE-VS-SAFE AUC per archetype family (positives = risk windows
     of that type; negatives = ALL safe windows) for IMU and IMU+sEMG, plus the
     delta. This isolates the lift by risk type.

Run from the repo root on the machine that produced the models (needs sklearn
/ joblib), e.g.:

    py scripts/decompose_rq2_by_archetype.py

Outputs a printed table and writes:
    results/p14_fullhybrid_corrected/evaluation/rq2_archetype_decomposition.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import roc_auc_score

# --- repo root on path so we can import the SAME feature definitions ----------
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from ml.training.train_classifier import (  # noqa: E402
    IMU_FEATURES,
    IMU_EMG_FEATURES,
    TARGET_COL,        # "risk_class"
    SESSION_COL,       # "session_id"
)

# --- paths / config -----------------------------------------------------------
DATA_DIR  = REPO / "data" / "real" / "protocol_train_full_hybrid_restcorrected"
MODEL_DIR = REPO / "ml" / "models" / "p14_fullhybrid_corrected"
EVAL_DIR  = REPO / "results" / "p14_fullhybrid_corrected" / "evaluation"
FEATURES_CSV = DATA_DIR / "combined_features.csv"
TARGET_SRC = "risk_class_protocol"     # --label_source protocol
GROUP_COL  = SESSION_COL               # --cv_group session  (P14 = one participant)

# Map movement_label archetypes -> interpretable risk families.
# LUMBAR_DOMINANT is the core lumbar-flexion behaviour; the rest are
# compensation / fatigue / dynamic behaviours where sEMG is hypothesised to add
# the most.
FAMILY = {
    "LUMBAR_DOMINANT": "core_flexion",
    "SHOULDER_DRIVEN": "compensation",
    "PICKUP_ASYM":     "compensation",
    "FATIGUE_FLEXION": "fatigue",
    "FAST_BEND":       "dynamic",
}


def build_matrices():
    """Replicate train_classifier.py's dataset construction exactly."""
    df = pd.read_csv(FEATURES_CSV, low_memory=False)

    # Drop unlabelled windows. risk_class_protocol uses -1 (UNLABELLED) as a
    # sentinel, NOT NaN -- train_classifier.py filters these before the
    # 'Loaded N labelled windows' print. Must replicate, else the 590 unlabelled
    # rows leak in as a third class (breaks AUC) and pollute the safe/risk masks.
    df = df[df[TARGET_SRC].isin([0, 1])].copy()

    # Target rename: risk_class_protocol -> risk_class (TARGET_COL).
    if TARGET_SRC != TARGET_COL:
        if TARGET_COL in df.columns:
            df = df.drop(columns=[TARGET_COL])
        df = df.rename(columns={TARGET_SRC: TARGET_COL})
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    def available(cols):
        return [c for c in cols if c in df.columns]

    imu_cols     = available(IMU_FEATURES)
    imu_emg_cols = available(IMU_EMG_FEATURES)

    # Keep movement_label + risk_criteria on the side, indexed identically.
    side = df[[SESSION_COL, TARGET_COL]].copy()
    for c in ("movement_label", "risk_criteria"):
        side[c] = df[c] if c in df.columns else "NA"

    # Same NaN-row policy as training (index PRESERVED -> aligns with `side`).
    df_imu  = df[[SESSION_COL, TARGET_COL] + imu_cols].dropna(subset=imu_cols)
    df_fuse = df[[SESSION_COL, TARGET_COL] + imu_emg_cols].dropna(subset=imu_emg_cols)

    return df_imu, df_fuse, imu_cols, imu_emg_cols, side


def oof_probs(dataset, feat_cols, model_prefix, sessions):
    """Reconstruct OOF probabilities by scoring each fold's held-out session
    with that fold's frozen RF model. Returns a Series indexed like `dataset`."""
    out = {}
    for fold_idx, test_session in enumerate(sessions):
        fold_no = fold_idx + 1
        model_path = MODEL_DIR / f"{model_prefix}_fold{fold_no}.joblib"
        te = dataset[dataset[GROUP_COL] == test_session]
        if len(te) == 0 or te[TARGET_COL].nunique() < 2:
            continue                      # matches training's skip rule
        if not model_path.exists():
            print(f"  [warn] missing {model_path.name} (fold {fold_no} / "
                  f"{test_session}) -- skipped")
            continue
        pipe = load(model_path)
        prob = pipe.predict_proba(te[feat_cols].values)[:, 1]
        for idx, p in zip(te.index, prob):
            out[idx] = p
    return pd.Series(out, name=model_prefix)


def one_vs_safe_auc(prob, y_true, mask_pos, mask_safe):
    """AUC separating one risk family (positives) from all safe windows."""
    sel = mask_pos | mask_safe
    y = y_true[sel].values
    s = prob[sel].values
    keep = ~np.isnan(s)
    y, s = y[keep], s[keep]
    if len(np.unique(y)) < 2:
        return np.nan, int(mask_pos.sum()), int(mask_safe.sum())
    return roc_auc_score(y, s), int((y == 1).sum()), int((y == 0).sum())


def main():
    df_imu, df_fuse, imu_cols, imu_emg_cols, side = build_matrices()
    sessions = sorted(df_imu[GROUP_COL].unique().tolist())
    print(f"Sessions (fold order): {sessions}")
    print(f"IMU matrix: {len(df_imu)} rows | IMU+sEMG matrix: {len(df_fuse)} rows")

    p_imu  = oof_probs(df_imu,  imu_cols,     "RF_IMU",     sessions)
    p_fuse = oof_probs(df_fuse, imu_emg_cols, "RF_IMU_EMG", sessions)

    # Common windows scored under BOTH conditions (like-for-like).
    common = p_imu.index.intersection(p_fuse.index)
    print(f"OOF windows scored under both conditions: {len(common)}")

    y    = side.loc[common, TARGET_COL]
    mv   = side.loc[common, "movement_label"]
    crit = side.loc[common, "risk_criteria"]
    fam  = mv.map(FAMILY).fillna("other")
    pi   = p_imu.loc[common]
    pf   = p_fuse.loc[common]

    # Reference: pooled OOF AUC (note: differs slightly from the mean-of-folds
    # 0.808 / 0.830 in summary_results.csv -- different estimator).
    print("\n--- pooled OOF AUC (reference; mean-of-folds is the headline) ---")
    print(f"  IMU      : {roc_auc_score(y, pi):.4f}")
    print(f"  IMU+sEMG : {roc_auc_score(y, pf):.4f}")

    safe_mask = (y == 0)
    rows = []

    def add(group_name, label, series):
        pos = (y == 1) & (series == label)
        if pos.sum() == 0:
            return
        auc_i, np_i, ns = one_vs_safe_auc(pi, y, pos, safe_mask)
        auc_f, _,    _  = one_vs_safe_auc(pf, y, pos, safe_mask)
        rows.append({
            "grouping": group_name, "subgroup": label,
            "n_risk": np_i, "n_safe": ns,
            "auc_imu": round(auc_i, 4), "auc_imu_emg": round(auc_f, 4),
            "delta_auc": round(auc_f - auc_i, 4),
            "flag": "LOW_N" if np_i < 50 else "",
        })

    for label in sorted(fam.unique()):
        add("family", label, fam)
    for label in sorted(mv.dropna().unique()):
        add("movement_label", label, mv)
    for label in sorted(crit.dropna().unique()):
        add("risk_criteria", label, crit)

    res = pd.DataFrame(rows)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = EVAL_DIR / "rq2_archetype_decomposition.csv"
    res.to_csv(out_csv, index=False)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_rows", 100)
    print("\n=== RQ2 one-vs-safe AUC decomposition (RF) ===")
    print(res.to_string(index=False))
    print(f"\nSaved -> {out_csv}")
    print("\nRead: small/near-zero delta_auc on core_flexion (LUMBAR_DOMINANT) with "
          "larger delta on compensation/fatigue would confirm the frozen "
          "conclusion -- sEMG helps confounded behaviours, not core flexion.")


if __name__ == "__main__":
    main()
