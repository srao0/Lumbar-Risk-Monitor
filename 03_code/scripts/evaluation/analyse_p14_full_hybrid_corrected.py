#!/usr/bin/env python3
"""
analyse_p14_full_hybrid_corrected.py
================================================================================
Faithful reconstruction of the P14 full-hybrid (Phase II.B) analysis on the
REST-CORRECTED data, regenerating the numbers + figures behind Chapter 8
Phase II.B. The original ad-hoc script was not saved; this reproduces the
methodology documented in results/participant_14_analysis/P14_full_hybrid_analysis_report.md.

Methodology (from the report, §3-§4 + Reproducibility):
  * Target  : risk_class_protocol (drop -1 UNLABELLED).
  * IMU set : available(IMU_FEATURES)  (19 features on P14 full-hybrid).
  * Hybrid  : IMU + available(EMG_FEATURES).
  * Model   : RandomForest(n_estimators=200, min_samples_leaf=3,
              class_weight='balanced', random_state=42) for CV, with
              PER-FOLD MEDIAN IMPUTATION (no row dropping -> all 19,342 windows).
  * CV      : leave-one-SESSION-out (13 P14 sessions).
  * Q1      : per-session + pooled AUC, IMU vs hybrid; wins; paired Wilcoxon
              (session unit); sEMG share of RF importance.
  * Decomp  : AUC on {all windows} / {exclude BASELINE_STATIC} /
              {hard subset = 5 risky vs confusable-safe}.
  * Confusable pairs: each risky movement vs its confusable-safe counterpart.
  * Reclassification: per-movement accuracy change at a fixed-sensitivity
              operating threshold.
  * Q2A     : Phase II.A cohort within-CV vs LOSO (read from the corrected
              fallback evaluation CSVs).
  * Q2B     : corrected population reduced model (no P14) applied to P14 vs a
              P14-personalised reduced LOSO model.

Confusable-safe set (report §3B.1 footnote): CLEAN_FLEXION, PICKUP_SYM,
SIT_TO_STAND_NORMAL, SIT_TO_STAND_FAST.

Run from repo root on the machine with sklearn/scipy/matplotlib:
    py scripts/analyse_p14_full_hybrid_corrected.py

Outputs -> results/participant_14_analysis_corrected/
    corrected_summary.json, q1_per_session.csv, decomposition.csv,
    confusable_pairs.csv, reclassification.csv, q2_personalisation.csv,
    and plots/fig1..fig10 (corrected).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import load
from scipy.stats import wilcoxon, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from ml.training.train_classifier import IMU_FEATURES, EMG_FEATURES  # noqa: E402

DATA = REPO / "data" / "real" / "protocol_train_full_hybrid_restcorrected" / "combined_features.csv"
RED_CSV = REPO / "results" / "fallback_analysis_sets_n9_corrected" / "reduced_pelvis_l3_features.csv"
POP_MODEL = REPO / "ml" / "models" / "fallback_final_n9_corrected" / "rf_reduced_pelvis_l3.joblib"
FB_EVAL = REPO / "results" / "fallback_analysis_sets_n9_corrected" / "evaluation"
OUT = REPO / "results" / "participant_14_analysis_corrected"
PLOTS = OUT / "plots"

TARGET = "risk_class_protocol"
SESSION = "session_id"
RISKY = ["LUMBAR_DOMINANT", "FATIGUE_FLEXION", "FAST_BEND", "SHOULDER_DRIVEN", "PICKUP_ASYM"]
CONFUSABLE_SAFE = ["CLEAN_FLEXION", "PICKUP_SYM", "SIT_TO_STAND_NORMAL", "SIT_TO_STAND_FAST"]
CONFUSABLE_PAIRS = [
    ("Fatigued vs clean flexion", "FATIGUE_FLEXION", "CLEAN_FLEXION"),
    ("Fast vs clean flexion", "FAST_BEND", "CLEAN_FLEXION"),
    ("Shoulder-driven vs clean flexion", "SHOULDER_DRIVEN", "CLEAN_FLEXION"),
    ("Asymmetric vs symmetric pickup", "PICKUP_ASYM", "PICKUP_SYM"),
    ("Lumbar-dominant vs clean flexion", "LUMBAR_DOMINANT", "CLEAN_FLEXION"),
]
SEED = 42

# Exact training order of the corrected reduced deployment model
# (train_fallback_analysis_models.py REDUCED_PELVIS_L3_FEATURES).
REDUCED_FEATURES = [
    "imu_angvel_peak", "imu_angvel_mean", "imu_time_high_velocity", "imu_ldlj",
    "imu_jerk_rms", "imu_jerk_peak", "imu_pelvis_angle_peak", "imu_pelvis_angle_mean",
    "imu_l3_accel_tilt_peak", "imu_l3_accel_tilt_mean", "imu_l3_accel_tilt_range",
    "imu_z_vel", "imu_z_ldlj",
]


def rf():
    """The fixed RF pipeline from the P14 report — per-fold median imputation (so no window is dropped) feeding a balanced 200-tree forest at the canonical seed."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                       class_weight="balanced", random_state=SEED, n_jobs=-1)),
    ])


def spec_at_sens(y, p, target_sens):
    """Specificity at the threshold giving >= target sensitivity (pooled)."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target_sens)[0]
    if len(ok) == 0:
        return np.nan
    i = ok[0]
    return 1.0 - fpr[i]


def load_data():
    """Load the corrected P14 full-hybrid combined features, dropping the -1 UNLABELLED windows and coercing the protocol target to int."""
    df = pd.read_csv(DATA, low_memory=False)
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)
    return df


def loso_oof(df, feat_cols):
    """Leave-one-session-out OOF probabilities + per-session AUC + RF importances."""
    sessions = sorted(df[SESSION].unique())
    oof = pd.Series(index=df.index, dtype=float)
    per_session = {}
    importances = []
    for s in sessions:
        tr = df[df[SESSION] != s]
        te = df[df[SESSION] == s]
        if te[TARGET].nunique() < 2 or tr[TARGET].nunique() < 2:
            continue
        pipe = rf()
        pipe.fit(tr[feat_cols].values, tr[TARGET].values)
        p = pipe.predict_proba(te[feat_cols].values)[:, 1]
        oof.loc[te.index] = p
        per_session[s] = roc_auc_score(te[TARGET].values, p)
        importances.append(pipe.named_steps["clf"].feature_importances_)
    imp = pd.Series(np.mean(importances, axis=0), index=feat_cols)
    return oof, per_session, imp


def main():
    """Run the full corrected Phase II.B analysis — Q1 IMU-vs-hybrid LOSO, the gain decomposition, confusable pairs, reclassification, and the Q2A/Q2B personalisation comparisons — writing the summary, CSVs and figures, then echoing pre-correction numbers for side-by-side sanity."""
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    df = load_data()
    imu = [c for c in IMU_FEATURES if c in df.columns]
    emg = [c for c in EMG_FEATURES if c in df.columns]
    hyb = imu + emg
    print(f"Windows: {len(df)} | IMU feats {len(imu)} | EMG feats {len(emg)} | sessions {df[SESSION].nunique()}")

    # ---- Q1: IMU vs hybrid, LOSO-session -----------------------------------
    oof_imu, ps_imu, _ = loso_oof(df, imu)
    oof_hyb, ps_hyb, imp_hyb = loso_oof(df, hyb)
    sess = sorted(set(ps_imu) & set(ps_hyb))
    a_imu = np.array([ps_imu[s] for s in sess])
    a_hyb = np.array([ps_hyb[s] for s in sess])
    delta = a_hyb - a_imu
    wins = int((delta > 0).sum())
    W = wilcoxon(a_hyb, a_imu)
    m = df[TARGET].values
    pooled_imu = roc_auc_score(m, oof_imu.values)
    pooled_hyb = roc_auc_score(m, oof_hyb.values)
    emg_share = float(imp_hyb[emg].sum() / imp_hyb.sum())
    q1 = {"mean_imu": float(a_imu.mean()), "mean_hyb": float(a_hyb.mean()),
          "mean_delta": float(delta.mean()), "wins": f"{wins}/{len(sess)}",
          "wilcoxon_p": float(W.pvalue), "pooled_imu": float(pooled_imu),
          "pooled_hyb": float(pooled_hyb), "emg_importance_share": emg_share}
    pd.DataFrame({"session": sess, "auc_imu": a_imu, "auc_hyb": a_hyb,
                  "delta": delta}).to_csv(OUT / "q1_per_session.csv", index=False)

    # ---- Decomposition (all / ex-baseline / hard subset) -------------------
    lab = df["movement_label"]
    def subset_auc(mask):
        y = df.loc[mask, TARGET].values
        if len(np.unique(y)) < 2:
            return (np.nan, np.nan, int(mask.sum()))
        return (roc_auc_score(y, oof_imu[mask].values),
                roc_auc_score(y, oof_hyb[mask].values), int(mask.sum()))
    masks = {
        "all_windows": pd.Series(True, index=df.index),
        "exclude_baseline": lab != "BASELINE_STATIC",
        "hard_subset": lab.isin(RISKY + CONFUSABLE_SAFE),
    }
    decomp = []
    for name, mk in masks.items():
        ai, ah, n = subset_auc(mk)
        decomp.append({"subset": name, "n": n, "imu": ai, "hyb": ah,
                       "delta": (ah - ai) if ai == ai else np.nan})
    pd.DataFrame(decomp).to_csv(OUT / "decomposition.csv", index=False)

    # ---- Confusable pairs ---------------------------------------------------
    pairs = []
    for label, risky, safe in CONFUSABLE_PAIRS:
        mk = lab.isin([risky, safe])
        y = df.loc[mk, TARGET].values
        if len(np.unique(y)) < 2:
            continue
        ai = roc_auc_score(y, oof_imu[mk].values)
        ah = roc_auc_score(y, oof_hyb[mk].values)
        pairs.append({"decision": label, "imu": ai, "hyb": ah, "delta": ah - ai})
    pd.DataFrame(pairs).to_csv(OUT / "confusable_pairs.csv", index=False)

    # ---- Specificity at fixed sensitivity (pooled) -------------------------
    spec_rows = []
    for name, mk in masks.items():
        y = df.loc[mk, TARGET].values
        for ts in (0.85, 0.90):
            spec_rows.append({"subset": name, "sens": ts,
                              "spec_imu": spec_at_sens(y, oof_imu[mk].values, ts),
                              "spec_hyb": spec_at_sens(y, oof_hyb[mk].values, ts)})
    pd.DataFrame(spec_rows).to_csv(OUT / "specificity.csv", index=False)

    # ---- Gain vs IMU strength (Spearman) -----------------------------------
    rho, pval = spearmanr(a_imu, delta)
    q1["gain_vs_strength_rho"] = float(rho)
    q1["gain_vs_strength_p"] = float(pval)

    # ---- Reclassification: per-movement accuracy at sens=0.85 (pooled thr) -
    fpr, tpr, thr = roc_curve(m, oof_hyb.values)
    ok = np.where(tpr >= 0.85)[0]
    thr_hyb = thr[ok[0]] if len(ok) else 0.5
    fpr2, tpr2, thr2 = roc_curve(m, oof_imu.values)
    ok2 = np.where(tpr2 >= 0.85)[0]
    thr_imu = thr2[ok2[0]] if len(ok2) else 0.5
    pred_imu = (oof_imu.values >= thr_imu).astype(int)
    pred_hyb = (oof_hyb.values >= thr_hyb).astype(int)
    recl = []
    for mv in df["movement_label"].unique():
        mk = (df["movement_label"] == mv).values
        y = m[mk]
        acc_i = (pred_imu[mk] == y).mean()
        acc_h = (pred_hyb[mk] == y).mean()
        recl.append({"movement": mv, "n": int(mk.sum()), "acc_imu": acc_i,
                     "acc_hyb": acc_h, "delta": acc_h - acc_i})
    pd.DataFrame(recl).sort_values("delta").to_csv(OUT / "reclassification.csv", index=False)

    # ---- Q2A: Phase II.A within vs LOSO (corrected fallback CSVs) -----------
    q2a = {"within_mean": float("nan"), "loso_mean": float("nan"), "delta": float("nan"),
           "wins": "NA", "wilcoxon_p": float("nan")}
    try:
        rw = pd.read_csv(FB_EVAL / "reduced_within.csv").rename(columns={"participant_id": "pid"})
        rl = pd.read_csv(FB_EVAL / "reduced_loso.csv").rename(columns={"held_out": "pid"})
        merged = rw[["pid", "auc"]].merge(rl[["pid", "auc"]], on="pid", suffixes=("_within", "_loso"))
        dq2a = merged["auc_within"].values - merged["auc_loso"].values
        Wq2a = wilcoxon(merged["auc_within"].values, merged["auc_loso"].values)
        q2a = {"within_mean": float(merged["auc_within"].mean()),
               "loso_mean": float(merged["auc_loso"].mean()),
               "delta": float(dq2a.mean()), "wins": f"{int((dq2a > 0).sum())}/{len(dq2a)}",
               "wilcoxon_p": float(Wq2a.pvalue)}
    except Exception as e:
        q2a["error"] = str(e)

    # ---- Q2B: corrected population reduced model vs P14-personalised --------
    q2b = {"pop_pooled": float("nan"), "pers_pooled": float("nan"),
           "delta_mean_persession": float("nan"), "wins": "NA", "wilcoxon_p": float("nan")}
    try:
        red_feats = [c for c in REDUCED_FEATURES if c in df.columns]
        pop = load(POP_MODEL)
        # population model = bare RF trained without P14 on REDUCED_FEATURES order.
        # Impute (report methodology) and keep a named DataFrame so the model's
        # feature_names_in_ alignment is exact.
        imp = SimpleImputer(strategy="median").fit(df[red_feats])
        Xred = pd.DataFrame(imp.transform(df[red_feats]), columns=red_feats, index=df.index)
        pop_pooled = roc_auc_score(m, pop.predict_proba(Xred)[:, 1])
        oof_red, ps_red, _ = loso_oof(df, red_feats)
        pers_pooled = roc_auc_score(m, oof_red.values)
        pop_ps = {}
        for s in sess:
            mk = (df[SESSION] == s).values
            y = m[mk]
            if len(np.unique(y)) < 2:
                continue
            pop_ps[s] = roc_auc_score(y, pop.predict_proba(Xred[mk])[:, 1])
        s2 = sorted(set(pop_ps) & set(ps_red))
        dq2b = np.array([ps_red[s] for s in s2]) - np.array([pop_ps[s] for s in s2])
        Wq2b = wilcoxon([ps_red[s] for s in s2], [pop_ps[s] for s in s2])
        q2b = {"pop_pooled": float(pop_pooled), "pers_pooled": float(pers_pooled),
               "delta_mean_persession": float(dq2b.mean()),
               "wins": f"{int((dq2b > 0).sum())}/{len(s2)}", "wilcoxon_p": float(Wq2b.pvalue)}
    except Exception as e:
        q2b["error"] = str(e)

    summary = {"q1": q1, "q2a": q2a, "q2b": q2b,
               "decomposition": decomp, "confusable_pairs": pairs,
               "imu_n_features": len(imu), "emg_n_features": len(emg)}
    (OUT / "corrected_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- Figures (corrected) -----------------------------------------------
    # fig1: per-session AUC IMU vs hybrid
    plt.figure(figsize=(9, 4))
    x = np.arange(len(sess))
    plt.plot(x, a_imu, "o--", label="IMU-only", color="#888")
    plt.plot(x, a_hyb, "o-", label="IMU+sEMG", color="#1f77b4")
    plt.xticks(x, [s.split("__")[-1] for s in sess], rotation=45, ha="right", fontsize=7)
    plt.ylabel("AUC"); plt.title(f"Q1 per-session LOSO  (mean Δ={delta.mean():+.3f}, {wins}/{len(sess)}, p={W.pvalue:.4f})")
    plt.legend(); plt.tight_layout(); plt.savefig(PLOTS / "fig1_q1_per_session_auc.png", dpi=150); plt.close()

    # fig7: AUC decomposition
    plt.figure(figsize=(7, 4))
    names = [d["subset"] for d in decomp]
    plt.bar(np.arange(len(decomp)) - 0.2, [d["imu"] for d in decomp], 0.4, label="IMU", color="#888")
    plt.bar(np.arange(len(decomp)) + 0.2, [d["hyb"] for d in decomp], 0.4, label="IMU+sEMG", color="#1f77b4")
    plt.xticks(np.arange(len(decomp)), names, fontsize=8); plt.ylabel("AUC"); plt.ylim(0.5, 0.9)
    plt.title("Q1 gain decomposition"); plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS / "fig7_auc_decomposition.png", dpi=150); plt.close()

    # fig8: confusable pairs
    plt.figure(figsize=(8, 4))
    ylab = [p["decision"] for p in pairs]
    plt.barh(np.arange(len(pairs)), [p["delta"] for p in pairs], color="#1f77b4")
    plt.yticks(np.arange(len(pairs)), ylab, fontsize=8); plt.xlabel("Δ AUC (hybrid − IMU)")
    plt.title("Confusable-pair discrimination"); plt.tight_layout()
    plt.savefig(PLOTS / "fig8_confusable_pairs.png", dpi=150); plt.close()

    # fig10: reclassification
    rdf = pd.DataFrame(recl).sort_values("delta")
    plt.figure(figsize=(8, 5))
    colors = ["#d62728" if d < 0 else "#2ca02c" for d in rdf["delta"]]
    plt.barh(np.arange(len(rdf)), rdf["delta"], color=colors)
    plt.yticks(np.arange(len(rdf)), rdf["movement"], fontsize=7)
    plt.xlabel("Δ accuracy at sens≈0.85 (hybrid − IMU)"); plt.title("Reclassification by movement")
    plt.tight_layout(); plt.savefig(PLOTS / "fig10_reclassification.png", dpi=150); plt.close()

    # fig5: personalised vs population (Q2A cohort + Q2B P14)
    plt.figure(figsize=(7, 4))
    plt.bar([0, 1], [q2a["loso_mean"], q2a["within_mean"]], 0.5, color=["#888", "#1f77b4"])
    plt.bar([3, 4], [q2b["pop_pooled"], q2b["pers_pooled"]], 0.5, color=["#888", "#1f77b4"])
    plt.xticks([0, 1, 3, 4], ["II.A pop", "II.A pers", "P14 pop", "P14 pers"], fontsize=8)
    plt.ylabel("AUC"); plt.title("Personalised vs population"); plt.tight_layout()
    plt.savefig(PLOTS / "fig5_personalised_vs_population.png", dpi=150); plt.close()

    # ---- console report ----------------------------------------------------
    print("\n================ CORRECTED P14 PHASE II.B ================")
    print(f"Q1  IMU {q1['mean_imu']:.3f} -> hybrid {q1['mean_hyb']:.3f}  "
          f"(Δ {q1['mean_delta']:+.3f}; {q1['wins']}; p={q1['wilcoxon_p']:.4f}; "
          f"pooled {pooled_imu:.3f}->{pooled_hyb:.3f}; sEMG importance {emg_share:.0%})")
    print(f"    gain-vs-IMU-strength Spearman rho={rho:+.2f} p={pval:.2f}")
    print("Decomposition:")
    for d in decomp:
        print(f"    {d['subset']:<18} n={d['n']:>6}  IMU {d['imu']:.3f}  hyb {d['hyb']:.3f}  Δ {d['delta']:+.3f}")
    print("Confusable pairs:")
    for p in pairs:
        print(f"    {p['decision']:<36} Δ {p['delta']:+.3f}  ({p['imu']:.3f}->{p['hyb']:.3f})")
    print(f"Q2A  within {q2a['within_mean']:.3f} vs LOSO {q2a['loso_mean']:.3f}  "
          f"Δ {q2a['delta']:+.3f}  {q2a['wins']}  p={q2a['wilcoxon_p']:.4f}")
    print(f"Q2B  pop {q2b['pop_pooled']:.3f} vs personalised {q2b['pers_pooled']:.3f}  "
          f"per-session Δ {q2b['delta_mean_persession']:+.3f}  {q2b['wins']}  p={q2b['wilcoxon_p']:.4f}")
    print(f"\nSaved -> {OUT}")
    print("Compare against report (pre-correction): Q1 Δ+0.045 (0.799->0.844, pooled 0.794->0.835, 12/13, p=0.0005),")
    print("  hard-subset Δ+0.021, LUMBAR Δ+0.009, FATIGUE Δ+0.046, Q2B Δ+0.138.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="P14 Phase II.B analysis (corrected by default; "
                                             "point --data_dir at pre-correction data to validate).")
    ap.add_argument("--data_dir", default=str(DATA.parent),
                    help="dir containing combined_features.csv")
    ap.add_argument("--pop_model", default=str(POP_MODEL))
    ap.add_argument("--fb_eval", default=str(FB_EVAL))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    DATA = Path(a.data_dir) / "combined_features.csv"
    POP_MODEL = Path(a.pop_model)
    FB_EVAL = Path(a.fb_eval)
    OUT = Path(a.out)
    PLOTS = OUT / "plots"
    main()
