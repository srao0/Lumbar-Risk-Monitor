#!/usr/bin/env python3
"""Evaluate prepared fallback analysis sets with corrected methodology.

Trains the IMU-only fallback RF on each frozen analysis set and reports
within-participant (temporal 80/20) and LOSO discrimination. The primary vs
reduced Pelvis-L3 comparison feeds the deployment recommendation; the exact
paired Wilcoxon below keeps the p-value honest at the small n the cohort allows.
"""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve

# Full 4-IMU sensor suite (trunk + pelvis + L3); only the cleaned participants
# that retain all four IMUs qualify for this set.
PRIMARY_FEATURES = [
    "imu_trunk_angle_peak","imu_trunk_angle_mean","imu_angvel_peak","imu_angvel_mean",
    "imu_time_in_risk_zone","imu_time_high_velocity","imu_ldlj","imu_jerk_rms","imu_jerk_peak",
    "imu_ldlj_multiaxis","imu_compensation_index","imu_lumbopelv_ratio",
    "imu_pelvis_angle_peak","imu_pelvis_angle_mean",
    "imu_z_flex","imu_z_vel","imu_z_ldlj",
]
# Reduced Pelvis-L3 set: drops trunk-derived features so participants with
# T12/T4 IMU dropout still qualify. This is the recommended deployment model.
REDUCED_FEATURES = [
    "imu_angvel_peak","imu_angvel_mean","imu_time_high_velocity",
    "imu_ldlj","imu_jerk_rms","imu_jerk_peak",
    "imu_pelvis_angle_peak","imu_pelvis_angle_mean",
    "imu_l3_accel_tilt_peak","imu_l3_accel_tilt_mean","imu_l3_accel_tilt_range",
    "imu_z_vel","imu_z_ldlj",
]

def youden_threshold(y, p):
    """Operating threshold maximising Youden's J (sens + spec - 1)."""
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])

def compute_metrics(y, p, threshold=None):
    """AUC/sens/spec/F1 at the Youden threshold (or a supplied one)."""
    if threshold is None: threshold = youden_threshold(y, p)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    sens = tp/(tp+fn) if (tp+fn) else float("nan")
    spec = tn/(tn+fp) if (tn+fp) else float("nan")
    return {"auc": float(roc_auc_score(y, p)) if len(np.unique(y))==2 else float("nan"),
            "sens": float(sens), "spec": float(spec),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "youden_threshold": float(threshold),
            "n": int(len(y)), "safe_n": int((y==0).sum()), "risky_n": int((y==1).sum())}

def fit_predict(train, test, features, seed):
    """Fit the fallback RF on train and return risky-class probabilities for test."""
    clf = RandomForestClassifier(n_estimators=500, random_state=seed,
                                 class_weight="balanced", min_samples_leaf=3,
                                 n_jobs=-1)
    clf.fit(train[features], train["risk_class"].astype(int))
    return clf.predict_proba(test[features])[:,1]

def within_participant(df, features, seed):
    """Per-participant temporal 80/20 split (chronological, no window shuffling
    so future reps cannot leak into the past). Degenerate single-class splits
    are flagged rather than scored."""
    rows = []
    for pid, g in df.groupby("participant_id"):
        g = g.sort_values("window_centre_ms")
        cut = int(len(g)*0.8)
        tr, te = g.iloc[:cut], g.iloc[cut:]
        if tr["risk_class"].nunique()<2 or te["risk_class"].nunique()<2:
            rows.append({"participant_id": pid, "note":"degenerate"}); continue
        p = fit_predict(tr, te, features, seed)
        rows.append({"participant_id": pid, **compute_metrics(te["risk_class"].to_numpy(), p)})
    return pd.DataFrame(rows)

def loso(df, features, seed):
    """Leave-one-subject-out: each participant held out once to estimate
    cross-person generalisation."""
    rows = []
    for pid in sorted(df["participant_id"].unique()):
        tr = df[df["participant_id"]!=pid]; te = df[df["participant_id"]==pid]
        if tr["risk_class"].nunique()<2 or te["risk_class"].nunique()<2:
            rows.append({"held_out": pid, "note":"degenerate"}); continue
        p = fit_predict(tr, te, features, seed)
        rows.append({"held_out": pid, **compute_metrics(te["risk_class"].to_numpy(), p)})
    return pd.DataFrame(rows)

def exact_paired_wilcoxon_p(a, b):
    """Exact two-sided paired Wilcoxon signed-rank p by full sign enumeration.

    With only a handful of shared LOSO folds the normal approximation is
    unreliable, so the null is built by enumerating all 2**n sign flips. Returns
    (p, n_effective) where n_effective excludes zero-difference (tied) pairs.
    """
    diffs = [x-y for x,y in zip(a,b) if np.isfinite(x) and np.isfinite(y) and x!=y]
    n = len(diffs)
    if n==0: return 1.0, 0
    aso = sorted((abs(d), i) for i,d in enumerate(diffs))
    ranks = [0.0]*n; pos = 0
    while pos < n:
        end = pos
        while end+1 < n and aso[end+1][0]==aso[pos][0]: end += 1
        avg = (pos+1+end+1)/2.0
        for _,i in aso[pos:end+1]: ranks[i] = avg
        pos = end+1
    obs = min(sum(r for r,d in zip(ranks,diffs) if d>0),
              sum(r for r,d in zip(ranks,diffs) if d<0))
    cnt = 0; total = 2**n
    for s in itertools.product([-1,1], repeat=n):
        wp = sum(r for r,x in zip(ranks,s) if x>0)
        wm = sum(r for r,x in zip(ranks,s) if x<0)
        if min(wp,wm) <= obs + 1e-12: cnt += 1
    return cnt/total, n

def evaluate(path, features, out_dir, seed, tag):
    """Run both CV schemes on one analysis set; write per-fold CSVs and return
    its mean/SD AUC summary for the cross-set comparison."""
    df = pd.read_csv(path)
    df = df[df["risk_class"].isin([0,1])].copy()
    missing = sorted(set(features)-set(df.columns))
    if missing: raise ValueError(f"{path} missing: {missing}")
    df = df.dropna(subset=features+["risk_class","participant_id","window_centre_ms"])
    out_dir.mkdir(parents=True, exist_ok=True)
    w = within_participant(df, features, seed)
    l = loso(df, features, seed)
    w.to_csv(out_dir/f"{tag}_within.csv", index=False)
    l.to_csv(out_dir/f"{tag}_loso.csv", index=False)
    return {"tag":tag,"dataset":str(path),"features":features,
            "within_path":str(out_dir/f"{tag}_within.csv"),
            "loso_path":str(out_dir/f"{tag}_loso.csv"),
            "within_mean_auc":float(w["auc"].mean()) if "auc" in w else float("nan"),
            "within_std_auc": float(w["auc"].std())  if "auc" in w else float("nan"),
            "loso_mean_auc":  float(l["auc"].mean()) if "auc" in l else float("nan"),
            "loso_std_auc":   float(l["auc"].std())  if "auc" in l else float("nan")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_dir", type=Path, default=Path("results/fallback_analysis_sets"))
    ap.add_argument("--out_dir",      type=Path, default=Path("results/fallback_analysis_sets/evaluation_corrected"))
    ap.add_argument("--seed",         type=int,  default=42)
    a = ap.parse_args()
    primary = evaluate(a.analysis_dir/"primary_4imu_cleaned_features.csv", PRIMARY_FEATURES, a.out_dir, a.seed, "primary")
    reduced = evaluate(a.analysis_dir/"reduced_pelvis_l3_features.csv",  REDUCED_FEATURES, a.out_dir, a.seed, "reduced")
    p_l = pd.read_csv(primary["loso_path"]); r_l = pd.read_csv(reduced["loso_path"])
    shared = sorted(set(p_l["held_out"]) & set(r_l["held_out"]))
    pa = [p_l.set_index("held_out").loc[s,"auc"] for s in shared]
    rb = [r_l.set_index("held_out").loc[s,"auc"] for s in shared]
    pval, neff = exact_paired_wilcoxon_p(pa, rb)
    summary = {"primary":primary, "reduced":reduced,
               "wilcoxon":{"shared":shared,"primary_auc":pa,"reduced_auc":rb,
                           "two_sided_p":float(pval),"n_eff":int(neff),
                           "min_possible_p": float(2**(1-neff)) if neff>0 else 1.0}}
    (a.out_dir/"evaluation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
