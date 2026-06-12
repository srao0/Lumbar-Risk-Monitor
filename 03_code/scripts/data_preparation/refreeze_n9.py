#!/usr/bin/env python3
"""
Pass 0 re-freeze: extend the IMU-only fallback evidence from n=7 to n=9.

Faithful replica of:
  scripts/training/prepare_fallback_analysis_sets.py
  scripts/evaluation/evaluate_fallback_analysis_sets.py

Step 1 reproduces the frozen n=7 numbers as a TRUST CHECK. Only if those match
(primary within 0.6966 / LOSO 0.6542; reduced within 0.7867 / LOSO 0.6244;
Wilcoxon p=0.8125; primary rows 7294; reduced rows 10296) do we trust the n=9 run.

Step 2 builds n=6 primary (+P08) / n=9 reduced (+P08,P09) and re-evaluates.
Outputs go to results/fallback_analysis_sets_n9/ (the frozen n=7 dir is untouched).
"""
from __future__ import annotations
import json, hashlib, itertools, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "results" / "fallback_analysis_sets_n9"
SEED = 42
CAP  = 60.0

PRIMARY_FEATURES = ["imu_trunk_angle_peak","imu_trunk_angle_mean","imu_angvel_peak","imu_angvel_mean",
    "imu_time_in_risk_zone","imu_time_high_velocity","imu_ldlj","imu_jerk_rms","imu_jerk_peak",
    "imu_ldlj_multiaxis","imu_compensation_index","imu_lumbopelv_ratio",
    "imu_pelvis_angle_peak","imu_pelvis_angle_mean","imu_z_flex","imu_z_vel","imu_z_ldlj"]
REDUCED_FEATURES = ["imu_angvel_peak","imu_angvel_mean","imu_time_high_velocity",
    "imu_ldlj","imu_jerk_rms","imu_jerk_peak","imu_pelvis_angle_peak","imu_pelvis_angle_mean",
    "imu_l3_accel_tilt_peak","imu_l3_accel_tilt_mean","imu_l3_accel_tilt_range","imu_z_vel","imu_z_ldlj"]
ID_COLS = ["session_id","participant_id","window_centre_ms","movement_label","risk_class_protocol","risk_class"]

P = {
 "combined": ROOT/"data/real/protocol_train_fallback_2session/combined_features.csv",
 "p02": ROOT/"data/real/protocol_train_fallback_2session/participant_02/session_001/feature_matrix.pre_quality_exclusions_20260528_112742.csv",
 "p03": ROOT/"data/real/protocol_train_fallback/participant_03/session_001/feature_matrix.csv",
 "p08": ROOT/"data/real/protocol_train_fallback_recovered/participant_08/session_001/feature_matrix.csv",
 "p09": ROOT/"data/real/protocol_train_fallback/participant_09/session_001_restcorrected/feature_matrix.csv",
}

def load_session(path, pid, sess):
    df = pd.read_csv(path, low_memory=False)
    df.insert(0, "participant_id", pid)
    df.insert(1, "session_id", f"{pid}__{sess}")
    return df

def labelled(df):
    tgt = "risk_class_protocol" if "risk_class_protocol" in df.columns else "risk_class"
    out = df[df[tgt].isin([0,1])].copy()
    out["risk_class"] = out[tgt].astype(int)
    return out

def cap_pelvis(df):
    out = df.copy()
    for col in ["imu_pelvis_angle_peak","imu_pelvis_angle_mean"]:
        if col in out.columns:
            out[col] = out[col].astype(float).clip(upper=CAP)
    return out

def build_sets(combined, extra_full, extra_reduced):
    """Assemble the primary (full-4-IMU) and reduced (Pelvis-L3) analysis tables.

    Mirrors prepare_fallback_analysis_sets.py exactly so the frozen numbers reproduce:
    primary excludes P02 and skips feature-level dropna; reduced keeps only the
    Pelvis-L3 feature subset and drops rows missing any of them. Pelvis angles are
    capped at CAP to tame the rare physically implausible spikes. extra_full rows are
    relabelled from risk_class_protocol because the primary set filters on risk_class.
    """
    comb = combined.copy()
    for d in (extra_full or []):
        d = d.copy()
        if "risk_class_protocol" in d.columns:
            d["risk_class"] = d["risk_class_protocol"]   # primary filters on risk_class -> use protocol label
        comb = pd.concat([comb, d], ignore_index=True, sort=False)
    # primary: exclude P02, keep labelled, cap pelvis (NO feature dropna here, matching prepare)
    primary = comb[~comb["participant_id"].isin({"participant_02"})].copy()
    primary = primary[primary["risk_class"].isin([0,1])].copy()
    primary = cap_pelvis(primary)
    # reduced: concat combined + extra_reduced, labelled, keep cols, dropna on reduced features, cap
    reduced_base = pd.concat([comb] + list(extra_reduced), ignore_index=True, sort=False)
    reduced_base = labelled(reduced_base)
    keep = [c for c in ID_COLS + REDUCED_FEATURES if c in reduced_base.columns]
    reduced = reduced_base[keep].dropna(subset=REDUCED_FEATURES).copy()
    reduced = cap_pelvis(reduced)
    return primary, reduced

def fit_predict(tr, te, feats):
    clf = RandomForestClassifier(n_estimators=500, random_state=SEED, class_weight="balanced",
                                 min_samples_leaf=3, n_jobs=-1)
    clf.fit(tr[feats], tr["risk_class"].astype(int))
    return clf.predict_proba(te[feats])[:,1]

def within(df, feats):
    # Within-participant evaluation: first 80% of each person's time-ordered windows
    # train, last 20% test. Participants without both risk classes in either split are
    # marked degenerate (AUC undefined) rather than dropped silently.
    rows = []
    for pid, g in df.groupby("participant_id"):
        g = g.sort_values("window_centre_ms"); cut = int(len(g)*0.8)
        tr, te = g.iloc[:cut], g.iloc[cut:]
        if tr["risk_class"].nunique()<2 or te["risk_class"].nunique()<2:
            rows.append({"participant_id":pid,"auc":np.nan,"note":"degenerate"}); continue
        p = fit_predict(tr, te, feats)
        rows.append({"participant_id":pid,"auc":float(roc_auc_score(te["risk_class"], p)),"n":int(len(te))})
    return pd.DataFrame(rows)

def loso(df, feats):
    # Leave-one-subject-out: train on everyone but one participant, test on the held-out
    # one — the honest generalisation estimate since no held-out windows are ever seen.
    rows = []
    for pid in sorted(df["participant_id"].unique()):
        tr = df[df.participant_id!=pid]; te = df[df.participant_id==pid]
        if tr["risk_class"].nunique()<2 or te["risk_class"].nunique()<2:
            rows.append({"held_out":pid,"auc":np.nan,"note":"degenerate"}); continue
        p = fit_predict(tr, te, feats)
        rows.append({"held_out":pid,"auc":float(roc_auc_score(te["risk_class"], p)),"n":int(len(te))})
    return pd.DataFrame(rows)

def evaluate(df, feats):
    df = df[df["risk_class"].isin([0,1])].copy()
    df = df.dropna(subset=feats+["risk_class","participant_id","window_centre_ms"])
    return within(df, feats), loso(df, feats)

def exact_paired_wilcoxon_p(a, b):
    """Exact two-sided paired Wilcoxon signed-rank p-value by full sign enumeration.

    Used instead of scipy's normal approximation because the shared-participant count
    is tiny (n<10), where the exact 2**n permutation test is both feasible and correct.
    Zero differences are dropped (standard signed-rank handling); returns (p, n_eff).
    """
    diffs = [x-y for x,y in zip(a,b) if np.isfinite(x) and np.isfinite(y) and x!=y]
    n = len(diffs)
    if n==0: return 1.0, 0
    aso = sorted((abs(d), i) for i,d in enumerate(diffs))
    ranks=[0.0]*n; pos=0
    while pos<n:
        end=pos
        while end+1<n and aso[end+1][0]==aso[pos][0]: end+=1
        avg=(pos+1+end+1)/2.0
        for _,i in aso[pos:end+1]: ranks[i]=avg
        pos=end+1
    obs=min(sum(r for r,d in zip(ranks,diffs) if d>0), sum(r for r,d in zip(ranks,diffs) if d<0))
    cnt=0; total=2**n
    for s in itertools.product([-1,1],repeat=n):
        wp=sum(r for r,x in zip(ranks,s) if x>0); wm=sum(r for r,x in zip(ranks,s) if x<0)
        if min(wp,wm)<=obs+1e-12: cnt+=1
    return cnt/total, n

def summary(primary, reduced, w_p, l_p, w_r, l_r):
    shared = sorted(set(l_p["held_out"]) & set(l_r["held_out"]))
    pa = [float(l_p.set_index("held_out").loc[s,"auc"]) for s in shared]
    rb = [float(l_r.set_index("held_out").loc[s,"auc"]) for s in shared]
    pval, neff = exact_paired_wilcoxon_p(pa, rb)
    return {
      "primary":{"rows":int(len(primary)),"participants":sorted(primary.participant_id.unique().tolist()),
                 "within_mean_auc":float(w_p.auc.mean()),"within_std_auc":float(w_p.auc.std()),
                 "loso_mean_auc":float(l_p.auc.mean()),"loso_std_auc":float(l_p.auc.std())},
      "reduced":{"rows":int(len(reduced)),"participants":sorted(reduced.participant_id.unique().tolist()),
                 "within_mean_auc":float(w_r.auc.mean()),"within_std_auc":float(w_r.auc.std()),
                 "loso_mean_auc":float(l_r.auc.mean()),"loso_std_auc":float(l_r.auc.std())},
      "wilcoxon":{"shared":shared,"primary_auc":pa,"reduced_auc":rb,
                  "two_sided_p":float(pval),"n_eff":int(neff),
                  "min_possible_p":float(2**(1-neff)) if neff>0 else 1.0},
    }

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    combined = pd.read_csv(P["combined"], low_memory=False)
    # sanity: combined risk_class vs protocol agreement
    if "risk_class" in combined and "risk_class_protocol" in combined:
        agree = (combined["risk_class"]==combined["risk_class_protocol"]).mean()
        print(f"[chk] combined risk_class==risk_class_protocol: {agree*100:.2f}%")
    p02 = load_session(P["p02"], "participant_02", "session_001")
    p03 = load_session(P["p03"], "participant_03", "session_001")
    p08 = load_session(P["p08"], "participant_08", "session_001")
    p09 = load_session(P["p09"], "participant_09", "session_001")

    # STEP 1: reproduce n=7
    pr7, rd7 = build_sets(combined, extra_full=[], extra_reduced=[p02, p03])
    wp7, lp7 = evaluate(pr7, PRIMARY_FEATURES)
    wr7, lr7 = evaluate(rd7, REDUCED_FEATURES)
    s7 = summary(pr7, rd7, wp7, lp7, wr7, lr7)
    print("\n===== STEP 1: n=7 REPRODUCTION =====")
    print(json.dumps(s7, indent=2))
    exp = {"prim_rows":7294,"red_rows":10296,"prim_within":0.6966,"prim_loso":0.6542,
           "red_within":0.7867,"red_loso":0.6244,"p":0.8125}
    checks = {
      "primary_rows": s7["primary"]["rows"]==exp["prim_rows"],
      "reduced_rows": s7["reduced"]["rows"]==exp["red_rows"],
      "primary_within": abs(s7["primary"]["within_mean_auc"]-exp["prim_within"])<0.0005,
      "primary_loso": abs(s7["primary"]["loso_mean_auc"]-exp["prim_loso"])<0.0005,
      "reduced_within": abs(s7["reduced"]["within_mean_auc"]-exp["red_within"])<0.0005,
      "reduced_loso": abs(s7["reduced"]["loso_mean_auc"]-exp["red_loso"])<0.0005,
      "wilcoxon_p": abs(s7["wilcoxon"]["two_sided_p"]-exp["p"])<0.0005,
    }
    print("\n[REPRO CHECKS]", json.dumps(checks, indent=2))
    if not all(checks.values()):
        print("\n*** REPRODUCTION FAILED — n=9 output NOT trustworthy. Stopping. ***")
        sys.exit(2)
    print("\n*** REPRODUCTION PASSED — proceeding to n=9. ***")

    # STEP 2: build n=9
    pr9, rd9 = build_sets(combined, extra_full=[p08], extra_reduced=[p02, p03, p09])
    wp9, lp9 = evaluate(pr9, PRIMARY_FEATURES)
    wr9, lr9 = evaluate(rd9, REDUCED_FEATURES)
    s9 = summary(pr9, rd9, wp9, lp9, wr9, lr9)

    OUT.mkdir(parents=True, exist_ok=True)
    pr9.to_csv(OUT/"primary_4imu_cleaned_features.csv", index=False)
    rd9.to_csv(OUT/"reduced_pelvis_l3_features.csv", index=False)
    (OUT/"evaluation_corrected").mkdir(exist_ok=True)
    wp9.to_csv(OUT/"evaluation_corrected/primary_within.csv", index=False)
    lp9.to_csv(OUT/"evaluation_corrected/primary_loso.csv", index=False)
    wr9.to_csv(OUT/"evaluation_corrected/reduced_within.csv", index=False)
    lr9.to_csv(OUT/"evaluation_corrected/reduced_loso.csv", index=False)
    s9["sha256"] = {
        "primary_4imu_cleaned_features.csv": sha(OUT/"primary_4imu_cleaned_features.csv"),
        "reduced_pelvis_l3_features.csv": sha(OUT/"reduced_pelvis_l3_features.csv"),
    }
    s9["class_counts"] = {
        "primary": {"safe":int((pr9.risk_class==0).sum()),"risky":int((pr9.risk_class==1).sum())},
        "reduced": {"safe":int((rd9.risk_class==0).sum()),"risky":int((rd9.risk_class==1).sum())},
    }
    (OUT/"evaluation_corrected/evaluation_summary.json").write_text(json.dumps(s9, indent=2))
    print("\n===== STEP 2: n=9 RESULTS =====")
    print(json.dumps(s9, indent=2))
    print("\n[per-participant reduced within]"); print(wr9.to_string(index=False))
    print("\n[per-participant reduced loso]");   print(lr9.to_string(index=False))
    print("\n[per-participant primary within]"); print(wp9.to_string(index=False))
    print("\n[per-participant primary loso]");   print(lp9.to_string(index=False))
    import sklearn
    print(f"\n[env] sklearn={sklearn.__version__}")

if __name__ == "__main__":
    main()
