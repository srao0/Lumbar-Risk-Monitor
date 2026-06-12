#!/usr/bin/env python3
"""Pass 0 — DATA layer only (no RF). Faithfully replicate prepare_fallback_analysis_sets.py,
reproduce the frozen n=7 row/class counts as a trust check, then build + hash the n=9 sets."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "results" / "fallback_analysis_sets_n9"
CAP  = 60.0
REDUCED_FEATURES = ["imu_angvel_peak","imu_angvel_mean","imu_time_high_velocity","imu_ldlj","imu_jerk_rms",
    "imu_jerk_peak","imu_pelvis_angle_peak","imu_pelvis_angle_mean","imu_l3_accel_tilt_peak",
    "imu_l3_accel_tilt_mean","imu_l3_accel_tilt_range","imu_z_vel","imu_z_ldlj"]
ID_COLS = ["session_id","participant_id","window_centre_ms","movement_label","risk_class_protocol","risk_class"]
P = {
 "combined": ROOT/"data/real/protocol_train_fallback_2session/combined_features.csv",
 "p02": ROOT/"data/real/protocol_train_fallback_2session/participant_02/session_001/feature_matrix.pre_quality_exclusions_20260528_112742.csv",
 "p03": ROOT/"data/real/protocol_train_fallback/participant_03/session_001/feature_matrix.csv",
 "p08": ROOT/"data/real/protocol_train_fallback_recovered/participant_08/session_001/feature_matrix.csv",
 "p09": ROOT/"data/real/protocol_train_fallback/participant_09/session_001_restcorrected/feature_matrix.csv",
}
def load_session(path,pid,sess):
    """Read one per-session feature matrix and stamp it with the participant/session ids the combined sets key on (protocol session_id format pid__session)."""
    df=pd.read_csv(path, low_memory=False)
    df.insert(0,"participant_id",pid); df.insert(1,"session_id",f"{pid}__{sess}")
    return df
def labelled(df):
    """Keep only protocol-labelled windows (0/1), preferring risk_class_protocol over risk_class, and normalise the chosen target into an int risk_class."""
    tgt="risk_class_protocol" if "risk_class_protocol" in df.columns else "risk_class"
    out=df[df[tgt].isin([0,1])].copy(); out["risk_class"]=out[tgt].astype(int); return out
def cap_pelvis(df):
    """Clip the two pelvis-angle features at CAP (60°) to suppress Madgwick-drift outliers, returning the frame plus a per-column audit of how many windows were capped."""
    out=df.copy(); rep={}
    for col in ["imu_pelvis_angle_peak","imu_pelvis_angle_mean"]:
        if col in out.columns:
            v=out[col].astype(float); mask=v>CAP
            out[col]=v.clip(upper=CAP); rep[col]={"n_capped":int(mask.sum()),"pct":round(float(mask.mean()*100),3)}
    return out,rep
def build_sets(combined, extra_full, extra_reduced):
    """Build the primary and reduced tables (data layer only — no model is fitted here).

    Same assembly rules as refreeze_n9.py's build_sets, but also returns the pelvis-cap
    audit dicts so the n=7 trust check can confirm row/class counts before any n=9 write.
    """
    comb=combined.copy()
    for d in extra_full:
        d=d.copy()
        if "risk_class_protocol" in d.columns: d["risk_class"]=d["risk_class_protocol"]
        comb=pd.concat([comb,d],ignore_index=True,sort=False)
    primary=comb[~comb["participant_id"].isin({"participant_02"})].copy()
    primary=primary[primary["risk_class"].isin([0,1])].copy()
    primary,pcap=cap_pelvis(primary)
    base=pd.concat([comb]+list(extra_reduced),ignore_index=True,sort=False)
    base=labelled(base)
    keep=[c for c in ID_COLS+REDUCED_FEATURES if c in base.columns]
    reduced=base[keep].dropna(subset=REDUCED_FEATURES).copy()
    reduced,rcap=cap_pelvis(reduced)
    return primary,reduced,pcap,rcap
def desc(df):
    """Summarise a set's row/class/participant counts — the numbers the n=7 trust check matches against and the manifest records."""
    return {"rows":int(len(df)),"safe":int((df.risk_class==0).sum()),"risky":int((df.risk_class==1).sum()),
            "participants":sorted(df.participant_id.unique().tolist()),
            "per_participant":{p:int((df.participant_id==p).sum()) for p in sorted(df.participant_id.unique())}}
def sha(p):
    """SHA-256 of a written CSV so the frozen sets can be hash-verified against the manifest."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

combined=pd.read_csv(P["combined"],low_memory=False)
agree=(combined["risk_class"]==combined["risk_class_protocol"]).mean()*100 if "risk_class_protocol" in combined else -1
print(f"[chk] combined risk_class==protocol: {agree:.2f}%  combined rows={len(combined)}")
p02=load_session(P["p02"],"participant_02","session_001")
p03=load_session(P["p03"],"participant_03","session_001")
p08=load_session(P["p08"],"participant_08","session_001")
p09=load_session(P["p09"],"participant_09","session_001")

# n7 data reproduction — gate the n=9 write on exactly reproducing the frozen n=7
# row and class counts. A mismatch means an input table has changed underneath us,
# so we refuse to overwrite the frozen sets rather than freeze drifted numbers.
pr7,rd7,_,_=build_sets(combined,[],[p02,p03])
d_pr7,d_rd7=desc(pr7),desc(rd7)
ok = (d_pr7["rows"]==7294 and d_pr7["safe"]==4304 and d_pr7["risky"]==2990
      and d_rd7["rows"]==10296 and d_rd7["safe"]==6046 and d_rd7["risky"]==4250)
print("\n===== n7 DATA REPRODUCTION =====")
print("primary:",d_pr7["rows"],d_pr7["safe"],d_pr7["risky"],d_pr7["participants"])
print("reduced:",d_rd7["rows"],d_rd7["safe"],d_rd7["risky"],d_rd7["participants"])
print("MATCH FROZEN n7:", ok)
if not ok:
    print("*** data reproduction failed — not writing n9 ***"); raise SystemExit(2)

# n9 build
pr9,rd9,pcap,rcap=build_sets(combined,[p08],[p02,p03,p09])
OUT.mkdir(parents=True,exist_ok=True)
pr9.to_csv(OUT/"primary_4imu_cleaned_features.csv",index=False)
rd9.to_csv(OUT/"reduced_pelvis_l3_features.csv",index=False)
out={"primary":desc(pr9),"reduced":desc(rd9),
     "primary_pelvis_cap":pcap,"reduced_pelvis_cap":rcap,
     "sha256":{"primary":sha(OUT/'primary_4imu_cleaned_features.csv'),
               "reduced":sha(OUT/'reduced_pelvis_l3_features.csv')},
     "n_cols":{"primary":pr9.shape[1],"reduced":rd9.shape[1]}}
(OUT/"analysis_set_summary_n9.json").write_text(json.dumps(out,indent=2))
print("\n===== n9 SETS BUILT =====")
print(json.dumps(out,indent=2))
