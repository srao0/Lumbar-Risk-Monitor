#!/usr/bin/env python3
"""Phase II.C verification figures. Run: python scripts/phase2c_plots.py  -> results/phase2c/figures/"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/"results"/"phase2c"
FIG=R/"figures"; FIG.mkdir(parents=True,exist_ok=True)
s=pd.read_csv(R/"phase2c_summary.csv")
s=s[pd.to_numeric(s["auc"],errors="coerce").notna()].copy(); s["auc"]=s["auc"].astype(float)
RLAB={"imu_only_fallback":"IMU-only (primary)","reduced_pelvis_l3":"Reduced Pelvis-L3","full_hybrid":"Full-hybrid (IMU+sEMG)"}
RCOL={"imu_only_fallback":"#4C78A8","reduced_pelvis_l3":"#59A14F","full_hybrid":"#E15759"}
CL={"P11_real_normal":"P11 (real,\nfully held-out)","P12_real_normal":"P12 (real,\nfully held-out)",
    "P03_real_varied":"P03 (real,\nboundary)","P14_real_conforming":"P14 (real,\nheld-out)","P14_synth_varied":"P14 synth\n(varied)"}
def auc_of(c,rt):
    """AUC for one condition x route cell of the summary, or NaN when that combination was not evaluated."""
    r=s[(s.condition==c)&(s.route==rt)]; return float(r.auc.iloc[0]) if len(r) else np.nan
def leak_of(c,rt):
    """True if that cell is in-sample (held_out flagged 'leak') — these bars get hatched so the figure never passes leakage off as genuine held-out performance."""
    r=s[(s.condition==c)&(s.route==rt)]; return ("leak" in str(r.held_out.iloc[0]).lower()) if len(r) else False

# fig1 AUC by condition x route
try:
    conds=["P11_real_normal","P12_real_normal","P14_real_conforming","P03_real_varied","P14_synth_varied"]
    routes=["imu_only_fallback","reduced_pelvis_l3","full_hybrid"]
    fig,ax=plt.subplots(figsize=(11,5.5)); x=np.arange(len(conds)); w=0.26
    for i,rt in enumerate(routes):
        v=[auc_of(c,rt) for c in conds]; lk=[leak_of(c,rt) for c in conds]
        bars=ax.bar(x+(i-1)*w,[0 if np.isnan(z) else z for z in v],w,label=RLAB[rt],color=RCOL[rt],edgecolor="black",lw=0.5)
        for b,z,l in zip(bars,v,lk):
            if np.isnan(z): continue
            ax.text(b.get_x()+b.get_width()/2,z+0.01,f"{z:.2f}",ha="center",va="bottom",fontsize=8)
            if l: b.set_hatch("////")
    ax.axhline(0.5,ls="--",c="grey",lw=1); ax.axhline(0.648,ls=":",c="#59A14F",lw=1.2)
    ax.text(4.4,0.655,"n=9 LOSO 0.648",color="#59A14F",fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([CL[c] for c in conds],fontsize=9); ax.set_ylim(0,1.05); ax.set_ylabel("AUC")
    ax.set_title("Phase II.C — held-out AUC by participant and operating route (hatched = in-sample leakage)")
    ax.legend(fontsize=8,ncol=3,loc="upper center"); fig.tight_layout(); fig.savefig(FIG/"fig1_auc_by_condition.png",dpi=150); plt.close(fig); print("fig1 OK")
except Exception as e: print("fig1 FAIL",e)

# fig2 traffic-light distribution (reduced)
try:
    conds=["P11_real_normal","P12_real_normal","P14_real_conforming","P14_synth_varied"]
    d=s[(s.route=="reduced_pelvis_l3")&(s.condition.isin(conds))].set_index("condition")
    safe=[];caut=[];risk=[]
    for c in conds:
        tot=d.loc[c,"tl_Safe"]+d.loc[c,"tl_Cautious"]+d.loc[c,"tl_Risky"]
        safe.append(d.loc[c,"tl_Safe"]/tot);caut.append(d.loc[c,"tl_Cautious"]/tot);risk.append(d.loc[c,"tl_Risky"]/tot)
    fig,ax=plt.subplots(figsize=(8,5)); x=np.arange(len(conds))
    ax.bar(x,safe,color="#59A14F",label="Safe (Green)"); ax.bar(x,caut,bottom=safe,color="#F1B61C",label="Cautious (Amber)")
    ax.bar(x,risk,bottom=np.array(safe)+np.array(caut),color="#E15759",label="Risky (Red)")
    ax.set_xticks(x); ax.set_xticklabels([CL[c] for c in conds]); ax.set_ylabel("proportion of windows"); ax.set_ylim(0,1)
    ax.set_title("Phase II.C — traffic-light distribution (Reduced Pelvis-L3)\nmost windows land in Amber: a conservative operating point")
    ax.legend(fontsize=8,loc="lower right"); fig.tight_layout(); fig.savefig(FIG/"fig2_traffic_light.png",dpi=150); plt.close(fig); print("fig2 OK")
except Exception as e: print("fig2 FAIL",e)

# fig3 flagged vs overflag (held-out reduced)
try:
    conds=["P11_real_normal","P12_real_normal","P14_real_conforming"]
    d=s[(s.route=="reduced_pelvis_l3")&(s.condition.isin(conds))].set_index("condition")
    fr=[d.loc[c,"flagged_recall"] for c in conds]; of=[d.loc[c,"overflag_rate"] for c in conds]
    fig,ax=plt.subplots(figsize=(8,5)); x=np.arange(len(conds)); w=0.38
    ax.bar(x-w/2,fr,w,label="flagged recall (risky caught as Amber+Red)",color="#59A14F")
    ax.bar(x+w/2,of,w,label="over-flag (safe flagged Amber+Red)",color="#E15759")
    for i,(a,b) in enumerate(zip(fr,of)): ax.text(i-w/2,a+0.01,f"{a:.2f}",ha="center",fontsize=8); ax.text(i+w/2,b+0.01,f"{b:.2f}",ha="center",fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([CL[c] for c in conds]); ax.set_ylim(0,1.05)
    ax.set_title("Phase II.C — Amber band catches most risk but over-flags safe movement\n(Reduced Pelvis-L3, held-out)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG/"fig3_flagged_vs_overflag.png",dpi=150); plt.close(fig); print("fig3 OK")
except Exception as e: print("fig3 FAIL",e)

# fig4 ROC (held-out reduced)
try:
    def roc(y,p):
        """Self-contained ROC curve + trapezoidal AUC (no sklearn dependency) so the figure script stays light."""
        o=np.argsort(-p); y=y[o]; P=max((y==1).sum(),1); N=max((y==0).sum(),1)
        tpr=np.concatenate([[0],np.cumsum(y==1)/P]); fpr=np.concatenate([[0],np.cumsum(y==0)/N])
        a=float(np.sum((fpr[1:]-fpr[:-1])*(tpr[1:]+tpr[:-1])/2)); return fpr,tpr,a
    fig,ax=plt.subplots(figsize=(6.5,6.5))
    for name,f in {"P11":"phase2c_predictions_P11_real_normal_reduced_pelvis_l3.csv",
                   "P12":"phase2c_predictions_P12_real_normal_reduced_pelvis_l3.csv",
                   "P14":"phase2c_predictions_P14_real_conforming_reduced_pelvis_l3.csv"}.items():
        fp=R/f
        if not fp.exists(): print("ROC miss",name); continue
        d=pd.read_csv(fp); y=d["risk_class_protocol"].to_numpy(int); p=d["prob"].to_numpy(float)
        fpr,tpr,a=roc(y,p); ax.plot(fpr,tpr,lw=2,label=f"{name} (AUC={a:.3f})"); print(f"ROC {name} AUC={a:.3f}")
    ax.plot([0,1],[0,1],"--",c="grey",lw=1); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Phase II.C — ROC, Reduced Pelvis-L3 on held-out participants"); ax.legend(fontsize=9,loc="lower right")
    fig.tight_layout(); fig.savefig(FIG/"fig4_roc_heldout_reduced.png",dpi=150); plt.close(fig); print("fig4 OK")
except Exception as e: print("fig4 FAIL",e)

# fig5 sEMG route comparison
try:
    conds=["P14_real_conforming","P14_synth_varied"]; routes=["imu_only_fallback","reduced_pelvis_l3","full_hybrid"]
    fig,ax=plt.subplots(figsize=(8,5)); x=np.arange(len(conds)); w=0.26
    for i,rt in enumerate(routes):
        v=[auc_of(c,rt) for c in conds]; b=ax.bar(x+(i-1)*w,[0 if np.isnan(z) else z for z in v],w,label=RLAB[rt],color=RCOL[rt],edgecolor="black",lw=0.5)
        for bb,z in zip(b,v):
            if not np.isnan(z): ax.text(bb.get_x()+bb.get_width()/2,z+0.01,f"{z:.2f}",ha="center",fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["P14 (real)","P14 synth (varied)"]); ax.set_ylim(0,1.0); ax.set_ylabel("AUC")
    ax.set_title("Phase II.C — does sEMG help? full-hybrid does not beat IMU-only"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG/"fig5_semg_comparison.png",dpi=150); plt.close(fig); print("fig5 OK")
except Exception as e: print("fig5 FAIL",e)
print("DONE ->", FIG)
