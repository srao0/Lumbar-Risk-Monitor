#!/usr/bin/env python3
"""
plot_figure_work_order.py

Renders the data-ready figures listed in figure_work_order.md in the thesis house
style (labelled axes with units, legend, shaded/annotated regions, an on-figure
method note, caption-ready content).

Currently produces:
  F3 - sEMG rest-baseline normalisation, before vs after (P14, 3 sessions)
  F5 - P14 session 01 segmental sagittal flexion trace

Inputs : per-session feature_matrix.csv (restcorrected + emgnorm variants) and
         imu_data.csv under data/real/... (not bundled in this handover package).
Outputs: PNGs written to figures_work_order/ next to the data root.

Sits at the end of the pipeline: it consumes already-processed feature matrices
and produces the publication figures, it does not compute any results itself.

Example:
    py scripts/figures/plot_figure_work_order.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# parents[1] from scripts/figures/ resolves to 03_code/scripts. In the working
# repo this pointed at the project root holding data/ and figures_work_order/.
# It is only used to locate data/figure outputs (no sys.path manipulation); the
# referenced session data is not shipped in this package.
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures_work_order"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

RAW = ROOT / "data/real/protocol_train_full_hybrid_restcorrected/participant_14/session_01"
NORM = ROOT / "data/real/protocol_train_full_hybrid_emgnorm/participant_14/session_01"


def f3():
    """F3: sEMG resting-baseline normalisation before/after across three P14 sessions.

    A single session's before/after differ only in y-axis scale (division by one
    per-session constant leaves the shape unchanged). The method's value is
    cross-session comparability, so three sessions spanning low/mid/high resting
    level are plotted: scattered uV levels before, all collapsing onto rest = 1.0
    after.
    """
    ch = "emg_rms_RES"
    SESS = [("session_11", "#1f77b4"), ("session_01", "#ff7f0e"), ("session_13", "#2ca02c")]
    rootR = ROOT / "data/real/protocol_train_full_hybrid_restcorrected/participant_14"
    rootN = ROOT / "data/real/protocol_train_full_hybrid_emgnorm/participant_14"
    TMAX = 300.0           # first 300 s for legibility
    SM = 5                 # light rolling smoothing on the envelope (windows)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.5, 6.6), sharex=True)
    for sname, c in SESS:
        raw = pd.read_csv(rootR / sname / "feature_matrix.csv", low_memory=False)
        nrm = pd.read_csv(rootN / sname / "feature_matrix.csv", low_memory=False)
        t = raw["window_centre_ms"].to_numpy() / 1000.0
        m = t <= TMAX
        base = raw["movement_label"] == "BASELINE_STATIC"
        rest_uv = raw.loc[base, ch].mean() * 1000
        rraw = (raw[ch] * 1000).rolling(SM, center=True, min_periods=1).mean()
        rnrm = nrm[ch].rolling(SM, center=True, min_periods=1).mean()
        a1.plot(t[m], rraw[m], color=c, lw=0.9, label=f"{sname.replace('_',' ')}  (rest = {rest_uv:.0f} uV)")
        a1.axhline(rest_uv, color=c, ls=":", lw=0.9, alpha=0.7)
        a2.plot(t[m], rnrm[m], color=c, lw=0.9, label=f"{sname.replace('_',' ')}")

    a1.set_ylabel("RMS amplitude (uV)")
    a1.set_title("Before: raw sEMG RMS - resting level differs 16-78 uV between days (not comparable)")
    a1.legend(loc="upper right", fontsize=8); a1.set_ylim(0, 600)

    a2.axhline(1.0, color="#d62728", ls="--", lw=1.2, label="resting level = 1.0 (all sessions)")
    a2.set_ylabel("RMS (x rest)")
    a2.set_xlabel("Session time (s)")
    a2.set_title("After: / each session's own resting RMS - all rest at 1.0, peaks now comparable across days")
    a2.legend(loc="upper right", fontsize=8); a2.set_ylim(0, 25)

    fig.suptitle("F3  sEMG rest-baseline normalisation - P14, right erector spinae (RES), 3 sessions",
                 fontsize=12, y=0.985)
    fig.text(0.012, 0.01,
             "Method: each channel is divided by its mean RMS over that session's BASELINE_STATIC quiet-standing "
             "windows. Units change uV -> xrest (1.0 = that day's resting level).\nA single session's before/after "
             "are identical in shape (division by one constant); the method's value is cross-session - different "
             "electrode contact gives 16-78 uV at rest, which all map to 1.0.", fontsize=8, va="bottom")
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig.savefig(OUT / "F3_emg_rest_baseline_normalisation.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote F3_emg_rest_baseline_normalisation.png (multi-session)")


def f5():
    """F5: P14 session 01 segmental sagittal flexion (Madgwick-fused, drift-corrected).

    Plots the three inter-segment pitch angles relative to pelvis over the first
    legible window (baseline through the lumbar-dominant bends), with the 45 deg
    NIOSH reference line and grey shading over protocol movement blocks.
    """
    im = pd.read_csv(RAW / "imu_data.csv", low_memory=False)
    t = (im["timestamp_ms"] - im["timestamp_ms"].iloc[0]).to_numpy() / 1000.0
    segs = [("theta_PL_pitch", "#1f77b4", r"$\theta_{PL}$ pelvis-L3 (lumbar)"),
            ("theta_LT_pitch", "#ff7f0e", r"$\theta_{LT}$ L3-T12/L1 (thoracolumbar)"),
            ("theta_TU_pitch", "#2ca02c", r"$\theta_{TU}$ T12-T4/T6 (upper thoracic)")]
    # Focus on a legible representative segment: baseline -> first clean-flexion /
    # lumbar-dominant bends, capped near 260 s.
    lab = im["label"].astype(str)
    end = float(t[(lab == "LUMBAR_DOMINANT").to_numpy()].max()) if (lab == "LUMBAR_DOMINANT").any() else 250.0
    m = t <= min(end + 10, 260)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for col, c, lbl in segs:
        ax.plot(t[m], im.loc[m, col], color=c, lw=0.9, label=lbl)
    ax.axhline(45, color="#d62728", ls=":", lw=1, label="45 deg NIOSH risk-zone reference")
    # Shade the non-baseline (movement) label regions so rest reads as unshaded.
    shown = False
    tv = t[m]; lv = lab[m].to_numpy()
    in_mv = False; x0 = None
    for ti, l in zip(tv, lv):
        mv = l not in ("BASELINE_STATIC", "nan", "UNKNOWN", "")
        if mv and not in_mv: x0 = ti; in_mv = True
        elif not mv and in_mv:
            ax.axvspan(x0, ti, color="grey", alpha=0.10, label=None if shown else "movement block"); shown = True; in_mv = False
    if in_mv: ax.axvspan(x0, tv[-1], color="grey", alpha=0.10)
    ax.set_xlabel("Session time (s)"); ax.set_ylabel("Sagittal segment angle (deg)")
    ax.set_title("F5  P14 session 01 - segmental sagittal flexion (Madgwick-fused, N-pose-calibrated, drift-corrected)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.text(0.012, 0.005,
             "Pitch (sagittal) inter-segment angles relative to pelvis, from Madgwick-fused orientation "
             "(beta=0.033, 100 Hz), N-pose-calibrated and rest-anchor drift-corrected. Not raw gyro; yaw not used "
             "(no magnetometer). Grey = protocol movement blocks; rest is unshaded.", fontsize=8, va="bottom")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT / "F5_segmental_sagittal_flexion.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote F5_segmental_sagittal_flexion.png  (window 0-%.0fs)" % min(end + 10, 260))


if __name__ == "__main__":
    f3(); f5()
    print("done ->", OUT)
