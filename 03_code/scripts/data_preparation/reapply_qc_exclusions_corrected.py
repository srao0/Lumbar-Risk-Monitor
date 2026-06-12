#!/usr/bin/env python3
"""
reapply_qc_exclusions_corrected.py
================================================================================
Fix a data-integrity slip in the corrected fallback re-freeze: the corrected
analysis-set assembly did NOT carry over the frozen participant-level QC
exclusions, so it re-admitted windows the QC protocol excludes
(P07 +145 FATIGUE_FLEXION belt-slip windows pegged at the 60-deg pelvis cap;
P04 +51 T4-dropout; P05 +15).

Fix strategy = MEMBERSHIP MATCH. The windowing (2 s / 1 s stride) is identical
between the frozen and corrected sets; only the feature VALUES changed under
drift correction. So we keep the CORRECTED feature values but restrict each set
to the exact windows the FROZEN QC kept, keyed on
(participant_id, session_id, window_centre_ms). This reproduces the documented
QC composition (Ch7 Table 7.4) on corrected features, with no re-derivation of
QC rules.

Inputs  : results/fallback_analysis_sets_n9_corrected/{primary,reduced}_*.csv  (corrected values, contaminated membership)
          results/fallback_analysis_sets_n9/{primary,reduced}_*.csv            (frozen QC membership reference)
Outputs : results/fallback_analysis_sets_n9_corrected_qc/{primary,reduced}_*.csv (corrected values, QC membership)

Run from repo root (pandas only, no sklearn):
    py scripts/data_preparation/reapply_qc_exclusions_corrected.py
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

# parents[1] here is a DATA/results root, not a sys.path anchor: the script only
# reads/writes results CSVs, so it stays one level above this scripts/ tree.
ROOT = Path(__file__).resolve().parents[1]
CORR = ROOT / "results" / "fallback_analysis_sets_n9_corrected"
FROZEN = ROOT / "results" / "fallback_analysis_sets_n9"
OUT = ROOT / "results" / "fallback_analysis_sets_n9_corrected_qc"
KEY = ["participant_id", "session_id", "window_centre_ms"]
SETS = {
    "primary": "primary_4imu_cleaned_features.csv",
    "reduced": "reduced_pelvis_l3_features.csv",
}
LAB = "risk_class"


def counts(df):
    """Return (total rows, safe count, risky count) — labelled rows only for the class split."""
    rc = df[df[LAB].isin([0, 1])] if LAB in df else df
    return len(df), int((rc[LAB] == 0).sum()), int((rc[LAB] == 1).sum())


def main():
    """Restrict each corrected set to the frozen QC window membership and report the accounting."""
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, fname in SETS.items():
        corr = pd.read_csv(CORR / fname, low_memory=False)
        froz = pd.read_csv(FROZEN / fname, low_memory=False)
        keyset = set(map(tuple, froz[KEY].itertuples(index=False, name=None)))
        mask = corr[KEY].apply(lambda r: tuple(r) in keyset, axis=1)
        clean = corr[mask].copy()
        clean.to_csv(OUT / fname, index=False)

        tc, sc, rc = counts(corr)
        tk, sk, rk = counts(clean)
        tf, sf, rf = counts(froz)
        print(f"\n=== {tag.upper()} ===")
        print(f"  contaminated corrected : {tc} rows  (safe {sc} / risky {rc})")
        print(f"  QC-cleaned corrected   : {tk} rows  (safe {sk} / risky {rk})   [-{tc-tk} windows]")
        print(f"  frozen reference       : {tf} rows  (safe {sf} / risky {rf})")
        print(f"  membership matches frozen: {tk == tf}")
        # per-participant delta (contaminated -> cleaned) so the removed windows are attributable
        cpp = corr["participant_id"].value_counts()
        kpp = clean["participant_id"].value_counts()
        diffs = {p: int(cpp.get(p, 0) - kpp.get(p, 0)) for p in sorted(cpp.index) if cpp.get(p, 0) != kpp.get(p, 0)}
        print(f"  windows removed per participant: {diffs}")
        # sanity: the belt-slip windows pegged at the 60-deg pelvis cap should be gone
        if "imu_pelvis_angle_peak" in clean:
            capped = int((clean["imu_pelvis_angle_peak"] >= 59.999).sum())
            print(f"  pelvis-capped windows remaining in cleaned set: {capped} ({100*capped/len(clean):.1f}%)")

    print(f"\nCleaned sets written to {OUT}")
    print("Next (if impact warrants): re-run evaluate_fallback_analysis_sets + train_fallback_analysis_models on this dir, re-lock the manifest, then refresh Ch8/Ch7.")


if __name__ == "__main__":
    main()
