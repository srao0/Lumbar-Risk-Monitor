#!/usr/bin/env python3
"""
make_stage2b_qc_input.py
================================================================================
Build a QC-clean input directory for the Phase II.B personalised-calibration
(stage2b) experiment, so it no longer runs on the contaminated pre-QC cohort.

The contaminated source (data/real/fallback_corrected) re-admitted the windows
the QC protocol excludes. This script applies the SAME documented data-quality
exclusions used to assemble the frozen n=9 set, dropping only those window
ranges and keeping everything else (crucially the BASELINE_STATIC calibration
windows stage2b needs). It does NOT re-impose the analysis-set label filtering.

Exclusions applied: the window-range rules only (P04 / P05 T4 dropout, P07
pelvis slip = the 211 windows the QC bug re-admitted). P02's whole-session
"exclude_from_main_training" rule is SKIPPED, because the canonical frozen n=9
set keeps P02 as a valid participant (Appendix~G), and stage2b needs P02's
risky windows to form a calibration split. Validated drop = 211 windows;
P02 and all BASELINE_STATIC calibration windows retained.

Inputs  : data/real/fallback_corrected/combined_features.csv
          data/real/fallback_corrected/<participant>/session_*/labels.csv
          data/real/_intermediate_stages/protocol_train_fallback_2session/data_quality_exclusions.csv
Output  : data/real/fallback_corrected_qc/combined_features.csv
          data/real/fallback_corrected_qc/<participant>/session_*/labels.csv  (copied)

Sits in data preparation, upstream of run_personalised_stage2b.py.

Run on the machine with the data:  py scripts/data_preparation/make_stage2b_qc_input.py
Then:  py scripts/training/run_personalised_stage2b.py --data_dir data/real/fallback_corrected_qc \
           --results_dir results/personalised_stage2b_corrected_qc_n9 \
           --models_dir ml/models/personalised_stage2b_corrected_qc_n9
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

# parents[2] from scripts/data_preparation/ resolves to 03_code, the package data
# root holding data/. Only used to locate input/output data (no sys.path use).
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "real" / "fallback_corrected"
OUT = ROOT / "data" / "real" / "fallback_corrected_qc"
RULES_CSV = (ROOT / "data" / "real" / "_intermediate_stages"
             / "protocol_train_fallback_2session" / "data_quality_exclusions.csv")


def main() -> int:
    """Apply the window-range QC exclusions to combined_features.csv and copy the
    per-session labels.csv into the QC-clean output directory."""
    if not (SRC / "combined_features.csv").exists():
        print(f"ERROR: {SRC/'combined_features.csv'} not found"); return 1
    if not RULES_CSV.exists():
        print(f"ERROR: exclusions file not found: {RULES_CSV}"); return 1

    comb = pd.read_csv(SRC / "combined_features.csv", low_memory=False)
    rules = pd.read_csv(RULES_CSV)
    has_label = "movement_label" in comb.columns

    # Apply ONLY the window-range exclusions (recommended_action
    # "exclude_overlapping_*"): P04/P05/P07. The P02 rule is
    # "exclude_from_main_training" (whole session), which the canonical n=9
    # freeze did NOT apply (P02 is a valid participant in the frozen set,
    # Appendix~G), so it is skipped here to keep the same cohort. This drops
    # exactly the 211 re-admitted windows (P07 145, P04 51, P05 15).
    win_rules = rules[rules["recommended_action"].astype(str)
                      .str.startswith("exclude_overlapping")]
    skipped = rules[~rules["recommended_action"].astype(str)
                    .str.startswith("exclude_overlapping")]
    if len(skipped):
        print(f"  skipped whole-participant rules (kept, per canonical n=9): "
              f"{list(skipped['participant_id'])}")

    drop = pd.Series(False, index=comb.index)
    for _, r in win_rules.iterrows():
        m = (comb["participant_id"].astype(str) == str(r["participant_id"]))
        m &= comb["session_id"].astype(str).str.contains(str(r["session_id"]))
        m &= (comb["window_centre_ms"].astype(float) >= float(r["start_ms"]))
        m &= (comb["window_centre_ms"].astype(float) <= float(r["end_ms"]))
        if str(r["label"]) != "ALL" and has_label:
            m &= (comb["movement_label"].astype(str) == str(r["label"]))
        print(f"  {r['participant_id']} {r['label']} "
              f"[{r['start_ms']}-{r['end_ms']}] -> drops {int(m.sum())}")
        drop |= m

    clean = comb[~drop].copy()
    print(f"total dropped {int(drop.sum())} | kept {len(clean)} / {len(comb)}")
    if has_label:
        print(f"BASELINE_STATIC kept: "
              f"{int((clean['movement_label'] == 'BASELINE_STATIC').sum())}")

    OUT.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUT / "combined_features.csv", index=False)
    print(f"wrote {OUT / 'combined_features.csv'}")

    # Copy per-participant session label files (stage2b reads these for the split).
    n_copied = 0
    for pdir in sorted(SRC.glob("participant_*")):
        for sdir in sorted(pdir.glob("session_*")):
            lab = sdir / "labels.csv"
            if lab.exists():
                dest = OUT / pdir.name / sdir.name
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(lab, dest / "labels.csv")
                n_copied += 1
    print(f"copied {n_copied} labels.csv into {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
