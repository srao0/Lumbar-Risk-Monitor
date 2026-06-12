#!/usr/bin/env python3
"""
assemble_corrected_fallback_combined.py
================================================================================
Stitch the rest-corrected per-participant fallback feature matrices into the
`combined_features.csv + per-participant/<session>/labels.csv` layout that
scripts/run_personalised_stage2b.py (RQ4: personalised vs population) expects.

Source : data/real/fallback_corrected/participant_XX/session_001/feature_matrix.csv
         (+ labels.csv already present in each session folder)

Produces TWO data dirs:
  1. data/real/fallback_corrected        -> combined_features.csv over ALL 9
     participants (the n=9 EXTENSION run). labels.csv already in place.
  2. data/real/fallback_corrected_n3     -> combined_features.csv over
     participants 01,02,03 only (the like-for-like RE-FREEZE of the frozen
     3-participant RQ4 result), with their 3 labels.csv copied in.

Each per-participant feature_matrix.csv is 49 cols and lacks `session_id` and
`participant_id`; we inject those two (matching the 51-col frozen layout and the
`participant_XX__session_001` session_id format). risk_class_protocol == -1
(UNLABELLED) rows are LEFT IN -- run_personalised_stage2b.py excludes them
itself, exactly as in the frozen combined (which carried 712 such rows).

Run from repo root on the machine with the canonical data:
    py scripts/assemble_corrected_fallback_combined.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC  = REPO / "data" / "real" / "fallback_corrected"
OUT9 = SRC                                              # 9-participant extension
OUT3 = REPO / "data" / "real" / "fallback_corrected_n3"  # 3-participant re-freeze
N3_PARTICIPANTS = ["participant_01", "participant_02", "participant_03"]


def participant_dirs() -> list[Path]:
    """Discover the rest-corrected per-participant source folders, sorted so the combined order is deterministic across machines."""
    return sorted(p for p in SRC.glob("participant_*") if p.is_dir())


def build_combined(pids: list[str]) -> pd.DataFrame:
    """Concatenate the listed participants' feature matrices into one frame, injecting the session_id/participant_id columns the frozen 51-col layout expects and asserting every participant shares the same 49-col schema."""
    frames = []
    ref_cols = None
    for pid in pids:
        fm_path = SRC / pid / "session_001" / "feature_matrix.csv"
        fm = pd.read_csv(fm_path, low_memory=False)
        if ref_cols is None:
            ref_cols = list(fm.columns)
        elif list(fm.columns) != ref_cols:
            raise ValueError(
                f"Column mismatch in {pid}: "
                f"{set(fm.columns) ^ set(ref_cols)}"
            )
        fm.insert(0, "session_id", f"{pid}__session_001")
        fm.insert(1, "participant_id", pid)
        frames.append(fm)
        n_lab = int(fm["risk_class_protocol"].isin([0, 1]).sum())
        print(f"  {pid}: {len(fm)} windows ({n_lab} labelled)")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    """Write both data dirs — the n=9 extension combined and the like-for-like n=3 re-freeze (with its labels copied in) — that run_personalised_stage2b.py consumes for RQ4."""
    pids_all = [p.name for p in participant_dirs()]
    print(f"Found {len(pids_all)} corrected participants: {pids_all}")

    # ---- n=9 extension: write combined into the existing layout -------------
    print("\n[1/2] Assembling n=9 EXTENSION -> data/real/fallback_corrected/")
    combined9 = build_combined(pids_all)
    out9_csv = OUT9 / "combined_features.csv"
    combined9.to_csv(out9_csv, index=False)
    print(f"  wrote {out9_csv}  ({len(combined9)} rows, {combined9.shape[1]} cols)")
    print(f"  protocol-label balance: "
          f"{combined9['risk_class_protocol'].value_counts(dropna=False).to_dict()}")

    # ---- n=3 re-freeze: filtered combined + copied labels -------------------
    print("\n[2/2] Assembling n=3 RE-FREEZE -> data/real/fallback_corrected_n3/")
    missing = [p for p in N3_PARTICIPANTS if p not in pids_all]
    if missing:
        raise FileNotFoundError(f"n3 participants missing from source: {missing}")
    combined3 = combined9[combined9["participant_id"].isin(N3_PARTICIPANTS)].copy()
    OUT3.mkdir(parents=True, exist_ok=True)
    (OUT3 / "combined_features.csv").write_text(
        combined3.to_csv(index=False), encoding="utf-8"
    )
    for pid in N3_PARTICIPANTS:
        dst = OUT3 / pid / "session_001"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC / pid / "session_001" / "labels.csv", dst / "labels.csv")
    print(f"  wrote {OUT3 / 'combined_features.csv'}  ({len(combined3)} rows)")
    print(f"  copied labels.csv for {N3_PARTICIPANTS}")

    print("\nDone. Next:")
    print("  RE-FREEZE (n3): py scripts/run_personalised_stage2b.py "
          "--data_dir data/real/fallback_corrected_n3 "
          "--results_dir results/personalised_stage2b_corrected_n3 "
          "--models_dir ml/models/personalised_stage2b_corrected_n3 --seed 42")
    print("  EXTENSION (n9): py scripts/run_personalised_stage2b.py "
          "--data_dir data/real/fallback_corrected "
          "--results_dir results/personalised_stage2b_corrected_n9 "
          "--models_dir ml/models/personalised_stage2b_corrected_n9 --seed 42")


if __name__ == "__main__":
    main()
