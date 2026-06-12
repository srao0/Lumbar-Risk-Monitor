#!/usr/bin/env python3
"""
rerun_p10_augmentation_ablation.py
================================================================================
Re-run the Appendix C P10-augmentation ablation on the CORRECTED-QC base so the
table is consistent with the rest of the report (Chapter 8 / Appendix G).

Background
----------
Appendix C (Table tab:c2.1) asks: does adding Participant 10's salvageable
partial recording to the training pool change leave-one-subject-out (LOSO) AUC?
The published table's "standard" column was computed on the PRE-QC n=9 base, so
its per-participant LOSO values no longer match the corrected-QC numbers now used
everywhere else in the report (e.g. reduced LOSO mean 0.641, not 0.639). This
script recomputes both the "standard" and "+P10" columns on the corrected-QC
base, so the "standard" column reproduces results/fallback_analysis_sets_n9_
corrected_qc/evaluation/{primary,reduced}_loso.csv exactly and the "+P10" column
is the augmentation delta on that same base.

Methodology parity
------------------
Feature lists, the RandomForest configuration, the Youden-threshold metric helper
and the per-fold fit are IMPORTED from scripts/evaluation/evaluate_fallback_analysis_sets.py
(the canonical RQ1 evaluator), so this ablation cannot silently diverge from the
headline numbers. LOSO holds out each P01-P09 participant in turn; P10 is NEVER
held out, it is only added to the training pool for the augmented run.

Caveat (state in the appendix): P10's windows are the salvageable partial
recording processed in the fallback-sensitivity intermediate stage. They enter
the TRAINING pool only and are never evaluated, so the held-out P01-P09 results
remain pure corrected-QC; the standard column is identical to the canonical LOSO.

Run (on the machine with scikit-learn 1.8.0):
    py scripts/evaluation/rerun_p10_augmentation_ablation.py
Outputs (results/appendix_c_p10_ablation_corrected_qc/):
    p10_ablation_table.csv     - per-participant standard / +P10 / delta, both sets
    p10_ablation_summary.json  - means, deltas, sanity-check vs canonical LOSO
    p10_ablation_table.tex     - drop-in replacement for Table tab:c2.1
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# parents[2] from scripts/evaluation/ resolves to 03_code, which is put on
# sys.path so the package imports below resolve.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Import the CANONICAL methodology so this stays in lock-step with the headline run.
from scripts.evaluation.evaluate_fallback_analysis_sets import (  # noqa: E402
    PRIMARY_FEATURES, REDUCED_FEATURES, fit_predict,
)

# --- default data locations -------------------------------------------------
BASE_DIR = ROOT / "results/fallback_analysis_sets_n9_corrected_qc"
P10_PRIMARY = (ROOT / "data/real/_intermediate_stages/"
               "protocol_train_fallback_sensitivity_first14_4imu/participant_10/"
               "session_001/feature_matrix.csv")
P10_REDUCED = (ROOT / "data/real/_intermediate_stages/"
               "protocol_train_fallback_sensitivity_reduced_partial/participant_10/"
               "session_001/feature_matrix.csv")
CANON_LOSO = {  # for the sanity check that "standard" reproduces the headline run
    "primary": BASE_DIR / "evaluation/primary_loso.csv",
    "reduced": BASE_DIR / "evaluation/reduced_loso.csv",
}


def _clean(df: pd.DataFrame, features: list[str], require_pid: bool) -> pd.DataFrame:
    """Apply the canonical filtering: labelled windows, dropna over the columns used."""
    if "risk_class" not in df.columns:
        if "risk_class_protocol" in df.columns:
            df = df.assign(risk_class=df["risk_class_protocol"])
        else:
            raise ValueError("no risk_class / risk_class_protocol column")
    df = df[df["risk_class"].isin([0, 1])].copy()
    missing = sorted(set(features) - set(df.columns))
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    cols = features + ["risk_class"]
    if require_pid:
        cols += ["participant_id", "window_centre_ms"]
    return df.dropna(subset=[c for c in cols if c in df.columns])


def loso(base: pd.DataFrame, features: int, seed: int, augment: pd.DataFrame | None):
    """Leave-one-subject-out AUC over the base participants, optionally augmenting
    each training fold with the P10 windows (P10 is never held out)."""
    rows = []
    for pid in sorted(base["participant_id"].unique()):
        tr = base[base["participant_id"] != pid]
        te = base[base["participant_id"] == pid]
        if augment is not None:
            tr = pd.concat([tr, augment], ignore_index=True)
        if tr["risk_class"].nunique() < 2 or te["risk_class"].nunique() < 2:
            rows.append({"held_out": pid, "auc": float("nan")}); continue
        p = fit_predict(tr, te, features, seed)
        rows.append({"held_out": pid, "auc": float(roc_auc_score(te["risk_class"].astype(int), p))})
    return pd.DataFrame(rows)


def run_set(tag, base_csv, feats, p10_csv, seed):
    """Compute the standard and +P10 LOSO columns for one feature set and
    sanity-check that the standard column reproduces the canonical headline LOSO."""
    base = _clean(pd.read_csv(base_csv, low_memory=False), feats, require_pid=True)
    p10 = _clean(pd.read_csv(p10_csv, low_memory=False), feats, require_pid=False)
    p10 = p10.assign(participant_id="participant_10")
    print(f"[{tag}] base participants={base['participant_id'].nunique()} "
          f"rows={len(base)}; P10 windows added={len(p10)}")
    std = loso(base, feats, seed, augment=None).rename(columns={"auc": "standard"})
    aug = loso(base, feats, seed, augment=p10).rename(columns={"auc": "augmented"})
    m = std.merge(aug, on="held_out")
    m["delta"] = m["augmented"] - m["standard"]

    # Sanity check: does "standard" reproduce the canonical headline LOSO exactly?
    sanity = None
    canon = CANON_LOSO[tag]
    if canon.exists():
        c = pd.read_csv(canon)[["held_out", "auc"]].rename(columns={"auc": "canonical"})
        chk = m.merge(c, on="held_out")
        maxdiff = float((chk["standard"] - chk["canonical"]).abs().max())
        sanity = {"max_abs_diff_vs_canonical": maxdiff,
                  "reproduces_canonical": bool(maxdiff < 1e-6)}
        print(f"[{tag}] standard vs canonical LOSO: max|diff|={maxdiff:.2e} "
              f"({'MATCH' if maxdiff < 1e-6 else 'MISMATCH - investigate'})")
    return m, {"tag": tag, "n_p10_windows": int(len(p10)),
               "standard_mean": float(m["standard"].mean()),
               "augmented_mean": float(m["augmented"].mean()),
               "delta_mean": float(m["delta"].mean()), "sanity": sanity}


def emit_latex(primary, reduced, out):
    """Write the drop-in LaTeX replacement for Table tab:c2.1 (both feature sets,
    dashes for participants outside the relevant cohort, bold mean row)."""
    pids = [f"P0{i}" for i in range(1, 10)]
    pr = {r.held_out.replace("participant_", "P"): r for r in primary.itertuples()}
    rd = {r.held_out.replace("participant_", "P"): r for r in reduced.itertuples()}

    def cell(d, key):
        if key not in d:
            return "---", "---", "---"
        r = d[key]
        return f"{r.standard:.3f}", f"{r.augmented:.3f}", f"${r.delta:+.3f}$"
    lines = [
        r"\begin{table}[ht]", r"  \centering",
        r"  \caption{Per-participant LOSO AUC under standard and P10-augmented training "
        r"regimes (corrected-QC base). Dashes denote participants outside the relevant "
        r"feature-set cohort.}",
        r"  \label{tab:c2.1}", r"  \small", r"  \begin{tabular}{lcccccccc}", r"    \toprule",
        r"    & \multicolumn{3}{c}{\textbf{Primary 4-IMU (n=6)}} & & "
        r"\multicolumn{3}{c}{\textbf{Reduced Pelvis-L3 (n=9)}} \\",
        r"    \cmidrule(lr){2-4} \cmidrule(lr){6-8}",
        r"    \textbf{PID} & standard & + P10 & $\Delta$ & & standard & + P10 & $\Delta$ \\",
        r"    \midrule",
    ]
    for pid in pids:
        ps, pa, pd_ = cell(pr, pid)
        rs, ra, rd_ = cell(rd, pid)
        lines.append(f"    {pid}  & {ps} & {pa} & {pd_} & & {rs} & {ra} & {rd_} \\\\")
    pm = primary["standard"].mean(), primary["augmented"].mean(), primary["delta"].mean()
    rm = reduced["standard"].mean(), reduced["augmented"].mean(), reduced["delta"].mean()
    lines += [
        r"    \midrule",
        f"    \\textbf{{Mean}} & \\textbf{{{pm[0]:.3f}}} & \\textbf{{{pm[1]:.3f}}} & "
        f"$\\mathbf{{{pm[2]:+.3f}}}$ & & \\textbf{{{rm[0]:.3f}}} & \\textbf{{{rm[1]:.3f}}} & "
        f"$\\mathbf{{{rm[2]:+.3f}}}$ \\\\",
        r"    \bottomrule", r"  \end{tabular}", r"\end{table}",
    ]
    out.write_text("\n".join(lines) + "\n")


def main():
    """Parse args, run both feature sets, and write the CSV/JSON/LaTeX outputs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=Path, default=BASE_DIR)
    ap.add_argument("--p10_primary", type=Path, default=P10_PRIMARY)
    ap.add_argument("--p10_reduced", type=Path, default=P10_REDUCED)
    ap.add_argument("--out_dir", type=Path,
                    default=ROOT / "results/appendix_c_p10_ablation_corrected_qc")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    primary, ps = run_set("primary", a.base_dir / "primary_4imu_cleaned_features.csv",
                          PRIMARY_FEATURES, a.p10_primary, a.seed)
    reduced, rs = run_set("reduced", a.base_dir / "reduced_pelvis_l3_features.csv",
                          REDUCED_FEATURES, a.p10_reduced, a.seed)

    table = primary.add_suffix("_primary").merge(
        reduced.add_suffix("_reduced"),
        left_on="held_out_primary", right_on="held_out_reduced", how="outer")
    table.to_csv(a.out_dir / "p10_ablation_table.csv", index=False)
    (a.out_dir / "p10_ablation_summary.json").write_text(
        json.dumps({"primary": ps, "reduced": rs, "seed": a.seed}, indent=2))
    emit_latex(primary, reduced, a.out_dir / "p10_ablation_table.tex")

    print("\n=== SUMMARY (corrected-QC base) ===")
    print(f"Primary 4-IMU (n=6): standard {ps['standard_mean']:.4f} -> "
          f"+P10 {ps['augmented_mean']:.4f}  (delta {ps['delta_mean']:+.4f})")
    print(f"Reduced Pelvis-L3 (n=9): standard {rs['standard_mean']:.4f} -> "
          f"+P10 {rs['augmented_mean']:.4f}  (delta {rs['delta_mean']:+.4f})")
    print(f"\nWrote: {a.out_dir}/  (table.csv, summary.json, table.tex)")
    print("Paste p10_ablation_table.tex over Table tab:c2.1, and update the §C2.3 prose "
          "means/deltas to the SUMMARY values above.")


if __name__ == "__main__":
    raise SystemExit(main())
