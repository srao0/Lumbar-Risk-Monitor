#!/usr/bin/env python3
"""
freeze_emgnorm_artifacts.py
================================================================================
Lock the sEMG resting-baseline-normalisation re-freeze (2026-06-11) with
SHA-256 manifests, mirroring regenerate_manifest_n9_corrected_qc.py.

Writes TWO manifests:
  1. results/participant_14_analysis_emgnorm/FROZEN_MANIFEST.json
       Full-hybrid P14 arm: normalised combined_features, retrained fold models,
       Phase II.B analysis, Phase II.C emgnorm verification, and the scripts that
       produced them.
  2. results/personalised_stage2b_corrected_qc_n9/FROZEN_MANIFEST.json
       RQ4 personalisation pilot on the QC-clean cohort.

Headline numbers are read from the result JSON/CSV so the manifest cannot drift
from the artefacts it hashes.

Sits at the end of the dataset/freeze stage; consumes finished artefacts and
emits the SHA-256 provenance manifests the thesis claims reference.

Run on the machine with the artefacts (needs sklearn/joblib for version stamps):
    py scripts/datasets/freeze_emgnorm_artifacts.py
"""
from __future__ import annotations
import json, hashlib
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

# parents[2] from scripts/datasets/ resolves to 03_code, the artefact root that
# holds data/, ml/, results/ and the scripts hashed into the manifests.
ROOT = Path(__file__).resolve().parents[2]


def sha(p):
    """SHA-256 hex digest of a file's bytes."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    """Path relative to ROOT in Windows backslash form (manifest display)."""
    return str(Path(p).relative_to(ROOT)).replace("/", "\\")


def entry(p):
    """One manifest record for a file: relative path, size, SHA-256, mtime."""
    p = Path(p); st = p.stat()
    return {"path": rel(p), "size_bytes": st.st_size, "sha256": sha(p),
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()}


def hash_many(paths):
    """Hash each existing path; warn and skip any that are missing."""
    out = []
    for p in paths:
        if Path(p).exists():
            out.append(entry(p))
        else:
            print("  WARN missing:", rel(p) if str(p).startswith(str(ROOT)) else p)
    return out


def hash_glob(directory, pattern="*"):
    """Hash every file in a directory matching the glob pattern (sorted)."""
    d = Path(directory)
    return hash_many(sorted(d.glob(pattern))) if d.exists() else []


try:
    import sklearn, joblib
    VER = {"sklearn_version": sklearn.__version__, "joblib_version": joblib.__version__}
except Exception:
    VER = {"sklearn_version": None, "joblib_version": None}

NOW = datetime.now(timezone.utc).isoformat()

# ---------------------------------------------------------------- arm 1: full-hybrid emgnorm
DATA1 = ROOT / "data/real/protocol_train_full_hybrid_emgnorm"
MD1   = ROOT / "ml/models/p14_fullhybrid_emgnorm"
IIB   = ROOT / "results/participant_14_analysis_emgnorm"
IIC   = ROOT / "results/phase2c_emgnorm"

iib_summary = {}
if (IIB / "corrected_summary.json").exists():
    s = json.loads((IIB / "corrected_summary.json").read_text())
    q1 = s.get("q1", {})
    iib_summary = {"q1_mean_imu": q1.get("mean_imu"), "q1_mean_hyb": q1.get("mean_hyb"),
                   "q1_mean_delta": q1.get("mean_delta"), "q1_wins": q1.get("wins"),
                   "q1_wilcoxon_p": q1.get("wilcoxon_p"),
                   "emg_importance_share": q1.get("emg_importance_share")}

files1 = []
files1 += hash_many([DATA1 / "combined_features.csv", DATA1 / "dataset_manifest.json"])
files1 += hash_glob(MD1, "*.joblib")
files1 += hash_glob(MD1, "*.json")
files1 += hash_glob(IIB, "*.csv") + hash_glob(IIB, "*.json")
files1 += hash_glob(IIC, "*.csv") + hash_glob(IIC, "*.json") + hash_glob(IIC, "*.md")
# Source scripts that produced the arm-1 artefacts (nested-package layout).
files1 += hash_many([ROOT / "signal_processing/pipeline.py",
                     ROOT / "scripts/phase_runners/run_pipeline.py",
                     ROOT / "ml/training/train_classifier.py",
                     ROOT / "scripts/evaluation/analyse_p14_full_hybrid_corrected.py",
                     ROOT / "scripts/phase_runners/run_phase2c_emgnorm.py",
                     ROOT / "scripts/evaluation/phase2c_verification.py"])

manifest1 = {
    "frozen_at_utc": NOW,
    "purpose": "Frozen full-hybrid P14 evidence with sEMG resting-baseline-ratio "
               "amplitude normalisation (Phase II.B + II.C) for FYP Chapter 8 — CANONICAL "
               "(full-hybrid arm).",
    "supersedes": "results/participant_14_analysis_corrected_qc (raw-amplitude sEMG) and "
                  "results/phase2c_corrected_qc full_hybrid route, for the full-hybrid arm only. "
                  "IMU-only / reduced routes are unchanged and remain in fallback_final_n9_corrected_qc.",
    "normalisation": "emg_rms_*/emg_mav_* divided by per-session BASELINE_STATIC mean "
                     "(resting-baseline ratio); applied via run_pipeline "
                     "--emg_amplitude_norm resting_baseline_ratio. AI/CAI/ZCR unscaled.",
    "policy": [
        "All thesis full-hybrid numerical claims must reference a SHA-256 entry here.",
        "Re-running with seed=42 reproduces corrected_summary.json and the II.C summary.",
        "Regenerating any feature CSV or model invalidates this manifest; re-freeze required."],
    "evaluator_seed": 42,
    "phase_iib_headline": iib_summary,
    **VER,
    "files": files1,
}
out1 = IIB / "FROZEN_MANIFEST.json"
out1.write_text(json.dumps(manifest1, indent=2))
print(f"Wrote {rel(out1)}  ({len(files1)} files hashed)")
print(f"  Phase II.B headline: {iib_summary}")

# ---------------------------------------------------------------- arm 2: stage2b QC-clean
S2B   = ROOT / "results/personalised_stage2b_corrected_qc_n9"
S2BMD = ROOT / "ml/models/personalised_stage2b_corrected_qc_n9"
DATA2 = ROOT / "data/real/fallback_corrected_qc"

s2b_headline = {}
if (S2B / "summary_metrics.csv").exists():
    sm = pd.read_csv(S2B / "summary_metrics.csv")
    if "model_variant" in sm and "auc_mean" in sm:
        s2b_headline = {r["model_variant"]: round(float(r["auc_mean"]), 4)
                        for _, r in sm.iterrows()}

files2 = []
files2 += hash_many([DATA2 / "combined_features.csv"])
files2 += hash_glob(S2B, "*.csv") + hash_glob(S2B, "*.json") + hash_glob(S2B, "*.md")
files2 += hash_glob(S2BMD, "*.joblib") + hash_glob(S2BMD, "*.json")
# Source scripts that produced the arm-2 artefacts (nested-package layout).
files2 += hash_many([ROOT / "scripts/data_preparation/make_stage2b_qc_input.py",
                     ROOT / "scripts/training/run_personalised_stage2b.py"])

manifest2 = {
    "frozen_at_utc": NOW,
    "purpose": "Frozen RQ4 personalisation-calibration pilot (stage2b) on the QC-clean "
               "cohort — exploratory, IMU-only. Cited in Chapter 8 as a pilot, not a "
               "generalisation result.",
    "supersedes": "results/personalised_stage2b_corrected_n9 (contaminated: re-admitted "
                  "P04/P05/P07 QC-excluded windows).",
    "qc_filter": "make_stage2b_qc_input.py drops only the 211 re-admitted window-range "
                 "exclusions (P04/P05/P07); P02 kept to match the canonical frozen n=9.",
    "policy": [
        "Reported as an exploratory pilot; not population-level or longitudinal evidence.",
        "Re-running with seed=42 on fallback_corrected_qc reproduces summary_metrics.csv."],
    "evaluator_seed": 42,
    "headline_auc_mean": s2b_headline,
    **VER,
    "files": files2,
}
out2 = S2B / "FROZEN_MANIFEST.json"
out2.write_text(json.dumps(manifest2, indent=2))
print(f"Wrote {rel(out2)}  ({len(files2)} files hashed)")
print(f"  stage2b headline: {s2b_headline}")
