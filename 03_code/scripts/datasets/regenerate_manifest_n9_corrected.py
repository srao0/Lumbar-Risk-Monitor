#!/usr/bin/env python3
"""Lock the DRIFT-CORRECTED n=9 re-freeze.

Writes results/fallback_analysis_sets_n9_corrected/FROZEN_MANIFEST.json with SHA-256
over the corrected canonical CSVs, corrected evaluation outputs, corrected final
fallback models, and the generating scripts (including the rest-anchor drift
correction that defines this re-freeze).

This is the REPLACEMENT canonical manifest after the trunk-flex x Madgwick-drift
fix. The prior n=9 manifest (results/fallback_analysis_sets_n9/FROZEN_MANIFEST.json)
is NOT touched -- it stays as the pre-correction audit snapshot, per the
no-edit-in-place policy.

Run on the machine that produced the artefacts (so hashes + sklearn/joblib
versions are of the canonical 1.8.0 build). Resolves repo root from its location.
"""
from __future__ import annotations
import json, hashlib
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AS = ROOT / "results" / "fallback_analysis_sets_n9_corrected"     # corrected analysis set
MD = ROOT / "ml" / "models" / "fallback_final_n9_corrected"       # corrected final models
EV = AS / "evaluation"                                            # corrected eval (NB: not evaluation_corrected)


def sha(p):
    """SHA-256 of an artefact — the integrity anchor every thesis numerical claim must trace back to."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):
    """Repo-relative path with Windows separators so the manifest reads identically to the canonical build machine."""
    return str(Path(p).relative_to(ROOT)).replace("/", "\\")


def entry(p, with_rows=False):
    """Build one manifest record (path, size, hash, mtime) for a file; with_rows also captures row/col/participant/class counts for the feature CSVs."""
    p = Path(p); st = p.stat()
    e = {"path": rel(p), "size_bytes": st.st_size, "sha256": sha(p),
         "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()}
    if with_rows:
        df = pd.read_csv(p, low_memory=False)
        e["n_rows"] = int(len(df)); e["n_cols"] = int(df.shape[1])
        if "participant_id" in df:
            e["participants"] = sorted(df.participant_id.dropna().unique().tolist())
        if "risk_class" in df:
            rc = df[df.risk_class.isin([0, 1])]
            e["risk_class_counts"] = {"0": int((rc.risk_class == 0).sum()),
                                      "1": int((rc.risk_class == 1).sum())}
    return e


spec = [
    (AS / "primary_4imu_cleaned_features.csv", True),
    (AS / "reduced_pelvis_l3_features.csv", True),
    (AS / "analysis_set_summary_n9.json", False),
    (EV / "primary_within.csv", False), (EV / "primary_loso.csv", False),
    (EV / "reduced_within.csv", False), (EV / "reduced_loso.csv", False),
    (EV / "evaluation_summary.json", False),
    (MD / "rf_primary_4imu.joblib", False), (MD / "rf_reduced_pelvis_l3.joblib", False),
    (MD / "rf_primary_4imu_feature_importance.csv", False),
    (MD / "rf_reduced_pelvis_l3_feature_importance.csv", False),
    (MD / "fallback_model_metadata.json", False),
    # generating + correction provenance
    (ROOT / "scripts" / "apply_rest_anchor_correction.py", False),
    (ROOT / "scripts" / "refreeze_n9.py", False),
    (ROOT / "scripts" / "prepare_fallback_analysis_sets.py", False),
    (ROOT / "scripts" / "evaluate_fallback_analysis_sets.py", False),
    (ROOT / "scripts" / "train_fallback_analysis_models.py", False),
    (ROOT / "requirements.txt", False),
]
files = []
for f, wr in spec:
    if Path(f).exists():
        files.append(entry(f, wr))
    else:
        print("WARN missing:", f)

es = json.loads((EV / "evaluation_summary.json").read_text()) if (EV / "evaluation_summary.json").exists() else {}

import sklearn, joblib
manifest = {
    "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "Frozen DRIFT-CORRECTED n=9 evidence package (Phase II.A IMU-only "
               "fallback) for FYP Chapter 8, after the trunk-flex x Madgwick-drift fix.",
    "supersedes": "results/fallback_analysis_sets_n9/FROZEN_MANIFEST.json "
                  "(pre-correction n=9 — kept as audit snapshot)",
    "correction": "Rest-anchor drift correction (scripts/apply_rest_anchor_correction.py) "
                  "applied cohort-wide before feature regeneration; abs-sum trunk_flex "
                  "retained (signed-sum marginal on corrected data).",
    "policy": [
        "All thesis numerical claims must reference a SHA-256 entry in this manifest.",
        "Re-running the corrected pipeline with seed=42 reproduces evaluation_summary.json.",
        "If any feature CSV is regenerated, this manifest is invalidated and a re-freeze is required.",
        "Do not edit in place; bump frozen_at_utc and keep prior manifests as dated snapshots."],
    "evaluator_seed": 42,
    "headline_auc": {
        "primary_within_mean_auc": es.get("primary", {}).get("within_mean_auc"),
        "primary_loso_mean_auc":   es.get("primary", {}).get("loso_mean_auc"),
        "reduced_within_mean_auc": es.get("reduced", {}).get("within_mean_auc"),
        "reduced_loso_mean_auc":   es.get("reduced", {}).get("loso_mean_auc"),
        "wilcoxon_two_sided_p":    es.get("wilcoxon", {}).get("two_sided_p"),
        "wilcoxon_n_eff":          es.get("wilcoxon", {}).get("n_eff"),
        "wilcoxon_min_possible_p": es.get("wilcoxon", {}).get("min_possible_p")},
    "sklearn_version": sklearn.__version__, "joblib_version": joblib.__version__,
    "alignment_decision": "Option B: primary 17 feats; reduced 13 feats; RF n_estimators=500, "
                          "min_samples_leaf=3, class_weight=balanced, random_state=42.",
    "files": files,
}
out = AS / "FROZEN_MANIFEST.json"
out.write_text(json.dumps(manifest, indent=2))
print(f"Wrote {out}")
print(f"  files hashed: {len(files)}")
print(f"  headline: {manifest['headline_auc']}")
