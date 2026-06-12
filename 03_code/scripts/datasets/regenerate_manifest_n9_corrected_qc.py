#!/usr/bin/env python3
"""Lock the QC-CLEANED drift-corrected n=9 re-freeze (canonical).

Supersedes both the pre-correction n=9 manifest and the intermediate
(contaminated) fallback_analysis_sets_n9_corrected manifest, which re-admitted
the P04/P05/P07 windows the QC protocol excludes. This QC-cleaned set restores
the documented frozen membership (P04 1450 / P05 1486 / P07 1356) on top of the
drift-corrected feature values.

Run on the machine with the artefacts (needs sklearn/joblib for version stamps):
    py scripts/datasets/regenerate_manifest_n9_corrected_qc.py
"""
from __future__ import annotations
import json, hashlib
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

# parents[1] is a DATA/results root (results/, ml/, requirements.txt live one
# level above this scripts/ tree) — not a sys.path anchor, so it stays as-is.
ROOT = Path(__file__).resolve().parents[1]
AS = ROOT / "results" / "fallback_analysis_sets_n9_corrected_qc"
MD = ROOT / "ml" / "models" / "fallback_final_n9_corrected_qc"
EV = AS / "evaluation"


def sha(p):
    """SHA-256 of a file's bytes — the integrity stamp every manifest entry carries."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    """Path relative to the repo root, Windows-style, so the manifest is portable across checkouts."""
    return str(Path(p).relative_to(ROOT)).replace("/", "\\")


def entry(p, with_rows=False):
    """Build one manifest record (path, size, hash, mtime); optionally summarise CSV rows/labels."""
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
    (EV / "primary_within.csv", False), (EV / "primary_loso.csv", False),
    (EV / "reduced_within.csv", False), (EV / "reduced_loso.csv", False),
    (EV / "evaluation_summary.json", False),
    (MD / "rf_primary_4imu.joblib", False), (MD / "rf_reduced_pelvis_l3.joblib", False),
    (MD / "rf_primary_4imu_feature_importance.csv", False),
    (MD / "rf_reduced_pelvis_l3_feature_importance.csv", False),
    (MD / "fallback_model_metadata.json", False),
    (ROOT / "scripts" / "apply_rest_anchor_correction.py", False),
    (ROOT / "scripts" / "reapply_qc_exclusions_corrected.py", False),
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
    "purpose": "Frozen QC-cleaned drift-corrected n=9 evidence package (Phase II.A IMU-only "
               "fallback) for FYP Chapter 8 — CANONICAL.",
    "supersedes": "results/fallback_analysis_sets_n9_corrected/FROZEN_MANIFEST.json "
                  "(intermediate; re-admitted QC-excluded P04/P05/P07 windows).",
    "correction": "Rest-anchor drift correction (apply_rest_anchor_correction.py) applied "
                  "cohort-wide; then frozen participant-level QC exclusions re-applied "
                  "(reapply_qc_exclusions_corrected.py): P04 -51 (T4 dropout), P05 -15, "
                  "P07 -145 (FATIGUE_FLEXION pelvis belt slip).",
    "policy": [
        "All thesis numerical claims must reference a SHA-256 entry in this manifest.",
        "Re-running the corrected+QC pipeline with seed=42 reproduces evaluation_summary.json.",
        "If any feature CSV is regenerated, this manifest is invalidated and a re-freeze is required.",
        "Do not edit in place; bump frozen_at_utc and keep prior manifests as dated snapshots."],
    "evaluator_seed": 42,
    "headline_auc": {
        "primary_within_mean_auc": es.get("primary", {}).get("within_mean_auc"),
        "primary_loso_mean_auc":   es.get("primary", {}).get("loso_mean_auc"),
        "reduced_within_mean_auc": es.get("reduced", {}).get("within_mean_auc"),
        "reduced_loso_mean_auc":   es.get("reduced", {}).get("loso_mean_auc"),
        "wilcoxon_two_sided_p":    es.get("wilcoxon", {}).get("two_sided_p"),
        "wilcoxon_n_eff":          es.get("wilcoxon", {}).get("n_eff")},
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
