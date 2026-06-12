#!/usr/bin/env python3
"""
replay_from_features.py
================================================================================
Generate a dashboard-compatible replay (replay_predictions.csv +
replay_summary.json) by running the deployed FIS over a session's PRECOMPUTED
feature_matrix.csv, exactly as the offline Phase II.C verification does.

Why this exists: live_risk_pipeline.py recomputes features incrementally with a
rolling baseline, and that recompute produces a bad smoothness z-score
(imu_z_ldlj) on the held-out sessions, which collapses the fallback FIS's
smoothness membership and parks safe windows in Cautious/amber (R_total=0.500,
no rule fired / Medium catch-all). The offline path reads the correctly
baselined feature_matrix.csv and gives the right colours (e.g. P12 reduced =
51% Safe). This script reproduces the offline decisions for the replay demo, so
Phase III.A shows the same Safe/Cautious/Risky behaviour the evaluation reports.

Inputs : a processed session dir containing feature_matrix.csv, plus the deployed
         fallback model dir.
Outputs: replay_predictions.csv + replay_summary.json in --out_session.
Sits in Phase III.A (replay/demo), downstream of the pipeline and evaluation.

Run on the machine with sklearn:
    py scripts/demo/replay_from_features.py \
        --session data/real/phase2c_heldout_corrected/participant_12/session_003_stitched_labeltrimmed \
        --out_session data/real/replay_offline_p12_reduced \
        --mode imu_only_fallback \
        --models_dir ml/models/fallback_final_n9_corrected_qc
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# parents[2] from scripts/demo/ resolves to 03_code, placed on sys.path so the
# demo package import below resolves.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.demo.demo_risk_monitor import load_models, classify_window  # noqa: E402

LEVEL = {"GREEN": "Safe", "AMBER": "Cautious", "RED": "Risky"}


def main() -> int:
    """Classify every window of the session offline and write the replay CSV/JSON.

    With --light fis the traffic light is the Mamdani FIS R_total; with
    --light operating_point (IMU-only) it is R_IMU thresholded at 0.35/0.65 with a
    NIOSH deep-flexion escalation, matching the Phase II.C operating point.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="Session dir containing feature_matrix.csv")
    ap.add_argument("--out_session", required=True, help="Output dir for replay_predictions/summary")
    ap.add_argument("--mode", choices=["full_hybrid", "imu_only_fallback"], default="imu_only_fallback")
    ap.add_argument("--models_dir", default="ml/models/fallback_final_n9_corrected_qc")
    ap.add_argument("--light", choices=["fis", "operating_point"], default="fis",
                    help="Traffic-light source. 'fis' = Mamdani FIS R_total (default). "
                         "'operating_point' = R_IMU at 0.35/0.65 + NIOSH deep-flexion "
                         "escalation; the principled IMU-only mapping (matches Phase II.C).")
    a = ap.parse_args()

    sess = Path(a.session)
    fm = sess / "feature_matrix.csv"
    if not fm.exists():
        print(f"ERROR: {fm} not found"); return 1
    df = pd.read_csv(fm, low_memory=False).reset_index(drop=True)
    print(f"  Loaded {len(df)} windows from {fm}")

    imu_model, imu_features, emg_model, emg_features = load_models(Path(a.models_dir), a.mode)

    label_col = "risk_class_protocol" if "risk_class_protocol" in df else "risk_class"

    def operating_point_light(r_imu, time_zone):
        # R_IMU at the deployed operating threshold (0.35/0.65) + NIOSH deep-flexion
        # escalation: a window mostly in the >45 deg zone is not 'Safe' even if the
        # classifier is unsure. Reasons mirror the deployed engineering wording.
        base = "GREEN" if r_imu < 0.35 else ("AMBER" if r_imu < 0.65 else "RED")
        reason = {"GREEN": "Kinematic risk below the safe operating threshold",
                  "AMBER": "Moderate kinematic risk (above the safe threshold)",
                  "RED": "High kinematic risk"}[base]
        if base == "GREEN" and (time_zone or 0.0) >= 0.5:
            base, reason = "AMBER", "Sustained time in the >45 deg flexion zone (NIOSH)"
        level = {"GREEN": "Safe", "AMBER": "Cautious", "RED": "Risky"}[base]
        return base, level, reason

    rows = []
    for i, row in df.iterrows():
        res = classify_window(imu_model, imu_features, emg_model, emg_features, row, a.mode)
        rec = dict(row)  # keep all feature columns for the dashboard panels
        rec["window_idx"] = int(i)
        if "window_centre_ms" not in rec:
            rec["window_centre_ms"] = float((i + 1) * 1000)
        rec["R_IMU"] = res["R_IMU"]
        rec["R_EMG"] = res.get("R_EMG", float("nan"))
        if a.light == "operating_point" and a.mode == "imu_only_fallback":
            r_imu = float(res["R_IMU"])
            tz = float(row.get("imu_time_in_risk_zone", 0.0) or 0.0)
            light, level, reason = operating_point_light(r_imu, tz)
            rec["traffic_light"] = light
            rec["R_total"] = r_imu
            rec["predicted"] = level
            rec["predicted_binary"] = int(r_imu >= 0.65)
            rec["fis_reason"] = reason
        else:
            rec["traffic_light"] = res["label"]
            rec["R_total"] = res["p_risk"]
            rec["predicted"] = res["risk_level"]
            rec["predicted_binary"] = int(res["predicted"])
            rec["fis_reason"] = res.get("fis_reason", "")
        if label_col in row and pd.notna(row[label_col]):
            rec["true_risk_class"] = row[label_col]
        rows.append(rec)

    pred = pd.DataFrame(rows)
    out = Path(a.out_session)
    out.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out / "replay_predictions.csv", index=False)

    counts = pred["traffic_light"].map(LEVEL).value_counts().to_dict()
    risk_counts = {k: int(counts.get(k, 0)) for k in ("Safe", "Cautious", "Risky")}
    n = len(pred)
    risk_pct = {k: round(100 * v / n, 1) for k, v in risk_counts.items()} if n else {}
    acc = None
    if "true_risk_class" in pred:
        lab = pred[pred["true_risk_class"].isin([0, 1, 0.0, 1.0])]
        if len(lab):
            yhat = lab["predicted_binary"].astype(int)
            ytrue = lab["true_risk_class"].astype(float).astype(int)
            acc = {"n_labelled_windows": int(len(lab)),
                   "binary_accuracy": round(float((yhat.values == ytrue.values).mean()), 4)}
    hi = pred.loc[pred["R_total"].astype(float).idxmax()] if n else None

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_processed_session": str(sess).replace("/", "\\"),
        "out_session": str(out).replace("/", "\\"),
        "operating_mode": a.mode,
        "model_directory_used": str(Path(a.models_dir)).replace("/", "\\"),
        "generator": "replay_from_features.py (offline feature_matrix; reproduces Phase II.C decisions)",
        "emg_available": a.mode == "full_hybrid",
        "total_windows": n,
        "risk_counts": risk_counts,
        "risk_percentages": risk_pct,
        "accuracy_vs_protocol_labels": acc,
        "highest_risk_window": ({"window_idx": int(hi["window_idx"]),
                                  "R_total": round(float(hi["R_total"]), 4),
                                  "traffic_light": hi["traffic_light"],
                                  "movement_label": hi.get("movement_label", "")} if hi is not None else None),
        "generated_files": ["replay_predictions.csv", "replay_summary.json"],
        "disclaimer": "Movement-risk feedback within this system only. Not a medical diagnosis.",
    }
    (out / "replay_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Wrote {out}\\replay_predictions.csv + replay_summary.json")
    print(f"  risk_counts: {risk_counts}  ({risk_pct})")
    if acc:
        print(f"  binary accuracy vs protocol: {acc['binary_accuracy']} on {acc['n_labelled_windows']} windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
