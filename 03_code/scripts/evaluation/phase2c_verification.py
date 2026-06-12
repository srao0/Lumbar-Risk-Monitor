#!/usr/bin/env python3
"""
Phase II.C verification harness (frozen-pipeline check, NOT a generalisation proof).

Apply the FROZEN deployment models (ml/models/fallback_final) UNCHANGED to held-out
and synthetic varied sessions, to verify the trained decision pipeline executes end to
end and produces coherent risk output across the IMU-only fallback, reduced Pelvis-L3,
and full-hybrid (IMU+sEMG) routes. This is verification + Phase III preparation, NOT
evidence the system generalises to new human movement.

Integrity: no estimator is fitted; models are loaded read-only and only predict_proba is
called. Ground truth is the protocol label `risk_class_protocol`. Held-out status is
computed per (condition, model) from the training participant lists in
fallback_model_metadata.json: a real participant present in a model's training set is
reported as IN-SAMPLE for that model.

Run (machine with scikit-learn 1.8.0):
    python scripts/evaluation/phase2c_verification.py
    python scripts/evaluation/phase2c_verification.py --dry_run   # logic test, no sklearn needed

Outputs -> results/phase2c/ : phase2c_summary.csv/.md, per-condition predictions, run provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FALLBACK_DIR = ROOT / "ml" / "models" / "fallback_final_n9"
FALLBACK_META = FALLBACK_DIR / "fallback_model_metadata.json"
LR_EMG_GLOB = "LR_EMG_fold*.joblib"
LABEL_COL = "risk_class_protocol"
RESULTS_DIR = ROOT / "results" / "phase2c"

SAFE_MAX = 0.35
RISKY_MIN = 0.65  # == RISKY_THRESHOLD; overridden by the real FIS constant when importable.

CONDITIONS = [
    dict(key="P11_real_normal", kind="real", participant="participant_11",
         note="Real held-out participant; EMG ch4 saturated -> IMU-only routes only.",
         sources=["data/real/protocol_train/participant_11/session_001_restqc/feature_matrix.csv"],
         routes=["imu_only_fallback", "reduced_pelvis_l3"]),
    dict(key="P12_real_normal", kind="real", participant="participant_12",
         note="Real held-out participant; salvaged session_003_stitched_labeltrimmed.",
         sources=["data/real/protocol_train_fallback/participant_12/session_003_stitched_labeltrimmed/feature_matrix.csv"],
         routes=["imu_only_fallback", "reduced_pelvis_l3"]),
    dict(key="P03_real_varied", kind="real", participant="participant_03",
         note="Real VARIED movement; boundary/outlier sub-threshold mover; in reduced-train set.",
         sources=["data/real/varied_test_fallback/participant_03/session_001/feature_matrix.csv"],
         routes=["imu_only_fallback", "reduced_pelvis_l3"]),
    dict(key="P14_real_conforming", kind="real", participant="participant_14",
         note="Real held-out conforming full-hybrid participant (normal protocol, all sessions).",
         sources=["data/real/protocol_train_full_hybrid/participant_14/combined_features.csv"],
         routes=["imu_only_fallback", "reduced_pelvis_l3", "full_hybrid"]),
    dict(key="P14_synth_varied", kind="synthetic", participant="participant_14_synthetic",
         note="Synthetic VARIED A/B/C protocols (pipeline verification + Phase III substrate).",
         sources=[
             "data/synthetic/replay_dashboard_phase_iic_p14/replay_full_hybrid/session_A/feature_matrix.csv",
             "data/synthetic/replay_dashboard_phase_iic_p14/replay_full_hybrid/session_B/feature_matrix.csv",
             "data/synthetic/replay_dashboard_phase_iic_p14/replay_full_hybrid/session_C/feature_matrix.csv",
         ],
         routes=["imu_only_fallback", "reduced_pelvis_l3", "full_hybrid"]),
]

ROUTE_TO_MODEL = {
    "imu_only_fallback": "primary_4imu_cleaned",
    "reduced_pelvis_l3": "reduced_pelvis_l3",
    "full_hybrid": "primary_4imu_cleaned",
}

EMG_ALIASES = {"LOBL": "LMF", "ROBL": "RMF", "LMF": "LOBL", "RMF": "ROBL"}


def _rankdata(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    a_sorted = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a_sorted[j + 1] == a_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def auc_score(y, p):
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(p)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_metrics(y, pred, prob):
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * prec * sens / (prec + sens)) if prec and sens and not np.isnan(prec) and not np.isnan(sens) else 0.0
    acc = (tp + tn) / len(y) if len(y) else float("nan")
    return dict(n=len(y), n_safe=int((y == 0).sum()), n_risky=int((y == 1).sum()),
                auc=round(auc_score(y, prob), 4), sensitivity=round(sens, 4),
                specificity=round(spec, 4), precision=round(prec, 4), f1=round(f1, 4),
                accuracy=round(acc, 4), tp=tp, tn=tn, fp=fp, fn=fn,
                missed_risk=fn, false_alarms=fp)


def traffic_light(prob):
    """Map a fused risk probability to the deployment Safe/Cautious/Risky band
    the wearable surfaces to the user."""
    if prob < SAFE_MAX:
        return "Safe"
    if prob < RISKY_MIN:
        return "Cautious"
    return "Risky"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_condition_frame(cond):
    """Concatenate a condition's source feature CSVs, backfilling session_id and
    participant_id when a raw feature_matrix lacks them."""
    frames = []
    for i, rel in enumerate(cond["sources"]):
        p = ROOT / rel
        if not p.exists():
            raise FileNotFoundError(f"missing source for {cond['key']}: {p}")
        d = pd.read_csv(p)
        if "session_id" not in d.columns:
            d["session_id"] = Path(rel).parent.name or f"s{i+1}"
        if "participant_id" not in d.columns:
            d["participant_id"] = cond["participant"]
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def select_features(df, wanted):
    """Pull the model's expected feature columns from a frame, resolving the
    OBL<->MF channel-naming bug via EMG_ALIASES so older feature tables still
    line up with the frozen model's feature_names_in_. Missing values -> 0.0."""
    cols = {}
    for w in wanted:
        if w in df.columns:
            cols[w] = df[w]; continue
        aliased = w
        for a, b in EMG_ALIASES.items():
            if a in w and w.replace(a, b) in df.columns:
                aliased = w.replace(a, b); break
        if aliased in df.columns:
            cols[w] = df[aliased]
        else:
            raise KeyError(f"feature '{w}' not found (and no alias) in frame")
    return pd.DataFrame(cols).fillna(0.0)


def model_feature_list(model, fallback):
    names = getattr(model, "feature_names_in_", None)
    return list(names) if names is not None else list(fallback)


def main():
    ap = argparse.ArgumentParser(description="Phase II.C frozen-pipeline verification harness.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Skip sklearn/joblib; deterministic pseudo-probability to test data flow + outputs.")
    ap.add_argument("--results_dir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not FALLBACK_META.exists():
        print(f"[FAIL] missing {FALLBACK_META}"); return 1
    meta = json.loads(FALLBACK_META.read_text(encoding="utf-8-sig"))
    model_meta = {m["model_name"]: m for m in meta["models"]}

    risky_threshold = RISKY_MIN
    fis = None
    loaded = {}
    lr_emg_models = []
    sklearn_version = None
    if not args.dry_run:
        from joblib import load as joblib_load
        import sklearn
        sklearn_version = sklearn.__version__
        try:
            from ml.fuzzy.mamdani_fis import MamdaniFIS, RISKY_THRESHOLD
            risky_threshold = float(RISKY_THRESHOLD)
            fis = MamdaniFIS()
        except Exception as exc:
            fis = None
            print(f"[warn] could not import MamdaniFIS ({exc}); full_hybrid rows will be skipped.")
        for name, m in model_meta.items():
            mp = ROOT / m["model_file"].replace("\\", "/")
            loaded[name] = dict(model=joblib_load(mp), sha256=sha256(mp), path=str(mp))
        lr_paths = sorted((ROOT / "ml" / "models").glob(LR_EMG_GLOB))
        lr_emg_models = [joblib_load(p) for p in lr_paths]

    print(f"Phase II.C verification harness  (dry_run={args.dry_run}, RISKY_THRESHOLD={risky_threshold})")
    print("Frozen deployment models, read-only. No estimator is fitted.\n")

    rows, prov_models = [], {}
    for cond in CONDITIONS:
        try:
            df = load_condition_frame(cond)
        except Exception as exc:
            rows.append(dict(condition=cond["key"], route="-", model="-", status=f"LOAD_ERROR: {exc}")); continue
        if LABEL_COL not in df.columns:
            rows.append(dict(condition=cond["key"], route="-", model="-", status=f"NO_LABEL ({LABEL_COL})")); continue
        lab = df[df[LABEL_COL].isin([0, 1])].copy()
        if lab.empty:
            rows.append(dict(condition=cond["key"], route="-", model="-", status="NO_LABELLED_WINDOWS")); continue
        y = lab[LABEL_COL].to_numpy(int)

        for route in cond["routes"]:
            model_name = ROUTE_TO_MODEL[route]
            mmeta = model_meta[model_name]
            train_parts = set(mmeta.get("participants", []))
            if cond["kind"] == "synthetic":
                held = "n/a (synthetic)"
            else:
                held = "held-out" if cond["participant"] not in train_parts else "IN-SAMPLE (leak)"
            base = dict(condition=cond["key"], kind=cond["kind"], route=route, model=model_name,
                        held_out=held, note=cond["note"])
            try:
                feats = model_feature_list(loaded[model_name]["model"], mmeta["features"]) if not args.dry_run else mmeta["features"]
                X = select_features(lab, feats)
                if args.dry_run:
                    z = (X - X.mean()) / (X.std(ddof=0).replace(0, 1))
                    r_imu = 1.0 / (1.0 + np.exp(-z.mean(axis=1).to_numpy()))
                else:
                    r_imu = loaded[model_name]["model"].predict_proba(X.to_numpy())[:, 1]

                if route == "full_hybrid":
                    if cond["kind"] != "synthetic" and cond["participant"] == "participant_14":
                        base["emg_branch"] = "EMG held-out status UNVERIFIED (check LR_EMG provenance)"
                    if args.dry_run:
                        prob = r_imu; base["status"] = "DRY_OK (FIS skipped)"
                    elif fis is None or not lr_emg_models:
                        rows.append({**base, "status": "SKIPPED: FIS or LR_EMG unavailable"}); continue
                    else:
                        emg_feats = model_feature_list(lr_emg_models[0], [c for c in lab.columns if c.startswith("emg_")])
                        Xe = select_features(lab, emg_feats)
                        r_emg = np.vstack([m.predict_proba(Xe.to_numpy())[:, 1] for m in lr_emg_models]).mean(axis=0)
                        fin = lab.copy(); fin["R_IMU"] = r_imu; fin["R_EMG"] = r_emg
                        out = fis.infer_batch(fin)
                        prob = out["R_total"].to_numpy(float)
                else:
                    prob = r_imu

                pred = (prob >= risky_threshold).astype(int)
                m = binary_metrics(y, pred, prob)
                tl = pd.Series([traffic_light(p) for p in prob]).value_counts().to_dict()
                row = {**base, "status": base.get("status", "OK"), **m,
                       "tl_Safe": tl.get("Safe", 0), "tl_Cautious": tl.get("Cautious", 0), "tl_Risky": tl.get("Risky", 0)}
                rows.append(row)

                pred_df = lab[["session_id", "participant_id", LABEL_COL]].copy()
                pred_df["prob"] = prob; pred_df["pred"] = pred
                pred_df["traffic_light"] = [traffic_light(p) for p in prob]
                pred_df.to_csv(results_dir / f"phase2c_predictions_{cond['key']}_{route}.csv", index=False)
                if not args.dry_run and model_name in loaded:
                    prov_models[model_name] = loaded[model_name]["sha256"]
            except Exception as exc:
                rows.append({**base, "status": f"ERROR: {type(exc).__name__}: {exc}"})

    summary = pd.DataFrame(rows)
    summary.to_csv(results_dir / "phase2c_summary.csv", index=False)
    cols = [c for c in ["condition", "kind", "route", "model", "held_out", "status", "n",
                        "n_risky", "auc", "sensitivity", "specificity", "f1",
                        "missed_risk", "false_alarms", "tl_Safe", "tl_Cautious", "tl_Risky"] if c in summary.columns]
    md = ["# Phase II.C verification results", "",
          f"_Generated {datetime.now(timezone.utc).isoformat()} | dry_run={args.dry_run} | "
          f"RISKY_THRESHOLD={risky_threshold} | sklearn={sklearn_version}_", "",
          "Frozen deployment models applied read-only (no fitting). Ground truth = risk_class_protocol.",
          "Synthetic rows are pipeline/Phase-III verification, NOT generalisation evidence.", ""]
    try:
        md.append(summary[cols].to_markdown(index=False))
    except Exception:
        md.append(summary[cols].to_string(index=False))
    (results_dir / "phase2c_summary.md").write_text("\n".join(md), encoding="utf-8")

    prov = dict(generated=datetime.now(timezone.utc).isoformat(), dry_run=args.dry_run,
                sklearn_version=sklearn_version, risky_threshold=risky_threshold,
                label_column=LABEL_COL, models_used=prov_models,
                fallback_metadata=str(FALLBACK_META),
                note="Verification harness: frozen models, read-only, no estimator fitted.")
    (results_dir / "phase2c_run_provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")

    print(summary[cols].to_string(index=False))
    print("\n[OK] wrote " + str(results_dir) + "/phase2c_summary.(csv|md) + predictions + provenance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
