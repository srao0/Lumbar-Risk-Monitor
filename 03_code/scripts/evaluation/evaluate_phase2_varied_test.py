#!/usr/bin/env python3
"""
Aggregate Phase II.C evaluation using frozen Phase II.A models only.

Has no training route. It validates model provenance, applies the
shared feature pipeline to held-out varied sessions, and evaluates the final
deployed architecture using frozen Phase II.A RF_IMU and LR_EMG models followed
by the fixed Mamdani FIS.
"""

from __future__ import annotations

# Make the project package importable when this file is run directly.
import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = _Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PKG_ROOT))


import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.datasets.model_provenance import validate_phase2_model_provenance
from scripts.datasets.dataset_manifest import validate_dataset_manifest
from ml.fuzzy.mamdani_fis import MamdaniFIS, RISKY_THRESHOLD


FIS_AUXILIARY_COLUMNS = {
    "imu_z_ldlj",
    "imu_time_in_risk_zone",
    "emg_ai_ES",
    "imu_z_flex",
    "imu_z_vel",
}
OPERATING_MODES = {"full_hybrid", "imu_only_fallback"}


def risk_level_from_fixed_score(score: float) -> tuple[str, str]:
    """Map an IMU-only probability to the fixed traffic-light bands."""
    if score < 0.35:
        return "Safe", "Green"
    if score < 0.65:
        return "Cautious", "Amber"
    return "Risky", "Red"


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n_windows": int(len(y_true)),
        "n_safe": int((y_true == 0).sum()),
        "n_risky": int((y_true == 1).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "precision_risk": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_risk": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_frozen_models(models_dir: Path, model_type: str) -> list:
    paths = sorted(models_dir.glob(f"{model_type}_fold*.joblib"))
    if not paths:
        raise FileNotFoundError(
            f"No frozen {model_type} fold models found in {models_dir}"
        )
    print(f"  Loading {len(paths)} frozen model(s): {[path.name for path in paths]}")
    return [load(path) for path in paths]


def validate_session_modes(data_dir: Path, expected_mode: str) -> list[str]:
    """Reject aggregate evaluation sets containing a different declared mode."""
    errors = []
    metadata_paths = sorted(data_dir.rglob("session_metadata.json"))
    if not metadata_paths:
        return [f"No session_metadata.json files found under {data_dir}"]
    for path in metadata_paths:
        try:
            metadata = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            errors.append(f"Invalid session metadata JSON: {path}")
            continue
        recorded_mode = metadata.get("operating_mode")
        if recorded_mode != expected_mode:
            errors.append(
                f"Session {path.parent} records operating_mode={recorded_mode!r}; "
                f"cannot include it in {expected_mode!r} evaluation"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate Phase II.C evaluation using frozen Phase II.A models only."
    )
    parser.add_argument("--data_dir", default="data/real/varied_test")
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--results_dir", default="results/phase2_varied_test")
    parser.add_argument("--mode", choices=sorted(OPERATING_MODES), default="full_hybrid")
    parser.add_argument(
        "--skip_pipeline",
        action="store_true",
        help="Use an existing combined_features.csv rather than re-running shared feature extraction.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacement of existing Phase II.C feature/evaluation outputs.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    result_suffix = "full_hybrid" if args.mode == "full_hybrid" else "imu_only_fallback"
    protected_results = [
        results_dir / f"phase2_varied_predictions_{result_suffix}.csv",
        results_dir / f"phase2_varied_summary_{result_suffix}.csv",
        results_dir / f"phase2_varied_summary_{result_suffix}.json",
    ]
    existing_results = [path for path in protected_results if path.exists()]
    if existing_results and not args.force:
        print("[FAIL] Official Phase II.C results already exist; refusing overwrite without --force:")
        for path in existing_results:
            print(f"  - {path}")
        return 1

    print("\nPhase II.C: Frozen Aggregate Held-Out Evaluation")
    print("No model fitting or retraining is performed by this script.")
    print(f"Declared operating mode: {args.mode}")

    mode_errors = validate_session_modes(data_dir, args.mode)
    if mode_errors:
        print("[FAIL] Evaluation dataset mode separation")
        for error in mode_errors:
            print(f"  - {error}")
        return 1
    print("[OK] Evaluation dataset mode separation")

    metadata, errors = validate_phase2_model_provenance(
        models_dir,
        data_dir,
        expected_operating_mode=args.mode,
        allowed_cv_groups={"participant"} if args.mode == "full_hybrid" else {"participant", "session"},
    )
    if errors:
        print("[FAIL] Frozen Phase II.A model provenance validation")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("[OK] Frozen Phase II.A model provenance validation")

    model_types = set(metadata.get("model_types_trained", []))
    required_model_types = {"RF_IMU", "LR_EMG"} if args.mode == "full_hybrid" else {"RF_IMU"}
    missing_model_types = sorted(required_model_types - model_types)
    if missing_model_types:
        print(
            "[FAIL] Final hybrid evaluation requires frozen model types recorded "
            f"in provenance: {missing_model_types}"
        )
        return 1

    feature_records = metadata.get("feature_columns", {})
    imu_features = feature_records.get("IMU")
    emg_features = feature_records.get("EMG")
    if not imu_features or (args.mode == "full_hybrid" and not emg_features):
        print("[FAIL] Provenance metadata is missing feature columns required for the declared mode")
        return 1

    if not args.skip_pipeline:
        print(f"Processing held-out sessions through shared pipeline: {data_dir}")
        from signal_processing.pipeline import run_pipeline_batch
        run_pipeline_batch(
            str(data_dir),
            output_dir=str(data_dir),
            label_source="protocol",
            phase="Phase II.C",
            operating_mode=args.mode,
            force=args.force,
            command_used=" ".join([sys.executable, *sys.argv]),
        )

    _, manifest_errors = validate_dataset_manifest(
        data_dir,
        expected_phase="Phase II.C",
        expected_label_source="protocol",
        expected_operating_mode=args.mode,
    )
    if manifest_errors:
        print("[FAIL] Phase II.C dataset manifest validation")
        for error in manifest_errors:
            print(f"  - {error}")
        return 1

    feature_path = data_dir / "combined_features.csv"
    df = pd.read_csv(feature_path)
    required = set(imu_features) | {"risk_class_protocol", "session_id"}
    if args.mode == "full_hybrid":
        required |= set(emg_features) | FIS_AUXILIARY_COLUMNS
    missing = sorted(required - set(df.columns))
    if missing:
        print(f"[FAIL] Held-out feature file is missing columns: {missing}")
        return 1

    labelled = df[df["risk_class_protocol"].isin([0, 1])].copy()
    if labelled.empty:
        print("[FAIL] No protocol-labelled safe/risky windows found in held-out feature file")
        return 1

    imu_models = load_frozen_models(models_dir, "RF_IMU")
    x_imu = labelled[imu_features].fillna(0.0).to_numpy()
    r_imu = np.vstack([model.predict_proba(x_imu)[:, 1] for model in imu_models]).mean(axis=0)
    if args.mode == "full_hybrid":
        emg_models = load_frozen_models(models_dir, "LR_EMG")
        x_emg = labelled[emg_features].fillna(0.0).to_numpy()
        r_emg = np.vstack([model.predict_proba(x_emg)[:, 1] for model in emg_models]).mean(axis=0)
        fis_input = labelled.copy()
        fis_input["R_IMU"] = r_imu
        fis_input["R_EMG"] = r_emg
        fis_output = MamdaniFIS().infer_batch(fis_input)
        probability = fis_output["R_total"].to_numpy(dtype=float)
        risk_levels = fis_output["risk_level"].values
        colours = fis_output["colour"].values
        reasons = fis_output["reason"].values
    else:
        r_emg = np.full(len(labelled), np.nan)
        probability = r_imu
        mappings = [risk_level_from_fixed_score(score) for score in probability]
        risk_levels = [mapping[0] for mapping in mappings]
        colours = [mapping[1] for mapping in mappings]
        reasons = ["IMU-only fallback: frozen RF_IMU probability threshold mapping; FIS not used."] * len(labelled)
    prediction = (probability >= RISKY_THRESHOLD).astype(int)
    y_true = labelled["risk_class_protocol"].to_numpy(dtype=int)

    metrics = calculate_metrics(y_true, prediction, probability)
    per_window = labelled[["session_id", "risk_class_protocol"]].copy()
    if "participant_id" in labelled.columns:
        per_window.insert(1, "participant_id", labelled["participant_id"].values)
    per_window["R_IMU"] = r_imu
    per_window["operating_mode"] = args.mode
    per_window["primary_risk_score"] = probability
    if args.mode == "full_hybrid":
        per_window["R_EMG"] = r_emg
        per_window["R_total"] = probability
        per_window["fis_reason"] = reasons
    per_window["risk_level"] = risk_levels
    per_window["colour"] = colours
    per_window["risk_prediction"] = prediction
    per_window["correct"] = prediction == y_true

    results_dir.mkdir(parents=True, exist_ok=True)
    per_window_path = results_dir / f"phase2_varied_predictions_{result_suffix}.csv"
    summary_csv_path = results_dir / f"phase2_varied_summary_{result_suffix}.csv"
    summary_json_path = results_dir / f"phase2_varied_summary_{result_suffix}.json"
    metrics["operating_mode"] = args.mode
    per_window.to_csv(per_window_path, index=False)
    pd.DataFrame([metrics]).to_csv(summary_csv_path, index=False)
    summary_json = {
        "phase": "Phase II.C",
        "operating_mode": args.mode,
        "evaluation_type": (
            "frozen_phase2a_hybrid_rf_imu_lr_emg_mamdani_fis"
            if args.mode == "full_hybrid"
            else "frozen_phase2a_imu_only_fallback_rf_imu"
        ),
        "deployed_architecture": (
            "RF_IMU + LR_EMG + Mamdani FIS"
            if args.mode == "full_hybrid"
            else "IMU-only fallback: RF_IMU probability with fixed output mapping"
        ),
        "varied_data_dir": str(data_dir.resolve()),
        "feature_file": str(feature_path.resolve()),
        "models_dir": str(models_dir.resolve()),
        "model_types": ["RF_IMU", "LR_EMG"] if args.mode == "full_hybrid" else ["RF_IMU"],
        "comparator_model_not_used_for_primary_evaluation": "RF_IMU_EMG",
        "frozen_model_files": {
            "RF_IMU": sorted(path.name for path in models_dir.glob("RF_IMU_fold*.joblib")),
            **(
                {"LR_EMG": sorted(path.name for path in models_dir.glob("LR_EMG_fold*.joblib"))}
                if args.mode == "full_hybrid" else {}
            ),
        },
        "training_provenance": metadata,
        "risk_output_mapping": {
            "Safe": f"Green; {'R_total' if args.mode == 'full_hybrid' else 'R_IMU'} < 0.35",
            "Cautious": f"Amber; 0.35 <= {'R_total' if args.mode == 'full_hybrid' else 'R_IMU'} < 0.65",
            "Risky": f"Red; {'R_total' if args.mode == 'full_hybrid' else 'R_IMU'} >= 0.65",
        },
        "binary_risky_threshold": RISKY_THRESHOLD,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    summary_json_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    print(f"[OK] Aggregate held-out evaluation complete ({args.mode})")
    print(f"  Windows: {metrics['n_windows']} | Sensitivity: {metrics['sensitivity']:.3f} | Specificity: {metrics['specificity']:.3f}")
    print(f"  Summary JSON: {summary_json_path}")
    print(f"  Summary CSV: {summary_csv_path}")
    print(f"  Predictions CSV: {per_window_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
