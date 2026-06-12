#!/usr/bin/env python3
"""
Evaluate RF-only versus RF + IMU fallback FIS on frozen fallback datasets.

The comparison uses the same folds for both routes:

    RF-only:
        IMU features -> RF probability -> threshold

    RF + FIS:
        IMU features -> RF probability R_IMU
        R_IMU + IMU auxiliary features -> IMUFallbackFIS -> threshold

Thresholds are selected on the training fold using Youden's J, then applied to
the held-out fold. AUC is computed on the held-out fold scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.fuzzy.mamdani_fis import IMUFallbackFIS


PRIMARY_FEATURES = [
    "imu_trunk_angle_peak",
    "imu_trunk_angle_mean",
    "imu_angvel_peak",
    "imu_angvel_mean",
    "imu_time_in_risk_zone",
    "imu_time_high_velocity",
    "imu_ldlj",
    "imu_jerk_rms",
    "imu_jerk_peak",
    "imu_ldlj_multiaxis",
    "imu_compensation_index",
    "imu_lumbopelv_ratio",
    "imu_pelvis_angle_peak",
    "imu_pelvis_angle_mean",
    "imu_z_flex",
    "imu_z_vel",
    "imu_z_ldlj",
]

REDUCED_FEATURES = [
    "imu_angvel_peak",
    "imu_angvel_mean",
    "imu_time_high_velocity",
    "imu_ldlj",
    "imu_jerk_rms",
    "imu_jerk_peak",
    "imu_pelvis_angle_peak",
    "imu_pelvis_angle_mean",
    "imu_l3_accel_tilt_peak",
    "imu_l3_accel_tilt_mean",
    "imu_l3_accel_tilt_range",
    "imu_z_vel",
    "imu_z_ldlj",
]

AUXILIARY_FIS_COLUMNS = [
    "imu_time_in_risk_zone",
    "imu_z_flex",
    "imu_z_vel",
    "imu_z_ldlj",
]


@dataclass
class FoldOutput:
    metrics: list[dict]
    predictions: list[pd.DataFrame]


def youden_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.65
    fpr, tpr, thresholds = roc_curve(y_true, score)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def compute_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> dict:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "auc": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else float("nan"),
        "threshold": float(threshold),
        "sens": float(sens),
        "spec": float(spec),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "n": int(len(y_true)),
        "safe_n": int((y_true == 0).sum()),
        "risky_n": int((y_true == 1).sum()),
    }


def fit_rf(train: pd.DataFrame, features: list[str], seed: int) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=75,
        random_state=seed,
        class_weight="balanced",
        min_samples_leaf=5,
        n_jobs=-1,
    )
    model.fit(train[features], train["risk_class"].astype(int))
    return model


def _mean_abs_available(row: pd.Series, columns: list[str]) -> float:
    vals = []
    for col in columns:
        if col in row.index and not pd.isna(row[col]):
            vals.append(abs(float(row[col])))
    return float(np.clip(np.mean(vals), 0.0, 3.0)) if vals else 0.0


def fis_scores(df: pd.DataFrame, r_imu: np.ndarray) -> np.ndarray:
    """Run each window's RF probability plus its IMU auxiliary cues through the
    fixed Mamdani fallback FIS, returning the fused R_total per window.

    The FIS is rule-based, not trained, so this is purely a forward inference
    over the held-out windows. Missing auxiliary columns (e.g. the reduced set
    lacks time-in-risk-zone) default to 0, which the report flags as a caveat.
    """
    fis = IMUFallbackFIS(resolution=101)
    scores = []
    has_time = "imu_time_in_risk_zone" in df.columns
    for (_, row), prob in zip(df.iterrows(), r_imu):
        z_sal = float(np.clip(row.get("imu_z_ldlj", 0.0), -3.0, 3.0))
        time_in_risk_zone = (
            float(np.clip(row.get("imu_time_in_risk_zone", 0.0), 0.0, 1.0))
            if has_time else 0.0
        )
        z_imu_mean = _mean_abs_available(row, ["imu_z_flex", "imu_z_vel", "imu_z_ldlj"])
        result = fis.infer(
            R_IMU=float(prob),
            z_sal=z_sal,
            time_in_risk_zone=time_in_risk_zone,
            z_imu_mean=z_imu_mean,
        )
        scores.append(float(result["R_total"]))
    return np.asarray(scores, dtype=float)


def _prediction_frame(
    test: pd.DataFrame,
    fold_name: str,
    scheme: str,
    rf_score: np.ndarray,
    fis_score: np.ndarray,
    rf_threshold: float,
    fis_threshold: float,
) -> pd.DataFrame:
    cols = [
        col for col in [
            "participant_id",
            "session_id",
            "window_centre_ms",
            "movement_label",
            "risk_class",
        ] if col in test.columns
    ]
    out = test[cols].copy()
    out.insert(0, "scheme", scheme)
    out.insert(1, "fold", fold_name)
    out["rf_score"] = rf_score
    out["fis_score"] = fis_score
    out["rf_threshold"] = rf_threshold
    out["fis_threshold"] = fis_threshold
    out["rf_pred"] = (rf_score >= rf_threshold).astype(int)
    out["fis_pred"] = (fis_score >= fis_threshold).astype(int)
    return out


def evaluate_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    seed: int,
    fold_name: str,
    scheme: str,
) -> tuple[list[dict], pd.DataFrame]:
    """Score one fold under both routes on identical train/test data.

    Both the RF-only and RF+FIS routes pick their Youden threshold on the SAME
    training fold and apply it to the held-out fold, so any AUC difference is
    attributable to the FIS layer rather than to a threshold or data advantage.
    """
    model = fit_rf(train, features, seed)
    y_train = train["risk_class"].astype(int).to_numpy()
    y_test = test["risk_class"].astype(int).to_numpy()

    rf_train = model.predict_proba(train[features])[:, 1]
    rf_test = model.predict_proba(test[features])[:, 1]
    fis_train = fis_scores(train, rf_train)
    fis_test = fis_scores(test, rf_test)

    rf_threshold = youden_threshold(y_train, rf_train)
    fis_threshold = youden_threshold(y_train, fis_train)

    rf_metrics = {
        "scheme": scheme,
        "fold": fold_name,
        "route": "RF_only",
        **compute_metrics(y_test, rf_test, rf_threshold),
    }
    fis_metrics = {
        "scheme": scheme,
        "fold": fold_name,
        "route": "RF_plus_IMU_FIS",
        **compute_metrics(y_test, fis_test, fis_threshold),
    }
    predictions = _prediction_frame(
        test,
        fold_name,
        scheme,
        rf_test,
        fis_test,
        rf_threshold,
        fis_threshold,
    )
    return [rf_metrics, fis_metrics], predictions


def load_dataset(path: Path, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Load and clean an analysis set; also report which FIS auxiliary columns
    are absent so the report can disclose the reduced set's missing cues."""
    df = pd.read_csv(path)
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        raise ValueError(f"{path} missing model features: {missing_features}")

    df = df[df["risk_class"].isin([0, 1])].copy()
    needed = features + ["risk_class", "participant_id", "window_centre_ms"]
    df = df.dropna(subset=needed)
    df["risk_class"] = df["risk_class"].astype(int)
    missing_aux = [col for col in AUXILIARY_FIS_COLUMNS if col not in df.columns]
    return df, missing_aux


def evaluate_within(df: pd.DataFrame, features: list[str], seed: int, scheme: str) -> FoldOutput:
    metrics = []
    predictions = []
    for pid, group in df.groupby("participant_id"):
        group = group.sort_values("window_centre_ms")
        cut = int(len(group) * 0.8)
        train = group.iloc[:cut]
        test = group.iloc[cut:]
        if train["risk_class"].nunique() < 2 or test["risk_class"].nunique() < 2:
            metrics.append({
                "scheme": scheme,
                "fold": str(pid),
                "route": "SKIPPED",
                "note": "degenerate class distribution",
                "n": int(len(test)),
            })
            continue
        fold_metrics, fold_predictions = evaluate_fold(
            train, test, features, seed, str(pid), scheme
        )
        metrics.extend(fold_metrics)
        predictions.append(fold_predictions)
    return FoldOutput(metrics=metrics, predictions=predictions)


def evaluate_loso(df: pd.DataFrame, features: list[str], seed: int, scheme: str) -> FoldOutput:
    metrics = []
    predictions = []
    for pid in sorted(df["participant_id"].unique()):
        train = df[df["participant_id"] != pid]
        test = df[df["participant_id"] == pid]
        if train["risk_class"].nunique() < 2 or test["risk_class"].nunique() < 2:
            metrics.append({
                "scheme": scheme,
                "fold": str(pid),
                "route": "SKIPPED",
                "note": "degenerate class distribution",
                "n": int(len(test)),
            })
            continue
        fold_metrics, fold_predictions = evaluate_fold(
            train, test, features, seed, str(pid), scheme
        )
        metrics.extend(fold_metrics)
        predictions.append(fold_predictions)
    return FoldOutput(metrics=metrics, predictions=predictions)


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    valid = metrics[metrics["route"].isin(["RF_only", "RF_plus_IMU_FIS"])].copy()
    return (
        valid.groupby(["scheme", "cv", "route"])
        .agg(
            folds=("fold", "nunique"),
            mean_auc=("auc", "mean"),
            std_auc=("auc", "std"),
            mean_sens=("sens", "mean"),
            mean_spec=("spec", "mean"),
            mean_f1=("f1", "mean"),
            total_test_windows=("n", "sum"),
        )
        .reset_index()
    )


def write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    all_metrics: pd.DataFrame,
    dataset_notes: dict,
) -> Path:
    path = out_dir / "RF_VS_FIS_ABLATION_REPORT.md"

    def fmt(x: float) -> str:
        return "nan" if pd.isna(x) else f"{x:.3f}"

    lines = [
        "# RF-only vs RF + IMU FIS ablation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## What was tested",
        "",
        "Two decision routes were compared on the frozen fallback analysis datasets:",
        "",
        "- RF-only: IMU features -> Random Forest risk probability.",
        "- RF + IMU FIS: the same RF probability (`R_IMU`) plus available IMU auxiliary features -> `IMUFallbackFIS` risk score.",
        "",
        "For every fold, thresholds were selected on the training fold using Youden's J and then applied to the held-out fold. AUC was computed on held-out scores.",
        "",
        "## Dataset notes",
        "",
    ]
    for name, note in dataset_notes.items():
        missing = ", ".join(note["missing_auxiliary_columns"]) or "none"
        lines.extend([
            f"### {name}",
            "",
            f"- Rows after label/NaN cleaning: {note['rows']}",
            f"- Participants: {', '.join(note['participants'])}",
            f"- Missing FIS auxiliary columns: {missing}",
            "",
        ])

    lines.extend(["## Aggregate results", ""])
    lines.append("| Dataset | CV | Route | Folds | AUC mean +/- SD | Sens | Spec | F1 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['scheme']} | {row['cv']} | {row['route']} | {int(row['folds'])} | "
            f"{fmt(row['mean_auc'])} +/- {fmt(row['std_auc'])} | "
            f"{fmt(row['mean_sens'])} | {fmt(row['mean_spec'])} | {fmt(row['mean_f1'])} |"
        )

    lines.extend(["", "## Fold-level AUCs", ""])
    for (scheme, cv), group in all_metrics.groupby(["scheme", "cv"]):
        pivot = group.pivot_table(index="fold", columns="route", values="auc", aggfunc="first")
        if {"RF_only", "RF_plus_IMU_FIS"}.issubset(pivot.columns):
            pivot["delta_fis_minus_rf"] = pivot["RF_plus_IMU_FIS"] - pivot["RF_only"]
        lines.extend([f"### {scheme} / {cv}", ""])
        lines.append("| Fold | RF-only AUC | RF+FIS AUC | Delta |")
        lines.append("|---|---:|---:|---:|")
        for fold, row in pivot.iterrows():
            lines.append(
                f"| {fold} | {fmt(row.get('RF_only'))} | "
                f"{fmt(row.get('RF_plus_IMU_FIS'))} | {fmt(row.get('delta_fis_minus_rf'))} |"
            )
        lines.append("")

    lines.extend([
        "## Critical interpretation",
        "",
        "- The FIS should only be claimed as useful if it preserves RF discrimination or improves the operational trade-off. It is not a trained classifier.",
        "- If AUC drops materially, the thesis should present FIS as an interpretable decision layer rather than as a performance-improving layer.",
        "- The reduced Pelvis-L3 dataset does not contain `imu_time_in_risk_zone` or `imu_z_flex`, so the current reduced FIS cannot use those cues. This is a design issue to disclose or fix.",
        "- The primary 4-IMU dataset contains richer FIS auxiliary inputs, but only covers the cleaned five-participant set.",
        "",
        "## Files produced",
        "",
        "- `rf_vs_fis_metrics.csv`: fold-level metrics.",
        "- `rf_vs_fis_summary.csv`: aggregate metrics.",
        "- `rf_vs_fis_predictions.csv`: held-out predictions and scores.",
        "- `rf_vs_fis_summary.json`: machine-readable summary and dataset notes.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare RF-only and RF+IMU-FIS fallback routes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--analysis_dir", type=Path, default=Path("results/fallback_analysis_sets"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/fallback_analysis_sets/rf_vs_fis_ablation"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "primary_4imu_cleaned": (
            args.analysis_dir / "primary_4imu_cleaned_features.csv",
            PRIMARY_FEATURES,
        ),
        "reduced_pelvis_l3": (
            args.analysis_dir / "reduced_pelvis_l3_features.csv",
            REDUCED_FEATURES,
        ),
    }

    all_metrics = []
    all_predictions = []
    dataset_notes = {}

    for scheme, (path, features) in datasets.items():
        df, missing_aux = load_dataset(path, features)
        dataset_notes[scheme] = {
            "path": str(path),
            "rows": int(len(df)),
            "participants": sorted(df["participant_id"].unique().tolist()),
            "missing_auxiliary_columns": missing_aux,
            "features": features,
        }

        within = evaluate_within(df, features, args.seed, scheme)
        loso = evaluate_loso(df, features, args.seed, scheme)
        for item, cv_name in [(within, "within_temporal_80_20"), (loso, "loso")]:
            metric_df = pd.DataFrame(item.metrics)
            metric_df.insert(1, "cv", cv_name)
            all_metrics.append(metric_df)
            if item.predictions:
                pred_df = pd.concat(item.predictions, ignore_index=True)
                pred_df.insert(1, "cv", cv_name)
                all_predictions.append(pred_df)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    summary_df = summarise(metrics_df)

    metrics_path = args.out_dir / "rf_vs_fis_metrics.csv"
    predictions_path = args.out_dir / "rf_vs_fis_predictions.csv"
    summary_path = args.out_dir / "rf_vs_fis_summary.csv"
    json_path = args.out_dir / "rf_vs_fis_summary.json"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_notes": dataset_notes,
                "summary": summary_df.to_dict(orient="records"),
                "outputs": {
                    "metrics_csv": str(metrics_path),
                    "summary_csv": str(summary_path),
                    "predictions_csv": str(predictions_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path = write_report(args.out_dir, summary_df, metrics_df, dataset_notes)

    print(summary_df.to_string(index=False))
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
