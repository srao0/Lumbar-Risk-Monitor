#!/usr/bin/env python3
"""
Phase II.B personalised calibration experiment.

Compares three IMU-only RF models for each participant:
generic_other_participants, personal_calibration_only, and personal_augmented.
It uses only feature-level augmentation derived from each participant's
calibration windows and evaluates on held-out future repetitions.
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
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.training.train_classifier import IMU_FEATURES


TARGET_COL = "risk_class_protocol"
UNLABELLED = -1
CALIBRATION = "calibration"
FUTURE_TEST = "future_test"
SYNTHETIC = "synthetic_personalised"
REAL = "real"
MODEL_VARIANTS = (
    "generic_other_participants",
    "personal_calibration_only",
    "personal_augmented",
)


@dataclass(frozen=True)
class SplitSummary:
    participant_id: str
    calibration_windows: int
    future_test_windows: int
    calibration_safe: int
    calibration_risky: int
    future_safe: int
    future_risky: int
    synthetic_windows: int


def build_rf(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("sc", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=None,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


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


def load_features(data_dir: Path) -> pd.DataFrame:
    feature_path = data_dir / "combined_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing combined feature table: {feature_path}")

    df = pd.read_csv(feature_path)
    required = {
        "participant_id",
        "session_id",
        "window_centre_ms",
        "movement_label",
        TARGET_COL,
        *IMU_FEATURES,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"combined_features.csv is missing required columns: {missing}")
    return df


def load_labels_for_participant(data_dir: Path, participant_id: str) -> pd.DataFrame:
    participant_dir = data_dir / participant_id
    session_dirs = sorted(path for path in participant_dir.glob("session_*") if path.is_dir())
    if len(session_dirs) != 1:
        raise ValueError(
            f"Expected exactly one session folder for {participant_id}; found {len(session_dirs)}"
        )
    labels_path = session_dirs[0] / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels.csv for {participant_id}: {labels_path}")
    labels = pd.read_csv(labels_path)
    required = {"label", "rep", "start_ms", "end_ms", "risk_class"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"{labels_path} is missing required columns: {missing}")
    labels = labels.copy()
    labels["participant_id"] = participant_id
    labels["rep"] = labels["rep"].astype(int)
    labels["risk_class"] = labels["risk_class"].astype(int)
    labels["start_ms"] = labels["start_ms"].astype(float)
    labels["end_ms"] = labels["end_ms"].astype(float)
    return labels


def split_label_repetitions(labels: pd.DataFrame) -> dict[tuple[str, int], str]:
    """Assign each (movement, rep) to calibration or future_test.

    Splitting is chronological per movement (earlier reps calibrate, later reps
    test) so the personalised model is only ever evaluated on FUTURE repetitions
    it has not seen. BASELINE_STATIC is treated specially: its first segment
    calibrates and any later segments test.
    """
    split_map: dict[tuple[str, int], str] = {}
    labelled = labels[labels["risk_class"].isin([0, 1])].copy()

    for label, group in labelled.groupby("label", sort=False):
        ordered = group.sort_values(["start_ms", "end_ms", "rep"]).reset_index(drop=True)
        if label == "BASELINE_STATIC":
            if len(ordered) < 2:
                raise ValueError("BASELINE_STATIC needs two segments for calibration/test split")
            split_map[(label, int(ordered.iloc[0]["rep"]))] = CALIBRATION
            for _, row in ordered.iloc[1:].iterrows():
                split_map[(label, int(row["rep"]))] = FUTURE_TEST
            continue

        reps = ordered["rep"].astype(int).tolist()
        cut = int(np.ceil(len(reps) / 2.0))
        for rep in reps[:cut]:
            split_map[(label, rep)] = CALIBRATION
        for rep in reps[cut:]:
            split_map[(label, rep)] = FUTURE_TEST
    return split_map


def annotate_participant_windows(
    participant_df: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Tag each feature window with the labelled repetition it falls inside and
    its calibration/future_test role; windows matching no labelled rep are
    dropped."""
    split_map = split_label_repetitions(labels)
    rows = []
    usable_labels = labels[labels["risk_class"].isin([0, 1])].copy()

    for row in participant_df.itertuples(index=False):
        centre = float(row.window_centre_ms)
        matches = usable_labels[
            (usable_labels["start_ms"] <= centre) & (centre < usable_labels["end_ms"])
        ]
        if matches.empty:
            split_role = "excluded"
            rep = np.nan
            label_risk = UNLABELLED
        else:
            match = matches.iloc[0]
            rep = int(match["rep"])
            label_risk = int(match["risk_class"])
            split_role = split_map.get((str(match["label"]), rep), "excluded")

        row_dict = row._asdict()
        row_dict["protocol_rep"] = rep
        row_dict["split_role"] = split_role
        row_dict["source_type"] = REAL
        row_dict["label_risk_from_segment"] = label_risk
        rows.append(row_dict)

    annotated = pd.DataFrame(rows)
    annotated = annotated[annotated[TARGET_COL].isin([0, 1])].copy()
    annotated = annotated[annotated["split_role"].isin([CALIBRATION, FUTURE_TEST])].copy()
    annotated[TARGET_COL] = annotated[TARGET_COL].astype(int)
    return annotated


def prepare_split(df: pd.DataFrame, data_dir: Path) -> tuple[pd.DataFrame, list[SplitSummary]]:
    participants = sorted(df["participant_id"].dropna().unique().tolist())
    if len(participants) < 3:
        raise ValueError(f"Expected at least 3 participants; found {participants}")

    annotated_parts = []
    summaries: list[SplitSummary] = []
    for participant_id in participants:
        participant_df = df[df["participant_id"] == participant_id].copy()
        labels = load_labels_for_participant(data_dir, participant_id)
        annotated = annotate_participant_windows(participant_df, labels)

        cal = annotated[annotated["split_role"] == CALIBRATION]
        test = annotated[annotated["split_role"] == FUTURE_TEST]
        cal_counts = cal[TARGET_COL].value_counts().to_dict()
        test_counts = test[TARGET_COL].value_counts().to_dict()
        if set(cal_counts) != {0, 1}:
            raise ValueError(f"{participant_id} calibration split lacks both classes: {cal_counts}")
        if set(test_counts) != {0, 1}:
            raise ValueError(f"{participant_id} future-test split lacks both classes: {test_counts}")

        summaries.append(
            SplitSummary(
                participant_id=participant_id,
                calibration_windows=int(len(cal)),
                future_test_windows=int(len(test)),
                calibration_safe=int(cal_counts.get(0, 0)),
                calibration_risky=int(cal_counts.get(1, 0)),
                future_safe=int(test_counts.get(0, 0)),
                future_risky=int(test_counts.get(1, 0)),
                synthetic_windows=0,
            )
        )
        annotated_parts.append(annotated)

    return pd.concat(annotated_parts, ignore_index=True), summaries


def augment_calibration(
    calibration_df: pd.DataFrame,
    feature_cols: list[str],
    augment_factor: int,
    noise_scale: float,
    seed: int,
) -> pd.DataFrame:
    """Feature-level jitter augmentation of a participant's calibration windows.

    Replicates each calibration window `augment_factor` times with Gaussian
    noise scaled to the per-(movement, class) feature spread, clipped to that
    group's observed range plus a 10% margin. This is deliberately feature-only
    (no synthetic raw signal) so the personalised model gains denser calibration
    coverage without fabricating implausible movements.
    """
    if augment_factor <= 0:
        return calibration_df.iloc[0:0].copy()

    rng = np.random.default_rng(seed)
    synthetic_parts = []
    group_cols = ["participant_id", "movement_label", TARGET_COL]
    for _, group in calibration_df.groupby(group_cols, sort=False):
        values = group[feature_cols].astype(float)
        std = values.std(ddof=0).replace(0.0, np.nan)
        fallback = values.abs().mean().replace(0.0, np.nan) * 0.02
        std = std.fillna(fallback).fillna(1e-6)
        lower = values.min()
        upper = values.max()
        margin = (upper - lower).abs() * 0.10
        lower = lower - margin
        upper = upper + margin

        replicated = pd.concat([group.copy() for _ in range(augment_factor)], ignore_index=True)
        noise = rng.normal(
            loc=0.0,
            scale=(noise_scale * std.to_numpy()),
            size=(len(replicated), len(feature_cols)),
        )
        augmented_values = replicated[feature_cols].astype(float).to_numpy() + noise
        augmented_values = np.clip(augmented_values, lower.to_numpy(), upper.to_numpy())
        replicated.loc[:, feature_cols] = augmented_values
        replicated["source_type"] = SYNTHETIC
        replicated["split_role"] = CALIBRATION
        synthetic_parts.append(replicated)

    if not synthetic_parts:
        return calibration_df.iloc[0:0].copy()
    synthetic = pd.concat(synthetic_parts, ignore_index=True)
    synthetic["synthetic_parent"] = "calibration_group_jitter"
    return synthetic


def check_no_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame, participant_id: str, variant: str) -> None:
    """Hard guard against the two ways this experiment could cheat: the generic
    model must contain no rows from the held-out participant, and no variant may
    reuse a future-test window (keyed on participant/session/window) in training.
    Raises rather than silently producing inflated AUCs."""
    if variant == "generic_other_participants":
        overlap_participants = set(train_df["participant_id"].unique()) & {participant_id}
        if overlap_participants:
            raise ValueError(f"Generic model for {participant_id} includes held-out participant rows")

    test_keys = set(
        zip(
            test_df["participant_id"],
            test_df["session_id"],
            test_df["window_centre_ms"],
        )
    )
    train_keys = set(
        zip(
            train_df["participant_id"],
            train_df["session_id"],
            train_df["window_centre_ms"],
        )
    )
    leaked = test_keys & train_keys
    if leaked:
        raise ValueError(f"{variant} for {participant_id} leaks {len(leaked)} future-test windows")


def fit_and_evaluate(
    participant_id: str,
    variant: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    model_dir: Path,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    check_no_leakage(train_df, test_df, participant_id, variant)
    y_train = train_df[TARGET_COL].astype(int).to_numpy()
    y_test = test_df[TARGET_COL].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        raise ValueError(f"{variant} for {participant_id} has insufficient training classes")
    if len(np.unique(y_test)) < 2:
        raise ValueError(f"{variant} for {participant_id} has insufficient test classes")

    model = build_rf(seed)
    model.fit(train_df[feature_cols].fillna(0.0).to_numpy(), y_train)
    y_prob = model.predict_proba(test_df[feature_cols].fillna(0.0).to_numpy())[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    metrics.update(
        {
            "participant_id": participant_id,
            "model_variant": variant,
            "train_windows": int(len(train_df)),
            "train_real_windows": int((train_df["source_type"] == REAL).sum()),
            "train_synthetic_windows": int((train_df["source_type"] == SYNTHETIC).sum()),
            "test_windows": int(len(test_df)),
            "threshold": 0.5,
        }
    )

    model_path = model_dir / f"{participant_id}_{variant}.joblib"
    dump(model, model_path)
    metrics["model_path"] = str(model_path)

    predictions = test_df[
        [
            "participant_id",
            "session_id",
            "window_centre_ms",
            "movement_label",
            "protocol_rep",
            TARGET_COL,
        ]
    ].copy()
    predictions["model_variant"] = variant
    predictions["risk_probability"] = y_prob
    predictions["risk_prediction"] = y_pred
    predictions["threshold"] = 0.5
    predictions["evaluation_role"] = FUTURE_TEST
    return metrics, predictions


def summarise_results(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["auc", "accuracy", "sensitivity", "specificity", "precision_risk", "f1_risk"]
    summary = metrics_df.groupby("model_variant", as_index=False)[metric_cols].agg(["mean", "std"])
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col for col in summary.columns
    ]
    return summary.reset_index(drop=True)


def add_delta_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Per-participant metric deltas of the augmented personalised model against
    each baseline (generic and calibration-only) -- the RQ4 personalisation
    gain numbers."""
    rows = []
    metric_cols = ["auc", "accuracy", "sensitivity", "specificity", "precision_risk", "f1_risk"]
    for participant_id, group in metrics_df.groupby("participant_id"):
        by_variant = group.set_index("model_variant")
        augmented = by_variant.loc["personal_augmented"]
        for baseline in ("generic_other_participants", "personal_calibration_only"):
            base = by_variant.loc[baseline]
            row = {
                "participant_id": participant_id,
                "comparison": f"personal_augmented_minus_{baseline}",
            }
            for metric in metric_cols:
                row[f"delta_{metric}"] = float(augmented[metric] - base[metric])
            rows.append(row)
    return pd.DataFrame(rows)


def write_interpretation(metrics_df: pd.DataFrame, delta_df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Phase II.B Personalised Calibration Interpretation",
        "",
        "This is a pilot subject-specific feature augmentation experiment. It evaluates",
        "future repetitions from the same participant and should not be interpreted as",
        "population-level or longitudinal generalisation evidence.",
        "",
    ]

    for participant_id in sorted(metrics_df["participant_id"].unique()):
        participant_delta = delta_df[
            (delta_df["participant_id"] == participant_id)
            & (delta_df["comparison"] == "personal_augmented_minus_generic_other_participants")
        ].iloc[0]
        delta_auc = participant_delta["delta_auc"]
        if delta_auc > 0.02:
            verdict = "improved"
        elif delta_auc < -0.02:
            verdict = "worsened"
        else:
            verdict = "did not materially change"
        lines.append(
            f"- {participant_id}: personalised augmentation {verdict} AUC versus "
            f"the generic other-participants model (delta AUC={delta_auc:.3f})."
        )

    mean_delta = delta_df[
        delta_df["comparison"] == "personal_augmented_minus_generic_other_participants"
    ]["delta_auc"].mean()
    lines.extend(
        [
            "",
            f"Mean delta AUC versus generic baseline: {mean_delta:.3f}.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    model_dir = Path(args.models_dir)
    feature_cols = [col for col in IMU_FEATURES]

    df = load_features(data_dir)
    missing_features = sorted(set(feature_cols) - set(df.columns))
    if missing_features:
        raise ValueError(f"Missing IMU feature columns: {missing_features}")

    annotated, split_summaries = prepare_split(df, data_dir)
    all_cal = annotated[annotated["split_role"] == CALIBRATION].copy()
    all_test = annotated[annotated["split_role"] == FUTURE_TEST].copy()

    synthetic = augment_calibration(
        all_cal,
        feature_cols=feature_cols,
        augment_factor=args.augment_factor,
        noise_scale=args.noise_scale,
        seed=args.seed,
    )
    split_summary_df = pd.DataFrame([summary.__dict__ for summary in split_summaries])
    if not synthetic.empty:
        synth_counts = synthetic.groupby("participant_id").size().to_dict()
        split_summary_df["synthetic_windows"] = split_summary_df["participant_id"].map(synth_counts).fillna(0).astype(int)

    print("\nPhase II.B personalised calibration split summary")
    print(split_summary_df.to_string(index=False))
    print(f"\nIMU features: {len(feature_cols)}")
    print(f"Augment factor: {args.augment_factor}; noise_scale: {args.noise_scale}")
    print(f"Planned outputs: {results_dir} and {model_dir}")

    participants = sorted(annotated["participant_id"].unique().tolist())
    if args.dry_run:
        print("\nDry run complete; no files written.")
        return 0

    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    prediction_parts = []
    for participant_id in participants:
        test_df = all_test[all_test["participant_id"] == participant_id].copy()
        cal_df = all_cal[all_cal["participant_id"] == participant_id].copy()
        synth_df = synthetic[synthetic["participant_id"] == participant_id].copy()
        other_df = annotated[
            (annotated["participant_id"] != participant_id)
            & (annotated["split_role"].isin([CALIBRATION, FUTURE_TEST]))
        ].copy()

        train_sets = {
            "generic_other_participants": other_df,
            "personal_calibration_only": cal_df,
            "personal_augmented": pd.concat([cal_df, synth_df], ignore_index=True),
        }

        for variant in MODEL_VARIANTS:
            metrics, predictions = fit_and_evaluate(
                participant_id=participant_id,
                variant=variant,
                train_df=train_sets[variant],
                test_df=test_df,
                feature_cols=feature_cols,
                model_dir=model_dir,
                seed=args.seed,
            )
            metrics_rows.append(metrics)
            prediction_parts.append(predictions)

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(prediction_parts, ignore_index=True)
    summary_df = summarise_results(metrics_df)
    delta_df = add_delta_rows(metrics_df)

    metrics_df.to_csv(results_dir / "per_participant_metrics.csv", index=False)
    predictions_df.to_csv(results_dir / "per_window_predictions.csv", index=False)
    summary_df.to_csv(results_dir / "summary_metrics.csv", index=False)
    delta_df.to_csv(results_dir / "delta_metrics.csv", index=False)
    split_summary_df.to_csv(results_dir / "split_summary.csv", index=False)

    manifest = {
        "experiment": "personalised_feature_augmentation_stage2b",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir.resolve()),
        "results_dir": str(results_dir.resolve()),
        "models_dir": str(model_dir.resolve()),
        "participants": participants,
        "model_variants": list(MODEL_VARIANTS),
        "feature_columns": feature_cols,
        "target_column": TARGET_COL,
        "augmentation": {
            "level": "feature",
            "augment_factor": args.augment_factor,
            "noise_scale": args.noise_scale,
            "grouping": ["participant_id", "movement_label", TARGET_COL],
            "clip_margin": "group min/max plus 10 percent range",
        },
        "split_policy": {
            "baseline_static": "first baseline segment calibration; later baseline segment(s) future_test",
            "other_labels": "earlier half of repetitions calibration; later half future_test",
            "excluded": "risk_class_protocol=-1 and windows not matched to labelled repetitions",
        },
        "interpretation_limit": "pilot within-session future-repetition evaluation; not population-level or longitudinal evidence",
    }
    (results_dir / "augmentation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    write_interpretation(metrics_df, delta_df, results_dir / "interpretation.md")

    print("\nPer-participant metrics")
    print(
        metrics_df[
            ["participant_id", "model_variant", "auc", "accuracy", "sensitivity", "specificity", "f1_risk"]
        ].to_string(index=False, float_format="{:.3f}".format)
    )
    print("\nSummary metrics")
    print(summary_df.to_string(index=False, float_format="{:.3f}".format))
    print(f"\nSaved results to {results_dir}")
    print(f"Saved models to {model_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase II.B personalised calibration experiment."
    )
    parser.add_argument("--data_dir", default="data/real/protocol_train_fallback")
    parser.add_argument("--results_dir", default="results/personalised_stage2b")
    parser.add_argument("--models_dir", default="ml/models/personalised_stage2b")
    parser.add_argument("--augment_factor", type=int, default=8)
    parser.add_argument("--noise_scale", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
