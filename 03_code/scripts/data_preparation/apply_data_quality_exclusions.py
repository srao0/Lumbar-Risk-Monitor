#!/usr/bin/env python3
"""
Apply data-quality exclusions to generated feature tables.

Does not edit raw recordings, labels.csv, imu_data.csv, or emg_data.csv.
It only filters feature_matrix.csv files and combined_features.csv so known-bad
windows are not used for model training/evaluation.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _normalise_session_id(value: str) -> str:
    value = str(value)
    if "__" in value:
        return value.rsplit("__", 1)[-1]
    return value


def _backup(path: Path, tag: str) -> Path:
    backup_path = path.with_name(f"{path.stem}.{tag}{path.suffix}")
    if not backup_path.exists():
        backup_path.write_bytes(path.read_bytes())
    return backup_path


def _drop_for_rules(
    df: pd.DataFrame,
    rules: pd.DataFrame,
    participant_id: str | None,
    session_id: str | None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Drop windows matching the exclusion rules; return the kept rows and a per-rule report.

    Two rule kinds — an "exclude_from_main_training" action drops every window for
    that participant/session, otherwise only windows whose centre falls in
    [start_ms, end_ms] (optionally narrowed to a single movement_label) are dropped.
    Pass participant_id/session_id when filtering a single per-session feature_matrix;
    pass None/None for the pooled combined_features.csv so the participant/session
    masks come from the rows themselves.
    """
    if "window_centre_ms" not in df.columns:
        raise ValueError("Feature table is missing window_centre_ms")

    keep = pd.Series(True, index=df.index)
    report = []
    for _, rule in rules.iterrows():
        rule_pid = str(rule["participant_id"])
        rule_sid = str(rule["session_id"])
        action = str(rule["recommended_action"])

        if participant_id is not None and rule_pid != participant_id:
            continue
        if session_id is not None and rule_sid != session_id:
            continue

        if participant_id is None:
            pid_mask = df["participant_id"].astype(str) == rule_pid
        else:
            pid_mask = pd.Series(True, index=df.index)

        if session_id is None:
            sid_mask = df["session_id"].astype(str).map(_normalise_session_id) == rule_sid
        else:
            sid_mask = pd.Series(True, index=df.index)

        if action == "exclude_from_main_training":
            rule_mask = pid_mask & sid_mask
        else:
            start_ms = float(rule["start_ms"])
            end_ms = float(rule["end_ms"])
            rule_mask = (
                pid_mask
                & sid_mask
                & (df["window_centre_ms"].astype(float) >= start_ms)
                & (df["window_centre_ms"].astype(float) <= end_ms)
            )

            label = str(rule.get("label", "ALL"))
            if label and label != "ALL" and "movement_label" in df.columns:
                rule_mask = rule_mask & (df["movement_label"].astype(str) == label)

        to_drop = keep & rule_mask
        n_drop = int(to_drop.sum())
        keep.loc[to_drop] = False
        report.append({
            "participant_id": rule_pid,
            "session_id": rule_sid,
            "label": str(rule.get("label", "ALL")),
            "start_ms": float(rule["start_ms"]),
            "end_ms": float(rule["end_ms"]),
            "recommended_action": action,
            "reason": str(rule["reason"]),
            "dropped_windows": n_drop,
        })

    return df.loc[keep].copy(), report


def apply_exclusions(data_dir: Path, exclusions_csv: Path, backup: bool) -> dict:
    """Filter every feature_matrix.csv (and combined_features.csv) under data_dir in place.

    Overwrites each table with its filtered version; with backup=True the original is
    first copied to a timestamped sibling (e.g. feature_matrix.pre_quality_exclusions_*.csv).
    A run report is written to data_quality_exclusions_applied.json and returned.
    """
    rules = pd.read_csv(exclusions_csv)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_tag = f"pre_quality_exclusions_{timestamp}"

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "exclusions_csv": str(exclusions_csv),
        "files": [],
    }

    for feature_path in sorted(data_dir.glob("participant_*/session_*/feature_matrix.csv")):
        participant_id = feature_path.parents[1].name
        session_id = feature_path.parent.name
        df = pd.read_csv(feature_path)
        before = len(df)
        filtered, report = _drop_for_rules(df, rules, participant_id, session_id)
        after = len(filtered)
        backup_path = _backup(feature_path, backup_tag) if backup else None
        filtered.to_csv(feature_path, index=False)
        results["files"].append({
            "path": str(feature_path),
            "backup_path": str(backup_path) if backup_path else None,
            "rows_before": before,
            "rows_after": after,
            "rows_dropped": before - after,
            "rules": report,
        })

    combined_path = data_dir / "combined_features.csv"
    if combined_path.exists():
        df = pd.read_csv(combined_path)
        before = len(df)
        filtered, report = _drop_for_rules(df, rules, None, None)
        after = len(filtered)
        backup_path = _backup(combined_path, backup_tag) if backup else None
        filtered.to_csv(combined_path, index=False)
        results["files"].append({
            "path": str(combined_path),
            "backup_path": str(backup_path) if backup_path else None,
            "rows_before": before,
            "rows_after": after,
            "rows_dropped": before - after,
            "rules": report,
        })

    report_path = data_dir / "data_quality_exclusions_applied.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter known-bad feature windows from fallback feature tables.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/real/protocol_train_fallback_2session"))
    parser.add_argument("--exclusions", type=Path, default=None)
    parser.add_argument("--no_backup", action="store_true")
    args = parser.parse_args()

    exclusions = args.exclusions or (args.data_dir / "data_quality_exclusions.csv")
    result = apply_exclusions(args.data_dir, exclusions, backup=not args.no_backup)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
