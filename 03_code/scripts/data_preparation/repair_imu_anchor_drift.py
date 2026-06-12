#!/usr/bin/env python3
"""
Repair IMU pitch/roll drift using protocol rest anchors.

This is a post-processing repair for IMU-only fallback sessions where a single
BL1-to-BL2 linear correction is not enough. It keeps the raw recording intact,
backs up the existing imu_data.csv, then applies a piecewise-linear correction
to selected relative angle columns using windows where the protocol should be
upright/static.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ANGLE_COLS = [
    "theta_PL_pitch",
    "theta_PL_roll",
    "theta_LT_pitch",
    "theta_LT_roll",
    "theta_TU_pitch",
    "theta_TU_roll",
]

UPRIGHT_REST_LABELS = {
    "BASELINE_STATIC",
    "CLEAN_FLEXION",
    "LUMBAR_DOMINANT",
    "CLEAN_LATERAL_L",
    "CLEAN_LATERAL_R",
    "CLEAN_ROTATION_L",
    "CLEAN_ROTATION_R",
    "FAST_BEND",
    "SHOULDER_DRIVEN",
    "PICKUP_SYM",
    "PICKUP_ASYM",
}

SKIP_LABELS = {
    "SIT_TO_STAND_NORMAL",
    "SIT_TO_STAND_FAST",
    "FATIGUE_FLEXION",
}


def _build_anchor_windows(
    labels: pd.DataFrame,
    imu_end_ms: float,
    anchor_window_ms: float,
    bl2_start_ms: float | None,
    bl2_end_ms: float | None,
) -> list[dict]:
    """Pick the time windows where the subject should be upright/static — the drift anchors.

    Always seeds an anchor at the session start, then takes the trailing anchor_window_ms
    of each upright protocol block (skipping seated/fatigue blocks where the trunk is not
    reliably vertical), plus an optional explicit BL2 window. Returns the windows sorted
    and de-duplicated; the caller turns each into a zero-reference for the drift fit.
    """
    anchors = [{"name": "session_start", "start_ms": 0.0, "end_ms": min(anchor_window_ms, imu_end_ms)}]

    for _, row in labels.iterrows():
        label = str(row["label"])
        if label in SKIP_LABELS or label not in UPRIGHT_REST_LABELS:
            continue
        start_ms = float(row["start_ms"])
        end_ms = float(row["end_ms"])
        if end_ms <= 0 or end_ms > imu_end_ms:
            continue
        if label == "BASELINE_STATIC":
            win_start = max(start_ms, end_ms - anchor_window_ms)
        else:
            win_start = max(start_ms, end_ms - anchor_window_ms)
        anchors.append({
            "name": f"{label}_rep_{int(row['rep'])}",
            "start_ms": win_start,
            "end_ms": end_ms,
        })

    if bl2_start_ms is not None and bl2_end_ms is not None:
        anchors.append({"name": "BL2", "start_ms": float(bl2_start_ms), "end_ms": float(bl2_end_ms)})

    anchors = [a for a in anchors if a["end_ms"] > a["start_ms"]]
    anchors = sorted(anchors, key=lambda a: (a["start_ms"], a["end_ms"]))
    deduped = []
    seen = set()
    for anchor in anchors:
        key = (round(anchor["start_ms"], 1), round(anchor["end_ms"], 1), anchor["name"])
        if key not in seen:
            deduped.append(anchor)
            seen.add(key)
    return deduped


def repair_session(
    session_dir: Path,
    angle_cols: list[str],
    anchor_window_ms: float,
    bl2_start_ms: float | None,
    bl2_end_ms: float | None,
    backup: bool,
) -> dict:
    """Overwrite a session's imu_data.csv with anchor-corrected angle columns, in place.

    Treats each upright-anchor mean as the residual drift at that moment, interpolates
    it across the whole session, and subtracts it from the selected angle columns.
    With backup=True the original is renamed to imu_data.pre_anchor_repair_*.csv first;
    session_metadata.json is updated and the anchor table is dumped alongside. Needs at
    least three usable anchors so the piecewise correction has real curvature to follow.
    """
    imu_path = session_dir / "imu_data.csv"
    labels_path = session_dir / "labels.csv"
    metadata_path = session_dir / "session_metadata.json"
    if not imu_path.exists():
        raise FileNotFoundError(f"Missing {imu_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing {labels_path}")

    imu = pd.read_csv(imu_path)
    labels = pd.read_csv(labels_path)
    missing = [col for col in ["timestamp_ms"] + angle_cols if col not in imu.columns]
    if missing:
        raise ValueError(f"Cannot repair; missing IMU columns: {missing}")

    imu_end_ms = float(imu["timestamp_ms"].max())
    anchors = _build_anchor_windows(labels, imu_end_ms, anchor_window_ms, bl2_start_ms, bl2_end_ms)
    if len(anchors) < 3:
        raise ValueError(f"Need at least 3 anchors for piecewise repair; found {len(anchors)}")

    anchor_rows = []
    for anchor in anchors:
        mask = (imu["timestamp_ms"] >= anchor["start_ms"]) & (imu["timestamp_ms"] <= anchor["end_ms"])
        n = int(mask.sum())
        if n < 20:
            continue
        row = {
            "name": anchor["name"],
            "start_ms": anchor["start_ms"],
            "end_ms": anchor["end_ms"],
            "time_ms": float(imu.loc[mask, "timestamp_ms"].mean()),
            "n_samples": n,
        }
        for col in angle_cols:
            row[col] = float(imu.loc[mask, col].mean())
        anchor_rows.append(row)

    if len(anchor_rows) < 3:
        raise ValueError(f"Need at least 3 usable anchors; found {len(anchor_rows)}")

    anchor_df = pd.DataFrame(anchor_rows).sort_values("time_ms")
    repaired = imu.copy()
    t = repaired["timestamp_ms"].to_numpy(dtype=float)
    correction_summary = {}
    for col in angle_cols:
        anchor_t = anchor_df["time_ms"].to_numpy(dtype=float)
        residual = anchor_df[col].to_numpy(dtype=float)
        drift = np.interp(t, anchor_t, residual, left=residual[0], right=residual[-1])
        repaired[col] = repaired[col].to_numpy(dtype=float) - drift
        correction_summary[col] = {
            "max_abs_anchor_residual_deg": float(np.max(np.abs(residual))),
            "mean_abs_anchor_residual_deg": float(np.mean(np.abs(residual))),
        }

    if backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = session_dir / f"imu_data.pre_anchor_repair_{timestamp}.csv"
        imu_path.replace(backup_path)
    else:
        backup_path = None
    repaired.to_csv(imu_path, index=False)

    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["anchor_drift_repair"] = {
        "enabled": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": "piecewise_linear_zero_upright_rest_anchors",
        "anchor_window_ms": float(anchor_window_ms),
        "angle_columns": angle_cols,
        "n_anchors": int(len(anchor_df)),
        "backup_path": str(backup_path) if backup_path else None,
        "correction_summary": correction_summary,
        "anchors": anchor_df.to_dict(orient="records"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    anchor_csv = session_dir / "anchor_drift_repair_anchors.csv"
    anchor_df.to_csv(anchor_csv, index=False)

    return {
        "session_dir": str(session_dir),
        "imu_path": str(imu_path),
        "backup_path": str(backup_path) if backup_path else None,
        "anchor_csv": str(anchor_csv),
        "n_anchors": int(len(anchor_df)),
        "correction_summary": correction_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair IMU drift using protocol rest anchors.")
    parser.add_argument("--session_dir", required=True, type=Path)
    parser.add_argument("--anchor_window_ms", type=float, default=3000.0)
    parser.add_argument("--bl2_start_ms", type=float, default=None)
    parser.add_argument("--bl2_end_ms", type=float, default=None)
    parser.add_argument("--angle_cols", nargs="*", default=DEFAULT_ANGLE_COLS)
    parser.add_argument("--no_backup", action="store_true")
    args = parser.parse_args()

    result = repair_session(
        session_dir=args.session_dir,
        angle_cols=args.angle_cols,
        anchor_window_ms=args.anchor_window_ms,
        bl2_start_ms=args.bl2_start_ms,
        bl2_end_ms=args.bl2_end_ms,
        backup=not args.no_backup,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
