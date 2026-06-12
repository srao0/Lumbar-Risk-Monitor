#!/usr/bin/env python3
"""Apply rest-anchor zero correction to a processed IMU session.

This is intended for recovery/quality-improvement passes where the protocol
contains repeated still-rest subwindows.  It estimates the local standing
offset during formal baselines and standing rests, interpolates that offset
through the session, and subtracts it from derived relative-angle columns.
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
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.acquisition.session_timer import PROTOCOL  # noqa: E402


ANGLE_COLUMNS = [
    "theta_PL_pitch", "theta_PL_roll", "theta_PL_yaw",
    "theta_LT_pitch", "theta_LT_roll", "theta_LT_yaw",
    "theta_TU_pitch", "theta_TU_roll", "theta_TU_yaw",
]

SEATED_REST_LABELS = {"SIT_TO_STAND_NORMAL", "SIT_TO_STAND_FAST"}


def _protocol_by_label() -> dict[str, dict]:
    return {block["label"]: block for block in PROTOCOL}


def _rest_seconds(block: dict) -> float:
    # Length of a protocol block's trailing rest phase — 0 unless the final phase
    # is a REST / STAND STILL hold, since only those give a usable still anchor.
    if not block["phases"]:
        return 0.0
    phase_name, duration_s = block["phases"][-1]
    if "REST" not in phase_name.upper() and "STAND STILL" not in phase_name.upper():
        return 0.0
    return float(duration_s)


def discover_anchor_windows(labels: pd.DataFrame, min_rest_s: float = 5.0) -> list[dict]:
    """Return candidate still/rest windows from labels.csv."""
    by_label = _protocol_by_label()
    anchors: list[dict] = []

    for idx, row in labels.sort_values("start_ms").reset_index(drop=True).iterrows():
        label = str(row["label"])
        start = float(row["start_ms"])
        end = float(row["end_ms"])

        if label == "BASELINE_STATIC":
            duration = end - start
            if duration >= min_rest_s * 1000:
                # Skip first 5 s of long baselines for settling/filter convergence.
                skip = min(5000.0, duration * 0.2)
                anchors.append({
                    "label": label,
                    "rep": int(row.get("rep", 0)),
                    "start_ms": start + skip,
                    "end_ms": end,
                    "kind": "baseline",
                })
            continue

        if label in SEATED_REST_LABELS:
            continue

        block = by_label.get(label)
        if block is None:
            continue
        rest_s = _rest_seconds(block)
        if rest_s < min_rest_s:
            continue

        rest_start = max(start, end - rest_s * 1000.0)
        # Avoid transition into rest and use the quiet tail of the rest period.
        anchor_start = rest_start + min(2000.0, rest_s * 250.0)
        anchor_end = end - 500.0
        if anchor_end - anchor_start >= min_rest_s * 500.0:
            anchors.append({
                "label": label,
                "rep": int(row.get("rep", 0)),
                "start_ms": anchor_start,
                "end_ms": anchor_end,
                "kind": "standing_rest",
            })

    return anchors


def measure_anchors(
    imu: pd.DataFrame,
    anchors: list[dict],
    max_std_deg: float,
    angle_cols: list[str],
) -> tuple[pd.DataFrame, list[dict]]:
    """Measure the mean standing angle in each candidate window and keep the steady ones.

    A window is accepted only if every angle column's std stays within max_std_deg
    (i.e. the subject really was still) and it holds enough samples; its per-column
    means become the zero-offset for that point in time. Raises if fewer than two
    survive, since interpolation needs at least two anchors to define a trend.
    """
    accepted: list[dict] = []
    rejected: list[dict] = []

    for anchor in anchors:
        mask = (
            (imu["timestamp_ms"] >= anchor["start_ms"]) &
            (imu["timestamp_ms"] <= anchor["end_ms"])
        )
        chunk = imu.loc[mask, angle_cols]
        if len(chunk) < 100:
            rejected.append({**anchor, "reason": "too_few_samples", "samples": int(len(chunk))})
            continue

        means = chunk.mean()
        stds = chunk.std()
        max_std = float(stds.max())
        record = {
            **anchor,
            "centre_ms": float((anchor["start_ms"] + anchor["end_ms"]) / 2.0),
            "samples": int(len(chunk)),
            "max_std_deg": max_std,
        }
        for col in angle_cols:
            record[f"{col}_mean"] = float(means[col])
            record[f"{col}_std"] = float(stds[col])

        if max_std <= max_std_deg:
            accepted.append(record)
        else:
            rejected.append({**record, "reason": "unstable"})

    if len(accepted) < 2:
        raise ValueError(
            f"Need at least 2 accepted rest anchors; found {len(accepted)}. "
            "Try a larger --max_std_deg or inspect labels."
        )

    return pd.DataFrame(accepted).sort_values("centre_ms"), rejected


def apply_rest_anchor_correction(
    imu: pd.DataFrame,
    anchors: pd.DataFrame,
    angle_cols: list[str],
) -> pd.DataFrame:
    """Subtract the piecewise-linear drift implied by the accepted anchors from each angle.

    Offsets are linearly interpolated between anchor centres (held flat before the
    first and after the last), so the standing rests get pulled back toward zero and
    everything in between is corrected for the same slow drift.
    """
    corrected = imu.copy()
    t = corrected["timestamp_ms"].to_numpy(dtype=float)
    anchor_t = anchors["centre_ms"].to_numpy(dtype=float)

    for col in angle_cols:
        offsets = anchors[f"{col}_mean"].to_numpy(dtype=float)
        interp = np.interp(t, anchor_t, offsets, left=offsets[0], right=offsets[-1])
        corrected[col] = corrected[col].to_numpy(dtype=float) - interp

    return corrected


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply rest-anchor correction to processed IMU angles.")
    parser.add_argument("--session_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--max_std_deg", type=float, default=6.0)
    parser.add_argument("--min_rest_s", type=float, default=5.0)
    args = parser.parse_args()

    src = args.session_dir
    out = args.out_dir
    imu_path = src / "imu_data.csv"
    labels_path = src / "labels.csv"
    if not imu_path.exists() or not labels_path.exists():
        raise FileNotFoundError("session_dir must contain imu_data.csv and labels.csv")

    imu = pd.read_csv(imu_path)
    labels = pd.read_csv(labels_path)
    angle_cols = [c for c in ANGLE_COLUMNS if c in imu.columns]
    if not angle_cols:
        raise ValueError("No correctable angle columns found in imu_data.csv")

    candidates = discover_anchor_windows(labels, min_rest_s=args.min_rest_s)
    accepted, rejected = measure_anchors(imu, candidates, args.max_std_deg, angle_cols)
    corrected = apply_rest_anchor_correction(imu, accepted, angle_cols)

    out.mkdir(parents=True, exist_ok=True)
    for name in ["emg_data.csv", "labels.csv"]:
        src_file = src / name
        if src_file.exists():
            shutil.copy2(src_file, out / name)
    corrected.to_csv(out / "imu_data.csv", index=False)

    meta = {}
    meta_path = src / "session_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["session_id"] = out.name
    meta["rest_anchor_correction"] = {
        "enabled": True,
        "source_session": str(src),
        "applied_at": datetime.now().isoformat(),
        "method": "piecewise_linear_zero_offset_from_protocol_standing_rests",
        "angle_columns": angle_cols,
        "candidate_anchor_count": len(candidates),
        "accepted_anchor_count": int(len(accepted)),
        "rejected_anchor_count": int(len(rejected)),
        "max_anchor_std_deg": args.max_std_deg,
        "min_rest_s": args.min_rest_s,
        "note": (
            "Offsets are estimated from formal baselines and protocol standing "
            "rest tails, then linearly interpolated and subtracted from derived "
            "relative-angle columns. Labels are unchanged."
        ),
    }
    (out / "session_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    accepted.to_csv(out / "rest_anchor_report.csv", index=False)
    pd.DataFrame(rejected).to_csv(out / "rest_anchor_rejected.csv", index=False)

    print(f"Saved rest-anchor corrected session -> {out}")
    print(f"Candidate anchors: {len(candidates)}")
    print(f"Accepted anchors : {len(accepted)}")
    print(f"Rejected anchors : {len(rejected)}")


if __name__ == "__main__":
    main()
