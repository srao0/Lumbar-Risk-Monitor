#!/usr/bin/env python3
"""Check whether the final BL2 static baseline is actually still."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ANGLE_COLS = [
    "theta_PL_pitch",
    "theta_PL_roll",
    "theta_LT_pitch",
    "theta_LT_roll",
    "theta_TU_pitch",
    "theta_TU_roll",
]


def check_bl2(session_dir: Path, max_std_deg: float) -> dict:
    """Gauge whether the final standing baseline (BL2) was genuinely still.

    BL2 is taken as the last BASELINE_STATIC block in labels.csv (the post-fatigue one,
    which doubles as the drift re-zero reference downstream). The check passes only if
    every angle column's std over that window stays within max_std_deg — a noisy BL2
    means the re-zero offset it provides is untrustworthy. Read-only; nothing is written.
    """
    imu_path = session_dir / "imu_data.csv"
    labels_path = session_dir / "labels.csv"
    if not imu_path.exists():
        raise FileNotFoundError(f"Missing {imu_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing {labels_path}")

    labels = pd.read_csv(labels_path)
    baseline = labels[labels["label"].astype(str) == "BASELINE_STATIC"].copy()
    if baseline.empty:
        raise ValueError("No BASELINE_STATIC rows found in labels.csv")

    bl2 = baseline.sort_values("end_ms").iloc[-1]
    imu = pd.read_csv(imu_path, usecols=["timestamp_ms"] + ANGLE_COLS)
    mask = (imu["timestamp_ms"] >= float(bl2["start_ms"])) & (imu["timestamp_ms"] <= float(bl2["end_ms"]))
    window = imu.loc[mask]
    if window.empty:
        raise ValueError("BL2 label window does not overlap imu_data.csv")

    stats = {}
    failed = []
    for col in ANGLE_COLS:
        std = float(window[col].std())
        mean = float(window[col].mean())
        stats[col] = {"mean_deg": mean, "std_deg": std}
        if std > max_std_deg:
            failed.append(col)

    result = {
        "session_dir": str(session_dir),
        "bl2_start_ms": float(bl2["start_ms"]),
        "bl2_end_ms": float(bl2["end_ms"]),
        "samples": int(len(window)),
        "max_allowed_std_deg": float(max_std_deg),
        "passed": not failed,
        "failed_columns": failed,
        "stats": stats,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Check BL2 static quality for a converted session.")
    parser.add_argument("--session_dir", type=Path, required=True)
    parser.add_argument("--max_std_deg", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = check_bl2(args.session_dir, args.max_std_deg)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
