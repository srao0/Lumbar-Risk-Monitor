#!/usr/bin/env python3
"""
Extract the initial static baseline from an Arduino IMU recording.

The output is a raw-count CSV that can be passed to session_converter.py as
--still_cal. By default it takes the first 60 seconds, matching the formal
protocol's BASELINE_STATIC block.
"""

import argparse
from pathlib import Path

import pandas as pd


def extract_initial_still_cal(
    imu_csv: Path,
    out_csv: Path,
    seconds: float = 60.0,
    force: bool = False,
) -> Path:
    if seconds <= 0:
        raise ValueError("--seconds must be greater than zero")
    if not imu_csv.exists():
        raise FileNotFoundError(f"IMU CSV not found: {imu_csv}")
    if out_csv.exists() and not force:
        raise FileExistsError(f"Output exists: {out_csv}. Use --force to overwrite.")

    raw = pd.read_csv(imu_csv, comment="#", skipinitialspace=True)
    raw.columns = [str(c).strip() for c in raw.columns]
    if "t_ms" not in raw.columns:
        raise ValueError(f"Expected t_ms column in {imu_csv}")

    # Drop stray repeated header lines (the Arduino logger re-emits the header on reconnect).
    raw = raw[raw["t_ms"] != "t_ms"].copy()
    raw = raw.apply(pd.to_numeric, errors="coerce")
    raw = raw.dropna(how="all").reset_index(drop=True)
    if raw.empty:
        raise ValueError(f"No numeric IMU samples found in {imu_csv}")

    t0 = float(raw["t_ms"].iloc[0])
    t1 = t0 + seconds * 1000.0
    still = raw[(raw["t_ms"] >= t0) & (raw["t_ms"] <= t1)].copy()
    if still.empty:
        raise ValueError(f"No samples found in the first {seconds:g} seconds")

    # Refuse a short still segment (>10% under target) — usually a truncated or
    # mistimed recording, which would give a weak/biased calibration baseline.
    duration_s = (float(still["t_ms"].iloc[-1]) - float(still["t_ms"].iloc[0])) / 1000.0
    if duration_s < seconds * 0.9:
        raise ValueError(
            f"Extracted still segment is only {duration_s:.1f}s; expected about {seconds:g}s"
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    still.to_csv(out_csv, index=False)
    return out_csv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract first N seconds of raw Arduino IMU data as still_cal.csv."
    )
    parser.add_argument("--imu", required=True, help="Raw imu_arduino.csv path.")
    parser.add_argument("--out", required=True, help="Output still_cal.csv path.")
    parser.add_argument("--seconds", type=float, default=60.0, help="Seconds to extract. Default: 60.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output.")
    args = parser.parse_args()

    out = extract_initial_still_cal(
        Path(args.imu),
        Path(args.out),
        seconds=args.seconds,
        force=args.force,
    )
    print(f"[OK] Wrote initial still calibration CSV: {out}")
    print(f"     Duration source: first {args.seconds:g} s of {args.imu}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
