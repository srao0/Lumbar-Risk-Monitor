#!/usr/bin/env python3
"""
annotate_session.py
===================
Spinal Movement Risk Monitor -- FYP 2025/26

Post-hoc annotation tool. Use this when label_logger.py was not running
during a recording session and you need to assign labels to a session
directory after the fact.

The tool reads the processed imu_data.csv from a session directory,
shows movement summaries in the terminal (mean trunk flexion, peak angular
velocity, duration), and prompts you to label each detected segment.

Usage
-----
    # Interactive mode: step through auto-detected movement segments
    py scripts/acquisition/annotate_session.py --session data/real/processed/session_real_001

    # Pre-populate from an existing partial labels.csv and only fill gaps
    py scripts/acquisition/annotate_session.py --session data/real/processed/session_real_001 --existing labels.csv

    # Segment on angular velocity threshold (default 15 dps)
    py scripts/acquisition/annotate_session.py --session data/real/processed/session_real_001 --threshold 20

How segments are detected
--------------------------
    The tool uses a simple threshold on trunk angular velocity
    (Pelvis-L3 angular difference) to find movement onset/offset.
    Any continuous period above the threshold is treated as a candidate
    segment. Segments shorter than min_duration_s are skipped (likely
    noise). Segments are shown one at a time with a summary of key stats.

Output
------
    Writes or updates labels.csv in the session directory with columns:
        label, rep, start_ms, end_ms, risk_class, fatigue_fraction

    The output format matches synthetic_generator.py and label_logger.py.

Notes
-----
    Is a fallback for missed real-time annotation. For all new
    sessions, prefer label_logger.py which runs alongside recording and
    produces labels.csv with accurate per-rep timing.

Requirements
------------
    py -m pip install pandas numpy
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("[ERROR] numpy and pandas are required.")
    print("  Run: py -m pip install numpy pandas")
    sys.exit(1)


# Movement reference

MOVEMENT_MAP = {
    '1': ('BASELINE_STATIC',       0),
    '2': ('CLEAN_FLEXION',         0),
    '3': ('CLEAN_LATERAL_L',       0),
    '4': ('CLEAN_LATERAL_R',       0),
    '5': ('CLEAN_ROTATION_L',      0),
    '6': ('CLEAN_ROTATION_R',      0),
    '7': ('PICKUP_SYM',            0),
    '8': ('SIT_TO_STAND_NORMAL',   0),
    'a': ('LUMBAR_DOMINANT',       1),
    'b': ('FAST_BEND',             1),
    'c': ('SHOULDER_DRIVEN',       1),
    'd': ('PICKUP_ASYM',           1),
    'f': ('FATIGUE_FLEXION',       1),
    'g': ('SIT_TO_STAND_FAST',    -1),
    's': ('SKIP',                  None),  # skip this segment
}

MENU = """
  Movement labels:
    1 BASELINE_STATIC (safe)     2 CLEAN_FLEXION (safe)
    3 CLEAN_LATERAL_L (safe)     4 CLEAN_LATERAL_R (safe)
    5 CLEAN_ROTATION_L (safe)    6 CLEAN_ROTATION_R (safe)
    7 PICKUP_SYM (safe)          8 SIT_TO_STAND_NORMAL (safe)
    a LUMBAR_DOMINANT (risky)    b FAST_BEND (risky)
    c SHOULDER_DRIVEN (risky)    d PICKUP_ASYM (risky)
    f FATIGUE_FLEXION (risky)    g SIT_TO_STAND_FAST (ambiguous)
    s SKIP this segment          q QUIT and save
"""


# Segment detection

def detect_segments(
    imu_df: pd.DataFrame,
    threshold_dps: float = 15.0,
    min_duration_s: float = 0.5,
    gap_fill_ms: float = 500.0,
) -> list[dict]:
    """
    Detect movement segments from IMU data using angular velocity threshold.

    Uses the L3 gyroscope magnitude as a proxy for movement onset/offset.
    Merges segments separated by gaps shorter than gap_fill_ms.

    Parameters
    ----------
    imu_df : DataFrame with t_ms and L3_gx, L3_gy, L3_gz columns
    threshold_dps : angular velocity threshold in degrees/s
    min_duration_s : minimum segment length in seconds
    gap_fill_ms : merge segments with gaps shorter than this

    Returns
    -------
    List of dicts with keys: start_ms, end_ms, duration_s, peak_vel_dps,
        mean_flexion_deg (if pitch columns available)
    """
    # Compute angular velocity magnitude from the L3 sensor
    gyro_cols = [c for c in imu_df.columns if 'L3' in c and '_g' in c and
                 any(c.endswith(ax) for ax in ('x_dps', 'y_dps', 'z_dps'))]

    if not gyro_cols:
        # Fall back to any available gyro columns
        gyro_cols = [c for c in imu_df.columns if '_g' in c and
                     any(c.endswith(ax) for ax in ('x_dps', 'y_dps', 'z_dps'))]

    if not gyro_cols:
        print("  [WARN] No gyroscope columns found. Cannot auto-detect segments.")
        print("  Treating the entire recording as one segment.")
        t0 = float(imu_df['t_ms'].iloc[0])
        t1 = float(imu_df['t_ms'].iloc[-1])
        return [{'start_ms': t0, 'end_ms': t1,
                 'duration_s': (t1 - t0) / 1000.0,
                 'peak_vel_dps': 0.0, 'mean_flexion_deg': None}]

    vel_mag = np.sqrt((imu_df[gyro_cols] ** 2).sum(axis=1))
    moving = vel_mag > threshold_dps
    t_ms = imu_df['t_ms'].values

    # Find contiguous moving blocks
    raw_segments = []
    in_seg = False
    seg_start = None
    for i, m in enumerate(moving):
        if m and not in_seg:
            in_seg = True
            seg_start = t_ms[i]
        elif not m and in_seg:
            in_seg = False
            raw_segments.append((seg_start, t_ms[i]))
    if in_seg:
        raw_segments.append((seg_start, t_ms[-1]))

    # Merge close segments
    merged = []
    for seg in raw_segments:
        if merged and (seg[0] - merged[-1][1]) < gap_fill_ms:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(list(seg))

    # Filter short segments and compute stats
    results = []
    for start_ms, end_ms in merged:
        duration_s = (end_ms - start_ms) / 1000.0
        if duration_s < min_duration_s:
            continue

        mask = (t_ms >= start_ms) & (t_ms <= end_ms)
        seg_vel = vel_mag[mask]

        mean_flex = None
        flex_cols = [c for c in imu_df.columns if 'L3' in c and 'pitch' in c.lower()]
        if flex_cols:
            mean_flex = float(imu_df.loc[mask, flex_cols[0]].mean())

        results.append({
            'start_ms':        round(float(start_ms), 1),
            'end_ms':          round(float(end_ms),   1),
            'duration_s':      round(duration_s, 2),
            'peak_vel_dps':    round(float(seg_vel.max()), 1),
            'mean_flexion_deg': round(mean_flex, 1) if mean_flex is not None else None,
        })

    return results


# Terminal display

def _show_segment(idx: int, total: int, seg: dict) -> None:
    """Print one segment's timing and movement stats to help the user label it."""
    print()
    print(f"  Segment {idx + 1} / {total}")
    print(f"  Time:      {seg['start_ms']:.0f} ms  ->  {seg['end_ms']:.0f} ms  ({seg['duration_s']:.1f} s)")
    print(f"  Peak vel:  {seg['peak_vel_dps']:.1f} dps")
    if seg['mean_flexion_deg'] is not None:
        print(f"  Mean flex: {seg['mean_flexion_deg']:.1f} deg")


def _prompt(msg: str) -> str:
    """Read a normalised keypress; treat EOF/Ctrl+C as 'q' so partial work saves."""
    try:
        return input(msg).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return 'q'


# Main

def main() -> None:
    """Load a processed session, auto-detect segments, and step through labelling."""
    parser = argparse.ArgumentParser(
        description='Post-hoc annotation tool for recorded sessions.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--session', '-s',
        type=Path,
        required=True,
        help='Path to a processed session directory containing imu_data.csv',
    )
    parser.add_argument(
        '--existing', '-e',
        type=Path,
        default=None,
        help='Path to an existing partial labels.csv to append to',
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=15.0,
        help='Angular velocity threshold for segment detection in dps (default: 15)',
    )
    parser.add_argument(
        '--min_duration',
        type=float,
        default=0.5,
        help='Minimum segment duration in seconds (default: 0.5)',
    )
    args = parser.parse_args()

    session_dir = args.session
    imu_path = session_dir / 'imu_data.csv'
    out_path  = session_dir / 'labels.csv'

    if not imu_path.exists():
        print(f"[ERROR] imu_data.csv not found in {session_dir}")
        print("  Run session_converter.py first to produce the processed session.")
        sys.exit(1)

    if out_path.exists() and args.existing is None:
        resp = _prompt(f"  {out_path} already exists. Overwrite? [y/N]: ")
        if resp != 'y':
            print("  Aborted.")
            sys.exit(0)

    # Load existing labels if provided
    existing_rows: list[dict] = []
    if args.existing and args.existing.exists():
        existing_rows = list(csv.DictReader(open(args.existing)))
        print(f"  Loaded {len(existing_rows)} existing labels from {args.existing}")

    print(f"\n  Loading {imu_path} ...")
    imu_df = pd.read_csv(imu_path)
    print(f"  {len(imu_df)} rows, {imu_df['t_ms'].iloc[-1]:.0f} ms total")

    print(f"\n  Detecting movement segments (threshold {args.threshold} dps) ...")
    segments = detect_segments(imu_df, threshold_dps=args.threshold,
                               min_duration_s=args.min_duration)
    print(f"  Found {len(segments)} segments")

    if not segments:
        print("  No segments detected. Try lowering --threshold.")
        sys.exit(0)

    print(MENU)
    completed: list[dict] = list(existing_rows)
    rep_counters: dict[str, int] = {}

    # Resume rep numbering from the highest existing rep per label, so a
    # second annotation pass continues counting rather than restarting at 1.
    for row in completed:
        label = row['label']
        rep_counters[label] = max(rep_counters.get(label, 0), int(row.get('rep', 0)))

    for i, seg in enumerate(segments):
        _show_segment(i, len(segments), seg)
        print()

        while True:
            key = _prompt("  Label [1-8 / a-g / s=skip / q=quit]: ")
            if key == 'q':
                break
            if key not in MOVEMENT_MAP:
                print("  Unrecognised key. Try again.")
                continue
            label, risk_class = MOVEMENT_MAP[key]
            if label == 'SKIP':
                print("  Skipped.")
                break
            rep_num = rep_counters.get(label, 0) + 1
            rep_counters[label] = rep_num
            completed.append({
                'label':           label,
                'rep':             rep_num,
                'start_ms':        seg['start_ms'],
                'end_ms':          seg['end_ms'],
                'risk_class':      risk_class,
                'fatigue_fraction': '',
            })
            risk_str = {0: 'safe', 1: 'risky', -1: 'ambiguous', None: '?'}.get(risk_class, '?')
            print(f"  Labelled: {label} rep {rep_num} ({risk_str})")
            break

        if key == 'q':
            break

    # Save
    rows = sorted(completed, key=lambda r: float(r['start_ms']))
    fieldnames = ['label', 'rep', 'start_ms', 'end_ms', 'risk_class', 'fatigue_fraction']
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    risky = sum(1 for r in rows if str(r['risk_class']) == '1')
    safe  = sum(1 for r in rows if str(r['risk_class']) == '0')
    print()
    print(f"  Saved {n} labels -> {out_path}")
    print(f"  (safe={safe}, risky={risky})")
    print()


if __name__ == '__main__':
    main()
