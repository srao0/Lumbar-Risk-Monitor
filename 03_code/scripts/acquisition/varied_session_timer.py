#!/usr/bin/env python3
"""
Timer and label writer for Phase II.C varied-movement held-out sessions.

Run this alongside the IMU and OpenBCI recorders, just like session_timer.py.
The difference is evidential: these schedules are for frozen-model held-out
testing, not protocol training. The script writes labels.csv before the
countdown starts so the planned ground truth survives even if the timer is
interrupted.

Usage
-----
    py scripts/acquisition/varied_session_timer.py --variant A --out data/real/raw/varied_A/labels.csv
    py scripts/acquisition/varied_session_timer.py --variant B --out data/real/raw/varied_B/labels.csv
    py scripts/acquisition/varied_session_timer.py --variant C --out data/real/raw/varied_C/labels.csv
    py scripts/acquisition/varied_session_timer.py --variant A --preview
"""

from __future__ import annotations

# Make the project package importable when this file is run directly.
import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = _Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PKG_ROOT))


import argparse
import csv
import sys
import time
from pathlib import Path

try:
    from scripts.conversion.generate_protocol_labels import MOVEMENT_CATALOGUE
except ModuleNotFoundError:
    from scripts.conversion.generate_protocol_labels import MOVEMENT_CATALOGUE


# Each segment uses only labels already accepted by MOVEMENT_CATALOGUE so that
# session_converter.py and validate_phase2_dataset.py can process the output.
# risk is checked against the catalogue at startup.
VARIED_PROTOCOLS = {
    "A": {
        "name": "Mixed known movements",
        "purpose": "Held-out shuffled known movements with varied speed and order.",
        "segments": [
            ("BASELINE_STATIC", 60, "Stand still. Natural posture."),
            ("CLEAN_FLEXION", 12, "Two controlled hip-hinge bends."),
            ("REST", 6, "Reset to neutral."),
            ("FAST_BEND", 10, "Two fast comfortable forward bends."),
            ("REST", 6, "Reset to neutral."),
            ("PICKUP_SYM", 14, "Two symmetric pick-ups from the floor."),
            ("REST", 6, "Reset to neutral."),
            ("CLEAN_LATERAL_L", 10, "Two controlled lateral bends left."),
            ("REST", 5, "Reset to neutral."),
            ("PICKUP_ASYM", 14, "Two asymmetric pick-ups to one side."),
            ("REST", 6, "Reset to neutral."),
            ("CLEAN_ROTATION_R", 10, "Two controlled rotations right."),
            ("REST", 5, "Reset to neutral."),
            ("LUMBAR_DOMINANT", 12, "Two lumbar-dominant bends."),
            ("REST", 8, "Reset to neutral."),
            ("SIT_TO_STAND_NORMAL", 15, "Three controlled sit-to-stands."),
            ("REST", 8, "Stand still before final baseline."),
            ("BASELINE_STATIC", 60, "Final still stand."),
        ],
    },
    "B": {
        "name": "Transition-heavy sequence",
        "purpose": "Known tasks with frequent transitions and short neutral buffers.",
        "segments": [
            ("BASELINE_STATIC", 60, "Stand still. Natural posture."),
            ("SIT_TO_STAND_NORMAL", 18, "Stand and sit repeatedly at normal pace."),
            ("REST", 5, "Transition to standing."),
            ("CLEAN_FLEXION", 10, "Controlled bend and return."),
            ("REST", 4, "Transition."),
            ("CLEAN_ROTATION_L", 8, "Rotate left and return."),
            ("REST", 4, "Transition."),
            ("CLEAN_ROTATION_R", 8, "Rotate right and return."),
            ("REST", 4, "Transition."),
            ("SHOULDER_DRIVEN", 12, "Rounded shoulder-driven collapse and return."),
            ("REST", 5, "Transition."),
            ("PICKUP_SYM", 12, "Symmetric pick-up."),
            ("REST", 4, "Transition."),
            ("PICKUP_ASYM", 12, "Asymmetric pick-up."),
            ("REST", 4, "Transition."),
            ("FAST_BEND", 10, "Fast bend and return."),
            ("REST", 8, "Recover in neutral."),
            ("CLEAN_LATERAL_R", 8, "Controlled lateral bend right."),
            ("REST", 4, "Transition."),
            ("LUMBAR_DOMINANT", 12, "Lumbar-dominant bend and return."),
            ("REST", 8, "Stand still before final baseline."),
            ("BASELINE_STATIC", 60, "Final still stand."),
        ],
    },
    "C": {
        "name": "Prompted varied blocks",
        "purpose": "Longer self-paced blocks with less tidy repetitions.",
        "segments": [
            ("BASELINE_STATIC", 60, "Stand still. Natural posture."),
            ("CLEAN_FLEXION", 25, "Self-paced comfortable hip-hinge bends."),
            ("REST", 10, "Neutral standing."),
            ("PICKUP_SYM", 25, "Self-paced symmetric pick-ups."),
            ("REST", 10, "Neutral standing."),
            ("FAST_BEND", 20, "Several fast comfortable bends, not maximal depth."),
            ("REST", 10, "Neutral standing."),
            ("PICKUP_ASYM", 25, "Self-paced asymmetric reaches/pick-ups."),
            ("REST", 10, "Neutral standing."),
            ("SHOULDER_DRIVEN", 20, "Rounded upper-back-led bends."),
            ("REST", 10, "Neutral standing."),
            ("FATIGUE_FLEXION", 45, "Continuous comfortable repeated bends."),
            ("REST", 15, "Recover in neutral."),
            ("CLEAN_ROTATION_L", 12, "Controlled rotations left."),
            ("REST", 5, "Transition."),
            ("CLEAN_ROTATION_R", 12, "Controlled rotations right."),
            ("REST", 10, "Stand still before final baseline."),
            ("BASELINE_STATIC", 60, "Final still stand."),
        ],
    },
}


def hms(seconds: float) -> str:
    """Format a second count as m:ss for the schedule and countdown displays."""
    seconds = int(seconds)
    mins, secs = divmod(seconds, 60)
    return f"{mins}:{secs:02d}"


def risk_for(label: str) -> int:
    """Resolve a movement's risk_class from the shared catalogue.

    Raises rather than guessing so a typo in a VARIED_PROTOCOLS segment fails
    loudly at startup instead of writing a mislabelled held-out ground truth.
    """
    if label not in MOVEMENT_CATALOGUE:
        raise RuntimeError(f"Unsupported varied-session label: {label}")
    return int(MOVEMENT_CATALOGUE[label]["risk_class"])


def build_rows(segments: list[tuple[str, int, str]]) -> list[dict[str, object]]:
    """Lay segments end-to-end on a t=0 timeline into labels.csv rows.

    Each segment's end_ms becomes the next one's start_ms, so the planned
    schedule is contiguous with no gaps. rep is a per-label counter.
    """
    rows: list[dict[str, object]] = []
    t_ms = 0
    rep_counts: dict[str, int] = {}

    for label, duration_s, instruction in segments:
        risk = risk_for(label)
        rep_counts[label] = rep_counts.get(label, 0) + 1
        end_ms = t_ms + int(duration_s * 1000)
        rows.append(
            {
                "label": label,
                "rep": rep_counts[label],
                "start_ms": t_ms,
                "end_ms": end_ms,
                "risk_class": risk,
                "fatigue_fraction": "",
                "notes": instruction,
            }
        )
        t_ms = end_ms
    return rows


def write_labels(rows: list[dict[str, object]], out_path: Path) -> None:
    """Write the planned schedule to labels.csv (called before the countdown).

    Writing up front means the ground truth survives even if the timer is
    interrupted mid-session -- the recorded sensor data can still be aligned.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "rep",
        "start_ms",
        "end_ms",
        "risk_class",
        "fatigue_fraction",
        "notes",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_schedule(variant: str, rows: list[dict[str, object]]) -> None:
    """Print the full timeline and safe/risky/excluded tallies before running."""
    protocol = VARIED_PROTOCOLS[variant]
    total_s = rows[-1]["end_ms"] / 1000 if rows else 0
    safe = sum(1 for row in rows if row["risk_class"] == 0)
    risky = sum(1 for row in rows if row["risk_class"] == 1)
    excluded = sum(1 for row in rows if row["risk_class"] == -1)

    print()
    print("=" * 72)
    print(f"  Phase II.C varied session {variant}: {protocol['name']}")
    print(f"  Purpose : {protocol['purpose']}")
    print(f"  Duration: {hms(total_s)} | segments={len(rows)} | safe={safe} | risky={risky} | excluded={excluded}")
    print("=" * 72)
    for row in rows:
        start_s = row["start_ms"] / 1000
        end_s = row["end_ms"] / 1000
        risk = {0: "SAFE", 1: "RISKY", -1: "EXCLUDE"}[row["risk_class"]]
        print(f"  {hms(start_s)}-{hms(end_s)}  {row['label']:<20} {risk:<7} {row['notes']}")
    print()


def countdown_segment(row: dict[str, object], index: int, total: int) -> None:
    """Show a live progress bar for one segment, ticking once per second.

    Purely a participant cue -- the authoritative timing is already in the
    labels.csv that build_rows wrote, so a missed tick does not corrupt timing.
    """
    label = str(row["label"])
    duration_s = int((row["end_ms"] - row["start_ms"]) / 1000)
    risk = {0: "SAFE", 1: "RISKY", -1: "EXCLUDE"}[row["risk_class"]]

    print()
    print("=" * 72)
    print(f"  Segment {index}/{total}: {label} [{risk}]")
    print(f"  Duration: {duration_s}s")
    print(f"  Instruction: {row['notes']}")
    print("=" * 72)

    for tick in range(duration_s):
        remaining = duration_s - tick
        elapsed = tick
        fill = int(24 * elapsed / duration_s) if duration_s > 0 else 24
        bar = "#" * fill + "-" * (24 - fill)
        sys.stdout.write(f"\r  [{bar}] {remaining:3d}s remaining")
        sys.stdout.flush()
        time.sleep(1)
    print("\r  [" + "#" * 24 + "] complete          ")


def run(variant: str, out_path: Path | None, preview: bool = False) -> None:
    """Build and save the schedule, then (unless preview) run the guided timer."""
    segments = VARIED_PROTOCOLS[variant]["segments"]
    rows = build_rows(segments)
    print_schedule(variant, rows)

    if out_path is not None:
        write_labels(rows, out_path)
        print(f"  labels.csv written before countdown: {out_path}")
    else:
        print("  No --out provided; labels will not be saved.")

    if preview:
        return

    print()
    print("  Workflow:")
    print("   Terminal 1: start IMU recorder")
    print("   Terminal 2: start Cyton/Ganglion recorder, if full-hybrid")
    print("   Terminal 3: run this timer")
    print()
    input("  Press Enter once recordings are running: ")
    for n in (3, 2, 1):
        print(f"  Starting in {n}...")
        time.sleep(1)
    print("  GO!")

    for index, row in enumerate(rows, start=1):
        countdown_segment(row, index, len(rows))

    print()
    print("=" * 72)
    print("  Varied movement recording complete. Stop the sensor recorders now.")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Guided timer and labels.csv writer for Phase II.C varied-movement sessions."
    )
    parser.add_argument(
        "--variant",
        choices=sorted(VARIED_PROTOCOLS),
        required=True,
        help="Varied-session schedule to run: A=mixed known, B=transition-heavy, C=prompted varied.",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="Path to write labels.csv, usually inside the raw session folder.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print the schedule and write labels.csv if --out is supplied, but do not run countdown.",
    )
    args = parser.parse_args()
    run(
        variant=args.variant,
        out_path=Path(args.out) if args.out else None,
        preview=args.preview,
    )


if __name__ == "__main__":
    main()
