#!/usr/bin/env python3
"""
generate_protocol_labels.py: Phase 2 Protocol Label Generator
===============================================================
Spinal Movement Risk Monitor | FYP 2025/26 | Imperial College London

Generates a labels.csv for a real participant session following the
standardised Phase 2 data collection protocol.  The output file has
zero label-feature circularity because labels are assigned from the
experimental design (what the participant was asked to do and when),
not from any computed signal feature.

Why this script exists
----------------------
The synthetic pipeline uses signal-derived labels (label_window_from_signal)
where risk_class is computed from IMU features like imu_jerk_rms and
imu_compensation_index.  The ML classifier is then trained on these same
features, creating a circular dependency that artificially inflates IMU AUC
to ~0.9999 on synthetic data.

For Phase 2, labels must come from an external source.  Provides
two modes:

  1. GENERATE mode  (--mode generate)
     You specify the actual start/end times from your recording (from a
     synchronised video or a label_logger.py log) and the script writes a
     correctly-formatted labels.csv directly.

  2. TEMPLATE mode  (--mode template)
     The script writes a CSV template pre-filled with the protocol structure
     (baseline → safe blocks → risky blocks → cool-down) with placeholder
     timing.  You open the file, fill in the real timestamps, and save.
     Then run the pipeline normally.

Phase 2 Data Collection Protocol (standard session)
---------------------------------------------------
The following structure should be used for all Phase 2 participant sessions
to ensure consistent labelling across participants.  Each participant performs
the same sequence of movements in the same order.

  Phase          Duration  Movements                          risk_class
  ─────────────────────────────────────────────────────────────────────────
  Baseline       60 s      Stand still, normal breathing      0
  Safe block A   ~5 min    CLEAN_FLEXION ×10                  0
                           PICKUP_SYM ×10                     0
                           CLEAN_LATERAL_L ×5                 0
                           CLEAN_LATERAL_R ×5                 0
  Rest            2 min, skip
  Risky block A  ~5 min    FAST_BEND ×10                      1
                           LUMBAR_DOMINANT ×10                1
                           SHOULDER_DRIVEN ×10                1
  Rest            2 min, skip
  Safe block B   ~3 min    SIT_TO_STAND_NORMAL ×10            0
                           PICKUP_SYM ×5                      0
  Risky block B  ~3 min    PICKUP_ASYM ×10                    1
                           FATIGUE_FLEXION ×10                1
  Cool-down       2 min    BASELINE_STATIC                    0

Each movement "rep" is a complete cycle (bend + return), lasting ~3-5 s,
separated by ~3 s rest.  The researcher records actual start/end times
either from a video timestamp or from label_logger.py.

Risk class definitions
----------------------
  0  = safe movement (no biomechanical risk)
  1  = risky movement (exceeds occupational biomechanical thresholds)
 -1  = ambiguous / transition / exclude from analysis

Movement catalogue
------------------
  Safe (risk_class=0):
    BASELINE_STATIC: standing still, anatomical position
    CLEAN_FLEXION: controlled forward bend to ~60° and return
    CLEAN_LATERAL_L/R: lateral trunk bend, left/right
    CLEAN_ROTATION_L/R: trunk axial rotation, left/right
    PICKUP_SYM: symmetric two-handed lift from floor (~30 cm)
    SIT_TO_STAND_NORMAL: controlled sit-to-stand from standard chair

  Risky (risk_class=1):
    FAST_BEND: rapid forward bend (>60°/s, no control)
    LUMBAR_DOMINANT: forward bend driven by lumbar spine, not hip hinge
    SHOULDER_DRIVEN: forward bend initiated from upper back (thoracic compensation)
    PICKUP_ASYM: one-sided asymmetric lift with lateral trunk loading
    FATIGUE_FLEXION: repeated slow bends simulating end-of-shift fatigue

Usage
-----
  # Write a template for manual timing fill-in:
  python scripts/conversion/generate_protocol_labels.py \\
      --mode template \\
      --out_dir data/real/protocol_train/participant_01/session_001

  # Generate from known timings (interactive):
  python scripts/conversion/generate_protocol_labels.py \\
      --mode generate \\
      --out_dir data/real/protocol_train/participant_01/session_001

  # Generate from a timing CSV (batch, for scripted sessions):
  python scripts/conversion/generate_protocol_labels.py \\
      --mode generate \\
      --timings my_timings.csv \\
      --out_dir data/real/protocol_train/participant_01/session_001

Timing CSV format (--timings)
------------------------------
  label,rep,start_s,end_s
  BASELINE_STATIC,1,0,60
  CLEAN_FLEXION,1,65,72
  CLEAN_FLEXION,2,75,82
  ...

After generating labels.csv, run:
  python -m signal_processing.pipeline \\
      --data_dir data/real/protocol_train \\
      --label_source protocol

Then train:
  python ml/training/train_classifier.py \\
      --data_dir data/real/protocol_train \\
      --label_source protocol \\
      --cv_group participant
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Dict, Optional

try:
    import pandas as pd
except ImportError:
    print("[ERROR] pandas is required. Run: pip install pandas")
    sys.exit(1)


# MOVEMENT CATALOGUE

MOVEMENT_CATALOGUE: Dict[str, Dict] = {
    # Safe movements (risk_class = 0)
    "BASELINE_STATIC":     {"risk_class": 0, "category": "safe",  "duration_s": (3,  120), "description": "Standing still, anatomical position"},
    "CLEAN_FLEXION":       {"risk_class": 0, "category": "safe",  "duration_s": (3,  8),   "description": "Controlled forward bend to ~60° and return"},
    "CLEAN_LATERAL_L":     {"risk_class": 0, "category": "safe",  "duration_s": (3,  8),   "description": "Controlled lateral bend to the left"},
    "CLEAN_LATERAL_R":     {"risk_class": 0, "category": "safe",  "duration_s": (3,  8),   "description": "Controlled lateral bend to the right"},
    "CLEAN_ROTATION_L":    {"risk_class": 0, "category": "safe",  "duration_s": (3,  8),   "description": "Controlled axial rotation to the left"},
    "CLEAN_ROTATION_R":    {"risk_class": 0, "category": "safe",  "duration_s": (3,  8),   "description": "Controlled axial rotation to the right"},
    "PICKUP_SYM":          {"risk_class": 0, "category": "safe",  "duration_s": (4,  10),  "description": "Symmetric two-handed lift from ~30 cm height"},
    "SIT_TO_STAND_NORMAL": {"risk_class": 0, "category": "safe",  "duration_s": (3,  8),   "description": "Controlled sit-to-stand from standard chair"},
    # Risky movements (risk_class = 1)
    "FAST_BEND":           {"risk_class": 1, "category": "risky", "duration_s": (2,  6),   "description": "Rapid forward bend (>60°/s), no control"},
    "LUMBAR_DOMINANT":     {"risk_class": 1, "category": "risky", "duration_s": (3,  8),   "description": "Lumbar-driven bend without adequate hip hinge"},
    "SHOULDER_DRIVEN":     {"risk_class": 1, "category": "risky", "duration_s": (3,  8),   "description": "Thoracic-compensation bend (upper back initiates)"},
    "PICKUP_ASYM":         {"risk_class": 1, "category": "risky", "duration_s": (4,  10),  "description": "Asymmetric one-sided lift with lateral trunk load"},
    "FATIGUE_FLEXION":     {"risk_class": 1, "category": "risky", "duration_s": (4,  12),  "description": "Repeated slow fatigued bends, increasing lumbar load"},
    # Ambiguous / excluded
    "SIT_TO_STAND_FAST":   {"risk_class": -1, "category": "ambiguous", "duration_s": (2, 6), "description": "Fast sit-to-stand — borderline, excluded from analysis"},
    "REST":                {"risk_class": -1, "category": "exclude",   "duration_s": (30, 300), "description": "Rest period between blocks"},
}

# Standard Phase 2 protocol block order
STANDARD_PROTOCOL = [
    # (label, n_reps, rest_between_reps_s)
    ("BASELINE_STATIC",     1,  0),    # 60 s baseline
    ("CLEAN_FLEXION",       10, 3),    # 10 reps × ~5 s + 3 s rest
    ("PICKUP_SYM",          10, 3),
    ("CLEAN_LATERAL_L",     5,  3),
    ("CLEAN_LATERAL_R",     5,  3),
    ("REST",                1,  0),    # 2 min rest block
    ("FAST_BEND",           10, 3),
    ("LUMBAR_DOMINANT",     10, 3),
    ("SHOULDER_DRIVEN",     10, 3),
    ("REST",                1,  0),
    ("SIT_TO_STAND_NORMAL", 10, 3),
    ("PICKUP_SYM",          5,  3),
    ("PICKUP_ASYM",         10, 3),
    ("FATIGUE_FLEXION",     10, 3),
    ("BASELINE_STATIC",     1,  0),    # cool-down baseline
]

# Estimated movement durations (seconds) for template generation
TEMPLATE_DURATIONS = {
    "BASELINE_STATIC":     60,
    "CLEAN_FLEXION":        5,
    "PICKUP_SYM":           6,
    "CLEAN_LATERAL_L":      5,
    "CLEAN_LATERAL_R":      5,
    "CLEAN_ROTATION_L":     5,
    "CLEAN_ROTATION_R":     5,
    "SIT_TO_STAND_NORMAL":  4,
    "FAST_BEND":            3,
    "LUMBAR_DOMINANT":      5,
    "SHOULDER_DRIVEN":      5,
    "PICKUP_ASYM":          6,
    "FATIGUE_FLEXION":      7,
    "REST":               120,
    "SIT_TO_STAND_FAST":    3,
}

FIELDNAMES = ["label", "rep", "start_ms", "end_ms", "risk_class", "notes"]


# TEMPLATE GENERATOR

def generate_template(out_path: Path) -> List[Dict]:
    """
    Generate a labels.csv with estimated timings from the standard protocol.

    All timings are placeholder estimates based on average movement durations.
    The researcher should update start_ms and end_ms with actual recorded values
    (from video timestamps or label_logger.py) before running the pipeline.

    The template is colour-coded in the notes column to make manual editing easier.
    """
    rows = []
    t_ms  = 0.0
    rep_counters: Dict[str, int] = {}

    for label, n_reps, rest_between_s in STANDARD_PROTOCOL:
        duration_s = TEMPLATE_DURATIONS.get(label, 5)
        info       = MOVEMENT_CATALOGUE.get(label, {})
        rc         = info.get("risk_class", -1)

        for rep_i in range(n_reps):
            rep_counters[label] = rep_counters.get(label, 0) + 1
            rep_num = rep_counters[label]

            start_ms = round(t_ms, 1)
            end_ms   = round(t_ms + duration_s * 1000, 1)

            notes = "[TEMPLATE — update with real timestamps]"
            if label == "BASELINE_STATIC":
                notes = "[TEMPLATE — participant stands still, record actual duration]"
            elif label == "REST":
                notes = "[TEMPLATE — rest between blocks, adjust duration as needed]"
            elif rc == 1:
                notes = "[TEMPLATE — RISKY movement, record per-rep timing accurately]"

            rows.append({
                "label":      label,
                "rep":        rep_num,
                "start_ms":   start_ms,
                "end_ms":     end_ms,
                "risk_class": rc,
                "notes":      notes,
            })

            t_ms = end_ms + rest_between_s * 1000.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    total_safe  = sum(1 for r in rows if r["risk_class"] == 0 and r["label"] != "BASELINE_STATIC")
    total_risky = sum(1 for r in rows if r["risk_class"] == 1)
    total_min   = round(rows[-1]["end_ms"] / 60_000, 1)

    print(f"\n  ✓ Template labels.csv generated: {out_path}")
    print(f"    {len(rows)} segments | {total_safe} safe reps | {total_risky} risky reps")
    print(f"    Estimated session duration: ~{total_min} min")
    print()
    print("  NEXT STEPS:")
    print("  1. Open labels.csv in a spreadsheet")
    print("  2. Replace placeholder start_ms / end_ms with actual recording timestamps")
    print("     (use video timestamps × 1000, or export from label_logger.py)")
    print("  3. Delete 'notes' column or leave it — pipeline ignores it")
    print("  4. Save and run:")
    print("     python -m signal_processing.pipeline --data_dir data/real/protocol_train --label_source protocol")
    print()

    return rows


# TIMING CSV → LABELS.CSV

def from_timings_csv(timings_csv: Path, out_path: Path) -> List[Dict]:
    """
    Convert a researcher-authored timings CSV to labels.csv format.

    Timings CSV columns:
        label     : movement name (must be in MOVEMENT_CATALOGUE)
        rep       : repetition number (integer, starting from 1)
        start_s   : segment start time in seconds from recording start
        end_s     : segment end time in seconds from recording start

    Optional columns:
        notes     : free-text annotation (passed through)
        risk_class: override automatic risk_class from catalogue (use sparingly)
    """
    timings = pd.read_csv(timings_csv)
    required = {"label", "rep", "start_s", "end_s"}
    missing  = required - set(timings.columns)
    if missing:
        raise ValueError(f"Timings CSV is missing required columns: {missing}")

    rows = []
    warnings = []
    for i, row_in in timings.iterrows():
        label = str(row_in["label"]).strip()
        rep   = int(row_in["rep"])
        start_s = float(row_in["start_s"])
        end_s   = float(row_in["end_s"])

        if label not in MOVEMENT_CATALOGUE:
            warnings.append(f"  Row {i+2}: unknown label '{label}' — assigning risk_class=-1")
            rc = -1
        else:
            rc = MOVEMENT_CATALOGUE[label]["risk_class"]

        # Allow manual override of risk_class
        if "risk_class" in timings.columns and pd.notna(row_in.get("risk_class")):
            rc = int(row_in["risk_class"])

        duration_s = end_s - start_s
        if duration_s <= 0:
            warnings.append(f"  Row {i+2}: start_s >= end_s for {label} rep {rep} — skipped")
            continue
        if duration_s > 60:
            warnings.append(f"  Row {i+2}: {label} rep {rep} duration {duration_s:.1f}s seems long — check timestamps")

        rows.append({
            "label":      label,
            "rep":        rep,
            "start_ms":   round(start_s * 1000, 1),
            "end_ms":     round(end_s   * 1000, 1),
            "risk_class": rc,
            "notes":      str(row_in.get("notes", "")) if "notes" in timings.columns else "",
        })

    if warnings:
        print("\n  Warnings:")
        for w in warnings:
            print(w)

    rows.sort(key=lambda r: r["start_ms"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    safe  = sum(1 for r in rows if r["risk_class"] == 0)
    risky = sum(1 for r in rows if r["risk_class"] == 1)
    excl  = sum(1 for r in rows if r["risk_class"] == -1)
    print(f"\n  ✓ labels.csv written: {out_path}")
    print(f"    {len(rows)} segments | safe={safe} | risky={risky} | excluded={excl}")
    return rows


# INTERACTIVE GENERATOR

def interactive_generate(out_path: Path) -> List[Dict]:
    """
    Step through the standard protocol interactively, entering timestamps
    for each movement block as it is completed.

    The researcher starts recording, triggers the participant movements one at
    a time, and enters the start/end time (in seconds) for each rep as it
    finishes.  The labels.csv is written on exit.

    Press Ctrl-C or type 'q' to finish early and save what has been entered.
    """
    print("\n  Interactive protocol label generator")
    print("  ─────────────────────────────────────────────────────────────")
    print("  Start your recording BEFORE entering any times.")
    print("  Times are in SECONDS from recording start (not wall-clock time).")
    print("  Press Enter without a value to use the suggested default.")
    print("  Type 'q' at any prompt to finish and save.")
    print()

    rows = []
    rep_counters: Dict[str, int] = {}
    current_t = 0.0

    def prompt(msg: str) -> Optional[str]:
        try:
            v = input(f"  {msg}").strip()
            return None if v.lower() == "q" else (v if v else None)
        except (EOFError, KeyboardInterrupt):
            return None

    for label, n_reps, rest_between_s in STANDARD_PROTOCOL:
        info = MOVEMENT_CATALOGUE.get(label, {})
        rc   = info.get("risk_class", -1)
        cat  = info.get("category", "")
        desc = info.get("description", label)
        dur  = TEMPLATE_DURATIONS.get(label, 5)

        print(f"  ── {label}  [{cat}]  {desc}")

        for rep_i in range(n_reps):
            rep_counters[label] = rep_counters.get(label, 0) + 1
            rep_num = rep_counters[label]

            suggested_start = round(current_t, 1)
            suggested_end   = round(current_t + dur, 1)

            if label == "REST":
                val = prompt(f"  REST block — press Enter when rest is done (or 'q'): ")
                if val is None and val != "":
                    break
                current_t += dur
                rows.append({
                    "label":      "REST",
                    "rep":        rep_num,
                    "start_ms":   round(suggested_start * 1000, 1),
                    "end_ms":     round(current_t * 1000, 1),
                    "risk_class": -1,
                    "notes":      "rest between blocks",
                })
                break

            # Get start time
            val = prompt(f"  {label} rep {rep_num} — start_s [{suggested_start:.1f}]: ")
            if val is None:
                print("  Saving and exiting early...")
                _write_labels(rows, out_path)
                return rows
            try:
                start_s = float(val) if val else suggested_start
            except ValueError:
                print(f"    Invalid value — using {suggested_start:.1f}")
                start_s = suggested_start

            # Get end time
            val = prompt(f"  {label} rep {rep_num} — end_s   [{suggested_end:.1f}]: ")
            if val is None:
                print("  Saving and exiting early...")
                _write_labels(rows, out_path)
                return rows
            try:
                end_s = float(val) if val else suggested_end
            except ValueError:
                print(f"    Invalid value — using {suggested_end:.1f}")
                end_s = suggested_end

            rows.append({
                "label":      label,
                "rep":        rep_num,
                "start_ms":   round(start_s * 1000, 1),
                "end_ms":     round(end_s   * 1000, 1),
                "risk_class": rc,
                "notes":      "",
            })
            current_t = end_s + rest_between_s
            print(f"    Logged: {label} rep {rep_num}  ({start_s:.1f}–{end_s:.1f}s, {end_s - start_s:.1f}s)")

    _write_labels(rows, out_path)
    return rows


def _write_labels(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["start_ms"]))
    safe  = sum(1 for r in rows if r["risk_class"] == 0)
    risky = sum(1 for r in rows if r["risk_class"] == 1)
    excl  = sum(1 for r in rows if r["risk_class"] == -1)
    print(f"\n  ✓ labels.csv saved: {out_path}")
    print(f"    {len(rows)} segments | safe={safe} | risky={risky} | excluded/rest={excl}")


# VALIDATION

def validate_labels(labels_path: Path) -> bool:
    """
    Sanity-check a labels.csv before running the pipeline.

    Checks:
    - Required columns present
    - No segments with start >= end
    - No overlapping segments
    - At least one safe and one risky segment
    - Labels are in the movement catalogue
    """
    print(f"\n  Validating {labels_path} ...")
    df = pd.read_csv(labels_path)
    errors = []
    warnings = []

    # Required columns
    required = {"label", "rep", "start_ms", "end_ms", "risk_class"}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {missing}")

    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        return False

    # Duration checks
    bad_dur = df[df["end_ms"] <= df["start_ms"]]
    if len(bad_dur):
        errors.append(f"  {len(bad_dur)} segment(s) have end_ms <= start_ms")

    # Overlap check
    df_sorted = df.sort_values("start_ms").reset_index(drop=True)
    for i in range(1, len(df_sorted)):
        if df_sorted.loc[i, "start_ms"] < df_sorted.loc[i-1, "end_ms"]:
            warnings.append(
                f"  Overlap: {df_sorted.loc[i-1,'label']} ends at "
                f"{df_sorted.loc[i-1,'end_ms']:.0f}ms but "
                f"{df_sorted.loc[i,'label']} starts at {df_sorted.loc[i,'start_ms']:.0f}ms"
            )

    # Unknown labels
    unknown = [l for l in df["label"].unique() if l not in MOVEMENT_CATALOGUE]
    if unknown:
        warnings.append(f"  Unknown movement labels (will get risk_class=-1): {unknown}")

    # Class balance
    safe  = (df["risk_class"] == 0).sum()
    risky = (df["risk_class"] == 1).sum()
    excl  = (df["risk_class"] == -1).sum()
    if safe == 0:
        errors.append("No safe segments (risk_class=0) found")
    if risky == 0:
        errors.append("No risky segments (risk_class=1) found")

    # Template check, warn if placeholder text still present
    if "notes" in df.columns and df["notes"].astype(str).str.contains("TEMPLATE").any():
        warnings.append(
            "  'TEMPLATE' text found in notes — did you update the timestamps "
            "from the template? If yes, delete the notes column."
        )

    for e in errors:
        print(f"  ✗ {e}")
    for w in warnings:
        print(f"  ⚠ {w}")

    if not errors:
        print(f"  ✓ Validation passed")
        print(f"    {len(df)} segments | safe={safe} | risky={risky} | excluded={excl}")
        print(f"    Session span: {df['start_ms'].min()/1000:.1f}s – {df['end_ms'].max()/1000:.1f}s")
    return len(errors) == 0


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate protocol-based labels.csv for Phase 2 real participant sessions.\n"
            "These labels have zero circularity with IMU features and are the\n"
            "correct labelling strategy for real-data ML evaluation.\n\n"
            "Quick start:\n"
            "  # Step 1 — generate a template, fill in real timestamps:\n"
            "  python scripts/conversion/generate_protocol_labels.py --mode template --out_dir data/real/protocol_train/participant_01/session_001\n\n"
            "  # Step 2 — validate filled template:\n"
            "  python scripts/conversion/generate_protocol_labels.py --mode validate --out_dir data/real/protocol_train/participant_01/session_001\n\n"
            "  # Step 3 — run pipeline with protocol labels:\n"
            "  python -m signal_processing.pipeline --data_dir data/real/protocol_train --label_source protocol"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["template", "generate", "validate"],
        help=(
            "template  — write a labels.csv template with estimated timings to fill in\n"
            "generate  — interactively enter real timestamps as you record\n"
            "validate  — check an existing labels.csv for errors before running pipeline"
        ),
    )
    parser.add_argument(
        "--out_dir", required=True,
        help="Session directory (e.g. data/real/protocol_train/participant_01/session_001). "
             "labels.csv will be written here.",
    )
    parser.add_argument(
        "--timings", default=None,
        help="Path to a researcher-authored timings CSV (label, rep, start_s, end_s). "
             "Used with --mode generate when you have pre-recorded timings.",
    )
    args = parser.parse_args()

    out_dir    = Path(args.out_dir)
    labels_path = out_dir / "labels.csv"

    if args.mode == "template":
        generate_template(labels_path)

    elif args.mode == "generate":
        if args.timings:
            timings_path = Path(args.timings)
            if not timings_path.exists():
                print(f"[ERROR] Timings file not found: {timings_path}")
                sys.exit(1)
            from_timings_csv(timings_path, labels_path)
        else:
            interactive_generate(labels_path)

    elif args.mode == "validate":
        if not labels_path.exists():
            print(f"[ERROR] labels.csv not found at {labels_path}")
            print("  Run with --mode template first, then fill in timestamps.")
            sys.exit(1)
        ok = validate_labels(labels_path)
        sys.exit(0 if ok else 1)

    print()
    print("  Next step:")
    if args.mode in ("template",):
        print("  1. Open labels.csv, replace placeholder timestamps with real values")
        print("  2. python scripts/conversion/generate_protocol_labels.py --mode validate --out_dir", args.out_dir)
    elif args.mode == "generate":
        print("  1. python scripts/conversion/generate_protocol_labels.py --mode validate --out_dir", args.out_dir)
    if args.mode != "validate":
        print("  3. python -m signal_processing.pipeline --data_dir data/real/protocol_train --label_source protocol")
        print("  4. python ml/training/train_classifier.py --data_dir data/real/protocol_train --label_source protocol --cv_group participant")


if __name__ == "__main__":
    main()
