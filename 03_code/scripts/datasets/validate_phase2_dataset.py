#!/usr/bin/env python3
"""
Validate Phase II real protocol datasets before training or thesis reporting.

The checks are read-only and intentionally avoid heavy project imports. Use this
before running Phase II.A protocol processing and again before Phase II.C
held-out testing.
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
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from scripts.datasets.dataset_manifest import canonical_report_phase, validate_dataset_manifest
from scripts.conversion.generate_protocol_labels import MOVEMENT_CATALOGUE


REQUIRED_SESSION_FILES = {
    "imu_data.csv",
    "emg_data.csv",
    "labels.csv",
    "session_metadata.json",
}
PARTICIPANT_PATTERN = re.compile(r"^participant_\d+$")
SESSION_PATTERN = re.compile(r"^session_\d+$")
MIN_BASELINE_DURATION_MS = 60_000.0
WINDOW_MS = 2_000.0
STEP_MS = 1_000.0
MIN_BASELINE_WINDOWS = 5
LABEL_DURATION_TOLERANCE_MS = 2_000.0
SENSOR_DURATION_AGREEMENT_MS = 2_000.0
MIN_SENSOR_DURATION_MS = 60_000.0
SAMPLE_RATE_TOLERANCE_FRACTION = 0.10
EXPECTED_SAMPLE_RATES = {"IMU": 100.0, "EMG": 200.0}
OPERATING_MODES = {"full_hybrid", "imu_only_fallback"}
MIN_SAMPLE_COUNTS = {
    sensor: int(rate * MIN_SENSOR_DURATION_MS / 1000.0 * (1 - SAMPLE_RATE_TOLERANCE_FRACTION))
    for sensor, rate in EXPECTED_SAMPLE_RATES.items()
}
REQUIRED_OFFICIAL_PROTOCOL_LABELS = {
    label
    for label, details in MOVEMENT_CATALOGUE.items()
    if details["risk_class"] in (0, 1)
}

REQUIRED_LABEL_COLUMNS = {"label", "rep", "start_ms", "end_ms", "risk_class"}
REQUIRED_METADATA_KEYS = {
    "participant_id",
    "session_id",
    "phase",
    "protocol",
    "date",
}

REQUIRED_COMBINED_COLUMNS = {
    "session_id",
    "participant_id",
    "risk_class",
    "risk_class_protocol",
    "risk_class_signal",
    "movement_label",
    "imu_trunk_angle_peak",
    "emg_rms_LES",
}


@dataclass
class SessionCheck:
    path: Path
    participant_id: str
    session_id: str


@dataclass
class SensorQuality:
    name: str
    sample_count: int = 0
    duration_ms: float | None = None
    estimated_fs_hz: float | None = None
    is_stub: bool = False


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_csv_header(path: Path) -> set[str]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            return set(next(reader))
        except StopIteration:
            return set()


def inspect_sensor_quality(path: Path, sensor_name: str) -> tuple[SensorQuality, list[str]]:
    """Measure sensor completeness and reject empty, implausible or stub data."""
    quality = SensorQuality(name=sensor_name)
    problems = []
    rows = read_csv_rows(path)
    if not rows:
        return quality, [f"{path.name} is empty"]
    if "timestamp_ms" not in rows[0]:
        return quality, [f"{path.name} missing timestamp_ms column"]
    try:
        timestamps = [float(row["timestamp_ms"]) for row in rows if row.get("timestamp_ms", "") != ""]
    except ValueError:
        return quality, [f"{path.name} contains invalid timestamp_ms values"]
    if len(timestamps) < 2:
        return quality, [f"{path.name} has fewer than two timestamped samples"]

    quality.sample_count = len(timestamps)
    quality.duration_ms = max(timestamps) - min(timestamps)
    expected_fs = EXPECTED_SAMPLE_RATES[sensor_name]
    quality.estimated_fs_hz = (len(timestamps) - 1) / (quality.duration_ms / 1000.0) if quality.duration_ms > 0 else None
    if quality.duration_ms < MIN_SENSOR_DURATION_MS:
        problems.append(
            f"{path.name} duration is {quality.duration_ms / 1000:.1f}s; "
            f"requires at least {MIN_SENSOR_DURATION_MS / 1000:.0f}s"
        )
    if quality.sample_count < MIN_SAMPLE_COUNTS[sensor_name]:
        problems.append(
            f"{path.name} has {quality.sample_count} samples; "
            f"requires at least {MIN_SAMPLE_COUNTS[sensor_name]} for official collection"
        )
    if quality.estimated_fs_hz is None:
        problems.append(f"{path.name} has zero/invalid timestamp duration")
    else:
        lower = expected_fs * (1 - SAMPLE_RATE_TOLERANCE_FRACTION)
        upper = expected_fs * (1 + SAMPLE_RATE_TOLERANCE_FRACTION)
        if not (lower <= quality.estimated_fs_hz <= upper):
            problems.append(
                f"{path.name} estimated sample rate is {quality.estimated_fs_hz:.1f} Hz; "
                f"expected {expected_fs:.0f} Hz +/- {SAMPLE_RATE_TOLERANCE_FRACTION * 100:.0f}%"
            )

    if sensor_name == "EMG":
        signal_columns = ["emg_LES_mv", "emg_RES_mv", "emg_LOBL_mv", "emg_ROBL_mv"]
        missing_columns = [column for column in signal_columns if column not in rows[0]]
        if missing_columns:
            problems.append(f"{path.name} missing required EMG channels {missing_columns}")
        else:
            values = [row.get(column, "").strip().lower() for row in rows for column in signal_columns]
            quality.is_stub = all(value in {"", "nan"} for value in values)
    else:
        signal_columns = [
            "theta_PL_pitch", "theta_PL_roll", "theta_PL_yaw",
            "theta_LT_pitch", "theta_LT_roll", "theta_LT_yaw",
            "theta_TU_pitch", "theta_TU_roll", "theta_TU_yaw",
            "angvel_L3_sagittal",
        ]
        missing_columns = [column for column in signal_columns if column not in rows[0]]
        if missing_columns:
            problems.append(f"{path.name} missing required IMU motion channels {missing_columns}")
        else:
            try:
                values = [
                    float(row[column]) for row in rows for column in signal_columns
                    if row.get(column, "").strip() != ""
                ]
                finite_values = [value for value in values if math.isfinite(value)]
                quality.is_stub = not finite_values or all(value == 0.0 for value in finite_values)
            except ValueError:
                problems.append(f"{path.name} contains invalid motion signal values")
    if quality.is_stub:
        problems.append(f"{path.name} is a placeholder/stub sensor file and is forbidden in official Phase II")
    return quality, problems


def discover_sessions(root: Path) -> list[Path]:
    """Discover expected session directories, including incomplete sessions."""
    return sorted(
        [
            path
            for path in root.rglob("*")
            if path.is_dir()
            and (
                path.name.lower().startswith("session_")
                or any((path / name).exists() for name in REQUIRED_SESSION_FILES)
            )
            and not any(child.is_dir() for child in path.iterdir())
        ],
        key=lambda p: str(p.relative_to(root)).lower(),
    )


def infer_ids(root: Path, session_dir: Path) -> tuple[str, str]:
    """Derive (participant_id, session_id) from the folder layout, not metadata.

    Folder-derived IDs are the source of truth so that metadata can be
    cross-checked against them and tampering/mislabelling surfaces as a mismatch.
    """
    rel = session_dir.relative_to(root)
    session_id = "__".join(rel.parts)
    participant_id = rel.parts[-2] if len(rel.parts) >= 2 else session_dir.name
    return participant_id, session_id


def canonical_phase(value: str | None) -> str:
    value = str(value or "").strip().replace("Phase ", "")
    return f"Phase {value}" if value else ""


def validate_metadata(
    session_dir: Path,
    participant_id: str,
    session_id: str,
    expected_phase: str,
    expected_mode: str,
) -> list[str]:
    """Cross-check session_metadata.json against folder-derived IDs and the
    declared phase/mode, returning a list of human-readable problems.

    The mode-specific rules guard the two operating modes against silent
    contamination: full_hybrid demands real EMG and ``emg_available``, while
    imu_only_fallback must assert EMG was not used for inference.
    """
    problems = []
    path = session_dir / "session_metadata.json"
    if not path.exists():
        return ["missing session_metadata.json"]

    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"invalid session_metadata.json: {exc.msg}"]

    missing = sorted(REQUIRED_METADATA_KEYS - set(metadata))
    if missing:
        problems.append(f"metadata missing keys {missing}")

    if metadata.get("participant_id") != participant_id:
        problems.append(
            f"metadata participant_id={metadata.get('participant_id')!r} "
            f"does not match folder-derived {participant_id!r}"
        )

    if metadata.get("session_id") not in (session_id, session_dir.name):
        problems.append(
            f"metadata session_id={metadata.get('session_id')!r} "
            f"does not match folder-derived {session_id!r}"
        )
    if canonical_report_phase(canonical_phase(metadata.get("phase"))) != canonical_report_phase(expected_phase):
        problems.append(
            f"metadata phase={metadata.get('phase')!r} does not match {expected_phase!r}"
        )
    protocol = str(metadata.get("protocol", "")).strip()
    if not protocol:
        problems.append("metadata protocol must be populated")
    expected_report_phase = canonical_report_phase(expected_phase)
    if expected_report_phase == "Phase II.A" and protocol != "standard_phase2":
        problems.append(
            f"metadata protocol={protocol!r} must be 'standard_phase2' for Phase II.A"
        )
    if expected_report_phase in {"Phase II.A", "Phase II.C"}:
        if metadata.get("operating_mode") != expected_mode:
            problems.append(
                f"metadata operating_mode={metadata.get('operating_mode')!r} "
                f"does not match requested mode {expected_mode!r}"
            )
        if not metadata.get("imu_source"):
            problems.append("metadata imu_source is missing; official Phase II requires real IMU data")
        if expected_mode == "full_hybrid":
            if not metadata.get("emg_source"):
                problems.append("metadata emg_source is missing; full_hybrid Phase II requires real EMG data")
            if metadata.get("emg_available") is not True:
                problems.append("metadata emg_available must be true in full_hybrid mode")
        elif metadata.get("emg_used_for_inference") is not False:
            problems.append("metadata emg_used_for_inference must be false in imu_only_fallback mode")

    return problems


def validate_labels(
    session_dir: Path,
    recording_duration_ms: float | None = None,
    require_complete_protocol: bool = False,
) -> tuple[list[str], Counter, float, int]:
    """Validate labels.csv intervals, risk classes, and baseline coverage.

    Enforces that each label matches its catalogue risk_class, that intervals do
    not overlap or run past the sensor recording, that placeholder/UNKNOWN
    labels are rejected, and that there is enough static baseline to estimate the
    personal reference. ``require_complete_protocol`` additionally insists every
    official protocol block is present (used for Phase II.A). Returns the problem
    list plus risk-class counts, total baseline duration (ms) and baseline window
    count for downstream reporting.
    """
    problems = []
    labels_path = session_dir / "labels.csv"
    if not labels_path.exists():
        return ["missing labels.csv"], Counter(), 0.0, 0
    rows = read_csv_rows(labels_path)
    header = set(rows[0].keys()) if rows else read_csv_header(labels_path)

    missing = sorted(REQUIRED_LABEL_COLUMNS - header)
    if missing:
        return [f"labels.csv missing columns {missing}"], Counter(), 0.0, 0

    if not rows:
        return ["labels.csv is empty"], Counter(), 0.0, 0

    risk_counts: Counter = Counter()
    intervals: list[tuple[float, float, str]] = []
    baseline_duration_ms = 0.0
    baseline_windows = 0
    observed_labels: set[str] = set()

    for i, row in enumerate(rows, start=2):
        try:
            start_ms = float(row["start_ms"])
            end_ms = float(row["end_ms"])
        except ValueError:
            problems.append(f"labels.csv row {i}: non-numeric start_ms/end_ms")
            continue

        if end_ms <= start_ms:
            problems.append(f"labels.csv row {i}: end_ms <= start_ms")

        try:
            risk_class = int(float(row["risk_class"]))
        except ValueError:
            problems.append(f"labels.csv row {i}: invalid risk_class")
            continue

        if risk_class not in (-1, 0, 1):
            problems.append(f"labels.csv row {i}: risk_class must be -1, 0, or 1")

        label = row.get("label", "").strip()
        observed_labels.add(label)
        if not label or label.upper() == "UNKNOWN":
            problems.append(
                f"labels.csv row {i}: placeholder/UNKNOWN labels are forbidden in official Phase II"
            )
        catalogue_entry = MOVEMENT_CATALOGUE.get(label)
        if catalogue_entry is None:
            problems.append(f"labels.csv row {i}: unknown protocol task {label!r}")
        elif risk_class != catalogue_entry["risk_class"]:
            problems.append(
                f"labels.csv row {i}: {label!r} must have risk_class="
                f"{catalogue_entry['risk_class']}, found {risk_class}"
            )

        risk_counts[risk_class] += 1
        intervals.append((start_ms, end_ms, label))
        if label == "BASELINE_STATIC":
            duration_ms = max(0.0, end_ms - start_ms)
            baseline_duration_ms += duration_ms
            baseline_windows += max(0, int((duration_ms - WINDOW_MS) // STEP_MS) + 1)

    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1]:
            problems.append(
                f"labels.csv overlap: {previous[2]} ends at {previous[1]:.0f} ms "
                f"but {current[2]} starts at {current[0]:.0f} ms"
            )

    if intervals and recording_duration_ms is not None:
        final_label_end = max(interval[1] for interval in intervals)
        if final_label_end > recording_duration_ms + LABEL_DURATION_TOLERANCE_MS:
            problems.append(
                f"labels extend to {final_label_end / 1000:.1f}s but sensor data end at "
                f"{recording_duration_ms / 1000:.1f}s (tolerance "
                f"{LABEL_DURATION_TOLERANCE_MS / 1000:.0f}s)"
            )
    if require_complete_protocol:
        missing_blocks = sorted(REQUIRED_OFFICIAL_PROTOCOL_LABELS - observed_labels)
        if missing_blocks:
            problems.append(f"labels.csv missing expected official protocol blocks {missing_blocks}")

    if risk_counts[0] == 0:
        problems.append("labels.csv has no safe segments")
    if risk_counts[1] == 0:
        problems.append("labels.csv has no risky segments")
    if baseline_duration_ms < MIN_BASELINE_DURATION_MS:
        problems.append(
            f"baseline duration is {baseline_duration_ms / 1000:.1f}s; "
            f"requires at least {MIN_BASELINE_DURATION_MS / 1000:.0f}s"
        )
    if baseline_windows < MIN_BASELINE_WINDOWS:
        problems.append(
            f"baseline provides {baseline_windows} windows; "
            f"requires at least {MIN_BASELINE_WINDOWS}"
        )

    return problems, risk_counts, baseline_duration_ms, baseline_windows


def validate_session(
    root: Path,
    session_dir: Path,
    expected_phase: str,
    expected_mode: str = "full_hybrid",
) -> tuple[SessionCheck, list[str], Counter, dict[str, SensorQuality | float | None]]:
    """Run every per-session check and collect the results for one session.

    Combines folder-layout, metadata, sensor-quality and label checks. In
    imu_only_fallback mode EMG is not required, so EMG-quality problems and the
    IMU/EMG duration-agreement check are skipped. Returns the SessionCheck
    identity, the aggregated problem list, risk-class counts and a per-sensor
    quality report for printing.
    """
    participant_id, session_id = infer_ids(root, session_dir)
    problems = []
    rel = session_dir.relative_to(root)

    if len(rel.parts) != 2:
        problems.append("session folder must use participant_XX/session_YY layout")
    else:
        if not PARTICIPANT_PATTERN.fullmatch(rel.parts[0]):
            problems.append(f"invalid participant folder name {rel.parts[0]!r}")
        if not SESSION_PATTERN.fullmatch(rel.parts[1]):
            problems.append(f"invalid session folder name {rel.parts[1]!r}")

    required_files = REQUIRED_SESSION_FILES if expected_mode == "full_hybrid" else (
        REQUIRED_SESSION_FILES - {"emg_data.csv"}
    )
    missing_files = sorted(
        name for name in required_files
        if not (session_dir / name).exists()
    )
    if missing_files:
        problems.append(f"missing files {missing_files}")

    if (session_dir / "session_metadata.json").exists():
        problems.extend(validate_metadata(session_dir, participant_id, session_id, expected_phase, expected_mode))

    quality_report: dict[str, SensorQuality | float | None] = {
        "IMU": SensorQuality("IMU"),
        "EMG": SensorQuality("EMG"),
        "label_duration_ms": None,
    }
    sensor_durations = []
    imu_duration_ms = None
    emg_duration_ms = None
    for sensor_name, filename in (("IMU", "imu_data.csv"), ("EMG", "emg_data.csv")):
        sensor_path = session_dir / filename
        if sensor_path.exists():
            quality, quality_problems = inspect_sensor_quality(sensor_path, sensor_name)
            quality_report[sensor_name] = quality
            if sensor_name == "IMU" or expected_mode == "full_hybrid":
                problems.extend(quality_problems)
            if quality.duration_ms is not None:
                if sensor_name == "IMU":
                    imu_duration_ms = quality.duration_ms
                else:
                    emg_duration_ms = quality.duration_ms
                if sensor_name == "IMU" or expected_mode == "full_hybrid":
                    sensor_durations.append(quality.duration_ms)
    if expected_mode == "full_hybrid" and imu_duration_ms is not None and emg_duration_ms is not None and abs(imu_duration_ms - emg_duration_ms) > SENSOR_DURATION_AGREEMENT_MS:
        problems.append(
            f"IMU and EMG durations differ by {abs(imu_duration_ms - emg_duration_ms) / 1000:.1f}s; "
            f"maximum permitted difference is {SENSOR_DURATION_AGREEMENT_MS / 1000:.0f}s"
        )

    risk_counts = Counter()
    if (session_dir / "labels.csv").exists():
        recording_duration_ms = (
            min(sensor_durations) if expected_mode == "full_hybrid" and sensor_durations
            else imu_duration_ms
        )
        label_problems, risk_counts, _, _ = validate_labels(
            session_dir,
            recording_duration_ms,
            require_complete_protocol=(canonical_report_phase(expected_phase) == "Phase II.A"),
        )
        problems.extend(label_problems)
        try:
            label_rows = read_csv_rows(session_dir / "labels.csv")
            quality_report["label_duration_ms"] = max(float(row["end_ms"]) for row in label_rows)
        except (ValueError, KeyError):
            pass

    return SessionCheck(session_dir, participant_id, session_id), problems, risk_counts, quality_report


def validate_combined_features(
    root: Path,
    phase: str,
    min_class_fraction: float,
    expected_mode: str,
) -> list[str]:
    """Check the post-pipeline combined_features.csv schema, class balance and manifest.

    Runs only after signal_processing.pipeline has produced the windowed feature
    table: confirms the required columns exist, that both safe and risky
    protocol-labelled windows are present above ``min_class_fraction``, then
    defers to the dataset manifest validator for phase/mode integrity.
    """
    path = root / "combined_features.csv"
    if not path.exists():
        return ["combined_features.csv not found yet; run signal_processing.pipeline after session checks pass"]

    header = read_csv_header(path)
    missing = sorted(REQUIRED_COMBINED_COLUMNS - header)
    if missing:
        return [f"combined_features.csv missing columns {missing}"]
    rows = read_csv_rows(path)
    window_counts: Counter = Counter()
    for row in rows:
        try:
            value = int(float(row.get("risk_class_protocol", "-1")))
        except ValueError:
            return ["combined_features.csv contains invalid risk_class_protocol values"]
        if value in (0, 1):
            window_counts[value] += 1
    if window_counts[0] == 0 or window_counts[1] == 0:
        return ["combined_features.csv must contain safe and risky protocol-labelled windows"]
    minority_fraction = min(window_counts[0], window_counts[1]) / (
        window_counts[0] + window_counts[1]
    )
    if minority_fraction < min_class_fraction:
        return [
            "combined_features.csv has severe protocol-label window imbalance: "
            f"safe={window_counts[0]}, risky={window_counts[1]}, "
            f"minority_fraction={minority_fraction:.3f}"
        ]
    _, manifest_errors = validate_dataset_manifest(
        root,
        expected_phase=phase,
        expected_operating_mode=expected_mode,
    )
    return manifest_errors


def main() -> int:
    """CLI entry point: validate a Phase II dataset and report per-session + dataset verdicts.

    Discovers sessions, runs every per-session check, then aggregates
    participant count, dataset-level class balance and (unless ``--skip_combined``)
    the combined feature table. Prints a [QUALITY]/[OK]/[FAIL] log and returns a
    process exit code of 1 if any check fails, 0 otherwise — so it gates the
    phase runners and CI before training or thesis reporting.
    """
    parser = argparse.ArgumentParser(
        description="Read-only validation for Phase II real protocol datasets."
    )
    parser.add_argument(
        "--data_dir",
        default="data/real/protocol_train",
        help="Root Phase II dataset directory.",
    )
    parser.add_argument(
        "--expected_participants",
        type=int,
        default=10,
        help="Expected participant count for Phase II.A.",
    )
    parser.add_argument(
        "--phase",
        default="Phase II.A",
        choices=["Phase II.A", "Phase II.B", "Phase II.C", "Phase II.1", "Phase II.2"],
        help="Official study phase expected in session metadata and feature manifest.",
    )
    parser.add_argument(
        "--mode",
        default="full_hybrid",
        choices=sorted(OPERATING_MODES),
        help="Declared official system mode. Fallback sessions are validated separately from full-hybrid sessions.",
    )
    parser.add_argument(
        "--min_class_fraction",
        type=float,
        default=0.10,
        help="Hard-fail if the minority safe/risky segment fraction is below this value.",
    )
    parser.add_argument(
        "--skip_combined",
        action="store_true",
        help="Skip combined_features.csv schema check.",
    )
    args = parser.parse_args()

    root = Path(args.data_dir)
    if not root.exists():
        print(f"[FAIL] Dataset root not found: {root}")
        return 1

    sessions = discover_sessions(root)
    if not sessions:
        print(f"[FAIL] No session folders found under {root}")
        print("       Expected participant_XX/session_YY folders with imu_data.csv, emg_data.csv, labels.csv.")
        return 1

    failed = False
    participant_sessions: dict[str, list[str]] = defaultdict(list)
    total_risk_counts: Counter = Counter()

    print(f"[INFO] Found {len(sessions)} candidate session folder(s)")
    for session_dir in sessions:
        check, problems, risk_counts, quality_report = validate_session(root, session_dir, args.phase, args.mode)
        participant_sessions[check.participant_id].append(check.session_id)
        total_risk_counts.update(risk_counts)

        rel = session_dir.relative_to(root)
        imu_quality = quality_report["IMU"]
        emg_quality = quality_report["EMG"]
        label_duration_ms = quality_report["label_duration_ms"]
        print(f"\n[QUALITY] {rel}")
        print(f"  Mode: {args.mode}")
        for quality in (imu_quality, emg_quality):
            duration_text = (
                f"{quality.duration_ms / 1000:.1f}s" if quality.duration_ms is not None else "unavailable"
            )
            rate_text = (
                f"{quality.estimated_fs_hz:.1f} Hz" if quality.estimated_fs_hz is not None else "unavailable"
            )
            print(
                f"  {quality.name}: samples={quality.sample_count}, "
                f"duration={duration_text}, estimated_fs={rate_text}, stub={quality.is_stub}"
            )
        labels_text = (
            f"{label_duration_ms / 1000:.1f}s" if isinstance(label_duration_ms, float) else "unavailable"
        )
        print(f"  Labels: duration={labels_text}")
        if problems:
            failed = True
            print(f"[FAIL] {rel}")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print(
                f"[OK] {rel} "
                f"(safe={risk_counts[0]}, risky={risk_counts[1]}, excluded={risk_counts[-1]})"
            )

    participant_count = len(participant_sessions)
    if participant_count < args.expected_participants:
        failed = True
        print(
            f"[FAIL] Participant count is {participant_count}; "
            f"{args.phase} requires at least {args.expected_participants}."
        )
    else:
        print(f"[OK] Participant count: {participant_count}")

    if total_risk_counts[0] == 0 or total_risk_counts[1] == 0:
        failed = True
        print("[FAIL] Dataset-level labels must include both safe and risky segments")
    else:
        labelled_total = total_risk_counts[0] + total_risk_counts[1]
        minority_fraction = min(total_risk_counts[0], total_risk_counts[1]) / labelled_total
        if minority_fraction < args.min_class_fraction:
            failed = True
            print(
                f"[FAIL] Severe class imbalance: minority fraction={minority_fraction:.3f}; "
                f"minimum permitted is {args.min_class_fraction:.3f}"
            )
        status = "[FAIL]" if minority_fraction < args.min_class_fraction else "[OK]"
        print(
            f"{status} Dataset-level label balance: "
            f"safe={total_risk_counts[0]}, risky={total_risk_counts[1]}, excluded={total_risk_counts[-1]}"
        )

    if not args.skip_combined:
        combined_problems = validate_combined_features(
            root, args.phase, args.min_class_fraction, args.mode
        )
        if combined_problems:
            failed = True
            print("[FAIL] Combined feature table/manifest:")
            for problem in combined_problems:
                print(f"  - {problem}")
        else:
            print("[OK] combined_features.csv schema and dataset manifest integrity")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
