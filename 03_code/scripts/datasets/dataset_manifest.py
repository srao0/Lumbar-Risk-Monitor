#!/usr/bin/env python3
"""Create and validate derived-feature manifests for official Phase II runs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


MANIFEST_FILENAME = "dataset_manifest.json"
REQUIRED_SOURCE_FILES = ("imu_data.csv", "emg_data.csv", "labels.csv", "session_metadata.json")

PHASE_ALIASES = {
    "Phase II.1": "Phase II.A",
    "II.1": "Phase II.A",
    "Phase II.2": "Phase II.C",
    "II.2": "Phase II.C",
    "Phase II.A": "Phase II.A",
    "II.A": "Phase II.A",
    "Phase II.B": "Phase II.B",
    "II.B": "Phase II.B",
    "Phase II.C": "Phase II.C",
    "II.C": "Phase II.C",
    "Phase I": "Phase I",
}


def canonical_report_phase(value: str | None) -> str:
    """Map historical internal phase names to the official report phases."""
    raw = str(value or "").strip()
    if raw in PHASE_ALIASES:
        return PHASE_ALIASES[raw]
    without_prefix = raw.replace("Phase ", "")
    return PHASE_ALIASES.get(without_prefix, raw)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hash for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_complete_sessions(source_dir: Path) -> list[Path]:
    """Discover source session folders containing every required Phase II file."""
    return sorted(
        [
            path
            for path in source_dir.rglob("*")
            if path.is_dir()
            and all((path / name).exists() for name in REQUIRED_SOURCE_FILES)
        ],
        key=lambda path: str(path.relative_to(source_dir)).lower(),
    )


def discover_candidate_sessions(source_dir: Path) -> list[Path]:
    """Discover session-shaped folders even when collection output is incomplete."""
    return sorted(
        [
            path
            for path in source_dir.rglob("*")
            if path.is_dir()
            and (
                path.name.lower().startswith("session_")
                or any((path / name).exists() for name in REQUIRED_SOURCE_FILES)
            )
            and not any(child.is_dir() for child in path.iterdir())
        ],
        key=lambda path: str(path.relative_to(source_dir)).lower(),
    )


def participant_id_for_session(source_dir: Path, session_dir: Path) -> str:
    """Return participant id from a participant/session folder layout."""
    relative = session_dir.relative_to(source_dir)
    return relative.parts[-2] if len(relative.parts) >= 2 else session_dir.name


def write_dataset_manifest(
    source_dir: Path,
    session_dirs: list[Path],
    label_source: str,
    phase: str,
    operating_mode: str,
    feature_file: Path,
    command_used: str,
) -> Path:
    """Write metadata binding a combined feature table to its source sessions."""
    source_dir = source_dir.resolve()
    feature_file = feature_file.resolve()
    session_folders = [
        str(path.resolve().relative_to(source_dir)).replace("\\", "/")
        for path in session_dirs
    ]
    participant_ids = sorted(
        {participant_id_for_session(source_dir, path.resolve()) for path in session_dirs}
    )
    metadata = {
        "source_data_directory": str(source_dir),
        "session_folders_included": session_folders,
        "participant_ids": participant_ids,
        "session_count": len(session_folders),
        "participant_count": len(participant_ids),
        "label_source": label_source,
        "phase": phase,
        "operating_mode": operating_mode,
        "feature_file_path": str(feature_file),
        "feature_file_hash_sha256": file_sha256(feature_file),
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "command_used": command_used,
    }
    manifest_path = feature_file.parent / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return manifest_path


def validate_dataset_manifest(
    source_dir: Path,
    expected_phase: str,
    expected_label_source: str = "protocol",
    expected_operating_mode: str | None = None,
) -> tuple[dict | None, list[str]]:
    """Validate that a combined feature table is current and correctly scoped."""
    source_dir = source_dir.resolve()
    feature_file = source_dir / "combined_features.csv"
    manifest_path = source_dir / MANIFEST_FILENAME
    errors: list[str] = []

    if not feature_file.exists():
        errors.append(f"Missing combined feature file: {feature_file}")
    if not manifest_path.exists():
        errors.append(f"Missing dataset manifest: {manifest_path}")
        return None, errors

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        return None, errors + [f"Invalid dataset manifest JSON: {exc.msg}"]

    if str(manifest.get("source_data_directory", "")) != str(source_dir):
        errors.append("Manifest source_data_directory does not match requested data directory")
    if canonical_report_phase(manifest.get("phase")) != canonical_report_phase(expected_phase):
        errors.append(
            f"Manifest phase must be {expected_phase!r}; found {manifest.get('phase')!r}"
        )
    if manifest.get("label_source") != expected_label_source:
        errors.append(
            "Manifest label_source must be 'protocol' for official Phase II runs; "
            f"found {manifest.get('label_source')!r}"
        )
    if expected_operating_mode is not None and manifest.get("operating_mode") != expected_operating_mode:
        errors.append(
            f"Manifest operating_mode must be {expected_operating_mode!r}; "
            f"found {manifest.get('operating_mode')!r}"
        )

    if feature_file.exists():
        actual_hash = file_sha256(feature_file)
        expected_hash = manifest.get("feature_file_hash_sha256")
        if actual_hash != expected_hash:
            errors.append(
                "combined_features.csv SHA-256 does not match dataset_manifest.json; "
                "derived features are stale or have been modified"
            )
        if str(manifest.get("feature_file_path", "")) != str(feature_file):
            errors.append("Manifest feature_file_path does not match combined_features.csv")

    candidate_sessions = discover_candidate_sessions(source_dir)
    incomplete_sessions = [
        path
        for path in candidate_sessions
        if not all((path / name).exists() for name in REQUIRED_SOURCE_FILES)
    ]
    if incomplete_sessions:
        errors.append(
            "Source dataset contains incomplete session folders: "
            + ", ".join(str(path.relative_to(source_dir)) for path in incomplete_sessions)
        )
    sessions = discover_complete_sessions(source_dir)
    current_session_folders = [
        str(path.resolve().relative_to(source_dir)).replace("\\", "/") for path in sessions
    ]
    current_participants = sorted(
        {participant_id_for_session(source_dir, path.resolve()) for path in sessions}
    )
    if manifest.get("session_folders_included") != current_session_folders:
        errors.append("Manifest session folders do not match current source sessions")
    if manifest.get("participant_ids") != current_participants:
        errors.append("Manifest participant IDs do not match current source sessions")
    if manifest.get("session_count") != len(current_session_folders):
        errors.append("Manifest session count does not match current source sessions")
    if manifest.get("participant_count") != len(current_participants):
        errors.append("Manifest participant count does not match current source sessions")

    return manifest, errors
