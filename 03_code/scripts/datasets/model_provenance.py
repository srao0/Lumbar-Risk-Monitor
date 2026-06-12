#!/usr/bin/env python3
"""Read and validate Phase II.A frozen-model provenance metadata."""

from __future__ import annotations

# Make the project package importable when this file is run directly.
import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = _Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PKG_ROOT))


import argparse
import json
from pathlib import Path

from scripts.datasets.dataset_manifest import canonical_report_phase


PROVENANCE_FILENAME = "model_provenance.json"


def validate_phase2_model_provenance(
    models_dir: Path,
    varied_data_dir: Path | None = None,
    expected_operating_mode: str = "full_hybrid",
    allowed_cv_groups: set[str] | None = None,
) -> tuple[dict | None, list[str]]:
    """Return parsed metadata and validation failures for a frozen model folder."""
    errors: list[str] = []
    metadata_path = models_dir / PROVENANCE_FILENAME

    if not models_dir.exists():
        return None, [f"Models directory does not exist: {models_dir}"]
    if not metadata_path.exists():
        return None, [f"Missing provenance metadata: {metadata_path}"]

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        return None, [f"Invalid provenance metadata JSON: {exc.msg}"]

    allowed_cv_groups = allowed_cv_groups or {"participant"}
    expected = {
        "phase": "Phase II.A",
        "label_source": "protocol",
        "operating_mode": expected_operating_mode,
    }
    for field, value in expected.items():
        if field == "phase":
            matches = canonical_report_phase(metadata.get(field)) == canonical_report_phase(value)
        else:
            matches = metadata.get(field) == value
        if not matches:
            errors.append(
                f"Metadata {field} must be {value!r}; found {metadata.get(field)!r}"
            )
    if metadata.get("cv_group") not in allowed_cv_groups:
        errors.append(
            f"Metadata cv_group must be one of {sorted(allowed_cv_groups)!r}; "
            f"found {metadata.get('cv_group')!r}"
        )
    if expected_operating_mode == "full_hybrid" and metadata.get("contingency_only") is True:
        errors.append("Full-hybrid evaluation cannot use contingency-only fallback models")
    if expected_operating_mode == "imu_only_fallback" and metadata.get("contingency_only") is not True:
        errors.append("Fallback evaluation requires provenance marked contingency_only=true")

    training_data_dir = str(metadata.get("training_data_dir", ""))
    if not training_data_dir:
        errors.append("Metadata is missing training_data_dir")
    if "varied_test" in training_data_dir.lower():
        errors.append("Models were trained on varied_test data and are not valid frozen Phase II.A models")

    if varied_data_dir is not None:
        training_path = Path(training_data_dir)
        if training_path.resolve() == varied_data_dir.resolve():
            errors.append("Training data directory is the Phase II.C evaluation directory")

    model_types = metadata.get("model_types_trained")
    if not isinstance(model_types, list) or not model_types:
        errors.append("Metadata is missing model_types_trained")
    else:
        for model_type in model_types:
            if not list(models_dir.glob(f"{model_type}_fold*.joblib")):
                errors.append(f"No model file found for recorded model type {model_type!r}")

    return metadata, errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate that a model folder contains frozen Phase II.A models."
    )
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--varied_data_dir", default=None)
    parser.add_argument("--mode", choices=["full_hybrid", "imu_only_fallback"], default="full_hybrid")
    args = parser.parse_args()

    varied_dir = Path(args.varied_data_dir) if args.varied_data_dir else None
    allowed_groups = {"participant"} if args.mode == "full_hybrid" else {"participant", "session"}
    metadata, errors = validate_phase2_model_provenance(
        Path(args.models_dir),
        varied_dir,
        expected_operating_mode=args.mode,
        allowed_cv_groups=allowed_groups,
    )
    if errors:
        print("[FAIL] Frozen Phase II.A model provenance validation")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("[OK] Frozen Phase II.A model provenance validation")
    print(f"  Phase: {metadata['phase']}")
    print(f"  Mode: {metadata.get('operating_mode', 'unspecified')}")
    print(f"  CV group: {metadata.get('cv_group', 'unspecified')}")
    print(f"  Training data: {metadata['training_data_dir']}")
    print(f"  Models: {metadata['model_types_trained']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
