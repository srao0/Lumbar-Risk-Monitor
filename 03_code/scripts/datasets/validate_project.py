#!/usr/bin/env python3
"""
Read-only project validation checks for the Spinal Movement Risk Monitor.

Performs lightweight checks that do not write project files:
  - Python syntax check for all project .py files
  - optional dependency availability check
  - CSV schema checks for representative synthetic/real outputs
  - result/model artefact presence checks

It intentionally avoids importing scipy/sklearn-heavy modules so it can run
even before the full environment has been installed.
"""

from __future__ import annotations

import ast
import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

PY_SKIP_PARTS = {".git", "__pycache__", ".ipynb_checkpoints"}

REQUIRED_DEPS = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "joblib",
    "matplotlib",
    "serial",
    "brainflow",
]

REQUIRED_CSV_SCHEMAS = {
    "synthetic_combined": (
        ROOT / "data" / "synthetic" / "combined_features.csv",
        {
            "session_id",
            "window_centre_ms",
            "movement_label",
            "risk_class",
            "risk_class_protocol",
            "risk_class_signal",
            "imu_trunk_angle_peak",
            "emg_rms_LES",
        },
    ),
    "synthetic_session_labels": (
        ROOT / "data" / "synthetic" / "session_0001" / "labels.csv",
        {"label", "rep", "start_ms", "end_ms", "risk_class"},
    ),
}

OPTIONAL_CSV_SCHEMAS = {
    "real_combined": (
        [
            ROOT / "data" / "real" / "protocol_train" / "combined_features.csv",
            ROOT / "data" / "real" / "processed" / "combined_features.csv",
        ],
        {
            "session_id",
            "participant_id",
            "window_centre_ms",
            "movement_label",
            "risk_class",
            "risk_class_protocol",
            "risk_class_signal",
            "imu_trunk_angle_peak",
            "emg_rms_LES",
        },
    ),
}

ARTEFACTS = [
    ROOT / "ml" / "evaluation" / "loso_results.csv",
    ROOT / "ml" / "evaluation" / "summary_results.csv",
    ROOT / "ml" / "evaluation" / "feature_importance_RF.csv",
    ROOT / "ml" / "models",
]


def check_python_syntax() -> list[str]:
    """Parse every project .py file with ast to catch syntax errors.

    Uses ast.parse rather than importing modules so the check runs without the
    heavy scientific stack installed and without side effects from import-time
    code. Returns ``rel_path:line: message`` strings for any file that fails.
    """
    errors: list[str] = []
    for path in ROOT.rglob("*.py"):
        if any(part in PY_SKIP_PARTS for part in path.parts):
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            rel = path.relative_to(ROOT)
            errors.append(f"{rel}:{exc.lineno}: {exc.msg}")
    return errors


def check_dependencies() -> list[str]:
    missing = []
    for dep in REQUIRED_DEPS:
        if importlib.util.find_spec(dep) is None:
            missing.append(dep)
    return missing


def read_header(path: Path) -> set[str] | None:
    if not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            return set(next(reader))
        except StopIteration:
            return set()


def check_csv_schemas() -> tuple[list[str], list[str]]:
    """Check expected CSV outputs have the required columns.

    REQUIRED schemas (synthetic outputs that always ship with the repo) become
    hard problems; OPTIONAL schemas (real-data outputs that may not be present in
    a fresh clone) only ever become warnings. Returns (problems, warnings).
    """
    problems: list[str] = []
    warnings: list[str] = []

    for name, (path, required_cols) in REQUIRED_CSV_SCHEMAS.items():
        header = read_header(path)
        if header is None:
            problems.append(f"{name}: missing {path.relative_to(ROOT)}")
            continue
        missing = sorted(required_cols - header)
        if missing:
            problems.append(
                f"{name}: {path.relative_to(ROOT)} missing columns {missing}"
            )

    for name, (paths, required_cols) in OPTIONAL_CSV_SCHEMAS.items():
        existing = [path for path in paths if path.exists()]
        if not existing:
            choices = ", ".join(str(path.relative_to(ROOT)) for path in paths)
            warnings.append(f"{name}: not present yet; checked {choices}")
            continue

        path = existing[0]
        header = read_header(path)
        missing = sorted(required_cols - (header or set()))
        if missing:
            warnings.append(
                f"{name}: {path.relative_to(ROOT)} missing columns {missing}"
            )

    return problems, warnings


def check_artefacts() -> list[str]:
    missing = []
    for path in ARTEFACTS:
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
    return missing


def main() -> int:
    """CLI entry point: run all read-only project checks and report a verdict.

    Runs syntax, dependency, CSV-schema and artefact checks in turn. Only syntax
    errors and required-schema problems fail the run; missing dependencies and
    absent real-data artefacts are warnings so the check still passes on a fresh
    clone before the environment or real data exist. Returns 1 on failure, else 0.
    """
    failed = False

    syntax_errors = check_python_syntax()
    if syntax_errors:
        failed = True
        print("[FAIL] Python syntax")
        for err in syntax_errors:
            print(f"  - {err}")
    else:
        print("[OK] Python syntax")

    missing_deps = check_dependencies()
    if missing_deps:
        print("[WARN] Missing optional/runtime dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
    else:
        print("[OK] Dependencies available")

    csv_problems, csv_warnings = check_csv_schemas()
    if csv_problems:
        failed = True
        print("[FAIL] CSV schemas")
        for problem in csv_problems:
            print(f"  - {problem}")
    else:
        print("[OK] CSV schemas")
    if csv_warnings:
        print("[WARN] Optional CSV schemas:")
        for warning in csv_warnings:
            print(f"  - {warning}")

    missing_artefacts = check_artefacts()
    if missing_artefacts:
        print("[WARN] Missing result/model artefacts:")
        for item in missing_artefacts:
            print(f"  - {item}")
    else:
        print("[OK] Result/model artefacts present")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
