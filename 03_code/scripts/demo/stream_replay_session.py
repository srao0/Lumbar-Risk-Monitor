#!/usr/bin/env python3
"""
stream_replay_session.py
========================
Stream an already-generated replay session into a second folder one window at a
time.

This is a live-demo bridge, not a new inference path. It reads the completed
outputs produced by scripts/replay_recorded_session.py and progressively writes
replay_predictions.csv plus replay_summary.json into --out_session. The
dashboard can watch that output folder to show the session changing in front of
the examiner, while the conversion, feature extraction, model loading,
classify_window and FIS logic remain the existing offline pipeline.

Example
-------
    py scripts/stream_replay_session.py ^
        --source_session data/real/replay_full_hybrid/_smoketest_p11_s001_230s ^
        --out_session data/real/live_demo_full_hybrid/_smoketest_p11_s001_230s ^
        --speed 1

Then view:
    py -m streamlit run scripts/replay_dashboard.py -- ^
        --session data/real/live_demo_full_hybrid/_smoketest_p11_s001_230s
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RISK_LEVELS = ["Safe", "Cautious", "Risky"]
REQUIRED_FILES = ("replay_predictions.csv", "replay_summary.json")


def _replace_with_retry(tmp: Path, path: Path, attempts: int = 20) -> None:
    """Atomically swap tmp into place, retrying on PermissionError — the dashboard polls these files mid-write, and Windows briefly locks the target during a read."""
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text via a temp file + atomic replace so the watching dashboard never reads a half-written summary."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    _replace_with_retry(tmp, path)


def _atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    """Same temp-then-replace guard as _atomic_write_text, for the growing predictions CSV."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    _replace_with_retry(tmp, path)


def _risk_counts(pred: pd.DataFrame) -> dict[str, int]:
    """Tally windows per traffic-light level, always returning all three keys so the dashboard never KeyErrors on an empty/early frame."""
    if "predicted" not in pred.columns:
        return {level: 0 for level in RISK_LEVELS}
    counts = pred["predicted"].value_counts().to_dict()
    return {level: int(counts.get(level, 0)) for level in RISK_LEVELS}


def _risk_percentages(counts: dict[str, int], n: int) -> dict[str, float]:
    """Convert the level tally to percentages of the windows seen so far (0 when nothing has streamed yet)."""
    return {
        level: round(100.0 * counts[level] / n, 1) if n else 0.0
        for level in RISK_LEVELS
    }


def _highest_risk(pred: pd.DataFrame) -> dict | None:
    """Pull out the single highest-R_total window for the dashboard's headline, or None before any scored window exists."""
    if pred.empty or "R_total" not in pred.columns or pred["R_total"].isna().all():
        return None
    top = pred.loc[pred["R_total"].idxmax()]
    return {
        "window_idx": int(top.get("window_idx", 0)),
        "R_total": round(float(top.get("R_total", 0.0)), 4),
        "traffic_light": top.get("traffic_light", ""),
        "predicted": top.get("predicted", ""),
        "movement_label": top.get("movement_label", "UNKNOWN"),
        "window_start_ms": float(top.get("window_start_ms", top.get("window_start", 0.0))),
    }


def _duration_s(pred: pd.DataFrame) -> float:
    """Session span in seconds from the first to last visible window, tolerating either the _ms or bare window-start/end column names."""
    if pred.empty:
        return 0.0
    start_col = "window_start_ms" if "window_start_ms" in pred.columns else "window_start"
    end_col = "window_end_ms" if "window_end_ms" in pred.columns else "window_end"
    if start_col not in pred.columns or end_col not in pred.columns:
        return 0.0
    return round((float(pred[end_col].max()) - float(pred[start_col].min())) / 1000.0, 1)


def _build_live_summary(source_summary: dict, source_session: Path, out_session: Path, visible: pd.DataFrame) -> dict:
    """Recompute the replay_summary.json for the windows revealed so far — inherits the source summary's static fields and overlays the live counts/duration/highest-risk."""
    n = len(visible)
    counts = _risk_counts(visible)
    summary = dict(source_summary)
    summary.update(
        {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "out_session": str(out_session),
            "source_replay_session": str(source_session),
            "streaming_replay": True,
            "streaming_status": "running",
            "total_windows": int(n),
            "source_total_windows": int(source_summary.get("total_windows", n)),
            "duration_s": _duration_s(visible),
            "risk_counts": counts,
            "risk_percentages": _risk_percentages(counts, n),
            "highest_risk_window": _highest_risk(visible),
            "generated_files": ["replay_predictions.csv", "replay_summary.json"],
        }
    )
    return summary


def _copy_optional_assets(source_session: Path, out_session: Path) -> None:
    """Bring across the static side artefacts (timeline PNG, feature matrix) once, so the dashboard's Export/Signals tabs work against the live folder too."""
    for name in ("replay_timeline.png", "feature_matrix.csv"):
        src = source_session / name
        dst = out_session / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def stream_session(source_session: Path, out_session: Path, speed: float, loop: bool, reset: bool) -> None:
    """Reveal the pre-computed replay one window at a time into out_session at the requested speed — the live-demo bridge that re-uses offline outputs rather than re-running inference."""
    missing = [name for name in REQUIRED_FILES if not (source_session / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required file(s) in {source_session}: {', '.join(missing)}"
        )

    if out_session.exists() and reset:
        for name in ("replay_predictions.csv", "replay_summary.json"):
            path = out_session / name
            if path.exists():
                path.unlink()
    out_session.mkdir(parents=True, exist_ok=True)
    _copy_optional_assets(source_session, out_session)

    full_pred = pd.read_csv(source_session / "replay_predictions.csv")
    source_summary = json.loads((source_session / "replay_summary.json").read_text(encoding="utf-8"))
    delay_s = 1.0 / speed if speed > 0 else 0.0

    print("=" * 64)
    print("Spinal Movement Risk - Live Replay Stream")
    print(f"  Source replay : {source_session}")
    print(f"  Live output   : {out_session}")
    print(f"  Windows       : {len(full_pred)}")
    print(f"  Speed         : {speed:g} windows/sec")
    print("=" * 64)
    print("Open the dashboard against the live output folder and enable Watch session folder.")

    while True:
        empty = full_pred.iloc[:0].copy()
        _atomic_write_csv(out_session / "replay_predictions.csv", empty)
        _atomic_write_text(
            out_session / "replay_summary.json",
            json.dumps(_build_live_summary(source_summary, source_session, out_session, empty), indent=2),
        )

        for idx in range(len(full_pred)):
            visible = full_pred.iloc[: idx + 1].copy()
            _atomic_write_csv(out_session / "replay_predictions.csv", visible)
            _atomic_write_text(
                out_session / "replay_summary.json",
                json.dumps(_build_live_summary(source_summary, source_session, out_session, visible), indent=2),
            )
            row = visible.iloc[-1]
            print(
                f"  window {idx + 1:03d}/{len(full_pred):03d} "
                f"{row.get('traffic_light', ''):>5} "
                f"{row.get('predicted', ''):<8} "
                f"R_total={float(row.get('R_total', 0.0)):.3f}"
            )
            if delay_s:
                time.sleep(delay_s)

        final_summary = _build_live_summary(source_summary, source_session, out_session, full_pred)
        final_summary["streaming_status"] = "complete"
        _atomic_write_text(out_session / "replay_summary.json", json.dumps(final_summary, indent=2))
        print("Stream complete.")

        if not loop:
            return
        print("Looping from the start...")


def main(argv=None) -> int:
    """CLI entry — parse the source/output sessions and streaming options, then drive stream_session."""
    parser = argparse.ArgumentParser(
        description="Progressively write an existing replay session for live dashboard demonstration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source_session", required=True, help="Completed replay session directory.")
    parser.add_argument("--out_session", required=True, help="Live output folder watched by the dashboard.")
    parser.add_argument("--speed", type=float, default=1.0, help="Windows written per second. Use 1 for real-time replay.")
    parser.add_argument("--loop", action="store_true", help="Restart from the first window after reaching the end.")
    parser.add_argument("--no_reset", action="store_true", help="Do not delete existing live output files before starting.")
    args = parser.parse_args(argv)

    if args.speed < 0:
        parser.error("--speed must be >= 0")

    stream_session(
        source_session=Path(args.source_session),
        out_session=Path(args.out_session),
        speed=args.speed,
        loop=args.loop,
        reset=not args.no_reset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
