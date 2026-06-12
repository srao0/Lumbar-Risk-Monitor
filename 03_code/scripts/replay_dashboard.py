#!/usr/bin/env python3
"""
replay_dashboard.py
===================
Spinal Movement Risk Monitor — FYP 2025/26 | Imperial College London

A visible Streamlit dashboard for the final demonstration. It reads an *already
processed* replay session directory (produced by replay_recorded_session.py) and
renders it. It does NOT silently re-run conversion, feature extraction or
inference — it only loads replay_predictions.csv and replay_summary.json.

Run
---
    streamlit run scripts/replay_dashboard.py -- --session <replay_session_dir>

e.g.
    streamlit run scripts/replay_dashboard.py -- \
        --session data/real/replay_full_hybrid/participant_11/session_001

If --session is omitted you can type/paste the path in the sidebar.

Expected files in the session directory:
    replay_predictions.csv   (required)
    replay_summary.json      (required)
    replay_timeline.png      (optional)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "Streamlit, or one of its web-server dependencies, could not import.\n"
        "From the project root, install the pinned dashboard dependencies with:\n"
        "    python -m pip install -r requirements.txt\n"
        f"\nOriginal import error: {exc}\n"
    )
    raise


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (read-only, cached)
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_FILES = ("replay_predictions.csv", "replay_summary.json")

RISK_COLOURS = {"Safe": "#4CAF50", "Cautious": "#FF9800", "Risky": "#F44336"}
LIGHT_COLOURS = {"GREEN": "#4CAF50", "AMBER": "#FF9800", "RED": "#F44336"}
WINDOW_STRIDE_S = 1.0


def _default_session_from_argv() -> str:
    """Parse --session from the args after Streamlit's `--` separator."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--session", default="")
    args, _ = parser.parse_known_args()
    return args.session


def _session_cache_token(session_dir: str) -> tuple:
    """
    Return a small fingerprint for the files the dashboard reads.

    Streamlit caches by function arguments, so this lets a live-updating replay
    folder refresh when replay_predictions.csv or replay_summary.json changes.
    """
    d = Path(session_dir)
    token = []
    for fname in REQUIRED_FILES:
        path = d / fname
        if path.exists():
            stat = path.stat()
            token.append((fname, stat.st_mtime_ns, stat.st_size))
        else:
            token.append((fname, None, None))
    return tuple(token)


@st.cache_data(show_spinner=False)
def load_session(session_dir: str, cache_token: tuple):
    """Load predictions + summary from a processed replay directory (read-only)."""
    del cache_token
    d = Path(session_dir)
    pred_path = d / "replay_predictions.csv"
    summary_path = d / "replay_summary.json"
    missing = [f for f in REQUIRED_FILES if not (d / f).exists()]
    if missing:
        return None, None, missing
    try:
        pred = pd.read_csv(pred_path)
        pred = _normalise_predictions(pred)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, None, [f"could not read live replay outputs yet ({exc})"]
    return pred, summary, []


def _normalise_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    """
    Make the predictions dataframe robust to optional/missing columns.

    * Guarantee a `window_idx` column (charts and selection rely on it).
    * Prefer `window_start`/`window_end`; if absent, alias the `_ms` variants.
    """
    pred = pred.copy()
    if "window_idx" not in pred.columns:
        pred["window_idx"] = range(len(pred))
    if "window_start" not in pred.columns and "window_start_ms" in pred.columns:
        pred["window_start"] = pred["window_start_ms"]
    if "window_end" not in pred.columns and "window_end_ms" in pred.columns:
        pred["window_end"] = pred["window_end_ms"]
    return pred


def _has(df: pd.DataFrame, col: str) -> bool:
    """True only if the column exists and carries at least one real value — guards charts against all-NaN optional columns."""
    return col in df.columns and df[col].notna().any()


def _format_window_time(row: pd.Series) -> str:
    """Human-readable start–end span for a window, falling back to the _ms columns and degrading gracefully if neither is present."""
    start = row.get("window_start", row.get("window_start_ms", float("nan")))
    end = row.get("window_end", row.get("window_end_ms", float("nan")))
    try:
        return f"{float(start) / 1000.0:.1f}s to {float(end) / 1000.0:.1f}s"
    except (TypeError, ValueError):
        return "time unavailable"


def _score_text(row: pd.Series, col: str) -> str:
    """Format a risk score to 3 dp for a metric tile, showing '--' when the score is missing (e.g. R_EMG in IMU-only mode)."""
    val = row.get(col, float("nan"))
    return "--" if pd.isna(val) else f"{float(val):.3f}"


def _window_header(row: pd.Series) -> None:
    """Render the window heading with the predicted label coloured by its traffic light."""
    light = str(row.get("traffic_light", "")).upper()
    colour = LIGHT_COLOURS.get(light, "#777")
    st.markdown(
        f"### Window #{int(row['window_idx'])} - "
        f"<span style='color:{colour}'>{row.get('predicted', '?')}</span>",
        unsafe_allow_html=True,
    )


def _render_window(row: pd.Series, mode: str) -> None:
    """Draw the full per-window panel — scores, movement, and both the engineering and plain-English explanations; R_EMG reads 'unused' in IMU-only fallback mode."""
    light = str(row.get("traffic_light", "")).upper()
    _window_header(row)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Traffic light", light or "--")
    c2.metric("R_total", _score_text(row, "R_total"))
    c3.metric("Movement", str(row.get("movement_label", "?")))
    c4.metric("Window time", _format_window_time(row))

    d1, d2 = st.columns(2)
    d1.metric("R_IMU", _score_text(row, "R_IMU"))
    if mode == "imu_only_fallback":
        d2.metric("R_EMG", "unused")
    else:
        d2.metric("R_EMG", _score_text(row, "R_EMG"))

    st.subheader("Engineering explanation")
    st.write(row.get("engineering_reason", "--"))

    st.subheader("Plain-English explanation")
    st.info(row.get("layman_reason", "--"))


def _score_columns(pred: pd.DataFrame, mode: str) -> list[str]:
    """Pick which risk-score series to chart — R_EMG only joins R_total/R_IMU in full-hybrid mode, since the IMU-only fallback never produces it."""
    cols = [c for c in ["R_total", "R_IMU"] if _has(pred, c)]
    if mode == "full_hybrid" and _has(pred, "R_EMG"):
        cols.append("R_EMG")
    return cols


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Build and run the Streamlit dashboard — sidebar controls plus the six tabs over one processed replay session; pure presentation, no inference happens here."""
    st.set_page_config(page_title="Spinal Movement Risk Replay", layout="wide")
    st.title("Spinal Movement Risk Replay Dashboard")
    st.caption(
        "Movement-risk feedback within this system only — not a medical diagnosis."
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.header("Session")
    default_session = st.session_state.get("session_dir", _default_session_from_argv())
    session_dir = st.sidebar.text_input(
        "Processed replay session path", value=default_session,
        help="Directory containing replay_predictions.csv and replay_summary.json",
    )
    st.session_state["session_dir"] = session_dir

    if not session_dir:
        st.info("Enter a processed replay session path in the sidebar to begin.")
        st.stop()

    watch_updates = st.sidebar.toggle(
        "Watch session folder",
        value=False,
        help="Refresh automatically when replay outputs are being written by another process.",
    )
    refresh_s = st.sidebar.select_slider(
        "Watch refresh",
        options=[0.25, 0.5, 1.0, 2.0, 5.0],
        value=1.0,
        format_func=lambda x: f"{x:g}s",
        disabled=not watch_updates,
    )

    pred, summary, missing = load_session(session_dir, _session_cache_token(session_dir))
    if missing:
        st.error(
            f"Missing required file(s) in `{session_dir}`: {', '.join(missing)}.\n\n"
            "Generate them first with `scripts/replay_recorded_session.py`, "
            "`scripts/stream_replay_session.py`, or `scripts/live_risk_pipeline.py`."
        )
        if watch_updates:
            time.sleep(float(refresh_s))
            st.rerun()
        st.stop()

    mode = summary.get("operating_mode", "unknown")
    emg_available = bool(summary.get("emg_available", False))

    # Mode badge
    if mode == "imu_only_fallback":
        st.sidebar.warning("Mode: IMU-only fallback\n\n(decision uses movement only — no EMG)")
    else:
        st.sidebar.success(
            f"Mode: full hybrid\n\nEMG available: {'yes' if emg_available else 'no'}"
        )

    # A live-watched session can shrink between reruns, so re-clamp the stored
    # play/selection indices to the current window count to avoid out-of-range reads.
    n = len(pred)
    max_idx = max(n - 1, 0)
    if "play_idx" not in st.session_state or st.session_state["play_idx"] >= max(n, 1):
        st.session_state["play_idx"] = 0
    st.session_state["play_idx"] = min(max(int(st.session_state["play_idx"]), 0), max_idx)

    if "selected_window" not in st.session_state or st.session_state["selected_window"] >= max(n, 1):
        st.session_state["selected_window"] = st.session_state["play_idx"]
    st.session_state["selected_window"] = min(
        max(int(st.session_state["selected_window"]), 0),
        max_idx,
    )

    st.sidebar.header("Replay")
    play = st.sidebar.toggle("Play session", value=False, help="Step through saved windows like a live monitor.")
    replay_speed = st.sidebar.select_slider(
        "Replay speed",
        options=[0.25, 0.5, 1.0, 2.0, 4.0],
        value=1.0,
        format_func=lambda x: f"{x:g}x",
        help="1x advances one 1-second feature stride per second.",
    )
    col_prev, col_reset, col_next = st.sidebar.columns(3)
    if col_prev.button("Prev", disabled=not n):
        st.session_state["play_idx"] = max(0, st.session_state["play_idx"] - 1)
        st.session_state["selected_window"] = st.session_state["play_idx"]
    if col_reset.button("Reset", disabled=not n):
        st.session_state["play_idx"] = 0
        st.session_state["selected_window"] = 0
    if col_next.button("Next", disabled=not n):
        st.session_state["play_idx"] = min(max_idx, st.session_state["play_idx"] + 1)
        st.session_state["selected_window"] = st.session_state["play_idx"]

    if play:
        st.session_state["selected_window"] = st.session_state["play_idx"]

    sel = (
        st.sidebar.slider(
            "Selected window",
            0,
            max_idx,
            key="selected_window",
            disabled=play,
        )
        if n
        else 0
    )
    if n and not play:
        st.session_state["play_idx"] = int(sel)
    sel = int(st.session_state["play_idx"]) if n else 0
    st.sidebar.caption(f"{n} windows total")

    tab_live, tab_overview, tab_timeline, tab_window, tab_signals, tab_export = st.tabs(
        ["Live Replay", "Overview", "Timeline", "Window Explanation", "Signals", "Export"]
    )

    # ── Live Replay ──────────────────────────────────────────────────────────
    with tab_live:
        st.subheader("Session Playback")
        if not n:
            st.info("No windows to replay.")
        else:
            row = pred.iloc[sel]
            elapsed = row.get("window_start", row.get("window_start_ms", 0.0))
            try:
                elapsed_s = max(0.0, float(elapsed) / 1000.0)
            except (TypeError, ValueError):
                elapsed_s = 0.0

            st.progress((sel + 1) / n)
            p1, p2, p3 = st.columns(3)
            p1.metric("Playback position", f"{sel + 1} / {n}")
            p2.metric("Elapsed session time", f"{elapsed_s:.1f}s")
            p3.metric("Replay speed", f"{replay_speed:g}x")

            _render_window(row, mode)

            st.subheader("Risk timeline so far")
            visible = pred.iloc[: sel + 1].copy()
            score_cols = _score_columns(visible, mode)
            if score_cols:
                st.line_chart(visible.set_index("window_idx")[score_cols])
            else:
                st.caption("No score columns available for this session.")

            if play:
                # Auto-advance the playhead (wrapping at the end) and force a rerun;
                # Streamlit has no timer, so a sleep + st.rerun() is how the "live monitor"
                # animation is driven, paced by the chosen replay speed.
                if sel < n - 1:
                    st.session_state["play_idx"] = sel + 1
                else:
                    st.session_state["play_idx"] = 0
                time.sleep(max(0.05, WINDOW_STRIDE_S / float(replay_speed)))
                st.rerun()

    # ── Overview ─────────────────────────────────────────────────────────────
    with tab_overview:
        pct = summary.get("risk_percentages", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total windows", summary.get("total_windows", n))
        c2.metric("Duration (s)", summary.get("duration_s", "—"))
        c3.metric("Mode", mode)
        c4.metric("EMG available", "Yes" if emg_available else "No")

        st.subheader("Risk distribution")
        d1, d2, d3 = st.columns(3)
        d1.metric("Safe", f"{pct.get('Safe', 0)}%")
        d2.metric("Cautious", f"{pct.get('Cautious', 0)}%")
        d3.metric("Risky", f"{pct.get('Risky', 0)}%")
        counts = summary.get("risk_counts", {})
        st.bar_chart(pd.DataFrame(
            {"windows": [counts.get(k, 0) for k in ("Safe", "Cautious", "Risky")]},
            index=["Safe", "Cautious", "Risky"],
        ))

        h = summary.get("highest_risk_window")
        if h:
            st.subheader("Highest-risk window")
            st.write(
                f"Window **#{h.get('window_idx', '?')}** — {h.get('traffic_light', '?')} "
                f"({h.get('predicted', '?')}), R_total = **{h.get('R_total', '?')}**, "
                f"movement: *{h.get('movement_label', '?')}*"
            )

        acc = summary.get("accuracy_vs_protocol_labels")
        if acc:
            st.caption(
                f"Binary accuracy vs recorded protocol labels: "
                f"{acc.get('binary_accuracy', '?')} over "
                f"{acc.get('n_labelled_windows', '?')} labelled windows."
            )

        st.subheader("Model & provenance")
        st.code(summary.get("model_directory_used", "—"), language="text")
        st.write(f"Raw session: `{summary.get('raw_session', '—')}`")

        st.subheader("Preflight")
        pf = summary.get("preflight", {})
        st.write(f"Status: **{pf.get('status', '?')}**  ·  EMG board: `{pf.get('emg_board')}`")
        warns = pf.get("warnings", [])
        errs = pf.get("errors", [])
        if errs:
            for e in errs:
                st.error(e)
        if warns:
            for w in warns:
                st.warning(w)
        if not warns and not errs:
            st.success("No preflight warnings.")

    # ── Timeline ─────────────────────────────────────────────────────────────
    with tab_timeline:
        st.subheader("Risk class over time")
        if _has(pred, "predicted"):
            level_y = {"Safe": 0, "Cautious": 1, "Risky": 2}
            tl = pred[["window_idx", "predicted"]].copy()
            tl["risk_level_num"] = tl["predicted"].map(level_y)
            st.line_chart(tl.set_index("window_idx")[["risk_level_num"]])
            st.caption("0 = Safe, 1 = Cautious, 2 = Risky")

        st.subheader("Risk scores over time")
        score_cols = _score_columns(pred, mode)
        if score_cols:
            st.line_chart(pred.set_index("window_idx")[score_cols])
            if mode == "imu_only_fallback":
                st.caption("R_EMG is not shown — IMU-only fallback uses no EMG.")
            elif "R_EMG" not in score_cols:
                st.caption("R_EMG unavailable for this session.")

    # ── Window Explanation ────────────────────────────────────────────────────
    with tab_window:
        if not n:
            st.info("No windows to display.")
        else:
            row = pred.iloc[sel]
            _render_window(row, mode)

            st.subheader("Key contributing features")
            feat_cols = [c for c in pred.columns if c.startswith(("imu_", "emg_"))]
            if feat_cols:
                feat_view = pd.DataFrame({
                    "feature": feat_cols,
                    "value": [row.get(c) for c in feat_cols],
                })
                st.dataframe(feat_view, use_container_width=True, hide_index=True)

    # ── Signals ──────────────────────────────────────────────────────────────
    with tab_signals:
        st.subheader("IMU features over time")
        imu_cols = [c for c in pred.columns if c.startswith("imu_") and _has(pred, c)]
        default_imu = [c for c in ("imu_trunk_angle_peak", "imu_angvel_peak",
                                    "imu_time_in_risk_zone") if c in imu_cols]
        chosen_imu = st.multiselect("IMU features", imu_cols, default=default_imu or imu_cols[:3])
        if chosen_imu:
            st.line_chart(pred.set_index("window_idx")[chosen_imu])

        st.subheader("EMG features over time")
        if mode == "imu_only_fallback":
            st.caption("IMU-only fallback — no EMG features were used.")
        else:
            emg_cols = [c for c in pred.columns if c.startswith("emg_") and _has(pred, c)]
            if emg_cols:
                default_emg = [c for c in ("emg_rms_LES", "emg_rms_RES") if c in emg_cols]
                chosen_emg = st.multiselect("EMG features", emg_cols,
                                            default=default_emg or emg_cols[:2])
                if chosen_emg:
                    st.line_chart(pred.set_index("window_idx")[chosen_emg])
            else:
                st.caption("No EMG features available in this session.")

    # ── Export ───────────────────────────────────────────────────────────────
    with tab_export:
        st.subheader("Generated files")
        d = Path(session_dir)
        for fname in summary.get("generated_files", []):
            fpath = d / fname
            exists = fpath.exists()
            st.write(f"{'✅' if exists else '⚠️'} `{fname}`"
                     f"{'' if exists else '  (not found)'}")

        st.subheader("Summary JSON")
        st.json(summary)

        st.subheader("Download")
        pred_path = d / "replay_predictions.csv"
        if pred_path.exists():
            st.download_button(
                "Download replay_predictions.csv",
                data=pred_path.read_bytes(),
                file_name="replay_predictions.csv",
                mime="text/csv",
            )
        st.download_button(
            "Download replay_summary.json",
            data=json.dumps(summary, indent=2),
            file_name="replay_summary.json",
            mime="application/json",
        )

        timeline = d / "replay_timeline.png"
        if timeline.exists():
            st.subheader("Timeline figure")
            st.image(str(timeline), use_container_width=True)

    if watch_updates and not play:
        time.sleep(float(refresh_s))
        st.rerun()


if __name__ == "__main__":
    main()
