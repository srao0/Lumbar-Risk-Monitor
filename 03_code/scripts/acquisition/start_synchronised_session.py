#!/usr/bin/env python3
"""
Software-synchronised launcher for Phase II recording sessions.

Starts the IMU recorder, Ganglion recorder, and official session
timer from one host process. It gives all three subprocesses the same scheduled
Unix start timestamp so the session has a common host-side time origin.

This is not a hardware common clock. The IMU controller and OpenBCI board still
sample from their own clocks; the launcher reduces manual start offset and
writes metadata that makes the remaining timing uncertainty auditable.
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
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.acquisition.session_timer import PROTOCOL, total_duration
except ModuleNotFoundError:
    from scripts.acquisition.session_timer import PROTOCOL, total_duration


def iso_from_unix(ts: float) -> str:
    """Format a Unix timestamp as a millisecond-precision UTC ISO string for metadata."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds")


def rel(path: Path, root: Path) -> str:
    """Show a path relative to the repo root for tidier console output, falling back to absolute."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def command_text(cmd: list[str]) -> str:
    """Render a subprocess argv as a copy-pasteable shell line, quoting args with spaces."""
    return " ".join(f'"{part}"' if " " in part else part for part in cmd)


def refuse_existing(paths: list[Path], force: bool) -> None:
    """Abort before recording if any output already exists, unless --force was given.

    Guards against silently clobbering a real participant session.
    """
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        listed = "\n  - ".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing session files without --force:\n"
            f"  - {listed}"
        )


def terminate_process(proc: subprocess.Popen | None, name: str) -> None:
    """Stop a recorder subprocess gracefully, escalating to kill if it ignores terminate."""
    if proc is None or proc.poll() is not None:
        return
    print(f"[launcher] Terminating {name}...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print(f"[launcher] Killing {name}...")
        proc.kill()
        proc.wait(timeout=5)


def write_metadata(path: Path, metadata: dict) -> None:
    """Dump the session sync metadata to JSON (rewritten at each milestone)."""
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def select_protocol_blocks(skip_baseline: bool = False, start_section: str | None = None) -> list[dict]:
    """Slice the full PROTOCOL down to the blocks to run this session.

    Supports skipping the baseline (rehearsal) and resuming mid-protocol from a
    section number or label, e.g. after a hardware fault forced a restart.
    """
    blocks = PROTOCOL[:]
    if skip_baseline:
        blocks = [b for b in blocks if not b["section"].startswith("0")]
    if start_section:
        key = start_section.strip().lower()
        start_idx = None
        for i, block in enumerate(blocks):
            if block["section"].lower() == key or block["label"].lower() == key:
                start_idx = i
                break
        if start_idx is None:
            valid = ", ".join(f"{b['section']}:{b['label']}" for b in blocks)
            raise ValueError(f"Unknown --start_section {start_section!r}. Valid starts: {valid}")
        blocks = blocks[start_idx:]
    return blocks


def _alert_beep(repeats: int = 3) -> None:
    """Make an audible alarm without depending on the protocol timer."""
    def _run() -> None:
        try:
            import winsound

            for _ in range(repeats):
                winsound.Beep(1400, 220)
                time.sleep(0.08)
        except Exception:
            for _ in range(repeats):
                sys.stdout.write("\a")
                sys.stdout.flush()
                time.sleep(0.15)

    threading.Thread(target=_run, daemon=True).start()


IMU_FALLBACK_HEADER = (
    "t_ms,"
    "Pelvis_ax,Pelvis_ay,Pelvis_az,Pelvis_gx,Pelvis_gy,Pelvis_gz,"
    "L3_ax,L3_ay,L3_az,L3_gx,L3_gy,L3_gz,"
    "T12_ax,T12_ay,T12_az,T12_gx,T12_gy,T12_gz,"
    "T4_ax,T4_ay,T4_az,T4_gx,T4_gy,T4_gz"
).split(",")
IMU_SEGMENTS = ("Pelvis", "L3", "T12", "T4")
IMU_SEG_CHANS = ("ax", "ay", "az", "gx", "gy", "gz")
# Cyton/Ganglion ADS1299 full-scale rail in microvolts; samples pinned here mean
# the input is saturated (loose electrode, railed channel) rather than real sEMG.
ADS1299_RAIL_UV = 187500.0
EMG_SAT_UV = 100000.0


def _float_or_none(value: str) -> float | None:
    """Parse a CSV cell to a finite float, or None for blanks, NaN, or junk."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


class LiveCsvWatchdog:
    """Tail raw recorder CSVs and raise live alarms for acquisition failures."""

    def __init__(
        self,
        imu_csv: Path,
        emg_csv: Path,
        start_at_unix: float,
        events: list[dict],
        stop_event: threading.Event,
        imu_dead_run_s: float = 0.5,
        no_data_timeout_s: float = 4.0,
        emg_window_s: float = 2.0,
    ) -> None:
        self.imu_csv = imu_csv
        self.emg_csv = emg_csv
        self.start_at_unix = start_at_unix
        self.events = events
        self.stop_event = stop_event
        self.imu_dead_run_n = max(10, int(imu_dead_run_s * 100))
        self.no_data_timeout_s = no_data_timeout_s
        self.emg_window_n = max(100, int(emg_window_s * 250))
        self.imu_header: list[str] | None = None
        self.emg_header: list[str] | None = None
        self.imu_pos = 0
        self.emg_pos = 0
        self.imu_last_row_wall: float | None = None
        self.emg_last_row_wall: float | None = None
        self.imu_rows = 0
        self.emg_rows = 0
        self.imu_all_dead_run = 0
        self.imu_segment_dead_run = {seg: 0 for seg in IMU_SEGMENTS}
        self.emg_buffers: dict[str, list[float]] = {}
        self.alarmed_keys: set[str] = set()
        self.active_alarm_keys: set[str] = set()

    def alarm(self, key: str, message: str, level: str = "FAIL") -> None:
        """Raise a one-shot audible+console alarm for an acquisition fault.

        Keyed so each distinct fault fires only once; recovery is tracked
        separately via recovered().
        """
        if key in self.alarmed_keys:
            return
        self.alarmed_keys.add(key)
        event = {
            "level": level,
            "key": key,
            "message": message,
            "observed_iso_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        }
        self.events.append(event)
        self.active_alarm_keys.add(key)
        _alert_beep(4 if level == "FAIL" else 2)
        print()
        print("!" * 72)
        print(f"[LIVE WATCHDOG][{level}] {message}")
        print("!" * 72)
        print()

    def recovered(self, key: str, message: str) -> None:
        """Announce that a previously-alarmed fault is now clear (single soft beep)."""
        if key not in self.active_alarm_keys:
            return
        self.active_alarm_keys.remove(key)
        event = {
            "level": "RECOVERED",
            "key": key,
            "message": message,
            "observed_iso_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        }
        self.events.append(event)
        _alert_beep(1)
        print()
        print("-" * 72)
        print(f"[LIVE WATCHDOG][RECOVERED] {message}")
        print("-" * 72)
        print()

    def _read_new_lines(self, path: Path, pos: int) -> tuple[list[str], int]:
        """Tail a recorder CSV from byte offset `pos`, returning new lines and the new offset."""
        if not path.exists():
            return [], pos
        try:
            with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
                f.seek(pos)
                data = f.read()
                return data.splitlines(), f.tell()
        except OSError as exc:
            self.alarm(f"read-error:{path.name}", f"Could not read {path.name}: {exc}", level="WARN")
            return [], pos

    def _parse_csv_line(self, line: str) -> list[str] | None:
        if not line or line.startswith("#"):
            return None
        try:
            return next(csv.reader([line]))
        except csv.Error:
            return None

    def _process_imu_line(self, line: str) -> None:
        """Track per-segment all-zero runs and fire an alarm when a sensor goes dead.

        A segment reading exact zeros across all six channels means that IMU has
        dropped off the I2C bus -- exactly the P11/P12 failure mode this watchdog
        was added to catch live rather than in post-hoc QC.
        """
        fields = self._parse_csv_line(line)
        if not fields:
            return
        if fields[0] == "t_ms":
            self.imu_header = fields
            return
        if self.imu_header is None:
            if len(fields) == len(IMU_FALLBACK_HEADER) and _float_or_none(fields[0]) is not None:
                self.imu_header = IMU_FALLBACK_HEADER
            else:
                return
        if len(fields) != len(self.imu_header):
            return

        values = dict(zip(self.imu_header, fields))
        self.imu_rows += 1
        self.imu_last_row_wall = time.time()

        dead_segments = []
        for seg in IMU_SEGMENTS:
            chan_values = []
            for chan in IMU_SEG_CHANS:
                value = _float_or_none(values.get(f"{seg}_{chan}", ""))
                chan_values.append(value)
            is_dead = all(value == 0.0 for value in chan_values if value is not None) and all(
                value is not None for value in chan_values
            )
            if is_dead:
                dead_segments.append(seg)
                self.imu_segment_dead_run[seg] += 1
            else:
                self.imu_segment_dead_run[seg] = 0

        if len(dead_segments) == len(IMU_SEGMENTS):
            self.imu_all_dead_run += 1
        else:
            if "imu-all-zero" in self.active_alarm_keys:
                self.recovered("imu-all-zero", "IMU all-zero stream recovered; non-zero samples are arriving again.")
            self.imu_all_dead_run = 0

        t_ms = _float_or_none(values.get("t_ms", ""))
        t_label = f" at board t_ms={t_ms:.0f}" if t_ms is not None else ""
        if self.imu_all_dead_run >= self.imu_dead_run_n:
            self.alarm(
                "imu-all-zero",
                f"IMU stream is all zeros for >= {self.imu_dead_run_n / 100:.1f}s{t_label}. Stop/restart the IMU hardware.",
            )
        elif dead_segments:
            for seg, run in self.imu_segment_dead_run.items():
                if run >= self.imu_dead_run_n:
                    self.alarm(
                        f"imu-segment-zero:{seg}",
                        f"IMU segment {seg} is outputting all zeros for >= {self.imu_dead_run_n / 100:.1f}s{t_label}.",
                        level="WARN",
                    )
        for seg in IMU_SEGMENTS:
            key = f"imu-segment-zero:{seg}"
            if self.imu_segment_dead_run[seg] == 0 and key in self.active_alarm_keys:
                self.recovered(key, f"IMU segment {seg} recovered; non-zero samples are arriving again.")

    def _process_emg_line(self, line: str) -> None:
        """Watch a rolling sEMG window per channel for railing, DC drift, or flat/dead signal.

        Catches the saturated-channel and electrode-contact problems seen in the
        cohort before they ruin a whole recording.
        """
        fields = self._parse_csv_line(line)
        if not fields:
            return
        if any(field.lower().startswith("emg_ch") for field in fields):
            self.emg_header = fields
            self.emg_buffers = {
                field: [] for field in fields
                if field.lower().startswith("emg_ch") and field.lower() in {"emg_ch1", "emg_ch2", "emg_ch3", "emg_ch4"}
            }
            return
        if self.emg_header is None or len(fields) != len(self.emg_header):
            return

        values = dict(zip(self.emg_header, fields))
        self.emg_rows += 1
        self.emg_last_row_wall = time.time()
        for col, buf in self.emg_buffers.items():
            value = _float_or_none(values.get(col, ""))
            if value is None:
                continue
            buf.append(value)
            if len(buf) > self.emg_window_n:
                del buf[:len(buf) - self.emg_window_n]
            if len(buf) < self.emg_window_n:
                continue
            abs_vals = [abs(x) for x in buf]
            railed = sum(abs(abs(x) - ADS1299_RAIL_UV) < ADS1299_RAIL_UV * 0.01 for x in buf) / len(buf)
            saturated = sum(x > EMG_SAT_UV for x in abs_vals) / len(buf)
            mean_abs = sum(abs_vals) / len(abs_vals)
            mean_dc = sum(buf) / len(buf)
            if railed > 0.05 or saturated > 0.05:
                self.alarm(
                    f"emg-rail:{col}",
                    f"{col} is railing/saturating over the last ~{self.emg_window_n / 250:.1f}s "
                    f"(rail={railed:.0%}, sat={saturated:.0%}). Check electrode/contact.",
                )
            elif abs(mean_dc) > ADS1299_RAIL_UV * 0.70:
                self.alarm(
                    f"emg-dc:{col}",
                    f"{col} DC offset is approaching the Cyton rail ({mean_dc / 1000:.1f} mV).",
                    level="WARN",
                )
            elif mean_abs < 1.0:
                self.alarm(
                    f"emg-flat:{col}",
                    f"{col} looks flat/dead over the last ~{self.emg_window_n / 250:.1f}s.",
                    level="WARN",
                )
            else:
                for suffix in ("rail", "dc", "flat"):
                    key = f"emg-{suffix}:{col}"
                    if key in self.active_alarm_keys:
                        self.recovered(key, f"{col} recovered; recent sEMG window is back inside live thresholds.")

    def _check_no_data(self) -> None:
        """Alert if a recorder's file is missing, empty, or has stopped updating.

        Grace period of 5 s after the scheduled start lets the boards spin up
        before we start complaining about silence.
        """
        now = time.time()
        if now < self.start_at_unix + 5.0:
            return
        for name, path, last_wall, rows in (
            ("IMU", self.imu_csv, self.imu_last_row_wall, self.imu_rows),
            ("sEMG", self.emg_csv, self.emg_last_row_wall, self.emg_rows),
        ):
            if rows == 0 and not path.exists():
                self.alarm(f"missing:{name}", f"{name} output file has not appeared: {path}", level="WARN")
            elif rows == 0:
                self.alarm(f"no-rows:{name}", f"{name} output exists but no data rows have arrived.", level="WARN")
            elif last_wall is not None and now - last_wall > self.no_data_timeout_s:
                self.alarm(
                    f"stalled:{name}",
                    f"{name} data has not updated for {now - last_wall:.1f}s. Recorder may be stalled.",
                )
            else:
                for key in (f"missing:{name}", f"no-rows:{name}", f"stalled:{name}"):
                    if key in self.active_alarm_keys:
                        self.recovered(key, f"{name} data rows are arriving again.")

    def run(self) -> None:
        """Poll both recorder CSVs twice a second until told to stop (runs in a thread)."""
        print("[launcher] Live acquisition watchdog armed.")
        while not self.stop_event.is_set():
            imu_lines, self.imu_pos = self._read_new_lines(self.imu_csv, self.imu_pos)
            for line in imu_lines:
                self._process_imu_line(line)
            emg_lines, self.emg_pos = self._read_new_lines(self.emg_csv, self.emg_pos)
            for line in emg_lines:
                self._process_emg_line(line)
            self._check_no_data()
            self.stop_event.wait(0.5)


def main() -> int:
    """Launch the three recorders on a shared scheduled start, supervise them, and write sync metadata.

    Returns a process exit code: 0 on a clean session, non-zero if a recorder
    failed early, a watchdog FAILed, or an expected output file is missing.
    """
    parser = argparse.ArgumentParser(
        description="Start IMU, Ganglion, and protocol timer with a shared host-side start timestamp.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--session_dir", required=True,
                        help="Raw output folder, e.g. data/real/raw/participant_01/session_001.")
    parser.add_argument("--imu_port", required=True,
                        help="ESP32 IMU serial port, e.g. COM3.")
    parser.add_argument("--ganglion_port", default="",
                        help="Legacy alias for --emg_port when using Ganglion recordings.")
    parser.add_argument("--emg_port", default="",
                        help="OpenBCI dongle serial port, e.g. COM4 or COM15.")
    parser.add_argument("--emg_board", choices=["ganglion", "cyton"], default="ganglion",
                        help="OpenBCI board used for sEMG recording.")
    parser.add_argument("--ganglion_mac", default="",
                        help="Optional Ganglion MAC address.")
    parser.add_argument("--synthetic_ganglion", action="store_true",
                        help="Use BrainFlow synthetic board for the EMG recorder.")
    parser.add_argument("--cyton_bipolar_channel", default="0",
                        help="Cyton bipolar channel(s), e.g. '1' or '1,2,3,4'. Use 0 for defaults.")
    parser.add_argument("--imu_baud", type=int, default=115200,
                        help="ESP32 baud rate.")
    parser.add_argument("--prestart_delay", type=float, default=25.0,
                        help="Seconds between subprocess launch and scheduled start.")
    parser.add_argument("--settle_seconds", type=float, default=2.0,
                        help="Seconds to check recorder subprocesses before starting the visible timer.")
    parser.add_argument("--recording_duration", type=float, default=None,
                        help="Sensor recording duration. Defaults to protocol duration plus --post_padding.")
    parser.add_argument("--post_padding", type=float, default=5.0,
                        help="Extra seconds recorded after the protocol when --recording_duration is omitted.")
    parser.add_argument("--pause_padding", type=float, default=600.0,
                        help="Extra recorder time reserved for protocol pauses; unused padding is stopped after the timer ends.")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Pass through to session_timer.py for rehearsal only.")
    parser.add_argument("--start_section", default=None,
                        help="Resume protocol from this section/label, e.g. 5.3L or CLEAN_LATERAL_L.")
    parser.add_argument("--force", action="store_true",
                        help="Allow overwriting existing output files.")
    parser.add_argument("--no_live_watchdog", action="store_true",
                        help="Disable the live CSV watchdog during acquisition.")
    parser.add_argument("--watchdog_no_data_timeout", type=float, default=4.0,
                        help="Seconds without new CSV rows before a live watchdog stall alert.")
    parser.add_argument("--watchdog_imu_zero_seconds", type=float, default=0.5,
                        help="Sustained all-zero IMU duration needed before a live alert.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print planned commands and metadata without starting hardware.")
    args = parser.parse_args()

    emg_port = args.emg_port or args.ganglion_port
    if not args.synthetic_ganglion and not emg_port:
        parser.error("--emg_port is required unless --synthetic_ganglion is used.")
    if args.synthetic_ganglion and args.emg_board != "ganglion":
        parser.error("--synthetic_ganglion is only supported with --emg_board ganglion.")
    if args.prestart_delay < 0:
        parser.error("--prestart_delay must be non-negative.")
    if args.settle_seconds < 0:
        parser.error("--settle_seconds must be non-negative.")

    repo_root = Path(__file__).resolve().parents[2]
    session_dir = Path(args.session_dir)
    if not session_dir.is_absolute():
        session_dir = repo_root / session_dir
    session_dir.mkdir(parents=True, exist_ok=True)

    imu_csv = session_dir / "imu_arduino.csv"
    emg_csv = session_dir / ("cyton.csv" if args.emg_board == "cyton" else "ganglion.csv")
    labels_csv = session_dir / "labels.csv"
    sync_metadata = session_dir / "session_sync_metadata.json"
    imu_log = session_dir / "_imu_stdout.log"
    ganglion_log = session_dir / "_ganglion_stdout.log"

    selected_blocks = select_protocol_blocks(args.skip_baseline, args.start_section)
    protocol_seconds = total_duration(selected_blocks)
    recording_duration = args.recording_duration
    if recording_duration is None:
        recording_duration = protocol_seconds + args.post_padding + args.pause_padding
    if recording_duration <= 0:
        parser.error("Recording duration must be positive.")

    start_at_unix = time.time() + args.prestart_delay
    start_iso_utc = iso_from_unix(start_at_unix)
    start_local = datetime.fromtimestamp(start_at_unix).isoformat(timespec="milliseconds")

    if not args.dry_run:
        refuse_existing(
            [imu_csv, emg_csv, labels_csv, imu_log, ganglion_log],
            force=args.force,
        )

    imu_cmd = [
        sys.executable, "-u", str(repo_root / "scripts" / "acquisition" / "record_imu_serial.py"),
        "--port", args.imu_port,
        "--baud", str(args.imu_baud),
        "--duration", f"{recording_duration:.3f}",
        "--out", str(imu_csv),
        "--start_at_unix", f"{start_at_unix:.6f}",
    ]
    if args.force:
        imu_cmd.append("--force")

    emg_script = "cyton_stream.py" if args.emg_board == "cyton" else "ganglion_stream.py"
    ganglion_cmd = [
        sys.executable, "-u", str(repo_root / "scripts" / "acquisition" / emg_script),
        "--duration", f"{recording_duration:.3f}",
        "--out", str(emg_csv),
        "--start_at_unix", f"{start_at_unix:.6f}",
    ]
    if args.synthetic_ganglion:
        ganglion_cmd.append("--synthetic")
    else:
        ganglion_cmd.extend(["--port", emg_port])
    if args.emg_board == "cyton":
        ganglion_cmd.extend(["--bipolar_channel", args.cyton_bipolar_channel])
    if args.emg_board == "ganglion" and args.ganglion_mac:
        ganglion_cmd.extend(["--mac", args.ganglion_mac])
    if args.force:
        ganglion_cmd.append("--force")

    timer_cmd = [
        sys.executable, "-u", str(repo_root / "scripts" / "acquisition" / "session_timer.py"),
        "--out", str(labels_csv),
        "--auto_start",
        "--start_at_unix", f"{start_at_unix:.6f}",
    ]
    if args.skip_baseline:
        timer_cmd.append("--skip_baseline")
    if args.start_section:
        timer_cmd.extend(["--start_section", args.start_section])

    metadata = {
        "sync_method": "software_scheduled_start",
        "common_hardware_clock": False,
        "note": (
            "All subprocesses received the same host-side scheduled start timestamp. "
            "This reduces manual launch offset but does not synchronise the IMU and OpenBCI sample clocks."
        ),
        "session_dir": str(session_dir),
        "scheduled_start_unix": start_at_unix,
        "scheduled_start_iso_utc": start_iso_utc,
        "scheduled_start_local": start_local,
        "prestart_delay_s": args.prestart_delay,
        "settle_seconds_s": args.settle_seconds,
        "protocol_duration_s": protocol_seconds,
        "protocol_start_section": args.start_section or "",
        "protocol_sections": [b["section"] for b in selected_blocks],
        "recording_duration_s": recording_duration,
        "pause_padding_s": args.pause_padding,
        "outputs": {
            "imu_csv": str(imu_csv),
            "emg_csv": str(emg_csv),
            "emg_board": args.emg_board,
            "ganglion_csv": str(emg_csv) if args.emg_board == "ganglion" else "",
            "cyton_csv": str(emg_csv) if args.emg_board == "cyton" else "",
            "labels_csv": str(labels_csv),
            "imu_log": str(imu_log),
            "ganglion_log": str(ganglion_log),
        },
        "commands": {
            "imu": imu_cmd,
            "ganglion": ganglion_cmd,
            "timer": timer_cmd,
        },
        "live_watchdog": {
            "enabled": not args.no_live_watchdog,
            "no_data_timeout_s": args.watchdog_no_data_timeout,
            "imu_zero_seconds": args.watchdog_imu_zero_seconds,
        },
    }

    print()
    print("=" * 72)
    print("  Software-Synchronised Recording Session")
    print("=" * 72)
    print(f"  Session folder     : {rel(session_dir, repo_root)}")
    print(f"  Scheduled start    : {start_local}")
    print(f"  Protocol duration  : {protocol_seconds:.1f} s")
    print(f"  Recording duration : {recording_duration:.1f} s")
    print(f"  IMU output         : {rel(imu_csv, repo_root)}")
    print(f"  EMG board          : {args.emg_board}")
    print(f"  EMG output         : {rel(emg_csv, repo_root)}")
    print(f"  Labels output      : {rel(labels_csv, repo_root)}")
    print("=" * 72)
    print()
    print("Planned commands:")
    print(f"  IMU      : {command_text(imu_cmd)}")
    print(f"  EMG      : {command_text(ganglion_cmd)}")
    print(f"  Timer    : {command_text(timer_cmd)}")
    print()

    if args.dry_run:
        metadata["dry_run"] = True
        print("[dry-run] Metadata preview:")
        print(json.dumps(metadata, indent=2))
        print("[dry-run] No recorder/timer processes started and no metadata file written.")
        return 0

    write_metadata(sync_metadata, metadata)

    imu_proc = None
    ganglion_proc = None
    timer_proc = None
    imu_fp = None
    ganglion_fp = None
    spawn_t0 = time.monotonic()
    wall_launch_iso = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    stop_watch = None
    watcher_thread = None
    watcher_events: list[dict] = []
    live_watchdog_thread = None
    live_watchdog_events: list[dict] = []

    try:
        imu_fp = open(imu_log, "w", encoding="utf-8")
        ganglion_fp = open(ganglion_log, "w", encoding="utf-8")

        print("[launcher] Starting recorder subprocesses...")
        ganglion_proc = subprocess.Popen(
            ganglion_cmd,
            cwd=str(repo_root),
            stdout=ganglion_fp,
            stderr=subprocess.STDOUT,
            text=True,
        )
        ganglion_spawn_s = time.monotonic() - spawn_t0

        imu_proc = subprocess.Popen(
            imu_cmd,
            cwd=str(repo_root),
            stdout=imu_fp,
            stderr=subprocess.STDOUT,
            text=True,
        )
        imu_spawn_s = time.monotonic() - spawn_t0

        settle_s = min(args.settle_seconds, max(0.0, start_at_unix - time.time() - 0.5))
        if settle_s > 0:
            print(f"[launcher] Checking recorder subprocesses for {settle_s:.1f} s before starting timer...")
            time.sleep(settle_s)

        early_failures = []
        for name, proc, log_path in (
            ("Ganglion recorder", ganglion_proc, ganglion_log),
            ("IMU recorder", imu_proc, imu_log),
        ):
            exit_code = proc.poll()
            if exit_code is not None:
                early_failures.append({
                    "name": name,
                    "exit_code": exit_code,
                    "log": str(log_path),
                })

        if early_failures:
            if ganglion_fp is not None:
                ganglion_fp.flush()
            if imu_fp is not None:
                imu_fp.flush()
            metadata["early_start_failure"] = early_failures
            write_metadata(sync_metadata, metadata)
            print("[launcher] Recorder failed before the protocol timer started. Timer was not launched.")
            for failure in early_failures:
                print(f"  - {failure['name']} exited with code {failure['exit_code']}; see {failure['log']}")
            terminate_process(ganglion_proc, "Ganglion recorder")
            terminate_process(imu_proc, "IMU recorder")
            return 1

        print("[launcher] Starting visible protocol timer...")
        timer_proc = subprocess.Popen(timer_cmd, cwd=str(repo_root))
        timer_spawn_s = time.monotonic() - spawn_t0

        metadata["launcher"] = {
            "wall_launch_iso_utc": wall_launch_iso,
            "ganglion_spawn_s": ganglion_spawn_s,
            "imu_spawn_s": imu_spawn_s,
            "timer_spawn_s": timer_spawn_s,
            "early_recorder_check_s": settle_s,
            "ganglion_vs_imu_spawn_delta_ms": (imu_spawn_s - ganglion_spawn_s) * 1000.0,
            "timer_vs_ganglion_spawn_delta_ms": (timer_spawn_s - ganglion_spawn_s) * 1000.0,
        }
        write_metadata(sync_metadata, metadata)

        stop_watch = threading.Event()

        if not args.no_live_watchdog:
            live_watchdog = LiveCsvWatchdog(
                imu_csv=imu_csv,
                emg_csv=emg_csv,
                start_at_unix=start_at_unix,
                events=live_watchdog_events,
                stop_event=stop_watch,
                imu_dead_run_s=args.watchdog_imu_zero_seconds,
                no_data_timeout_s=args.watchdog_no_data_timeout,
            )
            live_watchdog_thread = threading.Thread(
                target=live_watchdog.run,
                name="live-acquisition-watchdog",
                daemon=True,
            )
            live_watchdog_thread.start()

        def watch_recorders() -> None:
            seen_exits = set()
            recorders = (
                ("Ganglion recorder", ganglion_proc, ganglion_log),
                ("IMU recorder", imu_proc, imu_log),
            )
            while not stop_watch.is_set():
                for name, proc, log_path in recorders:
                    if name in seen_exits:
                        continue
                    exit_code = proc.poll()
                    if exit_code is not None:
                        seen_exits.add(name)
                        event = {
                            "name": name,
                            "exit_code": exit_code,
                            "log": str(log_path),
                            "observed_iso_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                        }
                        watcher_events.append(event)
                        print(
                            f"\n[launcher][warning] {name} exited before the protocol timer finished "
                            f"(code {exit_code}). See {log_path}"
                        )
                if len(seen_exits) == len(recorders):
                    break
                stop_watch.wait(1.0)

        watcher_thread = threading.Thread(target=watch_recorders, name="recorder-watch", daemon=True)
        watcher_thread.start()

        print()
        print("[launcher] Sensor logs are being written to:")
        print(f"  - {rel(ganglion_log, repo_root)}")
        print(f"  - {rel(imu_log, repo_root)}")
        print("[launcher] Follow the protocol cues shown below.")
        print()

        timer_rc = timer_proc.wait()
        if stop_watch is not None:
            stop_watch.set()
        if watcher_thread is not None:
            watcher_thread.join(timeout=2)
        if live_watchdog_thread is not None:
            live_watchdog_thread.join(timeout=2)
        print(f"[launcher] Timer finished with exit code {timer_rc}. Stopping recorders...")
        recorders_stopped_after_timer = []
        for name, proc in (("Ganglion recorder", ganglion_proc), ("IMU recorder", imu_proc)):
            if proc.poll() is None:
                recorders_stopped_after_timer.append(name)
                terminate_process(proc, name)
        ganglion_rc = ganglion_proc.poll()
        imu_rc = imu_proc.poll()

        metadata["exit_codes"] = {
            "timer": timer_rc,
            "ganglion": ganglion_rc,
            "imu": imu_rc,
        }
        metadata["recorders_stopped_after_timer"] = recorders_stopped_after_timer
        metadata["watcher_events"] = watcher_events
        metadata["live_watchdog"]["events"] = live_watchdog_events
        metadata["live_watchdog"]["overall"] = (
            "FAIL" if any(e.get("level") == "FAIL" for e in live_watchdog_events)
            else ("WARN" if live_watchdog_events else "PASS")
        )
        metadata["output_sizes_bytes"] = {
            "labels_csv": labels_csv.stat().st_size if labels_csv.exists() else 0,
            "emg_csv": emg_csv.stat().st_size if emg_csv.exists() else 0,
            "imu_csv": imu_csv.stat().st_size if imu_csv.exists() else 0,
        }
        write_metadata(sync_metadata, metadata)

    except KeyboardInterrupt:
        print("\n[launcher] Interrupted. Stopping subprocesses.")
        if stop_watch is not None:
            stop_watch.set()
        if watcher_thread is not None:
            watcher_thread.join(timeout=2)
        if live_watchdog_thread is not None:
            live_watchdog_thread.join(timeout=2)
        terminate_process(timer_proc, "timer")
        terminate_process(ganglion_proc, "Ganglion recorder")
        terminate_process(imu_proc, "IMU recorder")
        metadata["interrupted"] = True
        write_metadata(sync_metadata, metadata)
        return 130
    finally:
        if imu_fp is not None:
            imu_fp.close()
        if ganglion_fp is not None:
            ganglion_fp.close()

    stopped_after_timer = set(metadata.get("recorders_stopped_after_timer", []))
    ok = (
        metadata.get("exit_codes", {}).get("timer") == 0
        and (
            metadata.get("exit_codes", {}).get("ganglion") == 0
            or "Ganglion recorder" in stopped_after_timer
        )
        and (
            metadata.get("exit_codes", {}).get("imu") == 0
            or "IMU recorder" in stopped_after_timer
        )
        and metadata.get("live_watchdog", {}).get("overall", "PASS") != "FAIL"
        and labels_csv.exists()
        and emg_csv.exists()
        and imu_csv.exists()
    )

    print()
    if ok:
        print("[OK] Software-synchronised recording complete.")
    else:
        print("[FAIL] One or more subprocesses failed or an output file is missing.")
    if metadata.get("live_watchdog", {}).get("enabled"):
        status = metadata.get("live_watchdog", {}).get("overall", "UNKNOWN")
        print(f"  Live watchdog: {status}")
    print(f"  Metadata: {rel(sync_metadata, repo_root)}")
    print()
    print("Next conversion command:")
    print("  py scripts/conversion/session_converter.py \\")
    print(f"    --emg {rel(emg_csv, repo_root)} \\")
    print(f"    --emg_board {args.emg_board} \\")
    print(f"    --imu {rel(imu_csv, repo_root)} \\")
    print(f"    --labels {rel(labels_csv, repo_root)} \\")
    print("    --mode full_hybrid \\")
    print(f"    --out_dir data/real/processed/{session_dir.name}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
