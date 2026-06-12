#!/usr/bin/env python3
"""
label_logger.py
===============
Spinal Movement Risk Monitor -- FYP 2025/26

Real-time annotation tool for recording sessions.

Run this alongside ganglion_stream.py and record_imu_serial.py. All three
scripts start their own clocks from launch. If you start them within a second
of each other, the timestamps will be close enough for manual alignment. For
sub-millisecond sync you need a hardware trigger pulse, which is out of scope
for this prototype.

Usage
-----
    # Start logging alongside the other recording scripts
    py scripts/acquisition/label_logger.py --out data/real/raw/session_001/labels.csv

    # Custom session duration reminder
    py scripts/acquisition/label_logger.py --out data/real/raw/session_001/labels.csv --duration 120

Controls
--------
    Movement keys -- press once to START a rep, press the same key or any
    other movement key to END it (the old rep closes, the new one opens).

        1  BASELINE_STATIC         safe  (risk=0)
        2  CLEAN_FLEXION           safe  (risk=0)
        3  CLEAN_LATERAL_L         safe  (risk=0)
        4  CLEAN_LATERAL_R         safe  (risk=0)
        5  CLEAN_ROTATION_L        safe  (risk=0)
        6  CLEAN_ROTATION_R        safe  (risk=0)
        7  PICKUP_SYM              safe  (risk=0)
        8  SIT_TO_STAND_NORMAL     safe  (risk=0)
        a  LUMBAR_DOMINANT         risky (risk=1)
        b  FAST_BEND               risky (risk=1)
        c  SHOULDER_DRIVEN         risky (risk=1)
        d  PICKUP_ASYM             risky (risk=1)
        f  FATIGUE_FLEXION         risky (risk=1)
        g  SIT_TO_STAND_FAST       ambiguous (risk=-1)

    Other keys:
        SPACE  End current rep without starting a new one (gap / rest period)
        r      Toggle risk_class of the LAST completed rep (0 -> 1 or 1 -> 0)
        z      Undo -- remove the last completed rep
        s      Save checkpoint to disk (does not quit)
        q      Save and quit

Output
------
    labels.csv with columns matching synthetic_generator.py output:
        label, rep, start_ms, end_ms, risk_class, fatigue_fraction

    fatigue_fraction is left blank (empty string) for real sessions. The
    pipeline accepts this -- it is only used by the fatigue model in synthetic
    data and is ignored for all other movement classes.

Notes
-----
    * Timestamps are milliseconds since this script was launched (t=0 at launch).
    * Rep numbers are per-label counters, not session-global counters.
      CLEAN_FLEXION rep 1 and LUMBAR_DOMINANT rep 1 are independent.
    * If you press 'q' without having ended the last rep, the script will
      auto-close it at the current timestamp before saving.
    * If the output file already exists, the script will ask before overwriting.

Requirements
------------
    No external packages required. Uses only the Python standard library.
    On Windows: msvcrt (built-in).
    On Linux/macOS: tty, termios, sys (built-in).
"""

import argparse
import csv
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


# Movement definitions

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
}

CONTROLS = ' '.join(f"{k}={v[0].split('_')[0].lower()}" for k, v in MOVEMENT_MAP.items())

HELP_LINES = [
    "",
    "  Movement keys (start/end rep):",
    "    1 BASELINE_STATIC (safe)     2 CLEAN_FLEXION (safe)",
    "    3 CLEAN_LATERAL_L (safe)     4 CLEAN_LATERAL_R (safe)",
    "    5 CLEAN_ROTATION_L (safe)    6 CLEAN_ROTATION_R (safe)",
    "    7 PICKUP_SYM (safe)          8 SIT_TO_STAND_NORMAL (safe)",
    "    a LUMBAR_DOMINANT (risky)    b FAST_BEND (risky)",
    "    c SHOULDER_DRIVEN (risky)    d PICKUP_ASYM (risky)",
    "    f FATIGUE_FLEXION (risky)    g SIT_TO_STAND_FAST (ambiguous)",
    "",
    "  Other:  SPACE=end rep   r=toggle risk   z=undo   s=save   q=quit",
    "",
]


# Cross-platform single character read (no external dependencies)

def _getch_windows():
    import msvcrt
    ch = msvcrt.getch()
    try:
        return ch.decode('utf-8').lower()
    except UnicodeDecodeError:
        return ''


def _getch_unix():
    import tty
    import termios
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch.lower()


if os.name == 'nt':
    _getch = _getch_windows
else:
    _getch = _getch_unix


# Logger state

class LabelLogger:
    """Tracks in-progress and completed reps during a recording session."""

    def __init__(self, out_path: Path):
        self.out_path = out_path
        self.start_wall = time.time()          # wall-clock at t=0
        self.completed: list[dict] = []        # finished reps
        self.rep_counters: dict[str, int] = {} # label -> next rep number
        self.active: dict | None = None        # currently open rep
        self._alive = True
        self.segment_label:     str   = ''
        self.segment_start_ms: float  = 0.0   # when current label block started
        self._timer_thread = threading.Thread(
            target=self._timer_worker, daemon=True)
        self._timer_thread.start()

    def elapsed_ms(self) -> float:
        return (time.time() - self.start_wall) * 1000.0

    def _now_str(self) -> str:
        return f"{self.elapsed_ms():.0f} ms"

    def start_rep(self, key: str):
        """Open a new rep for the keyed movement, closing any rep still running.

        Pressing one movement key after another chains reps with no gap, which
        matches how a participant flows through a protocol block.
        """
        label, risk_class = MOVEMENT_MAP[key]
        now = self.elapsed_ms()

        # Auto-close any open rep
        if self.active is not None:
            self._close_active(now)

        rep_num = self.rep_counters.get(label, 0) + 1
        self.rep_counters[label] = rep_num
        if label != self.segment_label:
            self.segment_label    = label
            self.segment_start_ms = now
        self.active = {
            'label':          label,
            'rep':            rep_num,
            'start_ms':       round(now, 1),
            'end_ms':         None,
            'risk_class':     risk_class,
            'fatigue_fraction': '',
        }
        risk_label = {0: 'SAFE', 1: 'RISKY', -1: 'AMBIGUOUS'}.get(risk_class, '?')
        print(f"  [{self._now_str()}]  START  {label}  rep {rep_num}  ({risk_label})")

    def end_rep(self):
        """Close the open rep at the current time (SPACE = end without a new rep)."""
        if self.active is None:
            print("  (no active rep to end)")
            return
        now = self.elapsed_ms()
        self._close_active(now)

    def _close_active(self, end_ms: float):
        self.active['end_ms'] = round(end_ms, 1)
        duration = (self.active['end_ms'] - self.active['start_ms']) / 1000.0
        self.completed.append(self.active)
        label = self.active['label']
        rep   = self.active['rep']
        print(f"  [{self._now_str()}]  END    {label}  rep {rep}  ({duration:.1f} s)")
        self.active = None

    def _timer_worker(self):
        """Background: update dual-timer status line every second."""
        while self._alive:
            time.sleep(1.0)
            now_ms = self.elapsed_ms()
            sess_s = int(now_ms / 1000)
            sess_m, sess_s2 = divmod(sess_s, 60)
            if self.active is not None:
                rep_s = (now_ms - self.active["start_ms"]) / 1000.0
                seg_s = (now_ms - self.segment_start_ms)   / 1000.0
                label = self.active["label"]
                rep   = self.active["rep"]
                sys.stdout.write(
                    f"\r  [{sess_m}:{sess_s2:02d}]"
                    f"  SEGMENT [{label}]: {seg_s:>5.0f}s"
                    f"  |  REP {rep}: {rep_s:>4.0f}s"
                    f"  |  SPACE=end  q=quit     "
                )
                sys.stdout.flush()
            else:
                # No active rep, still show session clock
                sys.stdout.write(
                    f"\r  [{sess_m}:{sess_s2:02d}]"
                    f"  (no active rep)  |  press a movement key to start"
                    f"                  "
                )
                sys.stdout.flush()

    def stop(self):
        self._alive = False

    def toggle_last_risk(self):
        """Flip the last completed rep between safe and risky (a mislabel fixer).

        Ambiguous reps (risk=-1, e.g. SIT_TO_STAND_FAST) are left untouched --
        there is no obvious flip target for them.
        """
        if not self.completed:
            print("  (no completed reps to toggle)")
            return
        last = self.completed[-1]
        old  = last['risk_class']
        if old == 0:
            last['risk_class'] = 1
        elif old == 1:
            last['risk_class'] = 0
        else:
            print(f"  (risk_class={old} is ambiguous -- not toggling)")
            return
        print(f"  Toggled {last['label']} rep {last['rep']} risk_class: {old} -> {last['risk_class']}")

    def undo_last(self):
        """Remove the most recent rep -- the open one if any, else the last closed one.

        Also rolls back that label's rep counter so the next rep reuses the number.
        """
        if self.active is not None:
            label = self.active['label']
            rep   = self.active['rep']
            # roll back rep counter
            self.rep_counters[label] = max(0, self.rep_counters.get(label, 1) - 1)
            self.active = None
            print(f"  Undid open rep: {label} rep {rep}")
            return
        if not self.completed:
            print("  (nothing to undo)")
            return
        last = self.completed.pop()
        self.rep_counters[last['label']] = max(0, self.rep_counters.get(last['label'], 1) - 1)
        print(f"  Undid: {last['label']} rep {last['rep']}")

    def save(self):
        """Write completed reps to labels.csv, sorted by start time."""
        rows = sorted(self.completed, key=lambda r: r['start_ms'])
        fieldnames = ['label', 'rep', 'start_ms', 'end_ms', 'risk_class', 'fatigue_fraction']
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved {len(rows)} reps -> {self.out_path}")

    def summary(self):
        n     = len(self.completed)
        risky = sum(1 for r in self.completed if r['risk_class'] == 1)
        safe  = sum(1 for r in self.completed if r['risk_class'] == 0)
        print(f"  Completed reps: {n}  (safe={safe}, risky={risky})")
        if self.active:
            print(f"  Open rep: {self.active['label']} rep {self.active['rep']} (not yet ended)")


# Main loop

def main():
    """Run the interactive keypress loop until the user saves and quits."""
    parser = argparse.ArgumentParser(
        description='Real-time label logger for spinal movement recordings.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n'.join(HELP_LINES),
    )
    parser.add_argument(
        '--out', '-o',
        type=Path,
        default=Path('labels.csv'),
        help='Path to write labels.csv (default: labels.csv in current dir)',
    )
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=None,
        help='Expected session duration in seconds (display only -- does not auto-stop)',
    )
    args = parser.parse_args()

    out_path = args.out
    if out_path.exists():
        print(f"\n  WARNING: {out_path} already exists.")
        print("  Overwrite? [y/N]: ", end='', flush=True)
        ch = _getch()
        print(ch)
        if ch.lower() != 'y':
            print("  Aborted.")
            sys.exit(0)

    logger = LabelLogger(out_path)

    print()
    print("=" * 60)
    print("  LABEL LOGGER -- Spinal Movement Risk Monitor")
    print("=" * 60)
    print(f"  Output:    {out_path}")
    if args.duration:
        print(f"  Duration:  {args.duration} s")
    print(f"  t=0:       {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    print()
    print("  Start ganglion_stream.py and record_imu_serial.py now,")
    print("  then annotate movements as they happen.")
    print()
    for line in HELP_LINES:
        print(line)
    print("=" * 60)
    print()

    try:
        while True:
            ch = _getch()

            if ch in MOVEMENT_MAP:
                logger.start_rep(ch)

            elif ch == ' ':
                logger.end_rep()

            elif ch == 'r':
                logger.toggle_last_risk()

            elif ch == 'z':
                logger.undo_last()

            elif ch == 's':
                logger.save()
                logger.summary()

            elif ch in ('q', '\x03'):  # q or Ctrl+C
                print()
                print("  Quitting...")
                logger.stop()
                if logger.active is not None:
                    print("  Auto-closing open rep before save.")
                    logger.end_rep()
                logger.save()
                logger.summary()
                print()
                break

    except KeyboardInterrupt:
        print()
        print("  Interrupted. Saving before exit...")
        logger.stop()
        if logger.active is not None:
            logger.end_rep()
        logger.save()
        logger.summary()
        print()


if __name__ == '__main__':
    main()
