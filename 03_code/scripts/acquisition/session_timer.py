#!/usr/bin/env python3
"""
session_timer.py  --  Spinal Movement Risk Monitor
====================================================
Guided countdown for the official supervised Phase II protocol.
Run this alongside the IMU and Ganglion recorders. This timer writes the
official protocol labels for scheduled supervised blocks.

Each movement block counts down rep-by-rep with per-phase breakdowns
(DESCEND / HOLD / RETURN / REST). Press Enter at any time to pause/resume.

Usage
-----
    py scripts/acquisition/session_timer.py
    py scripts/acquisition/session_timer.py --skip_baseline # rehearsal only; will not validate officially
"""

# Make the project package importable when this file is run directly.
import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = _Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PKG_ROOT))


import argparse
import sys
import threading
import time
from pathlib import Path

try:
    from scripts.conversion.generate_protocol_labels import MOVEMENT_CATALOGUE
except ModuleNotFoundError:
    from scripts.conversion.generate_protocol_labels import MOVEMENT_CATALOGUE

# Protocol definition
# Each block is a dict with:
# label        : movement label string (matches labels.csv)
# risk         : 0=safe, 1=risky, -1=ambiguous
# instruction  : displayed at block start
# n_reps       : number of reps
# phases       : list of (phase_name, duration_s) for ONE rep cycle
# phases run in order, repeated n_reps times
# section      : display section (e.g. "0A", "5.1")

PROTOCOL = [
    # Phase 0
    dict(
        section="0A", label="BASELINE_STATIC", risk=0,
        n_reps=1,
        phases=[
            ("STAND STILL  (eyes open)",  30),
            ("STAND STILL  (eyes closed)", 30),
        ],
        instruction=(
            "Stand feet shoulder-width apart, arms at sides, natural posture.\n"
            "  Eyes OPEN for 30 s, then CLOSE for 30 s.\n"
            "  Do not correct your posture — stand as you normally would."
        ),
    ),
    # Section 5 movements
    dict(
        section="5.1", label="CLEAN_FLEXION", risk=0,
        n_reps=8,
        phases=[
            ("DESCEND  (push hips back, spine long)", 3),
            ("HOLD at bottom",                        1),
            ("RETURN upright",                        3),
            ("REST standing",                        10),
        ],
        instruction=(
            "HIP-HINGE BENDS  (x8 reps)\n"
            "  Push hips back and hinge from the hip crease — keep spine long.\n"
            "  Lower until hands approach mid-shin. 3 s down, 1 s hold, 3 s return.\n"
            "  If you feel your lower back rounding, stop and restart the rep."
        ),
    ),
    dict(
        section="5.2", label="LUMBAR_DOMINANT", risk=1,
        n_reps=6,
        phases=[
            ("DESCEND  (pelvis STILL, lumbar bends)", 3),
            ("RETURN upright",                        3),
            ("REST standing",                        15),
        ],
        instruction=(
            "LUMBAR-DOMINANT BENDS  (x6 reps)  [RISKY CLASS]\n"
            "  Keep hips and pelvis AS STILL AS POSSIBLE.\n"
            "  Let your LOWER BACK do all the bending — reach toward your feet.\n"
            "  This is a designed compensation pattern — confirm pelvis is not tilting."
        ),
    ),
    dict(
        section="5.3L", label="CLEAN_LATERAL_L", risk=0,
        n_reps=6,
        phases=[
            ("BEND LEFT  (slide hand down thigh)", 2),
            ("HOLD",                               1),
            ("RETURN upright",                     2),
            ("REST",                              10),
        ],
        instruction=(
            "LATERAL BEND LEFT  (x6 reps)\n"
            "  Slide LEFT hand down the outside of the thigh toward the knee.\n"
            "  Keep shoulders and hips in the same plane — NO rotation, NO hip hike.\n"
            "  Target: 20-30 deg lateral trunk tilt."
        ),
    ),
    dict(
        section="5.3R", label="CLEAN_LATERAL_R", risk=0,
        n_reps=6,
        phases=[
            ("BEND RIGHT  (slide hand down thigh)", 2),
            ("HOLD",                                1),
            ("RETURN upright",                      2),
            ("REST",                               10),
        ],
        instruction=(
            "LATERAL BEND RIGHT  (x6 reps)\n"
            "  Same as left — slide RIGHT hand down the outside of thigh.\n"
            "  No hip hike. No rotation."
        ),
    ),
    dict(
        section="5.4L", label="CLEAN_ROTATION_L", risk=0,
        n_reps=6,
        phases=[
            ("ROTATE LEFT  (arms folded across chest)", 2),
            ("HOLD",                                    1),
            ("RETURN to centre",                        2),
            ("REST",                                   10),
        ],
        instruction=(
            "TRUNK ROTATION LEFT  (x6 reps)\n"
            "  Arms folded across chest (hands on opposite shoulders).\n"
            "  Rotate upper trunk LEFT smoothly. HIPS stay facing forward — do not pivot.\n"
            "  Target: 30-45 deg thoracic rotation."
        ),
    ),
    dict(
        section="5.4R", label="CLEAN_ROTATION_R", risk=0,
        n_reps=6,
        phases=[
            ("ROTATE RIGHT", 2),
            ("HOLD",         1),
            ("RETURN",       2),
            ("REST",        10),
        ],
        instruction=(
            "TRUNK ROTATION RIGHT  (x6 reps)\n"
            "  Same as left — rotate RIGHT this time. Hips forward, no foot pivot."
        ),
    ),
    dict(
        section="5.5", label="FAST_BEND", risk=1,
        n_reps=6,
        phases=[
            ("FAST FORWARD", 2),
            ("FAST RETURN",  2),
            ("REST",        15),
        ],
        instruction=(
            "FAST BENDS  (x6 reps)  [RISKY CLASS]\n"
            "  Bend forward as quickly as feels comfortable and natural, then return.\n"
            "  No depth target — trunk VELOCITY is the target feature, not range.\n"
            "  Do NOT rush to maximum depth."
        ),
    ),
    dict(
        section="5.6", label="SHOULDER_DRIVEN", risk=1,
        n_reps=5,
        phases=[
            ("ROUND SHOULDERS + COLLAPSE", 4),
            ("RETURN upright",             4),
            ("REST",                      15),
        ],
        instruction=(
            "SHOULDER-DRIVEN COLLAPSE  (x5 reps)  [RISKY CLASS]\n"
            "  Round your shoulders forward and let your upper body COLLAPSE down.\n"
            "  Reach toward the floor with a rounded back — minimal hip hinge.\n"
            "  The upper thoracic IMU should move disproportionately."
        ),
    ),
    dict(
        section="5.7A", label="PICKUP_SYM", risk=0,
        n_reps=5,
        phases=[
            ("BEND and PICK UP  (object straight ahead)", 4),
            ("RETURN upright, replace object",            4),
            ("REST",                                     15),
        ],
        instruction=(
            "SYMMETRIC PICK-UP  (x5 reps)\n"
            "  Object (0.5-1 kg) on floor directly in front.\n"
            "  Pick up and set down using any comfortable technique.\n"
            "  Object kept light — kinematics is the target, not muscular load."
        ),
    ),
    dict(
        section="5.7B", label="PICKUP_ASYM", risk=1,
        n_reps=5,
        phases=[
            ("REACH SIDE  (45 deg, 0.5 m out)", 4),
            ("RETURN upright, replace",          4),
            ("REST",                            15),
        ],
        instruction=(
            "ASYMMETRIC PICK-UP  (x5 reps)  [RISKY CLASS]\n"
            "  Object 45 deg to the SAME SIDE for all 5 reps, at ~0.5 m horizontal distance.\n"
            "  Pick up, return to upright, replace. Same side every rep.\n"
            "  This generates combined sagittal + coronal loading."
        ),
    ),
    dict(
        section="5.8A", label="SIT_TO_STAND_NORMAL", risk=0,
        n_reps=5,
        phases=[
            ("RISE to standing",    3),
            ("LOWER to seated",     3),
            ("REST seated",        10),
        ],
        instruction=(
            "SIT-TO-STAND — NORMAL PACE  (x5 reps)\n"
            "  Seated in chair (~45 cm), arms folded across chest (NO arm push).\n"
            "  Rise to full standing, lower back down. Comfortable self-paced.\n"
            "  Remove armrests or use an armrest-free chair."
        ),
    ),
    dict(
        section="5.8B", label="SIT_TO_STAND_FAST", risk=-1,
        n_reps=3,
        phases=[
            ("FAST RISE to standing", 2),
            ("LOWER to seated",       3),
            ("REST seated",          10),
        ],
        instruction=(
            "SIT-TO-STAND — FAST PACE  (x3 reps)  [AMBIGUOUS CLASS]\n"
            "  Same as above but rise as quickly as comfortable. No arm push.\n"
            "  This is labelled by extracted features, not movement name."
        ),
    ),
    # Rest buffer before fatigue block
    dict(
        section="REST", label="BASELINE_STATIC", risk=0,
        n_reps=1,
        phases=[("REST STANDING  (2 min break)", 120)],
        instruction=(
            "2-MINUTE REST BUFFER before the fatigue block.\n"
            "  Shake out any stiffness. Stay standing upright.\n"
            "  This is also captured as baseline windows."
        ),
    ),
    # Section 5.9
    dict(
        section="5.9", label="FATIGUE_FLEXION", risk=1,
        mode="timer",
        n_reps=20,          # target rep guidance only — NOT individually timed
        timer_s=120,        # safety cap (s); press any key to stop early and advance
        phases=[("BEND + RETURN  (no rest)", 8)],  # retained for duration estimate
        instruction=(
            "REPETITIVE FATIGUE BENDS  (target ~20 continuous)  [RISKY CLASS - escalating]\n"
            "  Continuous forward bending at your comfortable self-paced cadence.\n"
            "  NO rest between reps. A single timer runs for this whole section.\n"
            "  >> PRESS ANY KEY when your bends are done to stop and move on to the\n"
            "     final baseline. (Auto-advances at the safety cap if no key.)\n"
            "  STOP immediately at any discomfort >= 3/10 on Borg CR10."
        ),
    ),
    # Terminal drift-check baseline
    dict(
        section="BL2", label="BASELINE_STATIC", risk=0,
        n_reps=1,
        phases=[("FINAL STILL STAND  (BL2 drift check)", 60)],
        instruction=(
            "FINAL BASELINE / BL2 STATIC  (60 s)\n"
            "  Stand upright and still, feet shoulder-width apart, arms relaxed.\n"
            "  This is the end-of-session drift-check reference.\n"
            "  Do not talk, step, bend, adjust sensors, or remove the belt until the timer ends."
        ),
    ),
]

for _block in PROTOCOL:
    _approved = MOVEMENT_CATALOGUE.get(_block["label"])
    if _approved is None:
        raise RuntimeError(f"Official timer contains unsupported protocol task: {_block['label']}")
    if _approved["risk_class"] != _block["risk"]:
        raise RuntimeError(
            f"Official timer risk mismatch for {_block['label']}: "
            f"timer={_block['risk']}, catalogue={_approved['risk_class']}"
        )

# Helpers

def hms(secs: float) -> str:
    secs = int(secs)
    m, s = divmod(secs, 60)
    return f"{m}:{s:02d}"


def block_duration(block: dict) -> int:
    # Timer-mode blocks are bounded by their safety cap (timer_s), not by
    # n_reps * phases, so the launcher sizes recorders to the real worst case.
    if block.get("mode") == "timer":
        return int(block.get("timer_s",
                             block["n_reps"] * sum(d for _, d in block["phases"])))
    return block["n_reps"] * sum(d for _, d in block["phases"])


def total_duration(blocks) -> int:
    return sum(block_duration(b) for b in blocks)


RISK_TAG = {0: "SAFE", 1: "RISKY", -1: "AMBIGUOUS"}
RISK_COL = {0: "", 1: "  *** RISKY CLASS ***", -1: "  (ambiguous)"}


def cue_beep(duration_ms: int = 120, frequency_hz: int = 880):
    """Play a non-blocking audible cue."""
    try:
        import winsound

        threading.Thread(
            target=winsound.Beep,
            args=(frequency_hz, duration_ms),
            daemon=True,
        ).start()
    except Exception:
        sys.stdout.write("\a")
        sys.stdout.flush()


def rep_start_beep():
    """Play a short non-blocking cue for the start of a rep."""
    cue_beep(120, 880)


def phase_start_beep(phase_name: str):
    """Cue movement sub-phases: long for holds, short for descend/return."""
    name = phase_name.lower()
    if "hold" in name:
        cue_beep(350, 740)
    elif any(word in name for word in ("descend", "return", "bend", "rotate", "rise", "lower", "reach", "fast forward", "fast return")):
        cue_beep(60, 980)
    elif "eyes closed" in name:
        cue_beep(500, 660)


# Display

def show_block_header(block: dict, block_idx: int, n_blocks: int,
                      t_elapsed: int, t_total: int):
    risk_str = RISK_COL[block["risk"]]
    print()
    print("=" * 62)
    print(f"  [{hms(t_elapsed)}]  SECTION {block['section']}  "
          f"({block_idx}/{n_blocks}){risk_str}")
    print(f"  {block['label']}  —  {block['n_reps']} reps  "
          f"({block_duration(block)} s)  "
          f"| session end: {hms(t_total)}")
    print("=" * 62)
    for line in block["instruction"].split("\n"):
        print(f"  {line}")
    print()


def count_phase(phase_name: str, duration_s: int,
                rep_num: int, n_reps: int,
                t_elapsed: int,
                t_ms: int,
                key_reader,
                seg_elapsed: int = 0, seg_total: int = 0):
    """Count down one phase, showing rep timer + segment timer simultaneously."""
    phase_start_beep(phase_name)
    pause_ms = 0
    pause_rows = []
    for tick in range(duration_s):
        # rep/phase timer (counts down within this phase)
        phase_remain = duration_s - tick
        phase_fill   = int(12 * tick / duration_s) if duration_s > 1 else 12
        phase_bar    = "█" * phase_fill + "░" * (12 - phase_fill)

        # segment timer (counts up since block started)
        seg_so_far   = seg_elapsed + tick
        seg_pct      = seg_so_far / seg_total if seg_total > 0 else 0
        seg_fill     = int(12 * seg_pct)
        seg_bar      = "█" * seg_fill + "░" * (12 - seg_fill)

        rep_label = f"Rep {rep_num}/{n_reps}" if n_reps > 1 else "      "
        sys.stdout.write(
            f"\r  [{hms(t_elapsed + tick)}]  {rep_label}  "
            f"{phase_name:<28}"
            f"  REP [{phase_bar}] {phase_remain:2d}s"
            f"  |  SEG [{seg_bar}] {seg_so_far}/{seg_total}s"
        )
        sys.stdout.flush()
        second_start = time.monotonic()
        while time.monotonic() - second_start < 1.0:
            if any(k.lower() == "p" for k in key_reader()):
                pause_start_wall = time.monotonic()
                pause_start_ms = t_ms + tick * 1000 + pause_ms
                cue_beep(160, 520)
                print(f"\n  [PAUSED] Protocol paused during {phase_name}. Press P to resume.")
                while True:
                    time.sleep(0.1)
                    if any(k.lower() == "p" for k in key_reader()):
                        break
                elapsed_pause_ms = int(round((time.monotonic() - pause_start_wall) * 1000))
                pause_ms += elapsed_pause_ms
                pause_rows.append(dict(
                    label="PAUSE", rep=0,
                    start_ms=pause_start_ms, end_ms=pause_start_ms + elapsed_pause_ms,
                    risk_class=-1, fatigue_fraction="",
                ))
                cue_beep(100, 880)
                print("  [RESUMED] Protocol timer running.")
                second_start = time.monotonic()
            time.sleep(0.05)
    print()  # newline after phase completes
    return pause_ms, pause_rows


def _make_key_reader():
    """Return (read_keys, cleanup) for non-blocking single-key controls."""
    try:
        import msvcrt

        def read_keys():
            keys = []
            while msvcrt.kbhit():
                ch = msvcrt.getch()
                try:
                    keys.append(ch.decode("utf-8", errors="ignore"))
                except Exception:
                    pass
            return keys

        return read_keys, (lambda: None)
    except ImportError:
        import select
        import termios
        import tty

        if not sys.stdin.isatty():
            return (lambda: []), (lambda: None)

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        def read_keys():
            keys = []
            while select.select([sys.stdin], [], [], 0)[0]:
                keys.append(sys.stdin.read(1))
            return keys

        def cleanup():
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

        return read_keys, cleanup


def _make_key_poller():
    """Return (poll, cleanup).

    poll() -> True if any key has been pressed since the last call (non-blocking),
    draining the buffer so a single press registers once. cleanup() restores the
    terminal. Works on Windows (msvcrt) and POSIX (termios/select); degrades to a
    no-op poller if stdin is not an interactive TTY.
    """
    try:
        import msvcrt  # Windows

        def poll():
            pressed = False
            while msvcrt.kbhit():
                msvcrt.getch()
                pressed = True
            return pressed

        return poll, (lambda: None)
    except ImportError:
        import select
        import termios
        import tty

        if not sys.stdin.isatty():
            # Non-interactive (e.g. piped/automated), cannot read keys.
            return (lambda: False), (lambda: None)

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        def poll():
            pressed = False
            while select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.read(1)
                pressed = True
            return pressed

        def cleanup():
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

        return poll, cleanup


def run_timer_block(block: dict, t_elapsed: int, t_ms: int) -> tuple[int, int, list]:
    """Single count-up timer for a whole section, stoppable by any keypress.

    Returns the ACTUAL elapsed seconds so that labels.csv stays aligned with the
    real recording timeline. Auto-advances at block['timer_s'] if no key is hit.
    """
    timer_s = int(block.get("timer_s", block_duration(block)))
    phase_name = block["phases"][0][0]
    phase_start_beep(phase_name)
    key_reader, cleanup = _make_key_reader()
    start = time.monotonic()
    pause_ms = 0
    pause_rows = []
    stopped_early = False
    try:
        while True:
            elapsed = time.monotonic() - start
            if elapsed >= timer_s:
                break
            keys = key_reader()
            if any(k.lower() == "p" for k in keys):
                pause_start_wall = time.monotonic()
                pause_start_ms = t_ms + int(round(elapsed * 1000)) + pause_ms
                cue_beep(160, 520)
                print("\n  [PAUSED] Fatigue timer paused. Press P to resume.")
                while True:
                    time.sleep(0.1)
                    if any(k.lower() == "p" for k in key_reader()):
                        break
                elapsed_pause_ms = int(round((time.monotonic() - pause_start_wall) * 1000))
                pause_ms += elapsed_pause_ms
                pause_rows.append(dict(
                    label="PAUSE", rep=0,
                    start_ms=pause_start_ms, end_ms=pause_start_ms + elapsed_pause_ms,
                    risk_class=-1, fatigue_fraction="",
                ))
                cue_beep(100, 880)
                print("  [RESUMED] Fatigue timer running.")
                start = time.monotonic() - elapsed
            elif keys:
                stopped_early = True
                break
            seg_pct = elapsed / timer_s if timer_s > 0 else 0
            seg_fill = int(12 * seg_pct)
            seg_bar = "█" * seg_fill + "░" * (12 - seg_fill)
            sys.stdout.write(
                f"\r  [{hms(t_elapsed + elapsed)}]  {phase_name:<28}"
                f"  [{seg_bar}]  {int(elapsed):3d}s / {timer_s}s cap"
                f"   (press any key to finish)"
            )
            sys.stdout.flush()
            time.sleep(0.1)
    finally:
        cleanup()

    actual = max(1, int(round(min(time.monotonic() - start, timer_s))))
    print()
    why = "key pressed" if stopped_early else "reached safety cap"
    print(f"  [{hms(t_elapsed + actual)}]  Fatigue timer stopped after "
          f"{actual}s ({why}).")
    return actual, pause_ms, pause_rows


# Main

def run(
    skip_baseline: bool = False,
    out_path: Path = None,
    auto_start: bool = False,
    start_at_unix: float = None,
    start_section: str = None,
):
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

    t_total = total_duration(blocks)

    print()
    print("=" * 62)
    print("  SPINAL MOVEMENT RISK MONITOR  --  Full Session Timer")
    print(f"  Protocol:   {len(blocks)} blocks  "
          f"({sum(1 for b in blocks if b['risk']==0)} safe  +  "
          f"{sum(1 for b in blocks if b['risk']==1)} risky  +  "
          f"{sum(1 for b in blocks if b['risk']==-1)} ambiguous)")
    print(f"  Duration:   {t_total // 60} min {t_total % 60} s")
    print("=" * 62)
    print()
    out_note = str(out_path) if out_path else "(not saved -- use --out to auto-save)"
    print(f"  labels.csv  : {out_note}")
    print()
    print("  Workflow:")
    print("   Terminal 1:  py scripts/acquisition/record_imu_serial.py --port COM3 --duration <seconds> --out <path>")
    print("   Terminal 2:  py scripts/acquisition/ganglion_stream.py --port COM4 --duration <seconds> --out <path>")
    print("   Terminal 3:  py scripts/acquisition/session_timer.py --out <labels_path>  (this window)")
    print()
    if auto_start or start_at_unix is not None:
        if start_at_unix is None:
            start_at_unix = time.time()
        wait_s = start_at_unix - time.time()
        if wait_s > 0:
            print(f"  Auto-start armed for {time.strftime('%H:%M:%S', time.localtime(start_at_unix))}.")
            while True:
                remaining = start_at_unix - time.time()
                if remaining <= 0:
                    break
                if remaining > 3:
                    sys.stdout.write(f"\r  Starting in {remaining:5.1f} s")
                    sys.stdout.flush()
                    time.sleep(min(0.5, remaining))
                else:
                    sys.stdout.write(f"\r  Starting in {remaining:5.1f} s")
                    sys.stdout.flush()
                    time.sleep(min(0.1, remaining))
            print("\n\n  GO!\n")
        else:
            print(f"  Scheduled start has already passed by {-wait_s:.2f} s.")
            print("\n  GO!\n")
    else:
        print("  Start both recordings, then press Enter here.")
        input("\n  >> Press Enter when recording is running: ")

        print("\n  Starting in 3...")
        time.sleep(1)
        print("  Starting in 2...")
        time.sleep(1)
        print("  Starting in 1...")
        time.sleep(1)
        print("\n  GO!\n")

    t_elapsed = 0
    t_ms = 0
    rows = []  # labels.csv rows, captured live so they match the REAL timeline
    pause_events = []
    key_reader, key_cleanup = _make_key_reader()

    try:
      for block_idx, block in enumerate(blocks, 1):
        show_block_header(block, block_idx, len(blocks), t_elapsed, t_total)

        if block.get("mode") == "timer":
            # Single stoppable timer for the whole section (e.g. fatigue bends).
            actual_s, timer_pause_ms, timer_pause_rows = run_timer_block(block, t_elapsed, t_ms)
            pause_events.extend(timer_pause_rows)
            rows.append(dict(
                label=block["label"], rep=1,
                start_ms=t_ms, end_ms=t_ms + actual_s * 1000 + timer_pause_ms,
                risk_class=block["risk"], fatigue_fraction="",
            ))
            t_ms      += actual_s * 1000 + timer_pause_ms
            t_elapsed += actual_s
        else:
            seg_total   = block_duration(block)
            seg_elapsed = 0
            rep_total_s = sum(d for _, d in block["phases"])
            for rep in range(1, block["n_reps"] + 1):
                if block["n_reps"] > 1 and len(block["phases"]) > 1:
                    print(f"  -- Rep {rep}/{block['n_reps']} --")

                rep_start_ms = t_ms
                for phase_name, phase_s in block["phases"]:
                    phase_pause_ms, phase_pause_rows = count_phase(
                                phase_name, phase_s, rep, block["n_reps"],
                                t_elapsed, t_ms, key_reader,
                                seg_elapsed=seg_elapsed, seg_total=seg_total)
                    pause_events.extend(phase_pause_rows)
                    t_elapsed   += phase_s
                    seg_elapsed += phase_s
                    t_ms        += phase_s * 1000 + phase_pause_ms

                rows.append(dict(
                    label=block["label"], rep=rep,
                    start_ms=rep_start_ms, end_ms=t_ms,
                    risk_class=block["risk"], fatigue_fraction="",
                ))

        tag = RISK_TAG[block["risk"]]
        print(f"  [{hms(t_elapsed)}]  Section {block['section']} complete  [{tag}]")

        # Preview next block
        if block_idx < len(blocks):
            nxt = blocks[block_idx]
            print(f"\n  NEXT:  {nxt['section']}  {nxt['label']}"
                  f"  ({nxt['n_reps']} reps, {block_duration(nxt)} s)")
            time.sleep(1)
    finally:
        key_cleanup()

    print()
    print("=" * 62)
    print(f"  [{hms(t_elapsed)}]  RECORDING COMPLETE")
    print("  Stop the IMU and Ganglion recorders now.")
    print("=" * 62)

    # Write labels.csv (rows captured live during the run)
    # rows[] was built against the ACTUAL elapsed timeline, so the early-stopped
    # fatigue block and the final BL2 baseline stay correctly aligned.
    print()
    print("  start_ms,end_ms,label,rep,risk_class")
    for r in rows:
        print(f"  {r['start_ms']},{r['end_ms']},{r['label']},{r['rep']},{r['risk_class']}")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import csv
        fieldnames = ["label","rep","start_ms","end_ms","risk_class","fatigue_fraction"]
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\n  labels.csv written -> {out_path}")
        if pause_events:
            pause_path = out_path.with_name("pause_events.csv")
            with open(pause_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(pause_events)
            print(f"  pause_events.csv written -> {pause_path}")
    else:
        print("\n  (Use --out to auto-save labels.csv instead of copy-pasting)")
    print()


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full-protocol session timer for spinal movement recording."
    )
    parser.add_argument(
        "--skip_baseline", action="store_true",
        help="Skip required baseline for rehearsal only; resulting labels will fail official validation.",
    )
    parser.add_argument(
        "--out", "-o", default=None,
        help="Write labels.csv here automatically (e.g. data/real/raw/session_003/labels.csv)",
    )
    parser.add_argument(
        "--auto_start", action="store_true",
        help="Start without waiting for Enter. Use with --start_at_unix for synchronised launch.",
    )
    parser.add_argument(
        "--start_at_unix", type=float, default=None,
        help="Optional Unix timestamp for a software-synchronised protocol start.",
    )
    parser.add_argument(
        "--start_section", default=None,
        help="Resume/rehearsal start point, e.g. 5.3L or CLEAN_LATERAL_L. Labels start at 0 for this resumed file.",
    )
    args = parser.parse_args()
    run(
        skip_baseline=args.skip_baseline,
        out_path=Path(args.out) if args.out else None,
        auto_start=args.auto_start,
        start_at_unix=args.start_at_unix,
        start_section=args.start_section,
    )
