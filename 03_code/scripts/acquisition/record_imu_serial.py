#!/usr/bin/env python3
"""
record_imu_serial.py
====================
Spinal Movement Risk Monitor, FYP 2025/26

Records the CSV stream from imu_reader.ino over USB serial and saves it to a
file. Designed to run in parallel with ganglion_stream.py so both recordings
start at roughly the same time.

Usage
-----
    # Real hardware (ESP32-S3 on COM3 / /dev/ttyUSB0)
    py scripts/acquisition/record_imu_serial.py --port COM3 --out data/real/raw/session_001/imu_arduino.csv

    # With explicit duration (auto-stop)
    py scripts/acquisition/record_imu_serial.py --port COM3 --duration 120 --out data/real/raw/session_001/imu_arduino.csv

    # Verbose: show every line as it arrives
    py scripts/acquisition/record_imu_serial.py --port COM3 --out data/real/raw/session_001/imu_arduino.csv --verbose

Stop recording
--------------
    Press Ctrl+C at any time. The file is written incrementally (flushed every
    second), so a forced exit will not lose data.

Output file
-----------
    CSV with columns matching imu_reader.ino output:
        t_ms, Pelvis_ax, Pelvis_ay, Pelvis_az, Pelvis_gx, Pelvis_gy, Pelvis_gz,
              L3_ax, ..., T4_gz

    Comment lines from the Arduino (starting with #) are saved with a leading #
    for reference but are skipped by RawConverter.from_csv().

Find your COM port
------------------
    Windows  : Device Manager → Ports (COM & LPT) → look for "CP210x" or "CH340"
    Linux    : ls /dev/ttyUSB* or ls /dev/ttyACM*
    macOS    : ls /dev/cu.usbserial-*

Requirements
------------
    py -m pip install pyserial
"""

import argparse
import sys
import time
import signal
from datetime import datetime
from pathlib import Path


def _check_pyserial():
    """Fail early with an install hint if pyserial is missing — it is an optional acquisition-only dependency, not in the core pipeline requirements."""
    try:
        import serial  # noqa: F401
    except ImportError:
        print("[ERROR] pyserial is not installed.")
        print("  Run: py -m pip install pyserial")
        sys.exit(1)


# Known CSV header for the 4-IMU PLTU system (25 fields).
# Used as a fallback when the ESP32-S3 is already streaming before Python
# connects (native USB, no auto-reset on port open).
FALLBACK_HEADER = (
    "t_ms,"
    "Pelvis_ax,Pelvis_ay,Pelvis_az,Pelvis_gx,Pelvis_gy,Pelvis_gz,"
    "L3_ax,L3_ay,L3_az,L3_gx,L3_gy,L3_gz,"
    "T12_ax,T12_ay,T12_az,T12_gx,T12_gy,T12_gz,"
    "T4_ax,T4_ay,T4_az,T4_gx,T4_gy,T4_gz"
)
FALLBACK_FIELDS = len(FALLBACK_HEADER.split(","))  # 25


def record(
    port: str,
    output_path: str,
    baud: int = 115200,
    duration: float = None,
    verbose: bool = False,
    flush_interval: float = 1.0,
    start_at_unix: float = None,
    force: bool = False,
) -> None:
    """
    Connect to the ESP32 serial port and stream data to a CSV file.

    Parameters
    ----------
    port        : serial port name (e.g. "COM3" or "/dev/ttyUSB0")
    output_path : destination CSV file path
    baud        : baud rate (must match firmware, 115200)
    duration    : recording duration in seconds; None = run until Ctrl+C
    verbose     : if True, print every data line to stdout
    flush_interval : seconds between file flushes
    start_at_unix : optional Unix timestamp for scheduled recording start
    force       : if True, overwrite an existing output file
    """
    _check_pyserial()
    import serial

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"[ERROR] Refusing to overwrite existing output file: {out_path}")
        print("  Re-run with --force only if this is intentional.")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"IMU Serial Recorder")
    print(f"  Port       : {port}")
    print(f"  Baud       : {baud}")
    print(f"  Duration   : {'unlimited (Ctrl+C to stop)' if duration is None else f'{duration:.0f} s'}")
    print(f"  Output     : {out_path}")
    print(f"  Force      : {force}")
    if start_at_unix is not None:
        print(f"  Start at   : {datetime.fromtimestamp(start_at_unix).isoformat(timespec='milliseconds')}")
    print(f"{'='*55}\n")

    # Connect
    try:
        ser = serial.Serial(port, baud, timeout=2.0)
        # DTR=True signals "host is ready", ESP32-S3 native USB will not
        # transmit if DTR is False. Auto-reset at 115200 baud is not triggered
        # by DTR on native USB (only the 1200-baud bootloader trick does that).
        ser.dtr = True
        ser.rts = False
        print(f"[{_ts()}] Connected to {port} at {baud} baud.")
        print(f"[{_ts()}] Waiting 3 s for ESP32 to boot...")
        time.sleep(3.0)   # ESP32-S3 native USB needs time to reboot after port open
        ser.reset_input_buffer()   # discard any garbled bytes from boot
    except serial.SerialException as e:
        print(f"[ERROR] Could not open {port}: {e}")
        print("\nTroubleshooting:")
        print("  1. Check ESP32 is connected and powered (LED on)")
        print("  2. Verify the COM port in Device Manager / ls /dev/ttyUSB*")
        print("  3. Ensure no other program has the port open (Arduino IDE, etc.)")
        sys.exit(1)

    # State
    if start_at_unix is not None:
        wait_s = start_at_unix - time.time()
        if wait_s > 0:
            print(f"[{_ts()}] Armed. Waiting {wait_s:.2f} s for scheduled session start...")
            next_drain = time.time()
            while True:
                now = time.time()
                remaining = start_at_unix - now
                if remaining <= 0:
                    break
                if now >= next_drain:
                    ser.reset_input_buffer()
                    next_drain = now + 0.5
                time.sleep(min(0.05, remaining))
            ser.reset_input_buffer()
        else:
            print(f"[{_ts()}] Scheduled start has already passed by {-wait_s:.2f} s; starting now.")

    n_samples     = 0
    n_bad         = 0
    header_seen   = False
    header_line   = None
    t_start_wall  = time.time()
    t_last_flush  = t_start_wall
    t_last_status = t_start_wall
    stop_requested = False

    def _on_sigint(sig, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGINT, _on_sigint)

    print(f"[{_ts()}] Waiting for Arduino to start streaming...\n")

    # Record
    file_mode = "w" if force else "x"
    with open(out_path, file_mode, newline="", encoding="utf-8") as f_out:

        while not stop_requested:

            # Auto-stop after duration
            if duration is not None and (time.time() - t_start_wall) >= duration:
                print(f"\n[{_ts()}] Duration reached ({duration:.0f} s). Stopping.")
                break

            try:
                raw_line = ser.readline()
            except serial.SerialException as e:
                print(f"\n[ERROR] Serial read failed: {e}")
                break

            if not raw_line:
                continue

            try:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            except Exception:
                continue

            if not line:
                continue

            # Comment / header lines from Arduino
            if line.startswith("#"):
                print(f"  {line}")   # show firmware messages (startup, errors)
                f_out.write(line + "\n")
                continue

            # CSV header
            if not header_seen and line.startswith("t_ms"):
                header_line = line
                header_seen = True
                f_out.write(header + "\n" if (header := line) else "")
                expected_fields = len(line.split(","))
                print(f"[{_ts()}] Header received ({expected_fields} fields). Recording...\n")
                continue

            if not header_seen:
                # ESP32-S3 native USB: board is already streaming when Python
                # connects, so the boot-time header is missed.
                # Fallback: if this looks like a valid 25-field data row,
                # inject the known header and start recording immediately.
                fields_peek = line.split(",")
                if len(fields_peek) == FALLBACK_FIELDS:
                    try:
                        int(fields_peek[0])   # t_ms must be numeric
                        # Looks valid, inject fallback header
                        header_seen    = True
                        expected_fields = FALLBACK_FIELDS
                        f_out.write("# [FALLBACK] Header not received from board; "
                                    "using known PLTU header\n")
                        f_out.write(FALLBACK_HEADER + "\n")
                        print(f"[{_ts()}] ** Fallback header injected "
                              f"({FALLBACK_FIELDS} fields). Recording...\n")
                        # Fall through to write this line as a data row
                        f_out.write(line + "\n")
                        n_samples += 1
                        if verbose:
                            print(f"  {line}")
                    except ValueError:
                        pass   # first field not numeric — skip
                continue

            # Data line
            fields = line.split(",")
            if len(fields) != expected_fields:
                n_bad += 1
                if verbose:
                    print(f"  [SKIP] Malformed line ({len(fields)} fields): {line[:60]}")
                continue

            f_out.write(line + "\n")
            n_samples += 1

            if verbose:
                print(f"  {line}")

            # Periodic flush
            now = time.time()
            if now - t_last_flush >= flush_interval:
                f_out.flush()
                t_last_flush = now

            # Status line every 5 s
            if now - t_last_status >= 5.0:
                elapsed    = now - t_start_wall
                rate_hz    = n_samples / elapsed if elapsed > 0 else 0
                size_kb    = out_path.stat().st_size / 1024 if out_path.exists() else 0
                print(f"[{_ts()}]  {n_samples:>6} samples  |  "
                      f"{rate_hz:.1f} Hz  |  {elapsed:.0f} s  |  {size_kb:.1f} KB", end="")
                if n_bad:
                    print(f"  |  {n_bad} bad lines", end="")
                print()
                t_last_status = now

        # Final flush
        f_out.flush()

    ser.close()

    elapsed = time.time() - t_start_wall
    rate_hz = n_samples / elapsed if elapsed > 0 else 0
    size_kb = out_path.stat().st_size / 1024 if out_path.exists() else 0

    print(f"\n{'='*55}")
    print(f"Recording complete")
    print(f"  Samples    : {n_samples}")
    print(f"  Duration   : {elapsed:.1f} s")
    print(f"  Actual fs  : {rate_hz:.1f} Hz  (target: 100 Hz)")
    print(f"  Bad lines  : {n_bad}")
    print(f"  File       : {out_path}  ({size_kb:.1f} KB)")
    print(f"{'='*55}")

    if not header_seen:
        print("\n[WARNING] No valid header received. File may be empty or corrupt.")
        print("  Check that the ESP32 is running imu_reader.ino and the baud rate matches.")
    elif n_samples == 0:
        print("\n[WARNING] Header found but no data samples received.")
        print("  Check that all IMUs initialised correctly (look for 'OK' in startup messages).")
    else:
        actual_rate = n_samples / elapsed if elapsed > 0 else 0
        if actual_rate < 85:
            print(f"\n[WARNING] Sample rate {actual_rate:.1f} Hz is well below 100 Hz target.")
            print("  Possible causes:")
            print("  - Long USB cable introducing latency")
            print("  - Serial buffer overflow (try shorter recording, or increase baud to 230400)")
            print("  - I2C contention (check wiring, reduce cable length to sensors)")
        print(f"\n[OK] Use session_converter.py to convert this file to pipeline format:")
        print(f"  py scripts/conversion/session_converter.py --ganglion <ganglion.csv> --imu {out_path}")


def _ts() -> str:
    """Wall-clock HH:MM:SS stamp for the live status prints during a recording."""
    return datetime.now().strftime("%H:%M:%S")


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record ESP32 IMU serial stream to CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port",     required=True,
                        help="Serial port (e.g. COM3, /dev/ttyUSB0)")
    parser.add_argument("--out",      required=True,
                        help="Output CSV path (e.g. data/real/raw/session_001/imu_arduino.csv)")
    parser.add_argument("--baud",     type=int, default=115200,
                        help="Baud rate (must match firmware)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Recording duration in seconds (default: unlimited)")
    parser.add_argument("--verbose",  action="store_true",
                        help="Print every data line to stdout")
    parser.add_argument("--start_at_unix", type=float, default=None,
                        help="Optional Unix timestamp for a software-synchronised start.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite the output CSV if it already exists.")
    args = parser.parse_args()

    record(
        port        = args.port,
        output_path = args.out,
        baud        = args.baud,
        duration    = args.duration,
        verbose     = args.verbose,
        start_at_unix = args.start_at_unix,
        force       = args.force,
    )
