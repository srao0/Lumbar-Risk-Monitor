"""
ganglion_stream.py
------------------
Continuous BrainFlow acquisition for OpenBCI Ganglion recordings.

The recorded CSV preserves the existing processing-pipeline format while data
are persisted throughout capture. The BrainFlow ring buffer is drained at a
short interval, so long Phase II sessions do not depend on its fixed capacity.

Usage - real hardware:
    py scripts/acquisition/ganglion_stream.py --port COM4 --duration 1800 \
        --out data/real/raw/participant_01/session_001/ganglion.csv

Usage - no hardware:
    py scripts/acquisition/ganglion_stream.py --synthetic --duration 10 \
        --out data/real/raw/session_synthetic/ganglion.csv

Requirements:
    py -m pip install brainflow pandas numpy
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowError, BrainFlowInputParams


GANGLION_ID = BoardIds.GANGLION_BOARD.value      # Real hardware: 200 Hz
CYTON_ID = BoardIds.CYTON_BOARD.value            # Real hardware: 250 Hz
SYNTHETIC_ID = BoardIds.SYNTHETIC_BOARD.value    # Test board: 250 Hz
RING_BUFFER_SAMPLES = 45000
FAIL_COMPLETENESS_RATIO = 0.90
WARN_COMPLETENESS_RATIO = 0.98


def build_params(
    serial_port: str,
    mac_address: str = "",
    timeout: int = 15,
    native: bool = False,
) -> BrainFlowInputParams:
    """Build connection parameters for Ganglion dongle or native Bluetooth."""
    params = BrainFlowInputParams()
    if not native:
        params.serial_port = serial_port
    if mac_address:
        params.mac_address = mac_address
    params.timeout = 30 if native else timeout
    if native:
        params.other_info = "2"
    return params


def get_column_names(board_id: int) -> list:
    """Return the headed CSV schema consumed by session_converter.py."""
    exg_channels = BoardShim.get_exg_channels(board_id)
    accel_channels = BoardShim.get_accel_channels(board_id)
    timestamp_channel = BoardShim.get_timestamp_channel(board_id)

    name_map = {}
    for i, ch in enumerate(exg_channels):
        name_map[ch] = f"emg_ch{i + 1}"
    for i, ch in enumerate(accel_channels):
        name_map[ch] = ["accel_x", "accel_y", "accel_z"][i] if i < 3 else f"accel_{i}"
    name_map[timestamp_channel] = "timestamp_unix"

    return [name_map.get(i, f"col_{i}") for i in range(BoardShim.get_num_rows(board_id))]


def stream_and_save(
    board_id: int,
    serial_port: str,
    duration: float,
    output_path: str,
    synthetic: bool,
    mac_address: str = "",
    flush_interval: float = 1.0,
    force: bool = False,
    native: bool = False,
    board_name: str = "OpenBCI",
    config_strings=None,
    start_at_unix: float = None,
) -> bool:
    """Record continuously to a pipeline-compatible CSV and validate completeness."""
    BoardShim.disable_board_logger()

    out_path = Path(output_path)
    if duration <= 0:
        print("[ERROR] --duration must be greater than zero.")
        return False
    if flush_interval <= 0:
        print("[ERROR] --flush_interval must be greater than zero.")
        return False
    if out_path.exists() and not force:
        print(f"[ERROR] Output already exists: {out_path}")
        print("        Refusing to overwrite a recording. Use a new path or pass --force explicitly.")
        return False

    params = build_params(
        serial_port,
        mac_address=mac_address if not synthetic else "",
        native=native,
    )
    board = BoardShim(board_id, params)
    if synthetic:
        mode_label = "SYNTHETIC board"
    elif native:
        mode_label = f"{board_name} via native Bluetooth"
    else:
        mode_label = f"{board_name} on {serial_port}"
    expected_fs = BoardShim.get_sampling_rate(board_id)
    expected_samples = int(round(duration * expected_fs))

    print(f"[{datetime.now():%H:%M:%S}] Preparing session - {mode_label}")
    print(f"  Board ID:         {board_id}")
    print(f"  Expected fs:      {expected_fs} Hz")
    print(f"  Duration:         {duration:g} s")
    print(f"  Expected samples: {expected_samples}")
    print(f"  Output:           {out_path}")
    print(f"  Persistence:      append to CSV every <= {flush_interval:g} s")
    if start_at_unix is not None:
        print(f"  Start at:         {datetime.fromtimestamp(start_at_unix).isoformat(timespec='milliseconds')}")
    print()

    try:
        board.prepare_session()
    except BrainFlowError as exc:
        print(f"[ERROR] Could not connect: {exc}")
        if not synthetic:
            print(f"Check the dongle, COM port and {board_name} power.")
        return False

    time.sleep(2)

    if config_strings:
        print(f"[{datetime.now():%H:%M:%S}] Sending {len(config_strings)} board config command(s)...")
        for cfg in config_strings:
            try:
                response = board.config_board(cfg)
            except BrainFlowError as exc:
                print(f"[ERROR] Board rejected config '{cfg}': {exc}")
                board.release_session()
                return False
            response_text = response.strip() if isinstance(response, str) and response else "(no response)"
            print(f"  config '{cfg}' -> {response_text}")
            if isinstance(response, str) and "Failure" in response:
                print(f"[ERROR] Cyton firmware reported failure for config '{cfg}'. Aborting.")
                board.release_session()
                return False
        time.sleep(0.5)

    if start_at_unix is not None:
        wait_s = start_at_unix - time.time()
        if wait_s > 0:
            print(f"[{datetime.now():%H:%M:%S}] Armed. Waiting {wait_s:.2f} s for scheduled session start...")
            while True:
                remaining = start_at_unix - time.time()
                if remaining <= 0:
                    break
                time.sleep(min(0.05, remaining))
        else:
            print(f"[{datetime.now():%H:%M:%S}] Scheduled start has already passed by {-wait_s:.2f} s; starting now.")

    try:
        board.start_stream(RING_BUFFER_SAMPLES)
    except BrainFlowError as exc:
        print(f"[ERROR] Could not start stream: {exc}")
        board.release_session()
        return False

    col_names = get_column_names(board_id)
    timestamp_channel = BoardShim.get_timestamp_channel(board_id)
    columns = col_names + ["datetime"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(out_path, mode="w" if force else "x", index=False)

    first_timestamp = None
    last_timestamp = None
    previous_timestamp = None
    timestamps_monotonic = True
    n_samples = 0
    completed = True

    def append_buffered_data() -> int:
        nonlocal first_timestamp, last_timestamp, previous_timestamp, timestamps_monotonic, n_samples
        raw_data = board.get_board_data()
        if raw_data.shape[1] == 0:
            return 0

        timestamps = raw_data[timestamp_channel]
        if previous_timestamp is not None and timestamps[0] <= previous_timestamp:
            timestamps_monotonic = False
        if not np.all(np.diff(timestamps) > 0):
            timestamps_monotonic = False
        if first_timestamp is None:
            first_timestamp = float(timestamps[0])
        last_timestamp = float(timestamps[-1])
        previous_timestamp = float(timestamps[-1])

        frame = pd.DataFrame(raw_data.T, columns=col_names)
        frame["datetime"] = pd.to_datetime(frame["timestamp_unix"], unit="s")
        frame.to_csv(out_path, mode="a", header=False, index=False)
        n_samples += raw_data.shape[1]
        return raw_data.shape[1]

    print(f"[{datetime.now():%H:%M:%S}] Streaming...")
    started = time.monotonic()
    next_status = started + 10.0
    try:
        while True:
            remaining = duration - (time.monotonic() - started)
            if remaining <= 0:
                break
            time.sleep(min(flush_interval, remaining))
            append_buffered_data()
            if time.monotonic() >= next_status:
                print(f"  Saved so far: {n_samples} samples")
                next_status += 10.0
    except KeyboardInterrupt:
        completed = False
        print("\n[WARNING] Recording stopped by user. Preserving the partial recording.")
    finally:
        board.stop_stream()
        append_buffered_data()
        board.release_session()

    print(f"[{datetime.now():%H:%M:%S}] Done. Samples saved: {n_samples}")
    if n_samples == 0:
        print("[FAIL] No samples received; recording is invalid.")
        return False

    actual_duration = last_timestamp - first_timestamp if n_samples > 1 else 0.0
    actual_rate = n_samples / actual_duration if actual_duration > 0 else 0.0
    sample_ratio = n_samples / expected_samples if expected_samples else 0.0
    duration_ratio = actual_duration / duration if duration else 0.0
    output_nonempty = out_path.exists() and out_path.stat().st_size > 0

    print(f"  Expected samples:      {expected_samples}")
    print(f"  Actual samples:        {n_samples} ({sample_ratio * 100:.1f}% of expected)")
    print(f"  Duration (timestamps): {actual_duration:.2f} s ({duration_ratio * 100:.1f}% of requested)")
    print(f"  Effective fs:          {actual_rate:.1f} Hz (expected {expected_fs} Hz)")
    print(f"  Timestamps monotonic:  {timestamps_monotonic}")
    print(f"  Output non-empty:      {output_nonempty}")

    valid = completed and output_nonempty and timestamps_monotonic
    if sample_ratio < FAIL_COMPLETENESS_RATIO or duration_ratio < FAIL_COMPLETENESS_RATIO:
        print("[ERROR] Recording contains less than 90% of expected duration or samples.")
        valid = False
    elif sample_ratio < WARN_COMPLETENESS_RATIO or duration_ratio < WARN_COMPLETENESS_RATIO:
        print("[WARNING] Recording is below 98% completeness; inspect signal quality before use.")

    if not valid:
        print("[FAIL] Recording must not be used for formal Phase II collection.")
        return False

    print(f"\n[OK] Continuously saved complete recording: {out_path}")
    print(f"     Shape:   {n_samples} rows x {len(columns)} cols")
    print(f"     Columns: {columns}")
    if synthetic:
        print("\n  Note: This is SYNTHETIC data; use it only to validate the software acquisition path.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Continuously stream Ganglion (or synthetic) BrainFlow data to CSV."
    )
    parser.add_argument("--port", type=str, default="", help="Serial port, e.g. COM14. Not needed with --synthetic.")
    parser.add_argument("--mac", type=str, default="", help="Optional Ganglion MAC address.")
    parser.add_argument("--duration", type=float, required=True, help="Recording duration in seconds; required.")
    parser.add_argument("--out", type=str, default="data/real/raw/session_001/ganglion.csv", help="Output CSV path.")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic board; no hardware required.")
    parser.add_argument("--flush_interval", type=float, default=1.0, help="Seconds between disk flushes (default: 1.0).")
    parser.add_argument("--force", action="store_true", help="Explicitly overwrite an existing output file.")
    parser.add_argument("--start_at_unix", type=float, default=None,
                        help="Optional Unix timestamp for a software-synchronised start.")
    args = parser.parse_args()

    if not args.synthetic and not args.port:
        parser.error("--port is required unless using --synthetic mode.")

    selected_board = SYNTHETIC_ID if args.synthetic else GANGLION_ID
    succeeded = stream_and_save(
        selected_board,
        args.port,
        args.duration,
        args.out,
        args.synthetic,
        mac_address=args.mac,
        flush_interval=args.flush_interval,
        force=args.force,
        board_name="Ganglion",
        start_at_unix=args.start_at_unix,
    )
    sys.exit(0 if succeeded else 1)
