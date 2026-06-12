"""
cyton_stream.py
---------------
Continuous BrainFlow acquisition for OpenBCI Cyton recordings.

This writes the same pipeline-compatible raw CSV schema as ganglion_stream.py:
BrainFlow EXG channels are named emg_ch1..emg_ch8 and the board timestamp is
named timestamp_unix. The session converter can then map the first four EMG
channels to the project anatomy labels and resample to the pipeline rate.

Usage:
    py scripts/acquisition/cyton_stream.py --port COM4 --duration 1800 \
        --out data/real/raw/participant_01/session_001/cyton.csv
"""

# Make the project package importable when this file is run directly.
import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = _Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PKG_ROOT))


import argparse
import sys
from datetime import datetime

from brainflow.board_shim import BoardIds

try:
    from scripts.acquisition.ganglion_stream import stream_and_save
except ModuleNotFoundError:
    from scripts.acquisition.ganglion_stream import stream_and_save


def build_bipolar_config(target_channels):
    """Build ADS1299 config strings for true bipolar mode on one or more Cyton channels.

    Channels NOT listed are powered down using the OpenBCI shortcut chars '1'-'8'.
    Each target channel is configured with x N P G I B S Z X where:
        N = '1'..'8' (channel number)
        P = 0 (power on)
        G = 6 (24x gain, standard for sEMG)
        I = 0 (normal electrode input)
        B = 1 (include in bias drive calculation)
        S = 0 (SRB2 OPEN -> true bipolar: signal = NxP - NxN)
        Z = 0 (SRB1 off)

    Each target-channel string is e.g. 'x1060100X' for channel 1.

    Accepts either a single int (e.g. 1) or an iterable of ints (e.g. [1,2,3,4]).
    """
    if isinstance(target_channels, int):
        target_channels = [target_channels]
    target_channels = list(target_channels)
    if not target_channels:
        return []
    if not all(isinstance(c, int) and 1 <= c <= 8 for c in target_channels):
        raise ValueError(f"All target channels must be ints in 1-8, got {target_channels}")
    if len(set(target_channels)) != len(target_channels):
        raise ValueError(f"Duplicate channels not allowed: {target_channels}")
    target_set = set(target_channels)
    configs = []
    for i in range(1, 9):
        if i not in target_set:
            configs.append(str(i))
    for c in sorted(target_channels):
        configs.append(f"x{c}060100X")
    return configs


def record(
    serial_port: str,
    duration_s: float,
    output_path: str,
    flush_interval: float = 1.0,
    force: bool = False,
    synthetic: bool = False,
    bipolar_channels=None,
    start_at_unix: float = None,
) -> bool:
    """Record OpenBCI Cyton data using continuous CSV persistence.

    ``bipolar_channels`` may be an int (single channel) or an iterable of ints
    (multiple channels). Each listed channel is configured as true bipolar
    (SRB2 disconnected, 24x gain, normal electrode input, bias driven); all
    other Cyton channels are powered down. None / empty leaves the BrainFlow
    defaults (all 8 channels on, SRB2-referenced).
    """
    board_id = BoardIds.SYNTHETIC_BOARD.value if synthetic else BoardIds.CYTON_BOARD.value
    config_strings = None
    if bipolar_channels:
        if synthetic:
            print("[INFO] Ignoring bipolar config for synthetic board (no real ADS1299).")
        else:
            config_strings = build_bipolar_config(bipolar_channels)
    return stream_and_save(
        board_id,
        serial_port,
        duration_s,
        output_path,
        synthetic=synthetic,
        flush_interval=flush_interval,
        force=force,
        board_name="Synthetic" if synthetic else "Cyton",
        config_strings=config_strings,
        start_at_unix=start_at_unix,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuously record OpenBCI Cyton sEMG through BrainFlow.")
    parser.add_argument("--port", type=str, default="", help="Cyton USB dongle serial port, e.g. COM4.")
    parser.add_argument("--duration", type=float, required=True, help="Recording duration in seconds; required.")
    parser.add_argument(
        "--out",
        "--output",
        dest="output",
        type=str,
        default=f"cyton_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        help="Output CSV filename.",
    )
    parser.add_argument("--synthetic", action="store_true", help="Use BrainFlow synthetic board; no Cyton hardware required.")
    parser.add_argument("--flush_interval", type=float, default=1.0, help="Seconds between disk flushes.")
    parser.add_argument("--force", action="store_true", help="Explicitly overwrite an existing output file.")
    parser.add_argument("--start_at_unix", type=float, default=None,
                        help="Optional Unix timestamp for a software-synchronised start.")
    parser.add_argument(
        "--bipolar_channel", type=str, default="0",
        help=(
            "Reconfigure the Cyton into true bipolar mode (SRB2 off, 24x gain) on the given "
            "channel(s) before streaming; all other channels powered down. Accepts a single "
            "channel (e.g. '1') or a comma-separated list (e.g. '1,2,3,4'). '0' or empty "
            "leaves the BrainFlow defaults (all 8 channels on, SRB2-referenced)."
        ),
    )
    args = parser.parse_args()

    if not args.synthetic and not args.port:
        parser.error("--port is required unless using --synthetic mode.")

    raw = args.bipolar_channel.strip()
    if not raw or raw == "0":
        bipolar_channels = None
    else:
        try:
            bipolar_channels = [int(tok.strip()) for tok in raw.split(",") if tok.strip()]
        except ValueError:
            parser.error(f"--bipolar_channel must be comma-separated integers (got {raw!r}).")
        if not all(1 <= c <= 8 for c in bipolar_channels):
            parser.error(f"--bipolar_channel values must each be in 1-8 (got {bipolar_channels}).")
        if len(set(bipolar_channels)) != len(bipolar_channels):
            parser.error(f"--bipolar_channel values must be unique (got {bipolar_channels}).")

    succeeded = record(
        args.port, args.duration, args.output, args.flush_interval, args.force,
        synthetic=args.synthetic, bipolar_channels=bipolar_channels,
        start_at_unix=args.start_at_unix,
    )
    sys.exit(0 if succeeded else 1)
