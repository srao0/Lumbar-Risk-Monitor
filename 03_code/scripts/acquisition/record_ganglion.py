"""
Ganglion native-Bluetooth compatibility entry point.

This wrapper uses the same continuous CSV persistence and completeness
validation as ganglion_stream.py. Formal Phase II collection should follow
docs/PHASE_RUNBOOK.md and use an explicit duration and unique output path.
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


def record(
    duration_s: float,
    output_path: str,
    mac_address: str = "",
    flush_interval: float = 1.0,
    force: bool = False,
) -> bool:
    """Record native-Bluetooth Ganglion data using continuous persistence."""
    return stream_and_save(
        BoardIds.GANGLION_NATIVE_BOARD.value,
        "",
        duration_s,
        output_path,
        synthetic=False,
        mac_address=mac_address,
        flush_interval=flush_interval,
        force=force,
        native=True,
        board_name="Ganglion",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuously record Ganglion sEMG through native Bluetooth.")
    parser.add_argument("--duration", type=float, required=True, help="Recording duration in seconds; required.")
    parser.add_argument(
        "--output",
        type=str,
        default=f"ganglion_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        help="Output CSV filename.",
    )
    parser.add_argument("--mac", type=str, default="", help="Optional Ganglion MAC address.")
    parser.add_argument("--flush_interval", type=float, default=1.0, help="Seconds between disk flushes.")
    parser.add_argument("--force", action="store_true", help="Explicitly overwrite an existing output file.")
    args = parser.parse_args()

    succeeded = record(args.duration, args.output, args.mac, args.flush_interval, args.force)
    sys.exit(0 if succeeded else 1)
