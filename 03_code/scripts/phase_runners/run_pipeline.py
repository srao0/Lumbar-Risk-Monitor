#!/usr/bin/env python3
"""
Thin wrapper to run the signal processing pipeline without touching pipeline.py.
Usage:
    py scripts/phase_runners/run_pipeline.py --session_dir data/real/protocol_train/participant_01/session_001 --label_source protocol
    py scripts/phase_runners/run_pipeline.py --data_dir data/real/protocol_train --label_source protocol
"""
import argparse
import sys
from pathlib import Path

# parents[2] from scripts/phase_runners/ resolves to 03_code, put on sys.path so
# the signal_processing package import below resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from signal_processing.pipeline import run_pipeline, run_pipeline_batch

parser = argparse.ArgumentParser()
parser.add_argument("--session_dir", default=None)
parser.add_argument("--data_dir", default=None)
parser.add_argument("--output_dir", default=None)
parser.add_argument("--label_source", default="protocol",
                    choices=["signal", "protocol"])
parser.add_argument("--mode", default="full_hybrid",
                    choices=["full_hybrid", "imu_only_fallback"])
parser.add_argument("--no_notch", action="store_true")
parser.add_argument("--force", action="store_true",
                    help="Allow overwriting existing combined_features/feature_matrix "
                         "outputs in a batch run.")
parser.add_argument("--emg_amplitude_norm", default="none",
                    choices=["none", "resting_baseline_ratio"],
                    help="sEMG amplitude normalisation. 'none' keeps raw mV "
                         "(default, original behaviour); 'resting_baseline_ratio' "
                         "divides emg_rms_*/emg_mav_* by the per-session "
                         "BASELINE_STATIC mean.")
args = parser.parse_args()

if args.session_dir:
    feat_df = run_pipeline(
        session_dir=args.session_dir,
        output_dir=args.output_dir or args.session_dir,
        apply_notch=not args.no_notch,
        label_source=args.label_source,
        emg_amplitude_norm=args.emg_amplitude_norm,
    )
else:
    data_dir = args.data_dir or "data/real/protocol_train"
    feat_df = run_pipeline_batch(
        data_dir=data_dir,
        output_dir=args.output_dir or data_dir,
        label_source=args.label_source,
        operating_mode=args.mode,
        force=args.force,
        emg_amplitude_norm=args.emg_amplitude_norm,
    )

print(f"Shape: {feat_df.shape}")
print(f"Classes: {feat_df['risk_class'].value_counts().to_dict()}")
print(f"Accel-tilt features: {[c for c in feat_df.columns if 'tilt' in c]}")
