# Data

Enough data to see what the system works with, without exposing any on-body participant recordings.

## Contents

- `synthetic_phase1/session_0001…0005/`: the five **Phase I synthetic sessions**. Each is a simulated participant performing the full movement protocol, with raw IMU (100 Hz) and sEMG (200 Hz) streams, the windowed `feature_matrix.csv`, the segment `labels.csv`, and `session_config.json` (seed, sample rates, noise/drift parameters). These are the data behind Phase I and regenerate exactly via `../03_code/scripts/phase_runners/run_phase1_synthetic.py`.
- `sample_session/`: a small **real** raw→processed worked example (IMU + labels only), so the real-data shape is visible without identifying content.
- `illustrations/`: rendered figures of clean synthetic IMU-angle and sEMG signals, so the signal morphology is visible.
- `dataset_manifest.json`: provenance for the Phase I set (sources, SHA-256, generation command).

See `DATA_DICTIONARY.md` for every column and `labelling_protocol.md` for how labels were assigned.

## Not here

Real on-body cohort recordings (13 GB) and any identifying participant data are excluded by design. The aggregate results derived from them live in `../05_results/`.
