# scripts/: host-side acquisition, processing, and study orchestration

These scripts run from the package root (`03_code/`), e.g. `python scripts/phase_runners/run_phase1_synthetic.py`. Each file adds the package root to `sys.path` on import, so it runs directly from its subfolder while still importing siblings as `scripts.<area>.<module>`.

## By use case

### `acquisition/`: recording sensors and running timed capture sessions

- **`record_imu_serial.py`**: records the four-IMU ESP32 stream over USB serial to CSV.
- **`record_ganglion.py`**: native-Bluetooth wrapper to record OpenBCI Ganglion sEMG.
- **`ganglion_stream.py`**: BrainFlow Ganglion sEMG acquisition with stream-completeness checks.
- **`cyton_stream.py`**: BrainFlow Cyton sEMG acquisition (Ganglion-compatible output).
- **`session_timer.py`**: guided countdown for the supervised Phase II protocol; writes labels.csv.
- **`varied_session_timer.py`**: timer and label writer for the Phase II.C varied-movement sessions.
- **`start_synchronised_session.py`**: launches the IMU + Ganglion recorders and timer with a shared start timestamp.
- **`annotate_session.py`**: manually annotate/curate a recorded session.
- **`label_logger.py`**: live key-press event logger during a recording session.

### `conversion/`: turning raw streams into processed, calibrated, labelled sessions

- **`session_converter.py`**: raw IMU/EMG CSVs -> standard processed session (imu_data/emg_data/labels/metadata).
- **`extract_initial_still_cal.py`**: extracts the initial static baseline for gyro-bias calibration.
- **`check_bl2_static.py`**: checks whether the final BL2 static baseline is actually still.
- **`generate_protocol_labels.py`**: Phase II protocol-label generator and the MOVEMENT_CATALOGUE.
- **`synthetic_generator.py`**: generates synthetic IMU + sEMG sessions for the Phase I pipeline check.

### `datasets/`: manifests, model provenance, and dataset/project validation

- **`dataset_manifest.py`**: creates and validates the SHA-256 dataset manifest beside a feature matrix.
- **`model_provenance.py`**: reads and validates Phase II.A frozen-model provenance metadata.
- **`validate_phase2_dataset.py`**: mode-aware validation of a processed Phase II dataset before training/reporting.
- **`validate_project.py`**: read-only project-wide structural/health checks.

### `training/`: training the fallback and personalised models

- **`prepare_fallback_analysis_sets.py`**: assembles analysis-ready IMU-only fallback feature sets.
- **`train_fallback_analysis_models.py`**: trains the frozen IMU-only fallback models from the cleaned analysis sets.
- **`run_personalised_stage2b.py`**: Phase II.B personalised-vs-population calibration experiment.

### `evaluation/`: held-out and comparator evaluation

- **`evaluate_phase2_varied_test.py`**: aggregate Phase II.C evaluation using frozen Phase II.A models only.
- **`evaluate_fallback_analysis_sets.py`**: evaluates the prepared IMU-only fallback analysis sets.
- **`evaluate_rf_vs_fallback_fis.py`**: compares RF-only vs RF + IMU-fallback FIS on the frozen fallback datasets.
- **`phase2c_verification.py`**: Phase II.C frozen-pipeline verification harness (not a generalisation proof).

### `demo/`: replay-mode demonstration of the risk output

- **`demo_risk_monitor.py`**: replay-mode traffic-light risk demo with explanations (Phase III).
- **`replay_dashboard.py`**: Streamlit replay dashboard. NOTE: this file is truncated in the package (see top README).

### `phase_runners/`: one entry point per study phase (start here)

- **`run_phase1_synthetic.py`**: Phase I, validates the software pipeline end-to-end on synthetic data.
- **`run_phase2_protocol.py`**: Phase II.A full-hybrid protocol training/evaluation runner.
- **`run_phase2_fallback_protocol.py`**: Phase II.A IMU-only fallback route (the cohort's actual mode).
- **`run_phase2_varied_test.py`**: Phase II.C held-out varied-movement generalisation test.
- **`run_phase2_fallback_varied_test.py`**: Phase II.C IMU-only fallback varied-session test.
- **`run_pipeline.py`**: thin wrapper to run the signal-processing pipeline on a session.
- **`phase_runner_utils.py`**: shared helpers for the phase runners (run_step, python_cmd).

### `data_preparation/`: one-off per-participant correction scripts (historical; see its own README)

- **`apply_data_quality_exclusions.py`**: applies documented data-quality exclusions to feature tables.
- **`apply_drift_correction.py`**: post-hoc baseline (BL1<->BL2) drift correction.
- **`apply_rest_anchor_correction.py`**: applies rest-anchor zero correction to a processed IMU session.
- **`fix_participant_02.py`**: one-off P02 repair (T12 IMU dropout handling).
- **`refreeze_n9.py`**: Pass-0 re-freeze extending the fallback evidence from n=7 to n=9.
- **`refreeze_n9_data.py`**: Pass-0 data-layer re-freeze (no model retraining).
- **`repair_imu_anchor_drift.py`**: repairs IMU pitch/roll drift using protocol rest anchors.

## Finding a script cited in the report

The report refers to these by their original flat path (`scripts/<name>.py`). Each now lives in a use-case subfolder:

| Cited as | Now at |
|---|---|
| `scripts/annotate_session.py` | `scripts/acquisition/annotate_session.py` |
| `scripts/apply_data_quality_exclusions.py` | `scripts/data_preparation/apply_data_quality_exclusions.py` |
| `scripts/apply_drift_correction.py` | `scripts/data_preparation/apply_drift_correction.py` |
| `scripts/apply_rest_anchor_correction.py` | `scripts/data_preparation/apply_rest_anchor_correction.py` |
| `scripts/check_bl2_static.py` | `scripts/conversion/check_bl2_static.py` |
| `scripts/cyton_stream.py` | `scripts/acquisition/cyton_stream.py` |
| `scripts/dataset_manifest.py` | `scripts/datasets/dataset_manifest.py` |
| `scripts/demo_risk_monitor.py` | `scripts/demo/demo_risk_monitor.py` |
| `scripts/evaluate_fallback_analysis_sets.py` | `scripts/evaluation/evaluate_fallback_analysis_sets.py` |
| `scripts/evaluate_phase2_varied_test.py` | `scripts/evaluation/evaluate_phase2_varied_test.py` |
| `scripts/evaluate_rf_vs_fallback_fis.py` | `scripts/evaluation/evaluate_rf_vs_fallback_fis.py` |
| `scripts/extract_initial_still_cal.py` | `scripts/conversion/extract_initial_still_cal.py` |
| `scripts/fix_participant_02.py` | `scripts/data_preparation/fix_participant_02.py` |
| `scripts/ganglion_stream.py` | `scripts/acquisition/ganglion_stream.py` |
| `scripts/generate_protocol_labels.py` | `scripts/conversion/generate_protocol_labels.py` |
| `scripts/label_logger.py` | `scripts/acquisition/label_logger.py` |
| `scripts/model_provenance.py` | `scripts/datasets/model_provenance.py` |
| `scripts/phase2c_verification.py` | `scripts/evaluation/phase2c_verification.py` |
| `scripts/phase_runner_utils.py` | `scripts/phase_runners/phase_runner_utils.py` |
| `scripts/prepare_fallback_analysis_sets.py` | `scripts/training/prepare_fallback_analysis_sets.py` |
| `scripts/record_ganglion.py` | `scripts/acquisition/record_ganglion.py` |
| `scripts/record_imu_serial.py` | `scripts/acquisition/record_imu_serial.py` |
| `scripts/refreeze_n9.py` | `scripts/data_preparation/refreeze_n9.py` |
| `scripts/refreeze_n9_data.py` | `scripts/data_preparation/refreeze_n9_data.py` |
| `scripts/repair_imu_anchor_drift.py` | `scripts/data_preparation/repair_imu_anchor_drift.py` |
| `scripts/replay_dashboard.py` | `scripts/demo/replay_dashboard.py` |
| `scripts/run_personalised_stage2b.py` | `scripts/training/run_personalised_stage2b.py` |
| `scripts/run_phase1_synthetic.py` | `scripts/phase_runners/run_phase1_synthetic.py` |
| `scripts/run_phase2_fallback_protocol.py` | `scripts/phase_runners/run_phase2_fallback_protocol.py` |
| `scripts/run_phase2_fallback_varied_test.py` | `scripts/phase_runners/run_phase2_fallback_varied_test.py` |
| `scripts/run_phase2_protocol.py` | `scripts/phase_runners/run_phase2_protocol.py` |
| `scripts/run_phase2_varied_test.py` | `scripts/phase_runners/run_phase2_varied_test.py` |
| `scripts/run_pipeline.py` | `scripts/phase_runners/run_pipeline.py` |
| `scripts/session_converter.py` | `scripts/conversion/session_converter.py` |
| `scripts/session_timer.py` | `scripts/acquisition/session_timer.py` |
| `scripts/start_synchronised_session.py` | `scripts/acquisition/start_synchronised_session.py` |
| `scripts/synthetic_generator.py` | `scripts/conversion/synthetic_generator.py` |
| `scripts/train_fallback_analysis_models.py` | `scripts/training/train_fallback_analysis_models.py` |
| `scripts/validate_phase2_dataset.py` | `scripts/datasets/validate_phase2_dataset.py` |
| `scripts/validate_project.py` | `scripts/datasets/validate_project.py` |
| `scripts/varied_session_timer.py` | `scripts/acquisition/varied_session_timer.py` |

