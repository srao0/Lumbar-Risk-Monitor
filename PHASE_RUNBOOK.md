# Spinal Movement Risk Monitor Phase Runbook

This runbook defines the official report phases and the commands/files that
belong to each phase. Older repository artefacts may still contain historical
metadata values such as `Phase II.1` and `Phase II.2`; the current report
mapping is:

| Historical/internal label | Official report phase |
|---|---|
| `Phase II.1` | `Phase II.A: Real Protocol Data Collection` |
| `Phase II.2` | `Phase II.C: Held-Out Varied-Movement Evaluation` |
| `Stage II.B` | `Phase II.B: Personalised Calibration` |

The IMU-only fallback route is part of Phase II.A when it is used for protocol
data collection and part of Phase II.C when it is used for held-out
varied-movement evaluation. It is not a separate project phase.

## Workflow

Use these phase runners for a clean top-level demonstration of the project.
They call the shared processing/training code; they do not implement separate
synthetic and real feature pipelines.

```powershell
python scripts/run_phase1_synthetic.py
python scripts/run_phase2_protocol.py --mode full_hybrid
python scripts/run_phase2_fallback_protocol.py --cv_group participant
python scripts/run_personalised_stage2b.py
python scripts/run_phase2_varied_test.py --mode full_hybrid
python scripts/demo_risk_monitor.py --session data/real/protocol_train/participant_01/session_001 --mode full_hybrid
# Phase III.A replay dashboard (see docs/PHASE_III_RUNBOOK.md):
streamlit run scripts/replay_dashboard.py -- --session <replay_out_session>
# Phase III.B real-time foundations (full-refresh, NOT --incremental):
python scripts/run_live_dashboard_pipeline.py --raw_session <raw> --processed_session <proc> --out_session <live_out> --mode full_hybrid --emg_board cyton --duration_s 30
```

For a non-writing command preview:

```powershell
python scripts/run_phase1_synthetic.py --dry_run
python scripts/run_phase2_protocol.py --dry_run --mode full_hybrid
python scripts/run_phase2_fallback_protocol.py --dry_run
python scripts/run_phase2_varied_test.py --dry_run --mode full_hybrid
```

## Phase I: Synthetic Pipeline Validation

Purpose: verify that the software pipeline works end to end on controlled
synthetic IMU and sEMG data. This is not real hardware performance evidence.

Primary inputs:
- `data/synthetic/session_*/imu_data.csv`
- `data/synthetic/session_*/emg_data.csv`
- `data/synthetic/session_*/labels.csv`

Primary command:

```powershell
python scripts/run_phase1_synthetic.py
```

Equivalent component commands:

```powershell
python scripts/synthetic_generator.py --n_sessions 5 --seed 42 --out_dir data/synthetic
python -m signal_processing.pipeline --data_dir data/synthetic --label_source protocol
python ml/training/train_classifier.py --data_dir data/synthetic
python ml/evaluation/evaluate.py
python ml/evaluation/generate_extra_plots.py
```

Expected outputs:
- `data/synthetic/combined_features.csv`
- `ml/evaluation/loso_results.csv`
- `ml/evaluation/summary_results.csv`
- `ml/evaluation/feature_importance_RF.csv`
- `results/plots/`

Interpretation rule: describe Phase I as controlled synthetic validation of
the pipeline and model-comparison evidence, not as real participant or real
hardware accuracy.

## Phase II.A: Real Protocol Data Collection

Purpose: collect real participant data using the supervised movement protocol,
convert recordings into the standard session-folder structure, validate labels
and sensor quality, and train/evaluate the real protocol models where the
required streams are available.

Recommended target layout:
- `data/real/protocol_train/participant_XX/session_YY/`
- `data/real/protocol_train_fallback/participant_XX/session_YY/` when valid
  sEMG is unavailable and the session is explicitly declared
  `imu_only_fallback`.

Per-session required files after conversion:
- `imu_data.csv`
- `emg_data.csv` for full-hybrid sessions, or an explicit metadata declaration
  that EMG is not used in fallback sessions
- `labels.csv`
- `session_metadata.json`

Preferred acquisition command with software-synchronised start. The example
uses PowerShell line-continuation backticks; in other shells, run the same
arguments on one line.

```powershell
python scripts/start_synchronised_session.py `
  --session_dir data/real/raw/participant_01/session_001 `
  --imu_port COM3 `
  --ganglion_port COM4
```

This starts the IMU recorder, Ganglion recorder, and protocol timer from one
host process using a shared scheduled start timestamp. It writes
`session_sync_metadata.json` so the launch timing is auditable. This reduces
manual start offset, but it is not a shared hardware clock.

Use `--dry_run` to preview the planned commands without starting hardware or
writing session metadata.

Manual acquisition commands, retained as a fallback:

```powershell
python scripts/record_imu_serial.py --port COM3 --duration 1800 --out data/real/raw/participant_01/session_001/imu_arduino.csv
python scripts/ganglion_stream.py --port COM4 --duration 1800 --out data/real/raw/participant_01/session_001/ganglion.csv
python scripts/session_timer.py --out data/real/raw/participant_01/session_001/labels.csv
```

`session_timer.py` is the official scheduled protocol-label source for Phase
II.A. It emits only validator-approved supervised task labels. Signal-derived
labels can be used diagnostically, but they are not the official supervised
target for real-data claims.

### Operator note: fatigue block (Section 5.9) is operator-stopped

Section 5.9 (`FATIGUE_FLEXION`, repetitive fatigue bends) does **not** count
reps individually. It runs as a single count-up timer for the whole section.
When the participant has completed their continuous bends, the operator
**presses any key** in the `session_timer.py` terminal to end the section and
advance immediately to the final BL2 static baseline. If no key is pressed the
timer auto-advances at a safety cap (160 s). On Windows this uses any keypress
in the timer window; make sure that terminal has focus before pressing.

Implications:
- The fatigue section's real duration is variable. `session_timer.py` records
  its **actual** elapsed time, so `labels.csv` timestamps for the fatigue block
  and the following BL2 baseline stay aligned with the IMU/Ganglion recordings
  even when the section is stopped early.
- In `labels.csv` the fatigue block is written as a **single continuous row**
  (`rep=1`) spanning its real duration, not 20 fixed rep rows. `fatigue_fraction`
  is left blank and may be derived later from the continuous window.
- The IMU and Ganglion recorders still run for their fixed `--duration` window.
  Ending the fatigue block early simply means the labelled protocol finishes
  before the recorders' fixed duration elapses; stop the recorders when the
  timer reports `RECORDING COMPLETE`.

Full-hybrid processing:

```powershell
python scripts/run_phase2_protocol.py --mode full_hybrid
```

IMU-only fallback processing, when valid sEMG is unavailable:

```powershell
python scripts/run_phase2_fallback_protocol.py --cv_group participant
```

Fallback mode is never a silent substitute for full-hybrid evidence. It must
be declared with `--mode imu_only_fallback`, recorded in metadata and manifests,
and reported separately from full-hybrid results. In fallback mode:
- real IMU data is still required
- missing or unusable EMG is allowed only because it is not used for inference
- `LR_EMG`, `R_EMG`, and EMG-derived FIS inputs are not used
- the output is marked as `imu_only_fallback`

Official Phase II.A processing is overwrite-protected. Re-running official
processing or training over existing outputs requires an explicit `--force`
decision.

## Phase II.B: Personalised Calibration

Purpose: evaluate whether participant-specific calibration improves
performance compared with a generic population-style model.

Primary command:

```powershell
python scripts/run_personalised_stage2b.py
```

This phase compares generic, personal-calibration-only, and
personal-augmented models using participant-specific calibration windows and
held-out future repetitions. It is exploratory participant-level evidence. It
supports the argument for baseline-normalised features and strategy-aware
interpretation, but it does not replace full-hybrid population-level
validation.

Expected outputs:
- `results/personalised_stage2b/summary_metrics.csv`
- `results/personalised_stage2b/per_participant_metrics.csv`
- `results/personalised_stage2b/split_summary.csv`
- `results/personalised_stage2b/interpretation.md`

### Full-hybrid personalised evaluation — Participant 14 (frozen 2026-06-07)

Participant 14 is the chosen **full-hybrid (IMU + sEMG)** participant for Phase II.B
(other full-hybrid participants, including P13, produced fewer/poorer sessions).
This is the project's first accepted real full-hybrid evaluation artefact and
**supersedes the earlier "no accepted full-hybrid evaluation artefact" status**
for P14. It is single-participant evidence: report it as a personalised case
study, not as population-level full-hybrid validation.

Frozen dataset (corrected `LOBL/ROBL` oblique channel naming, post OBL-vs-MF fix):
- `data/real/protocol_train_full_hybrid/participant_14/combined_features.csv`
- `data/real/protocol_train_full_hybrid/participant_14/dataset_manifest.json`
- 14 sessions collected; **13 frozen** (session_05 excluded — IMU stream 53.2 s
  shorter than EMG, fails `SENSOR_DURATION_AGREEMENT_MS`; archived under
  `data/real/_excluded/`, see `EXCLUSION_NOTE_session_05.md`)
- 19,342 labelled windows, 37.0 % risky, SHA-256 `23fbab89…`
- IMU 100 Hz, EMG 200 Hz, `emg_used_for_inference = true`

Validation:

```powershell
python scripts/validate_phase2_dataset.py --data_dir <dir containing only participant_14> --phase "Phase II.B" --mode full_hybrid --expected_participants 1
```

Result: **13/13 sessions OK, 0 FAIL** (`VALIDATION_session_level.txt`). Note the
shared `protocol_train_full_hybrid/` directory also holds participant_13; scope
the validator to a directory containing only `participant_14`.

Findings (personalised leave-one-session-out; full report
`results/participant_14_analysis/P14_full_hybrid_analysis_report.md`, figures
`plots/fig1`–`fig10`):
- **Q1 IMU-only vs IMU+sEMG:** hybrid +0.045 AUC (0.799→0.844), 12/13 sessions,
  paired Wilcoxon p = 0.0005 *in aggregate*. But on the core lumbar-dominant-vs-
  hip-hinge construct sEMG adds only **+0.009**; its real value is fatigue
  (+0.046) and gross compensation, and it costs specificity on effortful-but-safe
  movements (symmetric pickup, sit-to-stand). EMG verdict: justify by design goal
  (worth it only if fatigue monitoring is explicit), not as a blanket improvement.
- **Q2B personalised vs population (reduced IMU-fallback features):** personalised
  0.725 vs population-model-on-P14 0.583, +0.138, 11/13, p = 0.0012 — reconfirms
  the Phase II.A personalisation finding on a fresh subject.

## Final Deployed Architecture And Comparators

The final deployed risk system is the hybrid interpretable path:

```text
IMU features -> frozen RF_IMU -> R_IMU --\
                                           -> fixed Mamdani FIS -> R_total -> Safe / Cautious / Risky
EMG features -> frozen LR_EMG -> R_EMG --/
```

The six Mamdani inputs are fixed before Phase II.C:
- `R_IMU`: frozen RF probability from IMU features
- `R_EMG`: frozen LR probability from EMG features
- smoothness abnormality from baseline-normalised `imu_z_ldlj`
- time-in-risk-zone abnormality from `imu_time_in_risk_zone`
- EMG asymmetry abnormality from `emg_ai_ES`
- baseline deviation magnitude from the IMU baseline-normalised summary

`RF_IMU_EMG` remains a valid benchmark model for testing direct feature-level
fusion. It is not the final deployed decision path.

## Phase II.C: Held-Out Varied-Movement Evaluation

Purpose: evaluate generalisation using held-out varied-movement sessions. The
models are frozen before this stage. No retraining, threshold tuning, feature
selection, membership-function tuning, or rule tuning may be performed on the
held-out data.

Recommended target layout:
- `data/real/varied_test/participant_XX/session_YY/`
- `data/real/varied_test_fallback/participant_XX/session_YY/` for explicitly
  declared fallback evidence.

Primary command:

```powershell
python scripts/run_phase2_varied_test.py --mode full_hybrid
```

Fallback held-out evaluation, reported separately:

```powershell
python scripts/run_phase2_fallback_varied_test.py
```

This phase should report missed-risk windows/events, false alarms, confusion
matrices, and mode-specific outputs. Fallback outputs are evidence for the
IMU-only route only.

## Phase III.A: Replay Feedback Evaluation

Purpose: demonstrate that the frozen system presents Safe/Cautious/Risky feedback
clearly, window by window, with supporting evidence (risk score, key feature values,
and engineering/layman explanation). This is the validated Phase III deliverable
(requirements M6/M7, SC3). Replay scores are exact: models are frozen and applied
read-only, never retrained. Detailed steps and the screenshot checklist are in
`docs/PHASE_III_RUNBOOK.md`.

```powershell
# Generate the dashboard contract, then launch the read-only dashboard.
python scripts/replay_recorded_session.py --raw_session data/real/raw/_smoketest_p11_s001_230s --out_session data/real/replay_full_hybrid/_smoketest --mode full_hybrid --emg_board cyton
python scripts/replay_recorded_session.py --raw_session data/real/raw/_smoketest_p11_s001_230s --out_session data/real/replay_imu_only/_smoketest --mode imu_only_fallback --models_dir ml/models/fallback_final_n9
streamlit run scripts/replay_dashboard.py -- --session data/real/replay_full_hybrid/_smoketest
```

Optional "live-looking" playback (writes precomputed windows progressively; no model
is re-run):

```powershell
python scripts/stream_replay_session.py --source_session data/real/replay_full_hybrid/_smoketest --out_session data/real/replay_stream/_smoketest --speed 1
```

Console alternative: `python scripts/demo_risk_monitor.py --session <processed_session> --mode full_hybrid`.
Fallback output must be labelled `IMU-only fallback`, never full-hybrid evidence.

## Phase III.B: Real-Time Foundations (prototype, not validated real-time)

Purpose: demonstrate the foundations of a streaming/live inference path that reuses the
identical frozen models, Mamdani FIS, explainer, and dashboard contract as Phase III.A.
This is an implemented prototype, NOT validated real-time deployment.

```powershell
python scripts/run_live_dashboard_pipeline.py --raw_session data/real/raw/_smoketest_p11_s001_230s --processed_session data/real/live_processed/_smoketest --out_session data/real/live_out/_smoketest --mode full_hybrid --emg_board cyton --duration_s 30
streamlit run scripts/replay_dashboard.py -- --session data/real/live_out/_smoketest
```

Claim boundaries (state wherever III.B appears):
- Full-refresh only; `--incremental` is experimental and not offline-parity-safe.
- The sEMG filter is zero-phase, so the latest window of a growing recording is
  provisional; completed/cached replay (III.A) is exact.
- Host-side, no shared hardware IMU/sEMG trigger; this mirrors the replay pipeline
  rather than achieving true low-latency real-time.
