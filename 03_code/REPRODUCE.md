# Reproducing the results

Everything runs from this `03_code/` folder with the pinned environment (`requirements.txt`: Python 3.11, scikit-learn 1.8.0, joblib 1.5.3). A fixed seed of **42** is used throughout, so a given step gives the same numbers on each run **within the same scikit-learn build**.

```bash
pip install -r requirements.txt
```

> ## Safe-run warning: read before running anything beyond Phase I
> Some per-participant correction scripts in `scripts/data_preparation/` **overwrite their input files in place** and write a timestamped backup alongside. If you run them more than once, or without the backups present, you can corrupt the prepared sessions. **Run each once, keep the backups, and do not re-run on already-corrected inputs.** Phase I (below) is fully self-contained and safe to run repeatedly.

## Phase I: synthetic validation (one command, safe)

```bash
python scripts/phase_runners/run_phase1_synthetic.py
```

Generates the synthetic sessions, runs the full pipeline, trains and evaluates the classifiers, and writes the plots. This is the self-contained check that the pipeline behaves correctly on a signal with known ground truth. The committed copies of these sessions are in `../04_data/synthetic_phase1/`.

## Phase II.A: the IMU-only fallback evidence (n = 9)

The frozen numbers in `../05_results/frozen_numbers/` are produced by this sequence (a sequence, not one command, because several sessions needed documented corrections before pooling):

1. **Convert** each raw recording to a session folder with `scripts/conversion/session_converter.py`.
2. **Extract features** per session: `scripts/phase_runners` / `signal_processing/pipeline.py` → one `feature_matrix.csv` per session.
3. **Apply per-participant corrections** with `scripts/data_preparation/` (drift/anchor corrections, P02 rebuild, quality exclusions). **These run in-place; see the warning above.** Affected participants are in `../05_results/participant_notes/`.
4. **Build the analysis sets**: assembles the reduced (pelvis + L3, nine participants) and primary (four-IMU, six participants) feature matrices. It first reproduces the previous freeze exactly as a self-check before emitting the new one.
5. **Evaluate**: computes within-participant and LOSO AUCs, the paired Wilcoxon test, and writes the summary JSON and per-participant CSVs.

## Phase II.B / II.C / III

- **II.B** (personalised vs population, incl. the P14 hybrid analysis with resting-baseline-normalised sEMG) and **III** (replay traffic-light demo) run from `scripts/training/`, `scripts/evaluation/` and `scripts/demo/`.
- **II.C** runs the frozen models read-only on held-out participants via `scripts/phase_runners/run_phase2c_emgnorm.py` (wraps `scripts/evaluation/phase2c_verification.py`). It provides genuine held-out evidence on unseen participants (P12, P14 conforming); the synthetic varied set is verification only, and P03's reduced route is an in-sample leak. See `../05_results/phase2c_held_out_varied.md`.

The canonical RQ1 evidence is `../05_results/frozen_numbers/` (mirrors `results/fallback_analysis_sets_n9_corrected_qc/`); RQ2 is `results/participant_14_analysis_emgnorm/`; the full-pipeline guide with exact commands and expected numbers is `../../RUN_THE_PIPELINE.md`.

## Things to know

- **Cohort:** nine participants (P01–P09); P10 excluded for hardware failure.
- **Line endings affect hashes, not AUCs.** Frozen SHA-256 hashes match byte-for-byte only on the same OS (Windows CRLF vs Linux LF).
- **Phase I vs real RF settings differ.** Phase I synthetic RF used `n_estimators=150, min_samples_leaf=5`; the deployed real-data RF uses `n_estimators=500, min_samples_leaf=3, class_weight='balanced'`. State per phase; do not conflate.
