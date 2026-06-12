# Running the whole pipeline, end to end

This is the start-to-finish guide: raw sensor data in, classified risk and frozen evidence out. A competent stranger should be able to follow it. Every command runs **from inside `03_code/`** unless noted, and a fixed seed of 42 is used throughout, so the numbers below are what you should see (within the same scikit-learn build). If a step's number doesn't match, stop and check before moving on.

> **Read first:** `03_code/REPRODUCE.md` has the SAFE-RUN warning. Some `scripts/data_preparation/` correction scripts overwrite their inputs in place and write a timestamped backup. Run those once and keep the backups. Phase I (step 5) is fully self-contained and safe to re-run.

---

## 1. Environment

```bash
cd 03_code
python -V                      # 3.11
pip install -r requirements.txt   # scikit-learn 1.8.0, joblib 1.5.3, numpy, pandas, matplotlib
```

The pinned scikit-learn version matters. Tree-model AUCs can shift by a few thousandths across sklearn builds, and the frozen SHA-256 hashes only match byte-for-byte on the same OS (Windows CRLF vs Linux LF).

## 2. Raw acquisition → session folders

Real sessions are recorded as raw IMU + (optionally) Cyton sEMG CSVs. Convert each raw recording into a standard session folder:

```bash
python scripts/conversion/session_converter.py --help
```

**In:** raw IMU CSV (+ EMG CSV for the full-hybrid route) and the protocol label timing.
**Out:** a session folder with `imu_data.csv`, `emg_data.csv`, `labels.csv`.
The synthetic equivalent (`scripts/conversion/synthetic_generator.py`) is what Phase I uses, so you can run the whole thing without any hardware.

## 3. Drift correction + calibration

Long sessions accumulate Madgwick orientation drift, and the trunk-flexion feature interacts with it. Two corrections fix this:

```bash
# gyro bias from a still N-pose hold, then rest-anchor drift correction (in place — keep backups)
python scripts/data_preparation/apply_rest_anchor_correction.py --help
```

**Why:** without this, the segments furthest up the chain (T12/T4) show inflated flexion, which contaminated an earlier "sEMG helps fatigue" result. After correction the primary 4-IMU set keeps a small residual L3–T12 drift sensitivity that the reduced pelvis+L3 set does not, which is one reason the reduced set holds up.

## 4. Feature extraction

Turn a processed session into a windowed feature matrix (2 s windows, 1 s hop, ≥50% overlap rule):

```bash
# IMU-only (the cohort's route):
python scripts/phase_runners/run_pipeline.py --help

# full-hybrid route (P14): normalise sEMG amplitude to each session's resting baseline
python scripts/phase_runners/run_pipeline.py --emg_amplitude_norm resting_baseline_ratio --force
```

**Why `--emg_amplitude_norm resting_baseline_ratio`:** raw sEMG amplitude drifts with electrode contact and skin condition between sessions; normalising each channel to its own resting baseline is what makes sEMG features comparable across P14's 13 sessions. IMU-only sessions ignore this flag.
**Out:** one `feature_matrix.csv` per session.

## 5. Phase I, synthetic validation (one command, safe)

```bash
python scripts/phase_runners/run_phase1_synthetic.py
```

Generates 5 synthetic sessions, runs the pipeline, trains and evaluates the classifiers, and writes plots. **Expect:** near-ceiling discrimination (AUROC ≈ 1.00), the self-contained proof the pipeline behaves on a signal with known ground truth. Committed copies of these sessions are in `04_data/synthetic_phase1/`.

## 6. Model training

```bash
# the headline IMU-only fallback evidence: build the analysis sets, train, evaluate
python scripts/training/prepare_fallback_analysis_sets.py
python scripts/training/train_fallback_analysis_models.py      # RF: 500 trees, min_samples_leaf=3, balanced
python scripts/evaluation/evaluate_fallback_analysis_sets.py
```

`train_classifier.py` (in `ml/training/`) is the general LOSO trainer for the three feature conditions (IMU-only, sEMG-only, IMU+sEMG) used in Phase I and the hybrid analysis. The fallback route trains the deployed RF only.

## 7. Evaluation, what you should see

**Phase II.A (IMU-only fallback, n=9)**, from `evaluate_fallback_analysis_sets.py`, mirrored in `05_results/frozen_numbers/`:

| Configuration | within-CV AUROC | LOSO AUROC |
|---|---|---|
| Reduced (pelvis + L3), recommended | **0.819 ± 0.087** | **0.641 ± 0.062** |
| Primary (4-IMU) | 0.755 ± 0.077 | 0.570 ± 0.090 |

Paired Wilcoxon (LOSO, 6 shared participants): **p = 0.0625**. Reduced matches/beats primary from half the sensors. Personalisation (within − LOSO): **+0.178, 9/9, p≈0.0039**.

**Phase II.B (P14 hybrid, normalised sEMG)**, from `scripts/evaluation/analyse_p14_full_hybrid_corrected.py` → `results/participant_14_analysis_emgnorm/corrected_summary.json`:
IMU 0.790 → hybrid **0.839, Δ +0.048, 11/13 sessions, p = 0.006** (sEMG importance 35%). Marginally meets SC2; benefit is **load asymmetry** (asym-pickup +0.049), not fatigue/compensation; **not adopted** for deployment.

**Phase II.C (held-out)**, from `scripts/phase_runners/run_phase2c_emgnorm.py`:
P14 reduced **0.693** > full-hybrid 0.655 (over-flag 0.95) > IMU 0.630; P12 reduced 0.583 / IMU 0.594; P03 IMU 0.520 (reduced 0.869 is an in-sample leak); P11 excluded (IMU dropout).

## 8. Extras

```bash
# Appendix C — P10 augmentation ablation (reproduces canonical LOSO, then adds P10)
python scripts/evaluation/rerun_p10_augmentation_ablation.py
#   expect: primary 0.570 → 0.581 (+0.011); reduced 0.641 → 0.630 (−0.011)

# Stage2b personalisation pilot (calibrate-on-first-use)
python scripts/training/run_personalised_stage2b.py
#   expect: generic 0.621 → personal-augmented 0.781 (+0.160, 8/9)

# Report figures (matplotlib only)
python scripts/figures/plot_figure_work_order.py

# Freeze the SHA-256 manifest over the canonical artefacts
python scripts/datasets/freeze_emgnorm_artifacts.py
```

## 9. Demo (replay, not live)

The real-time feedback is shown by **replaying** a recorded session through the deployed model. This is the honest scope, since a live wearable loop was not built:

```bash
# turn a processed session into the dashboard's replay contract; --light maps R_IMU to a
# traffic light at 0.35 / 0.65 with NIOSH escalation
python scripts/demo/replay_from_features.py --light operating_point

# then view it
streamlit run scripts/demo/replay_dashboard.py
# or the terminal traffic-light demo:
python scripts/demo/demo_risk_monitor.py
```

---

## Honest scope

Prototype scale: nine participants for the IMU-only evidence, one participant (P14) for the hybrid claim, replay rather than a live wearable loop. The numbers above are within-participant-strong and LOSO-modest by design. The central claim is "equal accuracy from half the sensors," not "better." Full caveats in `LIMITATIONS_AND_KNOWN_ISSUES.md`.

*Every command's flags were checked against the scripts' `argparse`; run any script with `--help` to see its full options.*
