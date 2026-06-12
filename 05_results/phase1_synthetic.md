# Phase I — Synthetic pipeline validation

**Question:** before touching real, noisy human data, can the *entire* software pipeline — orientation → features → windowing → classification → evaluation — be shown to work correctly on a signal whose ground truth is known by construction?

**Verdict: yes.** Phase I is the controlled sanity check. Five synthetic participant sessions (`../04_data/synthetic_phase1/`) are generated with known risk labels, pushed through the exact same pipeline used on real data, and evaluated. Because the ground truth is built in, any failure here is a pipeline bug, not a data problem.

## What was tested

- Synthetic IMU (100 Hz) + sEMG (200 Hz) with realistic noise, drift and inter-rep jitter (`session_config.json`: seed 42, `imu_drift_rate`, `emg_motion_art`, etc.).
- Full feature extraction (kinematic + EMG) into `feature_matrix.csv`.
- Three input conditions — IMU-only, EMG-only, IMU+EMG — classified with RF / LR / SVM.
- Phase I RF settings: `n_estimators=150, min_samples_leaf=5` (distinct from the deployed real-data RF — do not conflate).

## Figures (`figures/`)

| File | Shows |
|---|---|
| `roc_auc_comparison.png`, `roc_curves_per_condition.png` | Discrimination across the three input conditions |
| `confusion_matrices.png`, `per_class_recall_precision.png` | Per-class behaviour |
| `sensitivity_specificity.png` | Operating-point trade-off |
| `feature_importance_RF.png` | Which features drive the synthetic classification |
| `per_session_auc_RF.png` | Consistency across the 5 sessions |
| `fis_vs_rf_comparison.png` | Fuzzy-inference vs raw RF output |
| `statistical_significance.png` | Significance of the condition comparison |

## Reproduce

```bash
cd ../03_code
python scripts/phase_runners/run_phase1_synthetic.py
```

One command, fully self-contained, safe to re-run. It regenerates the sessions, runs the pipeline and writes these plots.

## How to read it

Phase I demonstrates **pipeline correctness, not real-world performance.** Strong AUCs here only confirm the machinery works on clean, known data; the meaningful performance evidence is Phase II.A on real participants.
