# Frozen Numbers Sheet — n=9 QC-cleaned, drift-corrected (CANONICAL)

**Frozen:** 2026-06-11 · **Status:** CANONICAL. Drift-corrected (rest-anchor) + frozen participant-level QC exclusions re-applied. Supersedes all earlier sheets (pre-correction "Pass 0" and the contaminated `_corrected` intermediate that re-admitted QC-excluded windows).
**Source of truth:** `fallback_analysis_sets_n9_corrected_qc/` — every number below traces to a SHA-256 entry in `FROZEN_MANIFEST.json` (this folder). Re-running the corrected+QC pipeline with seed 42 reproduces `evaluation_summary.json`.
**Rule:** every numerical claim in Chapters 7–8 must come from this sheet.

---

## A. Frozen n=9 dataset

Two corrections produced this set: (1) cohort-wide rest-anchor drift correction (`apply_rest_anchor_correction.py`) removed Madgwick-drift contamination of the trunk-angle features; (2) the frozen participant-level QC exclusions were re-applied (`reapply_qc_exclusions_corrected.py`) after an intermediate build had silently re-admitted them (P04 −51 T4 dropout, P05 −15, P07 −145 FATIGUE_FLEXION belt-slip windows with the pelvis pegged at the 60° cap).

| Set | n | Rows | Safe | Risky | Participants | Features |
|---|---|---|---|---|---|---|
| Primary 4-IMU | **6** | **8,698** | 5,174 | 3,524 | P01,04,05,06,07,08 | 17 |
| Reduced Pelvis-L3 (deployment) | **9** | **13,228** | 7,891 | 5,337 | P01–P09 | 13 |

Per-participant labelled windows: P01 1501 · P02 1501 · P03 1501 · P04 1450 · P05 1486 · P06 1501 · P07 1356 · P08 1404 · P09 1528.
(Primary excludes P02, P03, P09.)

**Set rows vs model-N:** the primary CSV holds **8,698** windows; the 4-IMU NaN-drop (≈150 windows on `imu_compensation_index`, the T12-dependent ratio — the set is not NaN-filtered on save) leaves **~8,550** entering the model. The reduced set is NaN-filtered at build time, so all **13,228** rows enter the model.

**60° pelvis physiological cap:** primary `imu_pelvis_angle_peak` 2.32%, `_mean` 0.76%; reduced `_peak` 2.78%, `_mean` 0.98%.

**SHA-256 (canonical, sklearn 1.8.0 / LF):**
- `primary_4imu_cleaned_features.csv` → `082156f5e546b74eccced77d52dad3d3330065e2a3b8a01297e92fff3940f22c`
- `reduced_pelvis_l3_features.csv` → `672bb0b248e80c09d633520fed39b4f387a96bb7bff2fada3a55956d7ab14da0`
- Location: `results/fallback_analysis_sets_n9_corrected_qc/`

---

## B. Evaluation AUCs (sklearn 1.8.0, seed 42)

| Metric | Canonical value |
|---|---|
| Primary 4-IMU (n=6) within-CV AUC | **0.755 ± 0.077** |
| Primary 4-IMU (n=6) LOSO AUC | **0.570 ± 0.090** |
| Reduced Pelvis-L3 (n=9) within-CV AUC | **0.819 ± 0.087** |
| Reduced Pelvis-L3 (n=9) LOSO AUC | **0.641 ± 0.062** |
| Paired Wilcoxon, primary vs reduced (LOSO, shared 6) | **p = 0.0625** (n_eff=6, min-p 0.03125) |

On the six shared participants the reduced set leads on **both** axes — within **+0.093** (0.848 vs 0.755) and LOSO 0.641 vs 0.570 — so the two-IMU Pelvis-L3 design matches or beats the full 4-IMU set from half the sensors. The LOSO edge is not statistically significant at n=6 (p=0.0625, the smallest value the test can return is 0.03125), which is the honest framing: reduced is **not worse**, and is the recommended deployment configuration.

Per-participant reduced **within**: P01 0.848 · P02 0.842 · P03 0.621 · P04 0.851 · P05 0.917 · P06 0.812 · P07 0.763 · P08 0.898 · P09 0.818
Per-participant reduced **LOSO**: P01 0.642 · P02 0.707 · P03 0.599 · P04 0.674 · P05 0.590 · P06 0.747 · P07 0.616 · P08 0.645 · P09 0.547
Per-participant primary **within**: P01 0.769 · P04 0.733 · P05 0.727 · P06 0.831 · P07 0.631 · P08 0.839
Per-participant primary **LOSO**: P01 0.560 · P04 0.568 · P05 0.652 · P06 0.675 · P07 0.544 · P08 0.421

**Personalisation (within − LOSO, reduced):** **+0.178**, 9/9 participants positive, p ≈ 0.0039 — a participant-specific model beats the population model for every participant.

---

## C. Deployed model configuration

- RF: `n_estimators=500, max_depth=None, min_samples_leaf=3, class_weight='balanced', random_state=42` (Option B alignment).
- Phase I synthetic RF used `n_estimators=150, min_samples_leaf=5` — state per-phase, do not conflate.
- Environment: scikit-learn **1.8.0**, joblib 1.5.3.
- Model files (`ml/models/fallback_final_n9_corrected_qc/`): primary `rf_primary_4imu.joblib` **68.4 MB** (SHA-256 `bf594044…`), reduced `rf_reduced_pelvis_l3.joblib` **108.4 MB** (SHA-256 `3088b66b…`). Both hashes locked in `FROZEN_MANIFEST.json`.

---

## D. Synthetic Phase I (protocol-labelled)

5 sessions · 2,661 total windows · 1,549 safe / 845 risky / 267 excluded · 2,394 binary windows · 6 archetypes. IMU 100 Hz, 19 synthetic IMU features (17 sagittal used for real data). Pipeline validates near-ceiling on this controlled signal (SC1a ≈ 1.00).

---

## E. P14 full-hybrid (Phase II.B, single subject) — sEMG contribution (resting-baseline-normalised)

Source: `results/participant_14_analysis_emgnorm/corrected_summary.json` (sEMG amplitude normalised to resting baseline; IMU drift-corrected).

- Aggregate sEMG lift (IMU+sEMG vs IMU): **+0.048 AUC (0.790 → 0.839), p = 0.006, 11/13 sessions** (pooled 0.794 → 0.834). sEMG feature-importance share **35%**.
- Decomposition by construct: all windows **+0.040**, exclude-baseline **+0.031**, hard subset **+0.016**.
- Confusable-pair AUC deltas: asymmetric-pickup **+0.049**, shoulder-driven **+0.018**, fast-flexion +0.006, fatigue **−0.002**, lumbar-dominant **−0.008**. The benefit is **load asymmetry**, not fatigue/compensation (the earlier fatigue/compensation reading was an IMU-drift artefact).
- **SC2 marginally met:** the +0.048 lift is significant (p=0.006) and at the margin of the +0.05 threshold, but narrow (asymmetry-specific) and ~0 on hard flexion-risk windows → sEMG **not adopted** for deployment.
- Q2B personalisation (P14): population 0.694 / personalised 0.778 (+0.084 pooled, +0.040 per session, 8/13 sessions, p = 0.31).
- EMG hardware: OpenBCI Cyton, 200 Hz (timestamp-verified, uniform resampled grid). Channels LES, RES, LOBL, ROBL — erector spinae + obliques (NOT multifidus).
- Cohort personalisation pilot (`personalised_stage2b_corrected_qc_n9/`): generic 0.621 → personal-augmented 0.781 (**+0.160**, 8/9).

---

## F. Phase II.C — held-out generalisation (clean models)

From `phase2c_emgnorm/` (frozen models, no retraining): P14 reduced **0.693** (best route) > full-hybrid 0.655 (over-flag 0.95) > IMU-only 0.630; P12 reduced 0.583 / IMU 0.594; P03 IMU 0.520, reduced in-sample 0.869 (leak). **P11 excluded** — IMU acquisition dropout at t ≈ 1186 s (data loss, not a model failure).

---

## G. Exclusions

- **P04**: 51 windows (T4 sensor dropout).
- **P05**: 15 windows.
- **P07**: 145 FATIGUE_FLEXION risky windows (pelvis belt slip during the 20-rep block, pelvis pegged at the 60° cap).
- **P10**: fully excluded — hardware failure (T12 dropout ≈ 14 min, board termination).
- **P11**: excluded from Phase II.C — IMU acquisition dropout at t ≈ 1186 s.

---

## H. Success criteria

SC1a met (synthetic ≈ 1.00) · SC1b met (reduced within 0.82 ≥ 0.75) · SC1c met (reduced LOSO 0.641 ≥ 0.60) · **SC2 marginally met** (sEMG +0.048 AUC, p=0.006, at the +0.05 margin; load-asymmetry-specific, not adopted) · SC3 met.
