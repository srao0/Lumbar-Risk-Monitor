# Participant 07: calibrated data analysis

> **Canonical numbers:** this note documents the exclusion *decision*; its per-participant AUCs predate the drift-correction and QC re-freeze. The authoritative frozen figures are in `../frozen_numbers/frozen_numbers_sheet.md`.

**Session:** participant_07 / session_001
**Date recorded:** 2026-05-28
**Operating mode:** `imu_only_fallback` (no EMG, Ganglion not collected this session)
**Duration:** 28.2 min IMU stream / 25.8 min protocol coverage
**Drift correction applied:** `linear_bl2_zero_reference` (per session_metadata.json)
**Feature matrix:** 1898 windows, 49 columns

---

## 1. Headline result

**Within-participant temporal 80/20 CV, sagittal-only IMU (20 features), protocol labels:**

| Metric | P07 | P01 (ref) | P02 (ref) |
|---|---|---|---|
| AUC | **0.917** | 0.844 | 0.840 |
| Sensitivity | 0.900 | 0.819 | 0.706 |
| Specificity | 0.950 | 0.837 | 0.830 |
| F1 | 0.926 | 0.834 | 0.761 |
| Youden threshold | 0.57 | 0.26 | 0.48 |

P07 is the strongest within-participant result to date. The top features by Gini importance (`imu_compensation_index` 0.102, `imu_pelvis_angle_peak` 0.099, `imu_lumbopelv_ratio` 0.090) match the pattern from P01/P02 and support the sagittal-only feature set.

The signal-label CV trivially returns AUC=1.00 (circular: features → signal labels → classify features). Reported here only for completeness; do **not** put it in the thesis.

---

## 2. Protocol additions (new vs P01–P03)

The protocol was extended with three movements that are not present in earlier participants and that the FIS / classifier were not trained on:

- `FATIGUE_FLEXION`: 20 reps (longest single block, ~160 s)
- `SIT_TO_STAND_NORMAL`: 5 reps
- `SIT_TO_STAND_FAST`: 3 reps

This is a meaningful protocol drift. If you intend to combine P07 with P01/P02 for LOSO, you have to either (a) exclude these movements from the comparison, or (b) accept that P07 trains/tests on a richer movement vocabulary than the others, which is not strictly LOSO.

---

## 3. Calibration issue: BL2 is not static

The `linear_bl2_zero_reference` correction treats the last 60 s of the stream (1546–1606 s) as a static neutral posture and subtracts a linear drift such that BL2 mean = 0°. The data show BL2 was **not** static:

| Channel | BL1 std (0–60 s) | BL2 std (1546–1606 s) | Comment |
|---|---|---|---|
| theta_PL_pitch | 1.16° | **33.36°** | range −85° to +31° during "static" window |
| theta_PL_roll  | 2.16° | **35.20°** | range −159° to +41° |
| theta_LT_pitch | 0.97° | 10.61° | also moving |
| theta_LT_roll  | 2.64° | 13.30° | also moving |
| theta_TU_pitch | 1.61° | 2.83° | OK |
| theta_TU_roll  | 0.35° | 7.86° | borderline |

A scan for the genuinely-most-static 30 s window in the post-protocol stream finds one at t≈1668 s with PL_pitch std=1.07°, but its mean PL_pitch is **+36°**, i.e. the participant was sitting/flexed, not in neutral stance. There is therefore **no clean zero-reference window** at the end of this session.

**Implication.** The drift correction is anchored to a moving reference. Pitch residuals between BL1 and BL2 are small (PL −1.08°, LT +0.45°, TU +1.76°) so the practical bias on sagittal flexion features is probably ≤2–3°. Roll residuals are larger (PL −6.35°, LT +8.84°), but lateral features are already dropped per the established P02/P03 protocol, so this affects nothing the classifier sees.

**For next sessions:** require the participant to hold a clean neutral standing posture for 30–60 s as the *last* action before sensor removal. Update `session_timer.py` to add an explicit `BL2_STATIC` segment and refuse to finalise the session if std > 2° on PL_pitch.

---

## 4. Suspicious peak angles

`FATIGUE_FLEXION` shows **PL_pitch peak = 116°** (per-movement summary, file 02). Pelvic sagittal tilt cannot physically exceed ~60°. Two candidates:

1. Late-session Madgwick drift accumulating non-linearly (the BL2-anchored linear correction would not catch a non-linear curve).
2. Sensor slip on the pelvis IMU during the fatigue block, plausible because 20 reps of repeated flexion is exactly when belt slip happens.

Either way this feature value is non-physical and the 6 windows it sits in will inflate `imu_pelvis_angle_peak`. Either trim the affected fatigue reps or apply a hard cap (e.g. `imu_pelvis_angle_peak = min(actual, 60°)`) before any LOSO use.

---

## 5. Signal-criteria over-firing on safe protocol movements

Crosstab of protocol vs signal labels on the 1501 labelled windows:

| | Signal=safe | Signal=risky | Total |
|---|---|---|---|
| Protocol=safe   | 386 | **485** | 871 |
| Protocol=risky  | 117 | 513 | 630 |

**Agreement: 59.9%**, substantially lower than the 95.1% the synthetic dataset achieved post-LUMBAR_DOMINANT fix.

The 485 windows where signal-criteria flag risk on protocol-defined safe movements concentrate on:

| Movement | Signal-risky / Protocol-safe count |
|---|---|
| PICKUP_SYM | 100 |
| CLEAN_LATERAL_L | 90 |
| CLEAN_ROTATION_R | 84 |
| SIT_TO_STAND_NORMAL | 68 |
| **BASELINE_STATIC** | **55** |
| CLEAN_ROTATION_L | 45 |
| CLEAN_LATERAL_R | 41 |
| CLEAN_FLEXION | 2 |

The 55 baseline windows firing as risky are the diagnostic finding: the participant was supposed to be standing still and the signal criteria (`postural`, `pattern`, `combined`) are firing anyway. Likely causes:

- Residual drift in PL_pitch/LT_pitch between BL1 and the windows-being-evaluated (the linear correction over-shoots near the start).
- Possibly the `time_in_risk_zone > 0.05` postural criterion firing on baseline tilt around 5–8° that drifts past the noise floor.

PICKUP and CLEAN_LATERAL/ROTATION firing as risky is more defensible, since they do produce large kinematic excursions even when not biomechanically "risky" in McGill's sense, but the **synthetic generator and the FIS rules were never calibrated for these movements**, so the criteria are arguably mis-applied to a richer real-world protocol. This is a known gap.

---

## 6. Euler wrap-around

| Channel | Samples > \|180°\| | % of stream |
|---|---|---|
| theta_PL_roll | 2,606 | 1.54% |
| theta_LT_roll | 3,801 | 2.24% |
| (all pitch channels) | 0 | 0.00% |

Wraps occur during ROTATION/PICKUP/SHOULDER_DRIVEN reps where yaw exceeds ±π and the ZYX Euler extraction folds. This is the same artefact you already excluded from the feature set (`imu_lat_angle_peak`, `imu_lat_angle_mean` were dropped post-P02). Pitch channels are unaffected. **No action required**; the sagittal-only feature set is unharmed.

---

## 7. What this session is and isn't usable for

**Usable for:**
- Within-participant CV result: strong AUC=0.917 supports a per-subject calibration story.
- Demo / dashboard playback (`scripts/demo_risk_monitor.py` replay mode).
- Showing the protocol covers the full movement vocabulary including fatigue.

**Not yet usable for:**
- LOSO with P01/P02/P03: the protocol vocabulary mismatch (added FATIGUE / SIT_TO_STAND) and the non-static BL2 mean P07 isn't a clean drop-in. Use the intersection of movements only.
- Any claim about sEMG: none was recorded.
- The `imu_pelvis_angle_peak` feature on FATIGUE_FLEXION windows without a 60° cap.

---

## 8. Concrete next actions (ranked)

1. **Cap `imu_pelvis_angle_peak` at 60°** in `signal_processing/pipeline.py` for real-data sessions, OR drop the 6 fatigue windows where it exceeds physiological range. Re-export the feature matrix and re-run the within-participant CV; the AUC should barely move because the cap affects only ~6/1898 windows.
2. **Patch `session_metadata.json`**, which currently has a trailing-comma JSON syntax error after the `notes` field (won't parse with `json.loads`). Add the missing closing brace.
3. **Update `session_timer.py`** to enforce an explicit 60-s `BL2_STATIC` segment at end of session, and validate `std(PL_pitch) < 2°` before finalising.
4. **Add a label-source ablation table to the report**: protocol vs signal vs archetype on P07, plus the same crosstab on P01/P02 once available. This is the cleanest way to defend the choice of protocol labels for the primary thesis claim.
5. **Decide P07's role in LOSO**. Options ranked:
   - (a) Run LOSO on intersection of P01/P02/P03/P07 movements only (cleanest comparison).
   - (b) Train on P01–P03 movements, test on P07 movements that intersect (biased but informative).
   - (c) Hold P07 out as a "richer protocol" demonstration outside LOSO (fine if Phase II.B is already specified to use only the original protocol).

---

## 9. Files generated

In `results/participant_07_analysis/`:

- `01_calibration_integrity.csv`: BL1 vs BL2 means, stds, residuals
- `02_per_movement_angles.csv`: peak/mean abs angle by movement
- `03_roll_wrap_diagnostic.csv`: Euler-wrap sample counts per channel
- `04_protocol_vs_signal_crosstab.csv`: label-source crosstab
- `05_feature_importance_protocol.csv`: RF Gini importances (protocol labels)
- `06_cv_summary.json`: AUC/Sens/Spec/F1 and Youden threshold
- `plot_01_calibrated_timeseries.png`: sagittal/roll/angvel traces with movement bands
- `plot_02_risk_labels_per_movement.png`: protocol vs signal risky-% per movement
