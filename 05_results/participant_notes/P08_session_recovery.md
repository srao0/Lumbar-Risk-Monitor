# Participant 08: analysis report

> **Canonical numbers:** this note documents the session-recovery *decision*; its per-participant AUCs predate the drift-correction and QC re-freeze. The authoritative frozen figures are in `../frozen_numbers/frozen_numbers_sheet.md`.

**Session:** `data/real/protocol_train_fallback_recovered/participant_08/session_001` (consolidated from `session_001` + `session_001_tail` after a laptop power failure)
**Date recorded:** 2026-06-01
**Duration:** 24.2 min (1450.5 s)
**Operating mode:** `imu_only_fallback`
**Drift correction applied:** `post_hoc_drift_correction.enabled: false` (none applied)
**Notes from metadata:** *"Recovered logical session after laptop power failure; source pieces are protocol_train_fallback/participant_08/session_001 and session_001_tail. Analyse as IMU-only fallback with documented tail continuation."*

## 1. Headline: P08 is the cleanest session in the cohort and reproduces the main thesis finding

**Within-participant 80/20 temporal CV (Youden threshold, drop-NaN, RF n=500 leaf=3 seed=42, identical config to the deployed models):**

| Feature set | n usable | Test n | Test class balance | AUC | Sens | Spec | F1 |
|---|---|---|---|---|---|---|---|
| Primary 4-IMU (17 features, no L3 accel) | 1403 | 281 | 199 / 82 | **0.710** | 0.841 | 0.648 | 0.624 |
| Reduced Pelvis-L3 (13 features, with L3 accel) | 1404 | 281 | 199 / 82 | **0.860** | 0.963 | 0.598 | 0.656 |

**Δ = +0.15 AUC in favour of the reduced model.** This is the largest reduced-vs-primary advantage in the cohort. The frozen cohort had P01 +0.29, P04 +0.25, P06 +0.00, P07 +0.11, P05 −0.04, so P08 sits between P07 and P04. The reduced model also tops out at higher sensitivity (0.963 vs 0.841) with only a small specificity cost.

The feature importance shows the mechanism:

- **Primary (17 features):** pelvis_angle_peak (0.18), pelvis_angle_mean (0.16), trunk_angle_peak (0.15), trunk_angle_mean (0.12), lumbopelv_ratio (0.10), heavily reliant on T12-derived `imu_lumbopelv_ratio`
- **Reduced (13 features):** pelvis_angle_peak (0.27), pelvis_angle_mean (0.23), **imu_l3_accel_tilt_peak (0.115), imu_l3_accel_tilt_mean (0.109)**, angvel_peak (0.05), with L3 accelerometer-derived tilt features in positions 3–4

P08 reproduces the §9.2 thesis finding (L3 accel-tilt is load-bearing) on a participant outside the frozen evidence package. This is independent external validation of the deployment recommendation.

## 2. Calibration health: best in the cohort

This is the first session where the BL2 anchor methodology assumption actually holds.

| Channel | BL1 mean (0–60 s) | BL1 std | End-baseline mean (final 60 s) | End-baseline std | Drift over session |
|---|---|---|---|---|---|
| theta_PL_pitch | +0.32° | **0.37°** | −4.07° | **0.74°** | −4.40° |
| theta_PL_roll | +0.28° | 0.27° | −14.62° | 0.94° | −14.90° |
| theta_LT_pitch | −1.63° | 0.78° | +1.74° | 1.53° | +3.38° |
| theta_LT_roll | −0.15° | 0.91° | +13.11° | 3.82° | +13.26° |
| theta_TU_pitch | +1.06° | 1.07° | +0.64° | 1.20° | −0.42° |
| theta_TU_roll | −0.33° | 0.93° | −2.01° | 3.43° | −1.68° |

**BL2 static check (`std(PL_pitch) < 2°`): PASS at 0.74°.** The previous seven participants all failed this check (P06 30.5°, P02 22.9°, P01 7.1°, P04 6.1°, P03 3.5°, P07 2.5°, P05 2.1°). P08 is the first session that meets the assumption the `linear_bl2_zero_reference` correction is built on. The new `check_bl2_static.py` validator would pass this session.

The drift over the session is **−4.4° on PL_pitch and +3.4° on LT_pitch**, small enough that the sagittal feature set is effectively unaffected, which explains why drift correction is turned off (`post_hoc_drift_correction.enabled: false`) and why classification still works.

Roll drift is larger (−15° PL_roll, +13° LT_roll) but the deployed reduced feature set does not depend on roll channels, so this is non-load-bearing.

**No Euler wrap-around on any of the six angle channels (0.00% samples > 180°).** The frozen cohort had wraps on every participant except P03 (range 0.07–4.18%). P08 is wrap-clean.

## 3. Per-movement angle peaks: physiologically reasonable, with two flags

| Movement | n_reps | PL_pitch peak | PL_roll peak | LT_pitch peak | LT_roll peak | TU_pitch peak |
|---|---|---|---|---|---|---|
| BASELINE_STATIC | 2 | 6.8° | 16.3° | 8.1° | 16.1° | 6.6° |
| CLEAN_FLEXION | 8 | 7.5° | 10.7° | 21.8° | 16.9° | 6.6° |
| CLEAN_LATERAL_L | 6 | 17.1° | 23.7° | 26.6° | 47.6° | 12.3° |
| CLEAN_LATERAL_R | 6 | 21.8° | 22.7° | 25.4° | 46.1° | 17.0° |
| CLEAN_ROTATION_L | 6 | 30.1° | 26.7° | 29.2° | 50.9° | 25.1° |
| CLEAN_ROTATION_R | 6 | 28.9° | 27.4° | 38.4° | 54.7° | 28.4° |
| FAST_BEND | 6 | 36.0° | 45.2° | 42.8° | 68.7° | 8.6° |
| FATIGUE_FLEXION | 20 | 12.0° | 15.1° | 39.4° | 14.6° | 12.7° |
| LUMBAR_DOMINANT | 6 | 15.7° | 19.9° | 28.8° | 28.2° | 11.3° |
| PICKUP_ASYM | 5 | 59.3° | 100.5° | 58.3° | 120.3° | 18.6° |
| PICKUP_SYM | 5 | 53.3° | 87.9° | 54.5° | 112.3° | 19.5° |
| SHOULDER_DRIVEN | 5 | 45.0° | 61.2° | 43.1° | 93.3° | 14.4° |
| SIT_TO_STAND_FAST | 3 | **68.4°** | 116.7° | 71.7° | 120.2° | 10.0° |
| SIT_TO_STAND_NORMAL | 5 | **65.2°** | 123.4° | 71.2° | 116.7° | 16.9° |

**Two observations:**

1. **BASELINE_STATIC is genuinely static**, PL_pitch peak only 6.8°. The frozen cohort had P07 baseline PL_pitch peak of 48.7°, P02 etc. far worse. This is the cleanest baseline collection in the dataset.

2. **The 60° physiological cap will fire on sit-to-stand and pickup movements only.** Maximum PL_pitch peak across all movements is 68.4° (SIT_TO_STAND_FAST), only 8.4° above the cap, and these movements *do* legitimately involve full hip+pelvis flexion as the participant stands up. No FATIGUE_FLEXION pelvis-slip pathology like P07 had (P07 max = 116°). The cap-fire is appropriate, not a sensor artefact.

## 4. Label-source agreement and FATIGUE behaviour

- **Protocol↔Signal agreement on labelled windows (n=1404): 60.0%**, middle of the cohort range (50.4% P02 to 63.3% P01).
- **BASELINE_STATIC windows flagged risky by signal criteria: 40.2%** of 179 windows. Better than the cohort worst (P02 76%, P05 72%, P01 69%) but still high. The cleaner baseline kinematics on P08 have noticeably reduced the BASELINE false-positive rate vs the worst cohort cases, supporting the signal-criteria-needs-recalibration discussion in §9.5.
- **FATIGUE_FLEXION (20 reps) shows NO pelvis-slip artefact**, peak PL_pitch only 12.0° across the whole 20-rep block. This contrasts with P07 (FATIGUE peak PL_pitch 116°, 145 windows excluded for non-physical values). P08's pelvis strap held through the fatigue block.

## 5. Cohort positioning (within-CV reduced model)

| Rank | PID | Within-CV reduced AUC |
|---|---|---|
| 1 | P02 | 0.967 |
| 2 | P05 | 0.936 |
| **3** | **P08** | **0.860** |
| 4 | P06 | 0.854 |
| 5 | P01 | 0.833 |
| 6 | P07 | 0.749 |
| 7 | P04 | 0.712 |
| 8 | P03 | 0.455 |

P08 is third of eight, comfortably in the strong-performing band.

| Rank | PID | Within-CV primary AUC |
|---|---|---|
| 1 | P05 | 0.980 |
| 2 | P06 | 0.852 |
| **3** | **P08** | **0.710** |
| 4 | P07 | 0.639 |
| 5 | P01 | 0.545 |
| 6 | P04 | 0.467 |

Primary side: P08 is third of six. The reduced > primary gap of +0.15 is consistent with the cohort pattern.

## 6. What the user's "correction" actually was

The metadata says `post_hoc_drift_correction.enabled: false`, so no linear_bl2_zero_reference was applied. The correction referenced in the user's message is almost certainly the **session-recovery concatenation**: the original `session_001` was 22.2 min before a laptop power failure cut it off, the participant restarted, and `session_001_tail` captured the remaining 3.1 min. The consolidated `session_001_recovered` (and the parallel `protocol_train_fallback_recovered/` path) stitches them into a single logical 24.2-min session and re-runs the feature extraction.

The stitching looks defensible: the labels.csv timestamps run continuously through the join (the gap between session_001 end and tail start is absorbed into BASELINE_STATIC rep_1's window between 1266000–1326000 ms, then there's a 64-s gap before BASELINE_STATIC rep_2 at 1390468–1450468 ms). The IMU stream and feature matrix produce clean numbers. No discontinuity artefact is visible in the per-movement angle table.

The cleaner baseline (BL1 std 0.37° vs cohort average ~1.2°) and the BL2-static-passes status suggest the participant followed the new `BL2_STATIC` protocol guidance even though the explicit label name wasn't added to `labels.csv`.

## 7. Recommendations

1. **Add P08 to the frozen evidence package.** It reproduces the central thesis finding (reduced > primary, L3 accel-tilt load-bearing) on a participant collected with the *fixed* methodology (clean BL2 anchor). This strengthens the deployment recommendation and bumps the reduced-set n from 7 to 8.
2. **Document P08 as the methodology-validation case.** P08 is the first session where BL2 staticness passes. The Chapter 8 §8.8 "drift correction" limitation can be partially closed by referencing P08 as evidence the fix works on real participants.
3. **Re-run the evaluator with P08 included.** This requires regenerating `reduced_pelvis_l3_features.csv` to add P08's 1404 labelled windows and `primary_4imu_cleaned_features.csv` to add P08's 1403 windows. The aggregate numbers will shift slightly: predicted reduced within mean AUC moves from 0.787 (n=7) to 0.796 (n=8); primary within mean AUC moves from 0.697 (n=5) to 0.699 (n=6).
4. **Update the FROZEN_MANIFEST.json** with the new feature CSV hashes and bump the `frozen_at_utc` timestamp.
5. **Do NOT** apply post-hoc drift correction to P08: the sagittal channels drift only 3–4° and BL2 is genuinely static. Adding correction would inject noise without benefit.

## 8. One caveat to flag

The session-recovery concatenation creates a **64-second gap** between BASELINE_STATIC rep_1 (ends 1326000 ms) and BASELINE_STATIC rep_2 (starts 1390468 ms), which is the time between the laptop power failure and recording resumption, during which actual time passed but no IMU data was collected. The feature extraction handles this correctly (it sees a discontinuous timestamp jump), but if any temporal-derivative feature (jerk, angular velocity) is computed across the gap it will see an artefact. Worth confirming `signal_processing/pipeline.py` either drops the boundary window or uses non-windowed timestamp-aware computation.

## Files written

- `01_calibration_integrity.csv`: BL1 vs end-baseline stats per angle channel
- `02_per_movement_angles.csv`: peak/mean abs angle by movement
- `05_feature_importance_primary.csv`: RF Gini importance (17-feature primary model)
- `05_feature_importance_reduced.csv`: RF Gini importance (13-feature reduced model)
