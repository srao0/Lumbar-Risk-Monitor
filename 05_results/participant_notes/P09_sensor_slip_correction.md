# Participant 09 — Analysis report

> **Canonical numbers:** this note documents the sensor-slip *correction decision*; its per-participant AUCs predate the drift-correction + QC re-freeze. The authoritative frozen figures are in `../frozen_numbers/frozen_numbers_sheet.md`.

**Session:** `data/real/protocol_train_fallback/participant_09/session_001_bl2corrected` (BL2 drift-correction applied)
**Sister variant:** `session_001` (uncorrected — kept for diagnostic comparison)
**Date recorded:** 2026-06-01
**Duration:** 26.2 min (1574.2 s)
**Operating mode:** `imu_only_fallback`
**Drift correction applied:** `linear_bl2_zero_reference` over BL2 window 1469000–1574232 ms (10,522 samples, ~105 s) — auto-extended from the standard 60 s

## 1. Verdict: P09 had substantial pelvis-IMU slip during the session; the BL2 correction is necessary but only partially repairs it

P09 is the dirtiest session in the cohort, and it is the strongest test case so far of whether the new BL2-anchored drift correction can salvage a session with sensor slip. The correction works for the sagittal-only deployed model but leaves residual artefacts on roll channels that would compromise the primary 4-IMU model if it depended on T12 features.

**Headline within-CV AUC, both feature sets, both variants:**

| Variant | Primary 4-IMU (17 features) | Reduced Pelvis-L3 (13 features) | Δ |
|---|---|---|---|
| Uncorrected `session_001` | 0.652 | 0.912 | +0.260 |
| Corrected `session_001_bl2corrected` | **0.855** | **0.882** | +0.027 |

The correction **rescues the primary model from 0.652 → 0.855 (+0.20)** while leaving the reduced model essentially unaffected (0.912 → 0.882, a small drop within noise). The reduced model was robust to the slip in the first place because it doesn't depend on T12 roll-derived features; the primary model needs the correction.

## 2. The drift profile is non-linear — this is sensor slip, not Madgwick drift

I sampled `theta_PL_roll` at 1-minute intervals across the uncorrected session:

| Time | PL_roll | Δ vs previous |
|---|---|---|
| 0.5 min | −0.13° | — |
| 8.5 min | −12.74° | gradual ramp |
| 12.5 min | −35.56° | acceleration |
| 19.5 min | −71.33° | continued |
| 22.5 min | −100.12° | continued |
| **23.5 min** | **−77.27°** | **REVERSE jump (+23°)** |
| 25.5 min | −135.09° | sudden continuation |
| 26.1 min | −137.68° | end |

The 23.5-min reverse jump is the smoking gun: a discrete sensor-slip event. Linear interpolation cannot model this. The `linear_bl2_zero_reference` correction assumes constant-rate drift and therefore:
- Over-corrects windows where actual drift was slow (first 8 min)
- Under-corrects windows where actual drift was fast (last 4 min)
- Cannot remove the artefact at the 23.5-min discontinuity

The same pattern shows on `theta_LT_roll` with opposite sign (the pelvis IMU rotation appears in both roll channels because LT_roll is computed relative to the pelvis frame), confirming the slip is at the pelvis IMU, not the lumbar one.

## 3. What the correction does well

- **BL2 staticness:** the BL2 window has PL_pitch std = 1.16° (passes the <2° validator threshold). Same window in the uncorrected data has std 1.39° — also static. So the BL2-as-zero-reference assumption holds for this session, even though the body of the session is broken.
- **BL1 mean shift is small:** after correction, BL1 PL_pitch sits at +1.87° (vs natural +1.21° uncorrected). The linear ramp's intercept-at-BL1 cost is bounded — small enough not to disturb feature distributions much.
- **BASELINE_STATIC signal-risky rate drops from 9.9% → 2.1%** — cohort best. This is by far the strongest single piece of evidence the correction helped.
- **Per-movement pitch ranges become physiologically reasonable:** corrected SHOULDER_DRIVEN peak PL_pitch is 66.6°, FAST_BEND 62.7°, LUMBAR_DOMINANT 58.6° — large but in-range.

## 4. What the correction does NOT fix

- **FATIGUE_FLEXION PL_roll peak is 145° in the corrected version** (vs 131.8° uncorrected — slightly worse). The linear ramp pushed the slipping segment further from zero rather than closer.
- **BASELINE_STATIC PL_roll peak is still 87.8° corrected** (vs 140.7° uncorrected). Better but still non-physical for static standing.
- **CLEAN_LATERAL and CLEAN_ROTATION show PL_roll 45–47°** post-correction — that's plausible for active lateral movement but also picks up the slip-residual.
- **SIT_TO_STAND_FAST shows PL_pitch peak only 21.0°** post-correction — suspiciously low for a fast sit-to-stand (P08 showed 68°). Likely the slip was active during this segment and the linear correction overshot it.

The deployed reduced 13-feature model uses none of the roll channels, so these residual artefacts do not enter the classifier. They are a problem for descriptive per-movement reporting and for any analysis using T12-roll-derived features (i.e. the primary set's `imu_compensation_index` and `imu_lumbopelv_ratio`).

## 5. Within-CV details (corrected variant)

**Primary 4-IMU:** n=1528, test 223 safe / 83 risky → AUC=0.855, Sens=0.892, Spec=0.767, F1=0.708. Top features: `imu_lumbopelv_ratio` (0.20), `imu_pelvis_angle_peak` (0.09), `imu_trunk_angle_peak` (0.08). The lumbopelv_ratio jumping to importance 0.20 is unusual — and it should be flagged as potentially over-fitting to the post-correction artefact pattern in T12 features. Worth a separate ablation.

**Reduced Pelvis-L3:** n=1528, same split → AUC=0.882, Sens=0.699, Spec=1.000, F1=0.823. Top features: `imu_l3_accel_tilt_mean` (0.14), `imu_pelvis_angle_peak` (0.13), `imu_pelvis_angle_mean` (0.12). **L3 accel-tilt features in top 3 again — P09 independently reproduces the §9.2 thesis finding for the second consecutive new participant** (after P08).

Note the asymmetric sens/spec: spec=1.000 with sens=0.699 means the model never falsely flags safe windows as risky but misses ~30% of true risky windows. The Youden threshold landed at a conservative cut. Acceptable trade-off, but reflects the test split's class imbalance (only 83 risky in test).

## 6. Cohort position (within-CV reduced AUC)

| Rank | PID | AUC |
|---|---|---|
| 1 | P02 | 0.967 |
| 2 | P05 | 0.936 |
| **3** | **P09** | **0.882** |
| 4 | P08 | 0.860 |
| 5 | P06 | 0.854 |
| 6 | P01 | 0.833 |
| 7 | P07 | 0.749 |
| 8 | P04 | 0.712 |
| 9 | P03 | 0.455 |

P09 sits **third of nine** on the reduced model — strong placement despite the sensor-slip pathology. The deployed model is robust to the slip pattern that broke the primary model.

## 7. Recommendations

1. **Use the corrected variant (`session_001_bl2corrected`)** as the canonical P09 data. The uncorrected variant should be archived and kept only for diagnostic purposes.
2. **Add P09 to the frozen evidence package only for the reduced model.** P09 is a fair test case for the deployment recommendation: the slip pathology is exactly the kind of failure mode the reduced model was designed to be robust to. Including it on the primary side risks over-stating primary performance because the `imu_lumbopelv_ratio` jump to 0.20 importance is suspicious — likely the classifier learning post-correction roll artefacts.
3. **Apply the 60° physiological cap** as documented — it will fire on SHOULDER_DRIVEN, FAST_BEND, LUMBAR_DOMINANT, PICKUP_SYM, CLEAN_FLEXION (PL_pitch peaks 57–67°). Several windows, but bounded.
4. **Flag P09 in §8.8 of the results chapter** as a documented sensor-slip case where the BL2 correction salvaged the deployed model's performance. This is a stronger evidence point than the calibration discussion currently has: it shows the BL2 fix works on a hard case.
5. **Investigate the FATIGUE_FLEXION + SIT_TO_STAND_FAST segments** before considering them for any per-movement analysis. The slip was likely active across these blocks (timestamps 22–26 min, which covers the FATIGUE block and approach to BL2).
6. **Do NOT re-train the deployed models with P09 included until the slip-segment windows are explicitly handled** (either excluded or flagged with a `slip_segment` column). Without that, the trained model will see post-correction roll artefacts as signal.

## 8. The correction did exactly what it was designed to do

The BL2-anchored linear correction is **upper-bounded by the linearity of the underlying drift**. P09's drift is not linear — there is a clear discrete slip event at minute 23. A linear correction cannot remove that. What it can do — and does — is:

- Restore the BL1 ↔ BL2 endpoints to a known reference frame
- Reduce the bias on time-averaged features
- Make the sagittal pitch channels approximately usable

For the deployed reduced 13-feature model, that is enough: AUC 0.882, third in the cohort. For the primary 4-IMU model it is not: the residual roll-channel artefacts inflate `imu_lumbopelv_ratio` importance to a suspicious degree.

P09 is therefore a case study that **strengthens the deployment recommendation**: when sensors slip, the reduced model degrades gracefully (0.912 → 0.882, ~0.03) while the primary model fails badly (0.652 uncorrected) and even after correction looks suspect.

## Files written

- `results/participant_09_analysis/` — drift profile, per-movement angles for both variants
