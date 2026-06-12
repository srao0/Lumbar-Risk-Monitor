# Limitations & Known Issues

This document states, up front, what did not go to plan and what was not completed. It is here so that nothing in the results needs to be discovered by inference. Where a limitation shaped a design decision, that is noted — several of them strengthen rather than weaken the central recommendation.

---

## 1. The EMG amplifier failed — the cohort is IMU-only

The system has two operating modes: **full hybrid** (IMU + sEMG) and **IMU-only fallback**. Partway through data collection the OpenBCI Ganglion sEMG amplifier developed a fault, so the **nine-participant cohort (P01–P09) was recorded in IMU-only fallback mode.** All headline real-data evidence is therefore IMU-only. This was declared as an explicit fallback route in the protocol rather than hidden.

- **Consequence:** the IMU-vs-(IMU+sEMG) comparison cannot be made on the full cohort.
- **Mitigation:** the hybrid path is demonstrated separately on one participant (**P14**, `05_results/phase2b_personalised_vs_population.md`), where sEMG was working. With sEMG resting-baseline-normalised, the hybrid lift is **+0.048 AUC (0.790→0.839), p = 0.006, 11/13 sessions** — **marginally meeting SC2**, but the benefit is narrow (load-asymmetry detection, not fatigue/compensation — that earlier reading was a drift artefact) and near-zero on the hardest flexion-risk windows (+0.016). sEMG is therefore a measurable but marginal add-on and is **not adopted** for the deployed system. n=1 for the hybrid claim.

## 2. Phase II.C (generalisation to held-out varied movements) was not completed

The frozen models **were** evaluated read-only on held-out participants (P12, and P14's conforming sessions) — genuine generalisation-to-unseen-person evidence, with the reduced Pelvis-L3 route strongest (P14 0.693, P12 0.583). What is **not** established is generalisation to truly *varied, unstructured* movements: the only real varied session (P03) is degenerate and leaks on the reduced route (it was in the training set), and the synthetic "varied" set is a **verification aid only — not generalisation evidence.** P11 was excluded for an IMU acquisition dropout. So: held-out generalisation to new participants is shown; generalisation to novel movement *types* remains a limitation. See `05_results/phase2c_held_out_varied.md`.

## 3. Small cohort (n = 9)

Nine participants is small for population-level (LOSO) claims. The within-participant results are more robust than the LOSO results, and the central claim is framed accordingly: the reduced set's within-participant advantage is the headline (+0.09 AUC on the matched participants), and its LOSO edge is present but **not statistically significant** (Wilcoxon p = 0.0625, the smallest value the test can return at n=6 short of 0.03125). P10 was excluded for a hardware failure during the session.

## 4. Madgwick orientation drift, and a feature-construction interaction

The Madgwick AHRS estimate drifts over long sessions, particularly in yaw. A separate issue was found in the trunk-flexion feature: an **absolute-sum construction interacted with this drift**, inflating apparent flexion on the segments furthest up the chain. After correction, the **primary (4-IMU) set carries a residual L3–T12 drift sensitivity that the reduced (pelvis + L3) set does not** — which is one mechanistic reason the reduced set holds up. The corrected pipeline is what produced the frozen numbers; the issue is documented here so the correction is traceable, not silent.

## 5. Participant-session data-quality issues (documented, not swept up)

Several sessions needed documented corrections before they could be pooled. Each has a case note in `05_results/participant_notes/`:
- **P07** — fatigue-driven exclusion of part of the session (audit figure included).
- **P08** — session recovery after an interruption.
- **P09** — sensor-slip correction.
- **P10** — full exclusion (hardware failure).
- **P11/P12** (not in the n=9 set) — the first full-hybrid sessions were compromised (saturated EMG channel + an IMU dropout), which is part of why the hybrid analysis settled on P14.

## 6. Ground-truth labelling is protocol-derived, not clinically adjudicated

"Risky" windows come from the movement protocol and a 45°-flexion / baseline-deviation rule (`04_data/labelling_protocol.md`), **not** from independent clinical annotation. "Risk" here means biomechanical-threshold exceedance or personal-baseline deviation — explicitly **not** injury or a medical diagnosis.

## 7. Hardware-validation accuracy caveats

The pole-rig used to validate IMU angle output was read against a manual protractor reference, so the angular-accuracy figures carry the protractor's own reading tolerance. The PCB (`02_hardware/`) is a rev-2 design; the cohort data was collected on the cabled breadboard build, not the wireless PCB.

## 8. Reproducibility caveats

- A fixed seed (42) is used throughout, so a given step reproduces on the **same scikit-learn build (1.8.0)**; a different build can shift AUCs slightly.
- SHA-256 manifests are computed over files as written, so hashes match byte-for-byte only on the same OS line endings (Windows CRLF vs Linux LF). **The AUCs themselves are unaffected** — only the hashes differ.
- Some per-participant correction scripts **overwrite their inputs in place** (with timestamped backups). See the SAFE-RUN warning in `03_code/REPRODUCE.md`.
