# Results

Results are split by study phase so each can be read on its own. Every phase folder has its own `figures/` directory containing both the figures that appear in the report and supplementary plots that don't, so all the evidence is visible rather than only the curated subset.

| Phase | Question | Verdict | Note |
|---|---|---|---|
| **I** | Does the full pipeline work end-to-end on synthetic data with known ground truth? | Validated | `phase1_synthetic.md` |
| **II.A** | Can real sessions be recorded/processed/evaluated (IMU-only fallback), and which sensor set wins? | **Reduced pelvis+L3 = full 4-IMU** | `phase2a_imu_fallback_n9.md` |
| **II.B** | Does personalised calibration beat a population model? | Personalised > population | `phase2b_personalised_vs_population.md` |
| **II.C** | Do frozen models generalise to held-out varied movements? | **Not completed** | `phase2c_held_out_varied.md` |
| **III** | Can the system give real-time replay traffic-light feedback with explanations? | Demonstrated | `phase3_replay_demo.md` |

## The main result

On nine participants, the reduced two-IMU configuration (pelvis + L3, 13 features) matches the full four-IMU configuration (17 features), and beats it within-participant. Half the sensors, equal accuracy. That is the design to build next.

| Configuration | Within-CV AUC | LOSO AUC |
|---|---|---|
| Reduced (Pelvis + L3), n=9 | **0.819 ± 0.087** | **0.641 ± 0.062** |
| Primary (4-IMU), n=6 | 0.755 ± 0.077 | 0.570 ± 0.090 |
| Paired Wilcoxon (LOSO) | n/a | **p = 0.0625 (n.s.)** |

## Supporting files
- `frozen_numbers/`: the canonical numbers sheet and per-participant CSVs. Every number in the report should trace to here.
- `participant_notes/`: case notes for each participant correction/exclusion (P07–P10, P14).

The numbers are framed plainly about cohort size and the IMU-only fallback. See `../LIMITATIONS_AND_KNOWN_ISSUES.md`.
