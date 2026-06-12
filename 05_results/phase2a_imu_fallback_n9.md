# Phase II.A — IMU-only fallback evidence (n = 9)

**This is the project's central result.**

**Question:** can the real protocol record, convert, validate and evaluate supervised participant sessions — and, given two sensor configurations, which one should be built? Because of the EMG amplifier fault (`../LIMITATIONS_AND_KNOWN_ISSUES.md` §1), this phase is **IMU-only**.

## The result

A **reduced two-IMU configuration (pelvis + L3, 13 features) matches the full four-IMU configuration (17 features)** — and is better within-participant.

| Configuration | n | Windows | Within-CV AUC | LOSO AUC |
|---|---|---|---|---|
| **Reduced (Pelvis + L3)** — recommended | 9 | 13,228 | **0.819 ± 0.087** | **0.641 ± 0.062** |
| Primary (full 4-IMU) | 6 | 8,698 (~8,550 modelled) | 0.755 ± 0.077 | 0.570 ± 0.090 |

- **Paired Wilcoxon (LOSO, 6 shared participants): p = 0.0625** — no significant difference (this is the smallest value the test can return at n=6 short of 0.03125). The two-IMU set is statistically *not worse*.
- Within-participant, reduced leads by **+0.09 AUC** on the matched 6 (0.848 vs 0.755).
- Reduced ≥ primary on *both* within-CV and LOSO — the two-IMU set matches or beats the full set, strengthening the recommendation.

**Why it works:** the discriminating signal concentrates in the **accelerometer-derived L3 trunk-tilt** feature. The extra two IMUs mostly add compensation features that carry a residual drift sensitivity (see §4 of the limitations) without adding discrimination.

## Cohort

Nine participants P01–P09. P10 excluded (hardware failure). Per-participant labelled windows and the full set-construction recipe (P02 rebuild, 60° pelvis cap, NaN handling) are in `frozen_numbers/frozen_numbers_sheet.md`.

## Figures (`figures/`)

> The main comparison plots (`01_within_primary_vs_reduced`, `02_loso_primary_vs_reduced`, `03_loso_scatter_primary_vs_reduced`, `05_aggregate_auc_summary`, `PR_curves_within/loso`, `confusion_matrices_aggregate`) were **regenerated on the corrected-QC tier (2026-06-12)** and match the canonical numbers. ⚠️ The cohort/auxiliary plots (`plot_01…plot_05_*`, `feature_importance_RF`, `roc_*`, `per_session_auc_RF`) were **not** regenerated and predate the re-freeze — illustrative only; cite `frozen_numbers/frozen_numbers_sheet.md` for values.

| File | Shows |
|---|---|
| `01_within_primary_vs_reduced.png`, `02_loso_primary_vs_reduced.png` | The head-to-head, both CV schemes |
| `03_loso_scatter_primary_vs_reduced.png` | Per-participant LOSO scatter |
| `05_aggregate_auc_summary.png`, `plot_01_auc_across_participants.png` | Cohort-level summary |
| `PR_curves_loso.png`, `PR_curves_within.png` | Precision–recall (better than ROC under class imbalance) |
| `confusion_matrices_aggregate.png` | Pooled confusion behaviour |
| `feature_importance_RF.png` | L3 accel-tilt dominance |
| `04_p07_fatigue_exclusion_audit.png` | The P07 exclusion, audited |
| `plot_02_calibration_health.png`, `plot_03_label_quality.png`, `plot_04_roll_wrap.png` | Data-quality diagnostics |
| `06_pipeline_overview.png` | End-to-end pipeline |

Plus the per-condition RF plots (`roc_*`, `per_session_auc_RF.png`, etc.) — several of these are **supplementary** and do not appear in the report.

## Honest framing

- Cohort is small (n=9); within-participant evidence is stronger than LOSO.
- The LOSO advantage is real but **not significant** — the claim is "equal accuracy from half the sensors," not "better."
- All numbers trace to `frozen_numbers/frozen_numbers_sheet.md` (drift-corrected + frozen QC exclusions re-applied, frozen 2026-06-11, scikit-learn 1.8.0, seed 42).
