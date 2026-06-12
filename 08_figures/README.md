# Figures

Result plots, hardware photos, and placement figures, named by what they show rather than by figure number, so they can be matched to the report without cross-referencing.

> The headline `results_plots/` comparison figures (`within_cv_primary_vs_reduced`, `loso_primary_vs_reduced`, `loso_scatter_primary_vs_reduced`, `aggregate_auc_summary`, `pr_curves_within_cv`, `pr_curves_loso`, `confusion_matrices`) were **regenerated on the corrected-QC tier (2026-06-12)** and match canonical numbers. Warning: the cohort/auxiliary plots (`auc_across_cohort`, `calibration_health`, `euler_roll_wrap`, `label_quality_agreement`, `loso_auc_cohort`, `p07_fatigue_exclusion_audit`, `p09_variant_feature_importance`, `per_movement_risk_probability`, `pipeline_overview`) were not regenerated and predate the re-freeze, so use `../05_results/frozen_numbers/frozen_numbers_sheet.md` for values.

## `results_plots/`

The evaluation figures: within-participant and LOSO comparisons of the reduced versus full configurations (`within_cv_primary_vs_reduced`, `loso_primary_vs_reduced`, `loso_scatter_primary_vs_reduced`), precision–recall curves (`pr_curves_within_cv`, `pr_curves_loso`), aggregate confusion matrices, per-movement risk probabilities, the aggregate AUC summary, the data-processing overview (`pipeline_overview`), the P07 fatigue-exclusion audit, the P09 variant feature-importance comparison, the P10 augmentation ablation, and the cohort-wide plots (AUC across participants, calibration health, label-quality agreement, Euler roll-wrap, LOSO AUC).

`results_plots/` also holds `C2_p10_augmentation_delta_auc`, the Appendix C P10 augmentation ablation on the corrected-QC base (primary +0.011, reduced −0.011).

## `preprocessing/` (current, regenerated 2026-06-12)

Signal-processing method figures, freshly generated: `F1_gyro_still_calibration` (gyro bias from a still hold), `F3_emg_rest_baseline_normalisation` (sEMG amplitude before/after resting-baseline normalisation), `F4_preprocessing_fused_vs_accel_gyro` (Madgwick-fused orientation vs raw accel/gyro), `F5_segmental_sagittal_flexion` (per-segment sagittal flexion), `F7_emg_envelope_window` (sEMG envelope window). These are current; F2 (drift reconstruction) and F6 are still outstanding.

## `circuits/`

Acquisition hardware. Schematics and layouts (`IMU_Breadboard_Circuit.png`, `Ganglion_circuit-layout.png`), photos of the IMU breadboard build (`imu_breadboard_build_1/2/3.jpg`), and labelled photos of the sEMG acquisition (`cyton_semg_board_labelled.png`, `semg_electrodes_full_setup_labelled.png`).

## `placement/`

Sensor-placement figures: the PLTU placement model (`pltu_imu_placement_model.png`) and a spine reference diagram (`Spine_diagram.webp`).
