# Appendix C — P10 augmentation ablation (corrected-QC base)

P10 was excluded from the cohort for a hardware failure, but its partial data was tried as synthetic-style augmentation to see whether it helps or hurts. Re-run on the canonical corrected-QC base (`rerun_p10_augmentation_ablation.py`, seed 42; the script reproduces the canonical LOSO means exactly as a parity check before adding P10).

Result (`p10_ablation_summary.json`, LOSO mean AUROC, standard → augmented):
- Primary 4-IMU: 0.570 → **0.581 (+0.011)**
- Reduced Pelvis-L3: 0.641 → **0.630 (−0.011)**

So adding P10's augmented windows nudges the primary set up slightly and the reduced set down slightly — both within noise. The takeaway: P10 augmentation is not worth including; the exclusion stands. Figure `C2_p10_augmentation_delta_auc.png`; table in `p10_ablation_table.csv` / `.tex`.
