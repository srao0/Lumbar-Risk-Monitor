# Stage2b personalisation pilot (RQ4, corrected-QC n=9)

A pilot test of *calibrate-on-first-use*: does fine-tuning the population model with a small slice of a participant's own data beat the generic model? Built on the QC-clean corrected base (`make_stage2b_qc_input.py`, then `run_personalised_stage2b.py`).

Headline (`summary_metrics.csv`, mean AUROC across participants):
- Generic (other participants only): **0.621**
- Personal, calibration-only: **0.767**
- Personal, calibration + augmentation: **0.781** (**+0.160 over generic, 8/9 participants improve**)

A short personal calibration lifts AUROC substantially, which is the strongest single argument for a calibrate-on-first-use deployment. Pilot scale (n=9, one session each); per-participant detail in `per_participant_metrics.csv`, augmentation recipe in `augmentation_manifest.json`, narrative in `interpretation.md`.
