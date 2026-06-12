# Phase II.B: Personalised vs population, and the hybrid (IMU + sEMG) check

**Two questions, one phase:**
1. Does a model calibrated to an individual outperform a generic population model?
2. Since the cohort was IMU-only, does adding sEMG actually help, demonstrated on the one participant (**P14**) where the full hybrid rig worked?

## 1. Personalised > population

Within-participant (personalised) AUCs consistently exceed leave-one-subject-out (population) AUCs, the same pattern visible in Phase II.A (reduced within-CV 0.819 vs LOSO 0.641). A model that has seen a person's own baseline and movement style classifies their risk better than one trained on a population. This is expected and supports a calibrate-on-first-use deployment design. For P14 specifically: population 0.694 → personalised 0.778 (+0.084 pooled, +0.040 per session, 8/13 sessions positive, p = 0.31). A separate stage2b personalisation pilot on the n=9 cohort (`stage2b_pilot/`) gives generic 0.621 → personal-augmented 0.781 (**+0.160**, 8/9 participants).

## 2. The hybrid check (P14, normalised sEMG)

P14 is the chosen full-hybrid participant (frozen 13-session set, s05 excluded). With sEMG resting-baseline-normalised and the IMU features drift-corrected, the comparison is IMU-only vs IMU+sEMG on the same person:

- **Aggregate sEMG lift:** **+0.048 AUC** (0.790 → 0.839), **11/13** sessions positive, **p = 0.006** (pooled 0.794 → 0.834). sEMG carries **35%** of the model's feature importance.
- **By construct (decomposition):** all windows +0.040, excluding baseline +0.031, hard subset **+0.016**. The gain shrinks toward the genuinely hard flexion-risk windows.
- **By movement archetype:** asymmetric-pickup **+0.049** and shoulder-driven **+0.018** improve; fatigue **−0.002** and lumbar-dominant **−0.008** do not. The benefit is load asymmetry, *not* fatigue/compensation. The earlier "sEMG helps fatigue/compensation" reading was an IMU-drift artefact and does not survive correction.
- **Conclusion:** sEMG **marginally meets SC2**. The +0.048 aggregate lift is statistically significant (p = 0.006) and sits at the margin of the +0.05 design threshold, but the benefit is narrow (load asymmetry) and near-zero on the hardest flexion-risk windows. So sEMG is a measurable but marginal add-on and **is not adopted for the deployed system**; IMU remains the baseline doing the heavy lifting.

## Figures (`figures/`)

**Canonical (emgnorm-refreshed):** `fig1_q1_per_session_auc.png` (per-session IMU vs hybrid), `fig5_personalised_vs_population.png` (the core comparison), `fig7_auc_decomposition.png` (IMU vs EMG contribution by construct), `fig8_confusable_pairs.png` (where sEMG helps, by archetype), `fig10_reclassification.png`. Values in these match `corrected_summary.json`.

Supplementary (`fig2_pooled_roc`, `fig3_per_movement`, `fig4_feature_importance`, `fig6_data_quality`, `fig9_gain_vs_imu_strength`) predate the emgnorm refresh, so they are illustrative only; cite `corrected_summary.json` for values.

## Framing

This is **n=1 for the hybrid claim** (P14). It is a demonstration that the hybrid path works and a directional result on where sEMG helps, not a population-level hybrid evaluation, which the amplifier fault made impossible. See `../LIMITATIONS_AND_KNOWN_ISSUES.md` §1.
