# Labelling protocol — how a window becomes "safe" or "risky"

A pedantic but essential question: *who decided what counts as risky, and by what rule?* This is the answer.

## What "risk" means here

Risk is **not injury** and not a clinical diagnosis. A window is labelled **risky** if it meets either condition:

1. **Threshold exceedance** — the movement crosses a biomechanical limit, principally **lumbar flexion > 45°**, and/or high angular velocity / poor smoothness in the risk-bearing window.
2. **Baseline deviation** — the movement departs markedly from the **same participant's** calibrated baseline (captured at the start of each session in an N-pose / quiet-standing rest).

## How labels were actually assigned

Labels are **protocol-derived**, not hand-annotated frame by frame. Each session follows a scripted movement protocol (baseline, clean flexion, fast flexion, twisted/loaded variants, fatigue blocks). The protocol defines, for each scripted movement, whether it is a safe or risky archetype. The segment table (`labels.csv`) records the start/end of each movement and its `risk_class`; the windowing step inherits that label for every 2-second window whose centre falls inside the segment.

This means:
- **Ground truth is as good as the protocol design**, not an independent clinical rater. It is consistent and reproducible, but it is a *constructed* target — stated plainly so it is not over-read.
- The feature matrix carries several label columns (`risk_class_protocol`, `risk_clinical`, `risk_layman`) representing the same event seen through different lenses; the modelling target is `risk_class`.

## Window definition

- **Window length:** 2 seconds.
- **Sampling into windows:** centred windows; a window's label is the protocol risk class of the movement at its centre time.
- **Baseline z-scores:** each window also carries features expressed as deviations from the participant's own baseline (`imu_z_*`, `emg_z_*`), which is how the "deviation from personal baseline" arm of the risk definition enters the model.

## Why this matters for the results

Because the labels are protocol-derived and per-participant baselines are used, **within-participant** evaluation is the most directly meaningful, and **LOSO** (generalising the labelling/feature relationship to an unseen person) is the harder, more uncertain test. The results in `../05_results/` report both and are framed with this in mind.
