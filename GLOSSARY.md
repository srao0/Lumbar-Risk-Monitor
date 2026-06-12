# Glossary

Terms and acronyms used throughout this repository, in plain language.

## System & anatomy
- **PLTU** — the four-IMU placement model: **P**elvis, **L**3 (lumbar), **T**12–L1 (thoracolumbar), **T**4–T6 (upper thoracic). The pelvis IMU is the reference; the others measure segment orientation relative to it.
- **IMU** — Inertial Measurement Unit. Combines accelerometer + gyroscope (+ magnetometer) to estimate orientation. Here, the BNO/ICM-class sensors sampled at 100 Hz.
- **sEMG / EMG** — surface Electromyography. Skin-electrode measurement of muscle electrical activity, sampled at ~200 Hz in the implemented pipeline.
- **ES / OBL** — muscle groups: **ES** = erector spinae (LES/RES = left/right), **OBL** = obliques (LOBL/ROBL = left/right). See the data-dictionary note on legacy `MF` naming.
- **ESP32-S3** — the microcontroller doing on-body data acquisition.
- **AHRS** — Attitude and Heading Reference System: the algorithm that turns raw IMU readings into an orientation estimate.
- **Madgwick filter** — the specific AHRS gradient-descent orientation filter used.

## Movement & risk features
- **Trunk-flexion angle** — how far the trunk is bent forward, derived from inter-segment orientation.
- **Angular velocity** — how fast the trunk is bending (fast bending is a risk cue).
- **Risk zone** — trunk flexion beyond a **45°** threshold; "time in risk zone" counts how long a window spends past it.
- **Smoothness / SAL / LDLJ** — movement-smoothness measures. **LDLJ** = Log Dimensionless Jerk; jerky movement is a risk cue. (SAL = Spectral Arc Length, the related frequency-domain measure.)
- **RMS / MAV / ZCR** — EMG time-domain features: Root Mean Square (activation level), Mean Absolute Value, Zero-Crossing Rate.
- **Asymmetry index (AI) / co-activation index (CAI)** — left–right imbalance and antagonist co-contraction from EMG.
- **Risk** — *not* injury. Defined as either exceeding a biomechanical threshold (e.g. >45° flexion) **or** deviating from the participant's own baseline. See `04_data/labelling_protocol.md`.

## Evaluation
- **AUC / AUROC** — Area Under the ROC Curve; classifier discrimination, 0.5 = chance, 1.0 = perfect.
- **Within-participant CV** — cross-validation where train/test come from the *same* person (measures personalised performance).
- **LOSO** — Leave-One-Subject-Out cross-validation; train on everyone *except* one person, test on them (measures generalisation to a new person).
- **Wilcoxon (signed-rank)** — the paired non-parametric test used to compare the two configurations.
- **FIS** — Fuzzy Inference System (Mamdani); the layer that turns a model probability into a traffic-light risk level with interpretable rules.
- **RF / LR / SVM** — Random Forest / Logistic Regression / Support Vector Machine, the classifiers compared.

## Study structure
- **Primary** — the full four-IMU, 17-feature configuration.
- **Reduced** — the pelvis + L3, 13-feature configuration (the recommended deployment set).
- **Full hybrid** — IMU + sEMG together.
- **IMU-only fallback** — IMU sensors alone; the mode the participant cohort was recorded in after the EMG amplifier fault.
- **Phase I / II.A / II.B / II.C / III** — the study phases; see `05_results/README.md`.
