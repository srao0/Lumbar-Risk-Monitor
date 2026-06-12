# Participant 14: full-hybrid (IMU + sEMG) analysis report

**Phase II.B · Spinal Movement Risk Monitor**
Frozen 2026-06-07 · Chosen full-hybrid participant · **14 sessions collected, 13 frozen** (session_05 excluded, §2.1) · 19,342 labelled 2 s windows · manifest SHA-256 `23fbab89…`

> **All numbers and all figures in this report are computed on the frozen 13-session dataset** (`data/real/protocol_train_full_hybrid/participant_14/combined_features.csv`, corrected `LOBL/ROBL` channels). They match the dataset manifest hashed for Tables 6.4/6.5. §3C documents that the pre-exclusion 14-session run gave a near-identical headline.

---

## 0. What this report answers

1. **Does adding surface EMG to the IMU stack improve risky-movement detection?** (sEMG + IMU vs IMU-only, within a *personalised* model trained on P14's own sessions.)
2. **Did personalised training beat population-level training in the IMU-fallback regime?** Answered on the frozen Phase II.A cohort (within-CV vs LOSO) and re-confirmed on P14 (P14-personalised IMU model vs the deployed population model applied to P14).

| Question | Result | Effect size | Significance |
|---|---|---|---|
| Q1, sEMG+IMU vs IMU-only (P14, LOSO-session, *aggregate*) | Hybrid wins **in aggregate** | +0.045 mean AUC (0.799→0.844); pooled 0.794→0.835 | 12/13 sessions, paired Wilcoxon **p = 0.0005** |
| Q2A, personalised vs population (Phase II.A reduced, n=7) | **Personalised wins** | +0.162 AUC (0.624→0.787) | 7/7, **p = 0.016** |
| Q2B, personalised vs population (P14 fallback, reduced) | **Personalised wins** | +0.138 AUC (0.583→0.725) | 11/13, **p = 0.0012** |

**Qualification on Q1 (see §3B).** The aggregate Q1 gain overstates operational value. Restricted to the decisions that matter (the hard, kinematically-confusable risky-vs-safe contrasts at the operating threshold), sEMG's contribution shrinks to ~+0.02 AUC, and **on the device's core construct (lumbar-dominant flexion vs hip-hinge) it adds only +0.009 AUC**. sEMG genuinely helps a *different* risk axis (fatigue +0.046, compensation/asymmetry) and *hurts* specificity on effortful-but-safe movements (symmetric pickup −0.26 accuracy, sit-to-stand −0.20). The defensible claim is conditional, not blanket.

The two answered questions are mechanistically linked: the single most important feature in every model is a **personalised baseline z-score** (`imu_z_flex`, importance 0.157), which is why a model trained on the individual beats one trained on the population, and why the personalisation argument carries from the IMU-only fallback into the hybrid system.

*Channel naming corrected 2026-06-07: the second EMG pair is reported as obliques `LOBL/ROBL`, not multifidus; the relabel left all feature values and model results byte-identical.*

---

## 1. Dataset and provenance

P14 is the project's chosen full-hybrid subject (other full-hybrid participants, including P13, yielded fewer/poorer sessions). Fourteen ~25-minute sessions of the movement protocol were recorded across two days: Day 1 (sessions 01–02) on the breadboard acquisition stack with stitched recordings; from session 03 on the soldered PCB with reinforced taping.

| Stream | Channels | Rate | Notes |
|---|---|---|---|
| IMU | Pelvis, L3, T12, T4 (quaternions + derived PLTU angles) | 100 Hz | `angvel_L3_sagittal`, L3 accel provided |
| sEMG | LES, RES, LOBL, ROBL | 200 Hz | OpenBCI (200 Hz ceiling); channels 3–4 are **obliques**, not multifidus |
| Labels | 13 protocol movement classes + fatigue fraction | per-segment | sessions 13–14 carry the extended 20-rep fatigue block |

Features were produced with the **unmodified project pipeline** (`run_pipeline.py --mode full_hybrid --label_source protocol`): the same 2 s / 50 %-overlap windowing, IMU kinematic features, L3-accelerometer tilt features, personalised z-scores and time-domain EMG features used in the frozen Phase II.A package. The **frozen 13-session matrix carries 19,342 labelled windows (12,184 safe / 7,158 risky; 37.0 % risky)**, by far the largest single-subject dataset in the project, which is what makes a personalised model viable.

### 1.1 sEMG channel-naming correction
The Phase II.A feature code named the second bilateral pair `LMF/RMF` (multifidus); P14's electrodes were on the **obliques** (`emg_LOBL_mv`/`emg_ROBL_mv`, surface, ~4 cm lateral at L5). Every `*_MF` feature was physically an oblique feature. The pipeline, feature module, training feature list and consumers were corrected to `LOBL/ROBL` (pair label `OBL`) on 2026-06-07; legacy `LMF/RMF` keys remain as backward-compatible aliases. Do not attribute oblique activity to deep multifidus anywhere in the thesis.

---

## 2. Data quality audit, and a correction to the hardware narrative

The expectation was: Day-1 breadboard sessions (S01–S02) are messy; the PCB makes Day-2 clean. Tested against objective per-session metrics, this does not hold.

![Per-session signal quality](plots/fig6_data_quality.png)

| Session | Day | IMU freeze % | max \|ω\| (°/s) | EMG baseline RMS (mV) | EMG SNR | LDLJ NaN % |
|---|---|---|---|---|---|---|
| S01 | 1 | 1.14 | 69 | 0.19 | 5.56 | 0.4 |
| S02 | 1 | 1.87 | 348 | 0.22 | 4.51 | 0.0 |
| S03 | 2 | 1.28 | 153 | 0.47 | 1.91 | 0.0 |
| S04 | 2 | 0.09 | 167 | 0.40 | 3.35 | 0.0 |
| **S05** | 2 | 0.00 | 300 | **3.55** | **0.29** | 0.1 |
| S06 | 2 | 0.26 | 463 | 0.50 | 3.41 | 0.0 |
| S07 | 2 | 0.38 | **1086** | 0.22 | 4.14 | 0.0 |
| S08 | 2 | 0.16 | **1297** | 0.28 | 3.97 | 0.0 |
| S09 | 2 | 1.62 | 47 | 0.31 | 3.49 | **49.9** |
| S10 | 2 | 0.57 | 171 | 0.22 | 6.24 | 0.0 |
| S11 | 2 | 0.00 | 243 | 1.10 | **0.97** | 0.0 |
| S12 | 2 | 1.29 | 66 | 0.31 | 2.16 | **79.9** |
| S13 | 2 | 0.06 | **907** | 0.48 | 4.55 | 0.0 |
| S14 | 2 | 0.26 | 196 | 0.21 | 5.33 | 0.0 |

**The clean breadboard→PCB story is not supported.** EMG quality is, if anything, *better* on Day 1 (S01/S02 baseline noise 0.19–0.22 mV, SNR 4.5–5.6); the poor-EMG sessions are all PCB-era (S05 SNR 0.29, S11 0.97, S03/S12 <2.2). IMU packet-freeze is modestly elevated on Day 1 (1.1–1.9 %) but equally bad on PCB sessions S03/S09/S12 (1.3–1.6 %). Two PCB-era failure modes are unrelated to the breadboard: angular-velocity glitch spikes (S07/S08/S13, up to 1297 °/s) and a smoothness-feature collapse (S09 50 %, S12 80 % LDLJ NaN). The framing for the thesis: *the PCB improved wiring robustness; it did not fix electrode contact or IMU sample corruption, which dominate session-to-session quality.*

### 2.1 Frozen-set exclusion: session_05
**session_05 is excluded from the frozen dataset** (hatched in fig6). It fails the acquisition contract in `validate_phase2_dataset.py` (its IMU stream is **53.2 s shorter than its EMG stream**, 1489.0 s vs 1542.2 s, exceeding the 2 s sensor-agreement tolerance) and it is independently the worst-quality session (EMG SNR 0.29, weakest IMU-only AUC 0.516). It is archived, not deleted (`data/real/_excluded/…`), following the P10 precedent. The frozen 13-session dataset passes the validator **13/13, 0 FAIL** (`VALIDATION_session_level.txt`); see `EXCLUSION_NOTE_session_05.md`.

---

## 3. Q1: IMU-only vs IMU + sEMG (personalised P14)

### 3.1 Design
Leave-one-session-out (LOSO-session) CV on P14's 13 frozen sessions: train on 12, test on the held-out session, rotate. This is the correct generalisation test for a personalised device (does a model calibrated on the person's past sessions work on a *new* session/day) and it spans the breadboard→PCB hardware change. **IMU-only** = 19 `IMU_FEATURES`; **IMU+sEMG** = those plus 18 EMG time-domain features (37 total). Classifier: `RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight='balanced', seed=42)` with leakage-safe per-fold median imputation. Both feature sets evaluated on identical windows (exactly paired).

### 3.2 Result: sEMG helps in aggregate

![Per-session AUC, IMU vs hybrid](plots/fig1_q1_per_session_auc.png)

| Metric | IMU-only | IMU + sEMG | Δ |
|---|---|---|---|
| Mean per-session AUC | 0.799 ± 0.098 | **0.844 ± 0.096** | +0.045 |
| Pooled AUC [95 % CI] | 0.794 [0.788, 0.800] | **0.835 [0.829, 0.841]** | +0.041 |
| Pooled AUPRC | 0.728 | **0.775** | +0.047 |
| Sessions hybrid wins | — | **12 / 13** | — |
| Paired Wilcoxon (AUC) | — | — | **p = 0.0005** |

The pooled 95 % CIs do not overlap and the per-session paired test is significant. sEMG's largest gains are on the weak-IMU sessions: S08 (0.672→0.828, +0.156), S12 (+0.079), S14 (+0.063), S01 (+0.061), S02 (+0.054); the only loss is S07 (−0.012). ![Pooled ROC](plots/fig2_pooled_roc.png)

*(Note: unlike the 14-session exploratory run, sEMG does **not** reduce between-session variance on the frozen set (IMU std 0.098 vs hybrid 0.096) because the excluded session_05 was the IMU low-outlier that previously inflated that effect.)*

### 3.3 Robustness
The frozen set already excludes the single worst session (S05). The gain is not carried by any one remaining session: 12 of 13 held-out sessions favour the hybrid, and the only negative (S07, −0.012) is within noise. §3C shows the pre-exclusion 14-session run gave a near-identical headline (+0.055).

### 3.4 Where does sEMG help (per-movement)

![Per-movement predicted risk](plots/fig3_per_movement.png)

sEMG sharpens the fatigue signal and suppresses false alarms on quiescent movements; it leaves LUMBAR_DOMINANT (the protocol's nominal "obvious risky case") essentially where the IMU left it. The decomposition in §3B makes this precise.

### 3.5 Why it works: feature importance

![Feature importance](plots/fig4_feature_importance.png)

sEMG accounts for **29.2 %** of total model importance, substantial rather than token. The most informative EMG features are amplitude- and personalisation-based: `emg_z_rms_r` (personalised RMS z-score), `emg_mav_ROBL` / `emg_mav_LES` / `emg_rms_ROBL` (oblique and erector-spinae amplitude). The single most important feature overall is `imu_z_flex` (0.157), a personalised kinematic z-score.

> **Caveat on magnitude.** This is a single participant. Q1 shows sEMG *can* add real within-subject value in a well-calibrated personalised model; it does not establish a population-level EMG benefit. n = 1 is the binding limitation.

---

## 3B. A decomposition: does sEMG help *where it counts*?

The §3 headline is an aggregate over all windows; aggregate AUC rewards ranking an obviously-still window below an obviously-bending one, which is not a decision anyone needs help with. The brief requires sEMG to "justify itself quantitatively." Pushed toward the **hard, clinically meaningful decisions at the operating threshold, on this clean frozen set, the case for sEMG weakens sharply and, on the core construct, collapses.**

### 3B.1 Most of the gain is easy-window inflation

![AUC gain decomposition](plots/fig7_auc_decomposition.png)

| Evaluation subset | n | IMU | IMU+sEMG | Δ |
|---|---|---|---|---|
| All windows | 19,342 | 0.794 | 0.835 | **+0.041** |
| Exclude BASELINE_STATIC | 16,315 | 0.755 | 0.786 | +0.032 |
| Hard subset (5 risky vs confusable-safe†) | 11,553 | 0.699 | 0.720 | **+0.021** |

†*confusable-safe = clean flexion, symmetric pickup, sit-to-stand.* Half the headline gain evaporates once easy static/clean windows are removed; on the hard subset both models are mediocre (≈0.70–0.72) and sEMG closes the gap by only 0.021.

### 3B.2 sEMG is silent on the core lumbar-flexion construct

![Confusable-pair discrimination](plots/fig8_confusable_pairs.png)

| Clinical decision | Risk dimension | Δ AUC (hybrid − IMU) |
|---|---|---|
| Fatigued vs clean flexion | physiological | **+0.046** (0.930→0.976) |
| Fast vs clean flexion | speed | +0.035 |
| Shoulder-driven vs clean flexion | compensation | +0.027 |
| Asymmetric vs symmetric pickup | load asymmetry | +0.019 |
| **Lumbar-dominant vs clean flexion** | **segmental (core)** | **+0.009** (0.783→0.792) |

This is the decisive result. sEMG helps the **physiological and compensatory** axes, fatigue (IMU-invisible) most of all. But **LUMBAR_DOMINANT vs CLEAN_FLEXION**, distinguishing excessive lumbar flexion from the same gross motion executed as a hip-hinge, is the one discrimination a *lumbar* monitor exists to make, and sEMG adds **+0.009**. The lumbar-vs-hip question is segmental-kinematic; surface-EMG amplitude does not encode it.

### 3B.3 At the operating point, the hard-case benefit is ~4 points

| Subset | sens | spec IMU | spec hybrid | Δspec |
|---|---|---|---|---|
| All windows | 0.85 | 0.524 | 0.649 | +0.125 |
| Exclude BASELINE | 0.85 | 0.424 | 0.538 | +0.115 |
| **Hard subset** | 0.85 | 0.335 | 0.377 | **+0.042** |
| Hard subset | 0.90 | 0.268 | 0.304 | +0.036 |

The "+0.13 specificity" is the model learning to keep rejecting easy static/clean windows. On the hard decisions at usable sensitivity, sEMG buys ~4 points of specificity.

### 3B.4 Concentration in weak-IMU sessions (directional only)

![Gain vs IMU strength](plots/fig9_gain_vs_imu_strength.png)

Weak-IMU sessions still gain about twice as much as strong ones (mean Δ **+0.071** for the 4 sessions with IMU AUC < 0.80 vs **+0.034** for the 9 with AUC ≥ 0.80), so sEMG behaves partly as a crutch for poor IMU sessions. **But on the frozen set the rank correlation is not significant** (Spearman ρ = −0.25, p = 0.41, n = 13): removing session_05 (the strongest case of this effect) collapses the correlation. State this as a directional tendency, not an established relationship. (On the 14-session exploratory set it was ρ = −0.78, which should not be quoted for the frozen dataset.)

### 3B.5 sEMG inflicts a false-alarm cost on effortful-but-safe movements

![Reclassification by movement](plots/fig10_reclassification.png)

At the operating threshold sEMG improves recall on risky movements (SHOULDER_DRIVEN +0.168, FAST_BEND +0.145, PICKUP_ASYM +0.141, FATIGUE_FLEXION +0.134 accuracy) but **collapses specificity on the two safe movements that involve real muscular effort**: symmetric pickup 0.391→**0.132** (−0.258) and sit-to-stand 0.293→**0.093** (−0.200). The mechanism is straightforward: **surface-EMG amplitude encodes muscular *effort*, and a safe symmetric heavy lift is high-effort.** sEMG therefore pushes effortful-but-safe movements toward "risky", producing false alarms exactly where a wearable's credibility is won or lost. Overall accuracy barely moves (0.724→0.740) because these false alarms offset the risky-movement gains.

### 3B.6 Statistical honesty
Per-window bootstrap CIs are narrow because 19,342 overlapping, autocorrelated windows from one participant are not independent; the appropriate unit is the **session (n = 13)**. No per-window p-value is population evidence from a single subject.

### 3B.7 Verdict on the hypothesis

**The instinct that "sEMG does not help where it counts" is substantially correct.** Precisely:

> For this study's main construct, segmental lumbar-dominant flexion vs a safe hip-hinge, sEMG adds essentially nothing (+0.009 AUC). The aggregate gain is inflated by easy static windows and tends to concentrate in low-quality sessions; at the operating point on hard decisions it is ~4 points of specificity; and sEMG actively degrades specificity on effortful-but-safe movements because it measures effort, not lumbar risk.

Where sEMG *does* earn its place is a **different risk axis**: **fatigue** (IMU-invisible, +0.046, best-separated contrast at 0.98 AUC) and **gross compensation/asymmetry**. The defensible thesis statement:

> *sEMG meaningfully improves detection of fatigue- and compensation-driven risk, but does not improve the kinematic lumbar-flexion discrimination that defines the device, and it introduces a false-positive liability on high-effort safe movements. Given that electrode contact is the system's dominant, least-controllable failure mode (§2), the marginal segmental-risk benefit of sEMG does not justify its hardware and reliability cost for an IMU-centric lumbar monitor unless fatigue monitoring is an explicit design goal.*

---

## 3C. Robustness to the session_05 exclusion (14-session vs 13-session)

To confirm the exclusion did not manufacture the result, the entire pipeline was also run on the pre-exclusion 14-session set (session_05 flagged but retained). The headline is near-identical and every conclusion holds; the "where it counts" skepticism is marginally *stronger* on the frozen set.

| Metric | 14-session (pre-exclusion) | **13-session (frozen, this report)** |
|---|---|---|
| Q1 mean ΔAUC | +0.055 | **+0.045** |
| Q1 pooled IMU / hybrid | 0.781 / 0.831 | **0.794 / 0.835** |
| Q1 wins / p | 13/14, 0.0004 | **12/13, 0.0005** |
| HARD-subset ΔAUC | +0.033 | **+0.021** |
| LUMBAR ΔAUC | +0.014 | **+0.009** |
| FATIGUE ΔAUC | +0.046 | **+0.046** |
| Q2B personalised − population | +0.121 | **+0.138** |

Excluding session_05 (a weak-IMU session where sEMG helped most) raises the IMU baseline and shrinks the sEMG advantage on hard decisions, leaving the fatigue benefit intact, so the exclusion is conservative for the IMU-vs-hybrid conclusion. Side-by-side data: `data/q_13session_vs_14session.csv`.

---

## 4. Q2: personalised vs population in the IMU-fallback regime

### 4.1 Q2A: the Phase II.A cohort (the motivation)
For each Phase II.A participant, *personalised* = within-participant 80/20 temporal CV; *population* = leave-one-subject-out (model never sees that person). From the frozen `evaluation_corrected` artefacts:

| Feature set | Personalised | Population | Δ | Wins | Wilcoxon |
|---|---|---|---|---|---|
| **Reduced Pelvis-L3 (n=7)** | 0.787 | 0.624 | **+0.162** | 7/7 | **p = 0.016** |
| Primary 4-IMU (n=5) | 0.697 | 0.654 | +0.042 | 3/5 | p = 0.81 (n.s.) |

![Personalised vs population](plots/fig5_personalised_vs_population.png)

For the deployed reduced feature set the personalised advantage is large and unanimous (+0.162, every participant). The primary-set null is an artefact of degenerate within-CV folds (P04's primary within-CV fold has only 7 safe windows); the reduced set is the trustworthy, deployed estimator. (The reproducible CSVs hold 7 participants, P01–P07; the master reference quotes n=9 adding P08/P09 from a separate evaluation, conclusion unchanged.)

### 4.2 Q2B: fresh confirmation on P14 (13-session frozen)
The strongest test is out-of-distribution: the **deployed population model** (reduced-feature RF trained on the Phase II.A cohort, no P14 data, earlier hardware era) applied to P14's IMU-fallback features, versus a **P14-personalised** reduced model evaluated by LOSO-session.

| Model | Pooled AUC | AUPRC |
|---|---|---|
| Population reduced RF, applied to P14 | 0.583 | 0.459 |
| **P14 personalised reduced (LOSO-session)** | **0.725** | — |

| Comparison | Δ | Wins | Wilcoxon |
|---|---|---|---|
| Personalised − population (per session) | **+0.138** | 11/13 | **p = 0.0012** |

The population model degrades to **AUC 0.58, barely above chance** on an unseen individual on different hardware, while the personalised model reaches 0.73. This independently reproduces the Phase II.A finding on a fresh subject: **a model calibrated on the individual is worth ~+0.12–0.16 AUC over a population model in the IMU-fallback regime.**

> Nuance: on P14 the *reduced* 2-IMU set (personalised 0.725) is weaker than the *full* 19-feature IMU set (0.794, §3.2). The Phase II.A "reduced > primary" deployment finding does **not** replicate on P14's hybrid-era hardware/placement; flag for re-evaluation if more hybrid participants are collected. Personalisation, the Q2 question, holds strongly either way.

---

## 5. Synthesis: the two findings are the same finding

The most discriminative features across every model are **personalised baseline z-scores** (`imu_z_flex` 0.157; `imu_z_vel`, `emg_z_rms_r` also top-tier). Q2 says personalisation is worth +0.12–0.16 AUC *because* the best features are person-relative; Q1's most useful EMG feature (`emg_z_rms_r`) is itself a personalised z-score. The project's two-stage logic is vindicated: Phase II.A showed personal calibration matters; the hybrid stage shows (a) the personalisation advantage reproduces on a fresh subject and hardware, and (b) once committed to a personalised model with enough per-subject data, sEMG becomes a worthwhile addition **for fatigue and compensation specifically**, not for the core lumbar-flexion discrimination.

---

## 6. Limitations
1. **n = 1 for the hybrid claim** (Q1): internally rigorous (13 sessions, 19.3 k windows, LOSO, p = 0.0005, robust to the §2.1 exclusion) but single-subject, with selection bias (P14 chosen as best case). Disclose both.
2. **EMG electrode = obliques, historically labelled multifidus**: fixed in code 2026-06-07 (§1.1); never claim multifidus.
3. **Quality is electrode- and sample-corruption-limited, not hardware-generation-limited** (§2).
4. **Reduced-set deployment recommendation does not transfer to P14** (§4.2).
5. **LDLJ smoothness feature failed on S09/S12**; the pipeline emits NaN silently — should raise an acquisition-time flag.
6. **Protocol labels** inherit the Phase II.A signal-vs-protocol gap; the LUMBAR_DOMINANT difficulty partly reflects label noise.

## 7. Concrete next steps
1. **Decide whether fatigue monitoring is an explicit device goal**, since that single decision determines whether sEMG earns its place (it is the one axis where the data clearly says yes).
2. **Re-run the reduced-vs-full feature-set comparison in the hybrid configuration** (P14 supports it) before fixing the deployed feature set.
3. **Add an acquisition-time stream-health monitor**: IMU packet-freeze > 1 %, |ω| > 800 °/s spikes, EMG SNR < 2, LDLJ-uncomputable windows, and IMU/EMG duration mismatch (the session_05 fault).
4. **Investigate a true deep-paraspinal electrode** for the LUMBAR_DOMINANT case the obliques do not resolve.
5. **Collect ≥1 corroborating hybrid participant** to convert the Q1 case study into a replicated pilot.

---

## 8. Files
All paths relative to `results/participant_14_analysis/`. Frozen dataset + manifest + per-session metadata + validator log live under `data/real/protocol_train_full_hybrid/participant_14/`.

| File | Contents |
|---|---|
| `data/SUMMARY.json` | All headline numbers (incl. frozen 13-session block), machine-readable |
| `data/q1_13session_per_session.csv` | Q1 IMU vs hybrid per held-out session (frozen) |
| `data/q2b_13session_per_session.csv` | Q2B personalised vs population per session (frozen) |
| `data/q_13session_vs_14session.csv` | §3C robustness comparison |
| `data/q1_deep_decomposition.csv` · `q1_confusable_pairs.csv` · `q1_reclassification.csv` | §3B tables (frozen) |
| `data/session_quality.csv` | Per-session quality metrics |
| `data/13s_*` | Raw frozen folds + OOF predictions (provenance) |
| `EXCLUSION_NOTE_session_05.md` · `VALIDATION_session_level.txt` | §2.1 exclusion + validator output |
| `plots/fig1…fig10` | All figures, regenerated on the frozen 13-session set |

**Reproducibility:** features via `run_pipeline.py --mode full_hybrid --label_source protocol` (unmodified, corrected LOBL/ROBL); RF(200 CV / 500 deploy, min_samples_leaf=3, balanced, seed 42) with per-fold median imputation; population model retrained from `reduced_pelvis_l3_features.csv` (no P14). Frozen matrix SHA-256 `23fbab89c228423f73293964f4e30dac71d94ddcab4ae4a9f1624e9803a5b44b`.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                