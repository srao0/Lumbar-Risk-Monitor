# Spinal Movement Risk Monitor — Project Handover

**Imperial College London — BEng/MEng Final Year Project**
A wearable system that detects risky lumbar movements in real time using inertial sensors (IMUs) along the spine, with an optional surface-EMG (sEMG) channel.

This repository is a guided handover of the project for someone who was **not** part of the day-to-day work. It is organised to be read top-to-bottom: each numbered folder is a step in understanding what was built, why, and how well it works. You do **not** need to run anything to follow it — but you can (see §"Running it").

---

## The one-paragraph version

Four IMUs sit along the spine — **P**elvis, **L**3 (lumbar), **T**12–L1 (thoracolumbar) and **T**4–T6 (upper thoracic): the *PLTU* model. From 2-second windows the pipeline extracts kinematic features (trunk-flexion angle, angular velocity, time past a 45° risk threshold, movement smoothness) and a Random Forest classifies each window as **safe** or **risky**. An optional sEMG channel adds muscle activation, asymmetry and fatigue. The headline finding: on nine participants, a **reduced two-IMU set (pelvis + L3) matches the full four-IMU set** — equal accuracy from half the sensors — so it is the recommended design to build next.

## Headline numbers (frozen, n=9)

| Configuration | Sensors | Features | Within-participant AUC | LOSO AUC |
|---|---|---|---|---|
| **Reduced (Pelvis + L3)** — recommended | 2 IMUs | 13 | **0.819 ± 0.087** | **0.641 ± 0.062** |
| Primary (full 4-IMU) | 4 IMUs | 17 | 0.755 ± 0.077 | 0.570 ± 0.090 |

Paired Wilcoxon (reduced vs primary, LOSO): **p = 0.0625** — no significant difference, i.e. the two-IMU set is *not* worse. Full provenance for every number is in `05_results/frozen_numbers/frozen_numbers_sheet.md`.

> **Read `LIMITATIONS_AND_KNOWN_ISSUES.md` early.** It states plainly what worked, what didn't, and what was not completed (including the EMG hardware fault that put the cohort in IMU-only mode, and the Phase II.C generalisation test that was not run). The results above are honest about their scope because of it.

---

## How to read this repository

| Folder | What's in it | Start here if you want to… |
|---|---|---|
| `01_overview/` | One-page system description + architecture diagram | understand the system |
| `02_hardware/` | PCB renders, schematic, layout, wiring, sensor placement | see the physical build |
| `03_code/` | Full pipeline, ML, firmware, acquisition scripts — runnable | inspect or run the code |
| `04_data/` | Phase I synthetic sessions, a worked example, data dictionary, signal illustrations | see what the data looks like |
| `05_results/` | Results **split by study phase**, each with its own figures (report + supplementary) | check the evidence |
| `06_demo/` | Replay-dashboard screenshots and video link | see the real-time feedback |
| `07_ethics/` | Participant information pack + ethics status | check governance |
| `08_figures/` | Report-aligned result plots + system diagrams, named by what they show | match figures to the report |

Supporting files at the root: `GLOSSARY.md` (every acronym), `LIMITATIONS_AND_KNOWN_ISSUES.md`, `requirements.txt`, `LICENSE`.

## Research questions → where they're answered

| RQ | Phase | Evidence |
|---|---|---|
| Can the full pipeline be validated on controlled synthetic data? | I | `05_results/phase1_synthetic.md` |
| Can the real protocol record/convert/evaluate participant sessions (with a declared IMU-only fallback)? | II.A | `05_results/phase2a_imu_fallback_n9.md` |
| Does participant-specific calibration beat a population model? | II.B | `05_results/phase2b_personalised_vs_population.md` |
| Do frozen models generalise to held-out varied movements? | II.C | `05_results/phase2c_held_out_varied.md` — **not completed; see note** |
| Can the system present replay traffic-light feedback with explanations? | III | `05_results/phase3_replay_demo.md` |

---

## Running it (optional)

The code runs from inside `03_code/` with a pinned environment (Python 3.11, scikit-learn 1.8.0, seed 42 throughout).

```bash
cd 03_code
pip install -r requirements.txt
python scripts/phase_runners/run_phase1_synthetic.py    # Phase I, end-to-end, one command
```

Phase I regenerates the synthetic sessions in `04_data/synthetic_phase1/`, runs the pipeline, trains and evaluates the classifiers, and writes the plots — a self-contained check on a signal with known ground truth. The real-data phases run on recorded sessions and are described in `03_code/REPRODUCE.md`. **Read the SAFE-RUN warning there first** — some correction scripts overwrite their inputs in place.

## What is deliberately *not* here

Raw on-body participant recordings (privacy), the 13 GB of real session data, heavy intermediate model artefacts, and the final report PDF are excluded by design. This repo is the evidence and the means to reproduce it, not an archive. Excluded paths are listed in `.gitignore`.
