# Phase III — Replay traffic-light feedback

**Question:** can the system close the loop — take a recorded session and present **real-time, interpretable risk feedback** a non-expert could act on?

**Verdict: demonstrated.** The replay dashboard streams a recorded session window-by-window, runs each window through the frozen model and the fuzzy-inference layer, and shows a **green / amber / red** risk level as the movement plays back — with two layers of explanation:

- a **scientific** explanation (which features drove the flag — e.g. flexion past 45° with high angular velocity), and
- a **lay** explanation (plain-language: "you bent too far, too fast").

This is what turns a classifier into a usable assistive device: the wearer sees *that* and *why* a movement was risky, as it happens.

## How it works

- `../03_code/scripts/demo/` runs the replay dashboard.
- The fuzzy layer (`../03_code/ml/fuzzy/`) maps the model probability to a traffic-light level via interpretable Mamdani rules rather than a raw threshold.
- The explanation layer (`../03_code/ml/explainability/`) produces the scientific and lay text per window.

## Media

The screen-recorded demo and dashboard screenshots are in `../06_demo/`.

> **⏳ Outstanding:** the demo video and dashboard screenshots are being produced and are the only gaps in this handover — see `../06_demo/README.md`.

## Honest framing

Phase III demonstrates the **feedback mechanism on replayed recordings**, not a deployed live-on-body trial. The live-acquisition path exists in the firmware and acquisition scripts but the validated demonstration is replay-based.
