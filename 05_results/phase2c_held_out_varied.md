# Phase II.C — Held-out generalisation

**Question:** do the frozen deployment models generalise to participants and sessions they were never trained on?

The frozen models were applied **read-only** (no fitting) to held-out data, with the full-hybrid route using resting-baseline-normalised sEMG (`phase2c_emgnorm`). Ground truth = `risk_class_protocol`. Full table: `phase2c_summary.md` / `phase2c_summary.csv` in this folder; provenance in `phase2c_run_provenance.json`.

## What was genuinely held-out (real generalisation evidence)

| Held-out participant | Route | AUC | Note |
|---|---|---|---|
| **P14 (conforming)** | reduced Pelvis-L3 | **0.693** | best route; > full-hybrid 0.655 (over-flag 0.95) > IMU-only 0.630 |
| **P12 (normal)** | reduced Pelvis-L3 | **0.583** | held-out, genuine |
| P12 (normal) | IMU-only fallback | 0.594 | held-out, genuine |

On an unseen participant's conforming sessions the reduced two-IMU model is the strongest route — consistent with the Phase II.A deployment recommendation, and notably the reduced set beats the full-hybrid route here too.

## What is NOT generalisation evidence (read the caveats)

- **P03 varied — reduced route is an in-sample leak.** P03 contributed to the reduced training set, so its reduced AUC (0.869) is *not* held-out. Its genuinely held-out IMU-only route scores 0.520 (≈ chance) on this degenerate varied session.
- **Synthetic "varied" (P14_synth) is verification only.** It exercises the inference path on out-of-distribution-style input; synthetic data cannot stand in for real human variability. Not generalisation evidence.
- **P11 excluded** — IMU acquisition dropout at t ≈ 1186 s (data loss, not a model failure).

## Honest takeaway

Phase II.C provides **real held-out evidence that the frozen models generalise to unseen participants on conforming movements** (P12, P14), with the reduced Pelvis-L3 route strongest — reinforcing the deployment recommendation. What it does **not** establish is generalisation to truly *varied, unstructured* movements: the one real varied session (P03) is degenerate and leaks on the reduced route, and the synthetic varied set is a verification aid only. Generalisation to novel movement types therefore remains a limitation. See `../LIMITATIONS_AND_KNOWN_ISSUES.md` §2.

> ⚠️ The PNGs in `figures/` predate the QC re-freeze and have not been regenerated at the `_corrected_qc` tier — use the table above / `phase2c_summary.md` for canonical numbers.

_Models applied read-only, scikit-learn 1.8.0, RISKY_THRESHOLD = 0.65. Source: `results/phase2c_emgnorm/`._
