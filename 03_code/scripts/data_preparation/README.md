# Data preparation — per-participant corrections

These scripts apply the documented, per-participant corrections that were made to the raw recordings before the analysis datasets were frozen. They are kept here for transparency: every adjustment to a participant's data is a script you can read, not an unrecorded manual edit.

| Script | What it does |
|---|---|
| `apply_drift_correction.py` | Removes slow orientation drift by re-zeroing against a baseline window. |
| `apply_rest_anchor_correction.py` | Re-anchors a session to a rest baseline when the standing baseline was unreliable (used for P09). |
| `repair_imu_anchor_drift.py` | Repairs anchor drift on a sensor that slipped during recording. |
| `fix_participant_02.py` | Rebuilds P02 after a T12 sensor slip ~186 s in, with a post-hoc orientation correction. |
| `apply_data_quality_exclusions.py` | Drops windows failing quality checks (e.g. non-physical pelvis values from belt slip). |
| `refreeze_n9.py`, `refreeze_n9_data.py` | Rebuild the nine-participant frozen sets, validating against the previous freeze before emitting new numbers. |

Each correction is also described in plain language in the relevant `../../../05_results/participant_notes/` file. The order in which they apply, and the exact inputs they expect, are set out in `../../REPRODUCE.md` — important, because several of them overwrite their inputs in place.
