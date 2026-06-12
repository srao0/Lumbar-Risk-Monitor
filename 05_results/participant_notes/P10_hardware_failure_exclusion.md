# Participant 10: raw recording audit

**Status: NOT processed. Raw IMU only. Significant data-quality issues; read this before running `session_converter.py`.**

## TL;DR

This recording is not usable as collected. There are two distinct hardware failures:

1. **T12 sensor died at ~14 min** and never came back during the session (≈6 min of T12-specific outage before the rest of the board went down).
2. **The entire board stopped streaming at ~20.5 min** (protocol clock). The protocol ran for 26.8 min, so **the last 6 min of the session has no IMU data at all**.

Net effect: roughly 24 of 90 movement reps lost completely, plus a further ~30 reps with T12 missing.

## Recording

- `data/real/raw/participant_10/session_001/imu_arduino.csv` (16.5 MB, 153,953 rows including a serial-fallback comment header)
- `data/real/raw/participant_10/session_001/labels.csv` (90 protocol rows, 26.8 min span)
- **No EMG**, **no `session_metadata.json`**, **no processed `imu_data.csv`/`feature_matrix.csv`**: `session_converter.py` has not been run.
- The CSV header was missed by the serial reader: file starts with `# [FALLBACK] Header not received from board; using known PLTU header`. The fallback header is correct (25 cols, t_ms + 4 sensors × 6 channels), so this is recoverable, just worth flagging.

## Failure-mode timeline (protocol clock, minutes)

| Window | Pelvis | L3 | T12 | T4 | Notes |
|---|---|---|---|---|---|
| 0 – 14 | OK | OK | OK | OK | All four sensors streaming cleanly |
| 14 – 20.5 | OK | OK | **DEAD (all zeros)** | OK | T12 dropped out, no recovery |
| 20.5 – 26.8 | **DEAD** | **DEAD** | **DEAD** | **DEAD** | Whole board stopped; final 6.3 min lost |

- T12-specific zero-rows: **36,607** (≈ 6.1 min of T12-only outage)
- Total all-sensors-zero trailing block: **31,421 rows** starting at protocol time **20.47 min**
- Sample rate when streaming: clean 100 Hz, median dt = 10 ms, max gap 358 ms (acceptable)

## Movements lost completely (after 20.47 min IMU termination, 24 reps)

| Movement | Reps lost |
|---|---|
| FATIGUE_FLEXION | 20 / 20 (all) |
| SIT_TO_STAND_FAST | 2 / 3 |
| BASELINE_STATIC | 2 / 3 (the BL2 anchor pair) |

## Movements with T12 sensor missing (14 – 20 min window, T12-dead samples)

| Movement | T12-dead samples |
|---|---|
| BASELINE_STATIC | 13,843 (the BL2 anchor periods) |
| FATIGUE_FLEXION | 12,304 |
| PICKUP_SYM | 11,479 |
| PICKUP_ASYM | 11,430 |
| SIT_TO_STAND_NORMAL | 7,810 |
| SHOULDER_DRIVEN | 5,302 |
| SIT_TO_STAND_FAST | 3,595 |

(FATIGUE and the SIT_TO_STAND_FAST counts overlap with the full board-death block above; the T12 dropout had already taken those samples down before the board crashed.)

## Salvageable subset (first 14 min of protocol, all 4 sensors)

| Movement | Reps in salvageable window |
|---|---|
| BASELINE_STATIC | 1 (the BL1 anchor) |
| CLEAN_FLEXION | 8 |
| LUMBAR_DOMINANT | 6 |
| CLEAN_LATERAL_L | 6 |
| CLEAN_LATERAL_R | 6 |
| CLEAN_ROTATION_L | 6 |
| CLEAN_ROTATION_R | 6 |
| FAST_BEND | 6 |

About 45 of 90 reps are fully covered with 4-sensor data. The reduced 13-feature Pelvis-L3 model could be evaluated on this subset.

## What is wrong with this session for the project's needs

1. **No BL2 anchor.** The terminal `BASELINE_STATIC` pair (rep 2 and the implicit BL2 segment) lost coverage. The `linear_bl2_zero_reference` drift correction cannot be applied to this session, because there is no end-of-session static reference. P09's BL2 fix and §8.8 cannot be cited here.
2. **FATIGUE_FLEXION fully lost.** This was one of the new protocol additions and is part of the P04–P09 collection comparison. Cannot include P10 in any analysis that depends on FATIGUE.
3. **T12 dropout makes the primary 4-IMU set unusable** even on the 14–20 min portion. `imu_compensation_index` and `imu_lumbopelv_ratio` both depend on T12. They will be either NaN or biased by the zeros.
4. **No EMG was collected**, same as P01, P05–P09. No hybrid analysis possible.
5. **Session metadata file is missing**, so the `prepare_fallback_analysis_sets.py` script (which keys on `session_metadata.json` for drift-correction flags) will fail.

## Recommendations, ranked

1. **Don't fold P10 into the frozen evidence package as currently recorded.** Even with truncation, it loses BL2, FATIGUE, half of SHOULDER_DRIVEN/PICKUP/SIT_TO_STAND, and any T12-dependent features. The contribution to mean AUC would be marginal and the cleaning logic would need bespoke handling.
2. **If you want to salvage anything, restrict to the first 14 min** and run only the reduced 13-feature Pelvis-L3 model on that subset. This is roughly equivalent to a "protocol-without-fatigue-and-without-stand" cohort case. Useful only for sensitivity analysis, not for the headline numbers.
3. **Diagnose the hardware failure before P11.** Two distinct issues happened:
   - T12 specifically dropped at minute 14; that channel's strap or wiring may be marginal. Worth inspecting the TCA9548A channel-2 wiring before the next session.
   - The whole board went down at minute 20.5. This could be the laptop USB sleeping, the ESP32-S3 hanging, or power-budget on the 4-IMU + Ganglion daisy-chain. Worth checking your laptop power settings and adding a serial-reader watchdog before the next session.
4. **If a re-record is possible with the same participant**, prefer that over salvaging the partial recording. P10 has ≈45 of 90 reps usable, vs the cohort's typical 87 of 90, a sample-size penalty large enough to skew any per-participant comparison.
5. **Add a stream-health monitor to `record_imu_serial.py`** that prints a warning when any sensor has produced (0,0,0,0,0,0) for >100 consecutive samples, and a hard alert when all four sensors do. P10's failures would both have been visible in real time with such a monitor and the session could have been aborted and restarted before 24 reps were lost.

## What I'd write in a `session_metadata.json` for this session (if you choose to keep it)

```json
{
  "participant_id": "participant_10",
  "session_id": "session_001",
  "phase": "II.A",
  "operating_mode": "imu_only_fallback",
  "date": "2026-06-01",
  "imu_available": true,
  "emg_available": false,
  "post_hoc_drift_correction": { "enabled": false, "reason": "no BL2 anchor, board died at 20.5 min" },
  "data_quality_flags": {
    "t12_sensor_dropout": { "start_protocol_ms": 840000, "end_protocol_ms": 1228500, "duration_ms": 388500, "note": "T12 channel reported all-zero" },
    "full_board_termination": { "at_protocol_ms": 1228500, "lost_protocol_duration_ms": 377500, "lost_reps": ["FATIGUE_FLEXION×20","SIT_TO_STAND_FAST×2","BASELINE_STATIC×2"] }
  },
  "usable_for_primary_4imu": false,
  "usable_for_reduced_pelvis_l3": "partial, first 14 min only",
  "notes": "Recording had two hardware failures: T12 sensor dropped at ≈14 min and the board stopped streaming entirely at ≈20.5 min. BL2 anchor not captured. Do not use for final thesis claims without explicit window-level exclusions."
}
```

## Bottom line

P10 is a partially-broken recording. It should not be treated the same as P01–P09. If you have time before the deadline, prefer re-recording the participant; if you don't, restrict to the first 14 min, reduced feature set, and report it explicitly as a partial-coverage sensitivity check rather than a full participant.
