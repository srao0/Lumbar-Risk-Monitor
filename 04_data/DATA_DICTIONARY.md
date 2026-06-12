# Data Dictionary

Every column in every CSV in this folder, with units. The Phase I synthetic sessions in `synthetic_phase1/` are the reference example; the real-session feature matrices use the same `feature_matrix.csv` schema.

## `synthetic_phase1/session_000X/`

Each session is one simulated participant performing the movement protocol. Files:

### `imu_data.csv` — raw IMU stream (100 Hz)
| Column | Unit | Meaning |
|---|---|---|
| `timestamp_ms` | ms | Sample time from session start |
| `label` | — | Protocol movement active at this sample (e.g. `BASELINE_STATIC`, `CLEAN_FLEXION`) |
| `rep` | int | Repetition index of that movement |
| `risk_class` | 0/1 | Ground-truth risk for this sample (0 = safe, 1 = risky) |
| `pelvis_qw…qz`, `l3_qw…qz`, `t12_qw…qz`, `t4_qw…qz` | quaternion | Orientation of each of the 4 PLTU segments (w, x, y, z) |
| `theta_PL_pitch/roll/yaw` | degrees | Pelvis→L3 inter-segment angle (the lumbar-flexion-bearing pair) |
| `theta_LT_pitch/roll/yaw` | degrees | L3→T12 inter-segment angle |
| `theta_TU_pitch/roll/yaw` | degrees | T12→T4 inter-segment angle (upper trunk) |
| `angvel_L3_sagittal` | deg/s | Sagittal-plane angular velocity at L3 |

### `emg_data.csv` — raw sEMG stream (200 Hz)
| Column | Unit | Meaning |
|---|---|---|
| `timestamp_ms` | ms | Sample time |
| `label`, `rep`, `risk_class` | — | As above |
| `emg_LES_mv`, `emg_RES_mv` | mV | Left / right **erector spinae** |
| `emg_LOBL_mv`, `emg_ROBL_mv` | mV | Left / right **obliques** |

### `labels.csv` — movement segment table
| Column | Unit | Meaning |
|---|---|---|
| `label` | — | Movement name |
| `rep` | int | Repetition |
| `start_ms`, `end_ms` | ms | Segment bounds |
| `risk_class` | 0/1 | Ground-truth risk for the segment |
| `fatigue_fraction` | 0–1 | Fraction of the way into a fatigue protocol (blank if N/A) |

### `feature_matrix.csv` — windowed features (one row per 2 s window)
The model's actual input. Key columns:
| Column group | Examples | Meaning |
|---|---|---|
| Window key | `window_centre_ms`, `movement_label` | Window centre time and the movement it falls in |
| IMU kinematics | `imu_trunk_angle_peak/mean`, `imu_angvel_peak/mean`, `imu_time_in_risk_zone`, `imu_time_high_velocity` | Trunk angle, velocity, time past 45° |
| IMU smoothness | `imu_ldlj`, `imu_jerk_rms/peak`, `imu_ldlj_multiaxis` | Log-dimensionless-jerk smoothness measures |
| IMU compensation | `imu_compensation_index`, `imu_lumbopelv_ratio`, `imu_pelvis_angle_*`, `imu_lat_angle_*` | Compensation / lumbopelvic-rhythm features (the 4-IMU-only ones) |
| IMU L3 accel-tilt | `imu_l3_accel_tilt_peak/mean/range` | Accelerometer-derived L3 tilt — the dominant reduced-set feature |
| EMG | `emg_rms_*`, `emg_mav_*`, `emg_zcr_*`, `emg_ai_*`, `emg_cai_*` | RMS / MAV / ZCR / asymmetry / co-activation per channel |
| Labels | `risk_class`, `risk_clinical`, `risk_layman`, `risk_class_protocol` | Risk targets from different labelling lenses (see labelling protocol) |
| Baseline z-scores | `imu_z_flex`, `imu_z_vel`, `emg_z_rms_r`, `emg_z_ar` | Deviation from the participant's own baseline |

> **Naming note (important for the pedant):** in `feature_matrix.csv` the oblique EMG features carry the suffix `_LMF/_RMF` (a legacy "multifidus" label) while the **raw** stream correctly uses `LOBL/ROBL`. They refer to the **same oblique channels**. The raw-data naming was corrected; the feature-column suffix was left for frozen-result continuity.

## `sample_session/`
A small real raw→processed worked example: `imu_data.csv` (same schema as above, minus the synthetic-only columns) and `labels.csv`. Use it to see the real-data shape without any on-body identifying content.

## `illustrations/`
Two AI-/generator-rendered figures showing what clean synthetic **IMU angle** and **sEMG** signals look like — included so the signal morphology is visible without exposing any on-body recording.

## `dataset_manifest.json`
Provenance for the Phase I set: source path, the 5 session folders, feature-file SHA-256, generation timestamp, and the exact command used.
