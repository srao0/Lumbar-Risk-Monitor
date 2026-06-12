#!/usr/bin/env python3
"""
run_phase2c_emgnorm.py
================================================================================
Phase II.C verification for the sEMG RESTING-BASELINE-RATIO re-normalisation arm.
Identical to run_phase2c_corrected.py except the full-hybrid EMG branch and the
P14 source point at the EMG-normalised artefacts:
  * fallback RF (IMU-only / reduced)  -> ml/models/fallback_final_n9_corrected_qc  (UNCHANGED; IMU side untouched)
  * LR_EMG (full-hybrid branch)        -> ml/models/p14_fullhybrid_emgnorm/LR_EMG_fold*.joblib
  * P14 held-out source                -> data/real/protocol_train_full_hybrid_emgnorm/combined_features.csv
  * results dir                        -> results/phase2c_emgnorm

Only the P14 full_hybrid route changes vs phase2c_corrected_qc; P12/P03 and all
IMU-only/reduced routes are byte-identical (normalisation only touched emg_rms_*/
emg_mav_*). P11 stays EXCLUDED (uncorrectable IMU dropout).

This is a thin configuration wrapper: it overrides the module-level constants in
phase2c_verification and then calls its main(), so the evaluation logic stays in
one place.

NOTE: the synthetic P14 varied condition runs the normalised LR_EMG against
un-normalised synthetic EMG -> train/serve skew; its full_hybrid output is NOT
meaningful and is not cited (synthetic is Ch6/demo pipeline-verification only).

Run on the machine with sklearn:
    py scripts/phase_runners/run_phase2c_emgnorm.py --fis_max_rows 25000
"""
import sys
from pathlib import Path

# parents[2] from scripts/phase_runners/ resolves to 03_code, placed on sys.path
# so the evaluation package import below resolves.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.evaluation.phase2c_verification as p2c  # noqa: E402

# Override the verifier's defaults to point at the emgnorm artefacts.
p2c.FALLBACK_DIR = ROOT / "ml" / "models" / "fallback_final_n9_corrected_qc"
p2c.FALLBACK_META = p2c.FALLBACK_DIR / "fallback_model_metadata.json"
p2c.LR_EMG_GLOB = "p14_fullhybrid_emgnorm/LR_EMG_fold*.joblib"
p2c.RESULTS_DIR = ROOT / "results" / "phase2c_emgnorm"

p2c.CONDITIONS = [
    dict(key="P12_real_normal", kind="real", participant="participant_12",
         note="Real held-out; drift-corrected (rest-anchor); salvaged session_003. IMU-only (EMG unusable) -> unaffected by EMG normalisation.",
         sources=["data/real/phase2c_heldout_corrected/participant_12/session_003_stitched_labeltrimmed/feature_matrix.csv"],
         routes=["imu_only_fallback", "reduced_pelvis_l3"]),
    dict(key="P14_real_conforming", kind="real", participant="participant_14",
         note="Real held-out full-hybrid; drift-corrected + EMG resting-baseline-ratio normalised (13-session set).",
         sources=["data/real/protocol_train_full_hybrid_emgnorm/combined_features.csv"],
         routes=["imu_only_fallback", "reduced_pelvis_l3", "full_hybrid"]),
    dict(key="P03_real_varied", kind="real", participant="participant_03",
         note="Real VARIED; in reduced-train set (in-sample leakage control); IMU-only -> unaffected.",
         sources=["data/real/varied_test_fallback/participant_03/session_001/feature_matrix.csv"],
         routes=["imu_only_fallback", "reduced_pelvis_l3"]),
    # P11 EXCLUDED -- uncorrectable IMU acquisition dropout (t=1186.1s); see QC README.
    dict(key="P14_synth_varied", kind="synthetic", participant="participant_14_synthetic",
         note="Synthetic VARIED (pipeline verification only). full_hybrid output NOT cited under emgnorm: normalised LR_EMG vs un-normalised synthetic EMG = train/serve skew.",
         sources=[
             "data/synthetic/replay_dashboard_phase_iic_p14/replay_full_hybrid/session_A/feature_matrix.csv",
             "data/synthetic/replay_dashboard_phase_iic_p14/replay_full_hybrid/session_B/feature_matrix.csv",
             "data/synthetic/replay_dashboard_phase_iic_p14/replay_full_hybrid/session_C/feature_matrix.csv",
         ],
         routes=["imu_only_fallback", "reduced_pelvis_l3", "full_hybrid"]),
]

if __name__ == "__main__":
    # Hand the remaining CLI args straight to the verifier's argparse.
    sys.argv = ["phase2c_verification.py"] + sys.argv[1:]
    raise SystemExit(p2c.main())
