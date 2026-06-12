#!/usr/bin/env python3
"""
run_phase2c_corrected.py
================================================================================
Run the Phase II.C verification harness with the DRIFT-CORRECTED re-freeze models
and data, without editing the frozen phase2c_verification.py. It monkeypatches the
harness globals:
  * fallback RF models  -> ml/models/fallback_final_n9_corrected
  * LR_EMG (full-hybrid) -> ml/models/p14_fullhybrid_corrected/LR_EMG_fold*.joblib
  * held-out sources     -> drift-corrected feature matrices
  * results dir          -> results/phase2c_corrected

We patch the globals rather than fork the harness so the verification logic stays
the single audited copy in phase2c_verification.py — this wrapper only swaps in the
corrected artefacts.

P11 is EXCLUDED: its session_001 has a total IMU acquisition dropout at t=1186.1s
(all four IMUs flatline to zero -> Madgwick open-loop). This is DATA LOSS, not
correctable drift; the rest-anchor correction is a verified no-op on it
(103.2 deg -> 103.4 deg, risk-zone 1.00 unchanged). Documented in its QC README.

P03 (in-sample leakage control) and the synthetic varied set keep their original
data (P03 is a control; synthetic needs no drift correction); they run against the
corrected models.

Run on the machine with sklearn. Force the FIS on the large P14 set:
    py scripts/evaluation/run_phase2c_corrected.py --fis_max_rows 25000
"""
import sys
from pathlib import Path

# parents[2] resolves to the 03_code package root (this file sits at
# 03_code/scripts/evaluation/), which is both the import anchor we put on
# sys.path and the root the harness resolves its data/model paths against.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.evaluation.phase2c_verification as p2c  # noqa: E402

p2c.FALLBACK_DIR = ROOT / "ml" / "models" / "fallback_final_n9_corrected_qc"
p2c.FALLBACK_META = p2c.FALLBACK_DIR / "fallback_model_metadata.json"
p2c.LR_EMG_GLOB = "p14_fullhybrid_corrected/LR_EMG_fold*.joblib"
p2c.RESULTS_DIR = ROOT / "results" / "phase2c_corrected_qc"

p2c.CONDITIONS = [
    dict(key="P12_real_normal", kind="real", participant="participant_12",
         note="Real held-out; drift-corrected (rest-anchor); salvaged session_003.",
         sources=["data/real/phase2c_heldout_corrected/participant_12/session_003_stitched_labeltrimmed/feature_matrix.csv"],
         routes=["imu_only_fallback", "reduced_pelvis_l3"]),
    dict(key="P14_real_conforming", kind="real", participant="participant_14",
         note="Real held-out full-hybrid; drift-corrected (13-session frozen set).",
         sources=["data/real/protocol_train_full_hybrid_restcorrected/combined_features.csv"],
         routes=["imu_only_fallback", "reduced_pelvis_l3", "full_hybrid"]),
    dict(key="P03_real_varied", kind="real", participant="participant_03",
         note="Real VARIED; in reduced-train set (in-sample leakage control); data not drift-corrected.",
         sources=["data/real/varied_test_fallback/participant_03/session_001/feature_matrix.csv"],
         routes=["imu_only_fallback", "reduced_pelvis_l3"]),
    # P11 EXCLUDED -- uncorrectable IMU acquisition dropout (t=1186.1s); see QC README.
    dict(key="P14_synth_varied", kind="synthetic", participant="participant_14_synthetic",
         note="Synthetic VARIED A/B/C (pipeline verification; synthetic needs no drift correction).",
         sources=[
             "data/synthetic/replay_dashboard_phase_iic_p14/replay_full_hybrid/session_A/feature_matrix.csv",
             "data/synthetic/replay_dashboard_phase_iic_p14/replay_full_hybrid/session_B/feature_matrix.csv",
             "data/synthetic/replay_dashboard_phase_iic_p14/replay_full_hybrid/session_C/feature_matrix.csv",
         ],
         routes=["imu_only_fallback", "reduced_pelvis_l3", "full_hybrid"]),
]

if __name__ == "__main__":
    # pass through CLI flags (e.g. --fis_max_rows 25000) to the harness
    sys.argv = ["phase2c_verification.py"] + sys.argv[1:]
    raise SystemExit(p2c.main())
