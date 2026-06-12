# Code

The complete pipeline, from raw sensor streams to a classified risk output, plus the embedded firmware and the scripts used to collect and process data. The layout mirrors how the code imports itself, so it runs as-is from this folder.

## Layout

```
signal_processing/     Feature extraction (shared by synthetic and real data)
  imu/                 Madgwick AHRS orientation, angle conversion, LDLJ smoothness
  emg/                 Band-pass / notch filtering, time-domain EMG features
  pipeline.py          Windowing, feature extraction, baseline z-scores -> feature_matrix.csv
ml/
  training/            train_classifier.py: LOSO cross-validation, RF / LR / SVM,
                       three conditions (IMU-only, EMG-only, IMU+EMG)
  evaluation/          Metrics, plots
  fuzzy/               Mamdani fuzzy inference layer that turns model output into a risk level
  explainability/      Replay-time scientific / lay explanations
scripts/                 Host-side acquisition, processing, orchestration (index: scripts/README.md)
  phase_runners/         One entry point per study phase: run_phase*.py (start here)
  acquisition/           Sensor recording + timed capture (IMU, Ganglion, session/varied timers, sync launcher)
  conversion/            Raw -> processed sessions: converter, calibration, protocol labels, synthetic data
  datasets/              Manifests, model provenance, dataset/project validation
  training/              Fallback + personalised model training
  evaluation/            Held-out and comparator evaluation
  demo/                  Replay traffic-light demo + dashboard
  data_preparation/      Per-participant correction scripts (documented; see its README)
firmware/              ESP32-S3 sketches: IMU reader, PCB variant, sensor verification
requirements.txt       Pinned dependencies (Python 3.11, scikit-learn 1.8.0)
```

## Running it

Install dependencies, then run from this directory so the package imports resolve:

```
pip install -r requirements.txt
python scripts/phase_runners/run_phase1_synthetic.py        # Phase I: validates the pipeline on synthetic data, end to end
```

The other phases operate on recorded participant sessions:

- `run_phase2_protocol.py`: full hybrid (IMU + sEMG) protocol evaluation
- `run_phase2_fallback_protocol.py`: IMU-only fallback route (the cohort's actual mode)
- `run_personalised_stage2b.py`: personalised vs population models (Phase II.B)
- `run_phase2_varied_test.py`: held-out varied-movement evaluation (Phase II.C)
- `demo_risk_monitor.py`: replay-mode traffic-light demo with explanations (Phase III)

The headline fallback evidence is produced by `prepare_fallback_analysis_sets.py` → `train_fallback_analysis_models.py` → `evaluate_fallback_analysis_sets.py`; the exact sequence is documented in `REPRODUCE.md` (this folder).

## Notes

`train_classifier.py` carries a feature-availability check: if a feature column expected by a model is absent from the input matrix, it raises immediately rather than silently training on a smaller feature set. IMU features are always required; EMG features are required only when the run is not on the IMU-only fallback route.
