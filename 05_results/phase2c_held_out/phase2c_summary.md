# Phase II.C verification results

_Generated 2026-06-11T18:42:17.563702+00:00 | dry_run=False | RISKY_THRESHOLD=0.65 | sklearn=1.8.0_

Frozen deployment models applied read-only (no fitting). Ground truth is risk_class_protocol.
Synthetic rows are pipeline/Phase-III verification, NOT generalisation evidence.

          condition      kind             route                model         held_out status     n  n_risky    auc  sensitivity  flagged_recall  specificity  overflag_rate     f1  missed_risk  false_alarms  tl_Safe  tl_Cautious  tl_Risky
    P12_real_normal      real imu_only_fallback primary_4imu_cleaned         held-out     OK  1468      538 0.5942       0.1375          0.6338       0.9226         0.4806 0.2164          464            72      680          642       146
    P12_real_normal      real reduced_pelvis_l3    reduced_pelvis_l3         held-out     OK  1468      538 0.5828       0.1022          0.5539       0.9602         0.4505 0.1746          483            37      751          625        92
P14_real_conforming      real imu_only_fallback primary_4imu_cleaned         held-out     OK 19342     7158 0.6295       0.0937          0.7772       0.9904         0.6536 0.1689         6487           117     5815        12739       788
P14_real_conforming      real reduced_pelvis_l3    reduced_pelvis_l3         held-out     OK 19342     7158 0.6930       0.0997          0.8445       0.9930         0.6312 0.1795         6444            85     5606        12937       799
P14_real_conforming      real       full_hybrid primary_4imu_cleaned         held-out     OK 19342     7158 0.6547       0.1046          0.9989       0.9942         0.9529 0.1878         6409            71      582        17940       820
    P03_real_varied      real imu_only_fallback primary_4imu_cleaned         held-out     OK  1501      630 0.5204       0.0333          0.3651       0.9621         0.3456 0.0614          609            33      970          477        54
    P03_real_varied      real reduced_pelvis_l3    reduced_pelvis_l3 IN-SAMPLE (leak)     OK  1501      630 0.8693       0.4460          0.8635       0.9816         0.3984 0.6063          349            16      610          594       297
   P14_synth_varied synthetic imu_only_fallback primary_4imu_cleaned  n/a (synthetic)     OK   745      192 0.5550       0.1719          0.7812       0.8336         0.6257 0.2082          159            92      249          371       125
   P14_synth_varied synthetic reduced_pelvis_l3    reduced_pelvis_l3  n/a (synthetic)     OK   745      192 0.6186       0.0729          0.8125       0.9892         0.7631 0.1321          178             6      167          558        20
   P14_synth_varied synthetic       full_hybrid primary_4imu_cleaned  n/a (synthetic)     OK   745      192 0.6435       0.0573          0.9583       0.8987         0.8608 0.0849          181            56       85          593        67