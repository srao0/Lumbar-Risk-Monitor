# Misc experiments and supporting material

Side experiments and supporting figures that are referenced by the project but sit outside the main Phase I to III evidence in `05_results/`.

## `imu_pole_verification/`

The pole-rig check on IMU angle output. The IMU was mounted on a rig set to known angles and its reported angle was read against a manual protractor. This supports the hardware-verification chapter. The protractor reference carries its own reading tolerance (see `LIMITATIONS_AND_KNOWN_ISSUES.md` §7).

- `pole_rig_photo.png`: the physical rig.
- `pole_rig_labelled.png`: the rig with the measured angles marked.
- `pole_static_angle.png`: reported vs reference angle.

## `movement_archetypes/`

- `movement_archetype_scatter.png`: a scatter of the protocol movement types in feature space, used to show inter-movement separation.

## `replay_dashboard/future_design/`

Design concepts for the replay dashboard, not part of the implemented system.

- `future_design_concept.png`: a mock-up of a future dashboard layout.
- `spine_risk_dashboard.html`: a static prototype page for the same idea.

## `representative_diagrams/`

Illustrative schematics drawn to explain a method. These are not real experiments and contain no measured data.

- `fis_fuzzification_defuzzification.png`: a worked single-window example of the Mamdani FIS (fuzzification, inference, defuzzification), built from the deployed fuzzy system for illustration.
- `window_to_label_protocol.png`: a schematic of the 2 s window, 1 s hop, and at-least-50%-overlap labelling rule, drawn with a synthetic representative trace and generic safe/risky blocks. It does not use any participant's real data or labels.
