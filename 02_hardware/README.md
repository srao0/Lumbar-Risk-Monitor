# Hardware

The physical sensing system. Four IMUs sit along the spine (PLTU model), multiplexed to an ESP32-S3, with an optional OpenBCI sEMG channel.

## Contents

- `sensor_placement.png`: the PLTU IMU placement on the body (pelvis, L3, T12–L1, T4–T6).
- `wiring/`: the IMU/multiplexer wiring (Fritzing) for the cabled breadboard build that the cohort data was collected on.
- `pcb/`: the rev-2 PCB:
  - `schematic_full_export.png`: full schematic.
  - `pcb_layout_*.png`: board layout (top view, all layers, bottom copper).
  - `pcb_3d_render_*.png`: 3D renders (top, left, right angles).
  - `_esp.png`, `_u2_layout.png`, `_usbc.png`: sub-block layouts (MCU, mux, USB-C).
  - `PCB_Design_Guide_SpinalIMU.docx`: design rationale and component choices.
  - `REV2_change_list.md`: what changed from rev 1 to rev 2.

## Important caveat

The **nine-participant cohort data was collected on the cabled breadboard build**, not the wireless PCB. The PCB is the design's next-revision target. See `../LIMITATIONS_AND_KNOWN_ISSUES.md` §7.
