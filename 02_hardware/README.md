# Hardware

The physical sensing system. Four IMUs sit along the spine (PLTU model), multiplexed to an ESP32-S3, with an OpenBCI sEMG channel.

## Contents

- `imu/`: the IMU acquisition hardware.
  - `IMU_Breadboard_Circuit.png`: the breadboard circuit (ESP32-S3, IMUs, TCA9548A multiplexer).
  - `imu_build_layout.jpeg`, `imu_build_closeup.jpeg`: photos of the built breadboard.
- `emg/`: the sEMG acquisition hardware.
  - `Ganglion_circuit-layout.png`: the OpenBCI Ganglion wiring.
  - `Cyton_Hardware.png`, `Cyton_Hardware_2.png`: the OpenBCI Cyton board used for the P14 full-hybrid sessions.
- `pcb/`: the rev-2 PCB.
  - `schematic_full_export.png`: full schematic.
  - `pcb_layout_*.png`: board layout (top view, all layers, bottom copper).
  - `pcb_3d_render_*.png`: 3D renders (top, left, right angles).
  - `_esp.png`, `_usbc.png`: sub-block layouts (MCU, USB-C).

Sensor-placement figures are in `../08_figures/placement/`.

## Important caveat

The nine-participant cohort data was collected on the cabled breadboard build, not the wireless PCB. The PCB is the design's next-revision target. See `../LIMITATIONS_AND_KNOWN_ISSUES.md` §7.
