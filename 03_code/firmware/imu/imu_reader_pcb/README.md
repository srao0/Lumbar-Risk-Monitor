# IMU reader: PCB version

PCB variant of the breadboard acquisition firmware (`../imu_reader/imu_reader.ino`).
Same 4× MPU-6050 / TCA9548A / ESP32-S3 acquisition at 100 Hz, retargeted to the
custom PCB and with BLE streaming added so a data cable is no longer required.

## What changed vs the breadboard sketch

| | Breadboard (`imu_reader.ino`) | PCB (`imu_reader_pcb.ino`) |
|---|---|---|
| SDA / SCL | GPIO3 / GPIO4 | **GPIO8 / GPIO9** |
| TCA9548A mux channels | 6, 3, 4, 5 (0–2 unreliable on that breakout) | **0, 1, 2, 3** (headers U11–U14) |
| I2C pull-ups | external 4.7 kΩ | internal ~45 kΩ (on-board, short traces) |
| Transport | USB serial CSV only | **BLE binary + USB serial CSV mirror** |

Mux address (0x70), IMU address (0x68), sample rate (100 Hz), and the
MPU-6050 ranges (±2 g, ±500 dps, DLPF 94/98 Hz) are all unchanged.

The breadboard sketch is left untouched so you can still flash either board.

## Mux channel → spinal level

The PCB routes its four IMU headers to sequential mux channels:

```
CH0 → Pelvis (U11)   CH1 → L3 (U12)   CH2 → T12 (U13)   CH3 → T4 (U14)
```

If your board is wired in a different order, edit `CH_PELVIS / CH_L3 / CH_T12 /
CH_T4` near the top of `imu_reader_pcb.ino`.

## Libraries / IDE settings

- **Wire.h**: built in.
- **NimBLE-Arduino** (by h2zero): install via Arduino Library Manager. Tested
  with 1.4.x. Lighter and more stable on ESP32-S3 than the stock `BLEDevice`.
- Board: your ESP32-S3 board (e.g. Adafruit Feather ESP32-S3). Set
  **USB CDC On Boot = Enabled** so the serial mirror works over native USB.

## Recording a session

**Wireless (BLE, no data cable):**

```bash
py -m pip install bleak          # one-time
py scripts/record_imu_ble.py --scan                       # find the board
py scripts/record_imu_ble.py --out data/real/raw/session_001/imu_arduino.csv
```

**Cabled (USB serial, unchanged):**

```bash
py scripts/acquisition/record_imu_serial.py --port COM3 --out data/real/raw/session_001/imu_arduino.csv
```

Both produce the **same** `imu_arduino.csv` (25-field PLTU header), so the rest
of the pipeline is identical:

```bash
py scripts/conversion/session_converter.py --ganglion <ganglion.csv> --imu data/real/raw/session_001/imu_arduino.csv
```

## BLE protocol (for reference)

- Nordic UART Service `6E400001-…`; notify (TX) characteristic `6E400003-…`;
  advertised name `SpineMonitor`.
- One 52-byte little-endian packet per 100 Hz sample:
  `uint32 t_ms` + 24× `int16` in CSV column order
  (Pelvis ax,ay,az,gx,gy,gz → L3 → T12 → T4). The firmware requests MTU 128 so a
  packet fits in a single notification; `record_imu_ble.py` reassembles and
  writes the standard CSV rows.
