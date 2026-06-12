# IMU firmware watchdog patch — imu_reader_pcb.ino

Closes the P11 failure mode: at t=1186.1 s the I2C bus/TCA9548A hung, every
`mpu_read` failed, the loop wrote `0,0,0,0,0,0` for all four IMUs, and kept doing
so for ~7 min. The `imu_ok[]` flags were computed but **never transmitted**, so
the host could not distinguish a dead bus from genuine zeros.

Three additive changes (review, then flash). None alter the existing column order;
the status field is appended, so older parsers still work if they ignore trailing columns.

## 1. Transmit per-sample health (so the host/ingest watchdog sees it)
Append a 4-bit status mask to each CSV row and the BLE packet — bit i = IMU i alive.

CSV header (line ~270), append one column:
```c
    "T4_ax,T4_ay,T4_az,T4_gx,T4_gy,T4_gz,"
    "imu_status";          // NEW: bitmask, bit0=Pelvis .. bit3=T4 (0xF = all OK)
```

In `loop()`, after the read block builds `imu_ok[]`, emit it (line ~390, before the `\r\n`):
```c
    uint8_t status = 0;
    for (int i = 0; i < 4; i++) status |= (imu_ok[i] ? 1 : 0) << i;
    Serial.print(','); Serial.print(status);
    Serial.print("\r\n");
```
(For BLE, extend `PACKET_LEN` by 1 and `packet[51] = status;` in `build_packet`.)

## 2. Recover the bus on sustained failure (don't stream zeros for 7 minutes)
Replace the per-sample retry block (lines ~356–369) with a bus-level recovery that
triggers when *all* sensors fail for a run of samples:
```c
    int n_fail = 0;
    for (int i = 0; i < 4; i++) {
        tca_select(IMU_CHANNELS[i]);
        bool ok = mpu_read(&samples[i]);
        if (!ok) { delay(2); ok = mpu_init() && mpu_read(&samples[i]); }
        if (!ok) { samples[i] = {0,0,0,0,0,0}; n_fail++; }
        imu_ok[i] = ok;
    }
    tca_deselect_all();

    static uint16_t dead_run = 0;
    dead_run = (n_fail == 4) ? dead_run + 1 : 0;
    if (dead_run >= 50) {              // ~0.5 s of total silence -> recover the bus
        Serial.println("# ALARM: all IMUs dead — attempting I2C bus recovery");
        Wire.end();
        delay(5);
        Wire.begin(SDA_PIN, SCL_PIN);
        Wire.setClock(I2C_FREQ_HZ);
        for (int i = 0; i < 4; i++) { tca_select(IMU_CHANNELS[i]); delay(5); mpu_init(); }
        tca_deselect_all();
        dead_run = 0;
    }
```
If the TCA9548A `RESET#` pin is brought to a spare GPIO (currently tied high via
10 kΩ), pulse it low ~1 ms inside the recovery block for a hard mux reset — more
reliable than a soft `Wire` restart against a latched-up bus.

## 3. Hardware watchdog timer (last-resort, for a truly blocking I2C call)
A latched bus can make `Wire` calls block indefinitely. Guard the loop with the
ESP32 task WDT so the MCU reboots rather than hanging silently:
```c
#include "esp_task_wdt.h"
// in setup():  esp_task_wdt_init(3, true); esp_task_wdt_add(NULL);
// at end of loop():  esp_task_wdt_reset();
```
A reboot is visible (the `# Spinal Movement Risk Monitor ...` banner reprints), so
the operator knows to restart the trial — far better than 7 min of zeros.

## Operator-visible alarm
Add an onboard-LED blink (or a repeated `# ALARM` serial line) whenever `dead_run`
or a WDT reboot fires, so a hang is noticed at the bench in real time.

---
## Host / Cyton side
- IMU: parse the new `imu_status` column; abort or warn live when it != 0xF.
- Cyton (no firmware access): run OpenBCI's **impedance/railed check before every
  recording**, and run `scripts/acquisition_watchdog.py` at ingest — it flags rail
  saturation and DC-offset drift (it caught P11 ch4: railed 39%, 42% usable).
