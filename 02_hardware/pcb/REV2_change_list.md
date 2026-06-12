# Custom IMU Carrier Board — REV 2 Change List

**Project:** Spinal Movement Risk Monitor (FYP 2025/26)
**Board:** IMU Carrier (EasyEDA, JLCPCB)
**REV 1 status:** assembled, functional sensing path, but cannot be programmed or
charged conveniently. Never flashed (no wired programming path).
**REV 2 goal:** a board that programs, charges, and streams over its USB-C port
with no soldering to the bare module — a clean demo artifact for the 23rd.

> Scope note: all testing and data collection use the **breadboard** prototype.
> REV 2 is a hardware demonstrator, so changes are kept minimal and low-risk.
> The "skip for now" items below are deliberately deferred to avoid introducing
> new faults before the presentation.

---

## Summary of REV 1 problems this fixes

| # | REV 1 problem | Root cause | REV 2 fix |
|---|---|---|---|
| 1 | Can't flash firmware; can't read serial | USB-C data lines (D+/D−) not wired to the ESP32; native-USB pins IO19/IO20 left no-connect | Wire USB-C D+/D− to GPIO20/GPIO19 |
| 2 | No way to enter download mode / reset | No BOOT or RESET control | Add BOOT and RESET buttons + RC |
| 3 | USB-C plug won't seat | Connector set back from board edge | Move U10 flush to the edge |
| 4 | Board didn't power up from a known-good cell | Suspected reversed JST polarity | Verify/fix CN1 polarity vs the actual cell |
| 5 | Marginal I2C bus | No external pull-ups (internal ~45 kΩ only) | Add 4.7 kΩ pull-ups on SDA/SCL |
| 6 | "Is it even on?" — no indication on battery | Only charge-status LEDs exist | Add a power-on LED on the 3.3 V rail |

---

## ESSENTIAL CHANGES (do all of these)

### 1. Wire USB-C data to the ESP32 native USB  ← the important one

This single change makes USB-C do **charging + programming + serial**, so you
never solder to the module again.

On the ESP32-S3-WROOM-1 (U2), the native USB pins are:
- **GPIO20 = USB D+**
- **GPIO19 = USB D−**

(Both are currently no-connect.) On the USB-C receptacle (U10), the USB 2.0 data
pins are DP1/DP2 (D+) and DN1/DN2 (D−). For a 2.0-only sink in a reversible
USB-C connector, bridge the two D+ pins and the two D− pins so it works in either
plug orientation:

```
USB-C DP1 ─┬─ 22 Ω ──► ESP32-S3 GPIO20  (USB D+)
USB-C DP2 ─┘

USB-C DN1 ─┬─ 22 Ω ──► ESP32-S3 GPIO19  (USB D−)
USB-C DN2 ─┘
```

- The 22 Ω series resistors (0402) are standard USB practice; 0 Ω / direct also
  works if you'd rather not add parts, but keep the footprints.
- Leave the existing VBUS → TP4056 path and the two 5.1 kΩ CC resistors exactly
  as they are. Charging is unchanged; you're only adding the data pair.
- Route D+/D− as a tight pair, kept short, away from the power traces.
- No pin conflict: I2C stays on GPIO8/GPIO9; USB is on GPIO19/GPIO20.

**Optional but recommended:** add a USB ESD protection array (e.g. USBLC6-2SC6)
across D+/D− to GND near the connector. Cheap insurance for a port you'll plug
and unplug often.

### 2. BOOT and RESET buttons (+ reset RC)

Even with native USB, you want manual control of the two strapping/reset lines.

- **RESET button:** momentary SPST from **EN → GND**. EN already has its 10 kΩ
  pull-up; add **0.1 µF from EN → GND** for a clean reset edge.
- **BOOT button:** momentary SPST from **IO0 (GPIO0) → GND**. Add a **10 kΩ
  pull-up from IO0 → 3V3** (IO0 has an internal pull-up, but make it explicit).
  Optional **0.1 µF from IO0 → GND** for debounce.

Manual download-mode sequence (same as any ESP32): hold BOOT, tap RESET, release
BOOT. With native USB the IDE can usually auto-enter download without this, but
the buttons are your guaranteed fallback.

> Optional (only if you also want hands-free auto-flashing over a UART path):
> add the classic two-transistor auto-reset circuit (USB-serial DTR/RTS → EN/IO0).
> Not needed for native-USB flashing — skip it to keep the board simple.

### 3. Move the USB-C connector to the board edge

Mechanical/layout fix. Place **U10** so its mating face is flush with — or
slightly overhanging — the board outline, and make sure the board edge / any
enclosure cutout doesn't foul the plug's overmould. This is what stops the cable
from seating in REV 1.

### 4. Verify and fix the LiPo (CN1) polarity

REV 1 most likely didn't power up because the cell's JST polarity is reversed
relative to the board. JST-PH wiring is **not** standardised between vendors, and
the keyed housing does not guarantee correct polarity.

- Decide which exact LiPo cell(s) you will use for the demo.
- Measure that cell's JST contacts with a multimeter and note which pin is **+**.
- Set the **CN1** footprint so **pin 1 = VBAT(+)** mates with the cell's **+**,
  **pin 2 = GND** with the cell's **−**.
- Add silkscreen **+** and **−** next to CN1 so it's unambiguous at assembly.

(If your cells turn out to already match REV 1's CN1, no change — but confirm by
measurement, don't assume.)

### 5. External I2C pull-ups

Add **4.7 kΩ to 3V3 on SDA (GPIO8)** and **4.7 kΩ to 3V3 on SCL (GPIO9)**, near
the TCA9548A. This is exactly what your Section 2.5 flags as a known limitation;
the internal ~45 kΩ pull-ups are weak for a four-device bus.

### 6. Power-on LED

Add a simple **3V3 → 1 kΩ → LED → GND** indicator (pick a colour different from
the CHG/STDBY charge LEDs). Then "is the board powered?" is answered by looking
at it — instead of the multimeter-and-guesswork loop. Optionally drive it from a
spare GPIO instead of straight off the rail if you want firmware control.

---

## SKIP FOR NOW (deliberately deferred)

These are real improvements but add risk/complexity. Leave them for a later
revision so REV 2 stays a safe presentation board:

- **Load-sharing PMIC** (MCP73871 / BQ24074) to run while charging and run with
  no battery. Bigger power-stage redesign — not worth the risk on this deadline.
  Workaround stays: charge with the device off, run off the LiPo.
- **Buck-boost 3.3 V regulator** to hold 3.3 V across the full LiPo discharge.
  The AMS1117 is fine for a demo (ESP32-S3 runs down to 3.0 V).
- **Shared IMU/EMG hardware sync trigger** (Chapter 4 future work). Not needed
  for the demo; software-parallel starts remain adequate.

---

## BOM additions (REV 2)

All small, all JLCPCB Basic-library parts (search by value/package):

| Qty | Part | Value / type | Package | Purpose |
|---|---|---|---|---|
| 2 | Resistor | 22 Ω | 0402 | USB D+/D− series |
| 2 | Tactile switch | SMD momentary | small SMD tact | BOOT, RESET |
| 2 | Resistor | 10 kΩ | 0402 | IO0 pull-up (+ spare) |
| 2 | Capacitor | 0.1 µF | 0402 | EN and IO0 RC |
| 2 | Resistor | 4.7 kΩ | 0402 | I2C SDA/SCL pull-ups |
| 1 | LED | any colour ≠ charge LEDs | 0603 | power-on indicator |
| 1 | Resistor | 1 kΩ | 0402 | power LED limiting |
| 1 | USBLC6-2SC6 *(optional)* | USB ESD array | SOT-23-6 | D+/D− ESD protection |

(You already have 10 kΩ, 0.1 µF, 1 kΩ, and LED parts in the REV 1 BOM — reuse
those library parts where possible.)

---

## EasyEDA implementation order

1. **Schematic — USB-C data:** draw nets DP1+DP2 → 22 Ω → GPIO20, DN1+DN2 → 22 Ω
   → GPIO19. (Optional ESD array across the pair to GND.)
2. **Schematic — buttons:** BOOT (IO0–GND) + 10 kΩ to 3V3 + 0.1 µF; RESET
   (EN–GND) + 0.1 µF. EN pull-up already present.
3. **Schematic — pull-ups & LED:** 4.7 kΩ on SDA and SCL to 3V3; power LED + 1 kΩ
   on 3V3.
4. **Schematic — CN1:** set/confirm pin-1 = VBAT(+) to match your measured cell;
   add +/− silkscreen.
5. **Convert to PCB**, place new parts (buttons reachable, LED visible, USB parts
   near U10).
6. **Layout — USB-C:** move U10 flush to the board edge; route D+/D− as a short
   tight pair.
7. **DRC**, re-pour ground, re-export schematic + layout figures (these double as
   updated Figures 2.x in your report).
8. **Order:** JLCPCB, 2-layer FR-4, LeadFree HASL, SMT assembly. Re-use the REV 1
   process settings.

---

## Timeline (presentation 23 June)

- JLCPCB fab + assembly + DHL is typically ~1–2 weeks. To be safe, **finalise and
  order by ~9 June** and choose expedited build + express shipping.
- Because the breadboard handles all data collection, REV 2 is **not on the
  critical path for results** — if it slips, your testing is unaffected; you'd
  just demo the breadboard. Low-stress.

---

## Post-assembly bring-up checklist (REV 2)

1. **Visual/continuity:** check CN1 polarity against a cell with a multimeter
   *before* plugging a battery in.
2. **Power LED** lights on battery → 3.3 V rail is up.
3. **Plug USB-C into the PC:** a new USB serial / USB-JTAG device should enumerate
   (Device Manager). This confirms the data-line fix.
4. **Flash** `firmware/imu/imu_reader_pcb/imu_reader_pcb.ino` over USB-C
   (BOOT+RESET if it doesn't auto-enter download). No external adapter needed.
5. **Serial check:** open the port at 115200 — you should see the startup banner
   and CSV rows (the firmware's serial mirror).
6. **BLE check:** `py scripts/record_imu_ble.py --scan` → `SpineMonitor` appears.
7. **Charge check:** unplug data host, plug a charger into USB-C → CHG LED on,
   STDBY when complete.

---

## For the report (limitations → future work)

REV 1 demonstrated the sensing and integration concept but had three avoidable
hardware-bring-up faults: a power-only USB-C (data lines unrouted), no programming
interface or reset/boot control, and a connector set back from the board edge,
compounded by a battery-connector polarity mismatch. REV 2 resolves these by
routing USB-C to the ESP32-S3 native USB (enabling charge, program, and serial
through one port), adding BOOT/RESET control, edge-aligning the connector, and
correcting the JST polarity, plus the previously noted I2C pull-up and a power
indicator. This is a clean, well-evidenced "lessons learned" arc.
