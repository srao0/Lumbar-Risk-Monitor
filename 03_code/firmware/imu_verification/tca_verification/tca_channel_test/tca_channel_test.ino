/*
 * tca_channel_test.ino
 * ====================
 * Tests all 8 channels of a TCA9548A I2C multiplexer.
 * Plugs one MPU-6050 into each channel in turn and reports
 * whether a device responds at 0x68.
 *
 * HOW TO USE
 * ----------
 * 1. Upload this sketch.
 * 2. Plug your MPU-6050 into SD0/SC0 (channel 0), open Serial Monitor.
 * 3. Press RESET on the Feather — note the result for CH0.
 * 4. Move the MPU to SD1/SC1 (channel 1), press RESET again.
 * 5. Repeat for channels 2–7.
 *
 * OR: plug the MPU into one channel and leave it — the sketch
 * continuously scans all 8 channels and shows which ones see a device.
 *
 * Expected output (MPU on CH0):
 *   CH0 [SD0/SC0]: 0x68 FOUND ✓
 *   CH1 [SD1/SC1]: nothing
 *   CH2 [SD2/SC2]: nothing
 *   ...
 *
 * Hardware
 * --------
 *   ESP32-S3 SDA → GPIO 3
 *   ESP32-S3 SCL → GPIO 4
 *   TCA9548A A0/A1/A2 → GND  (address 0x70)
 */

#include <Wire.h>

#define SDA_PIN   3
#define SCL_PIN   4
#define TCA_ADDR  0x70

// ── TCA9548A helpers ─────────────────────────────────────────────────────────

void tca_select(uint8_t ch) {
    Wire.beginTransmission(TCA_ADDR);
    Wire.write(1 << ch);
    Wire.endTransmission();
    delay(5);
}

void tca_deselect() {
    Wire.beginTransmission(TCA_ADDR);
    Wire.write(0x00);
    Wire.endTransmission();
}

// ── Scan a single I2C address on the currently selected channel ──────────────

bool device_present(uint8_t addr) {
    Wire.beginTransmission(addr);
    return (Wire.endTransmission() == 0);
}

// ── Setup ────────────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    while (!Serial) { delay(10); }

    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(400000);
    delay(100);

    Serial.println("============================================");
    Serial.println("  TCA9548A Channel Test");
    Serial.println("  Spinal Movement Risk Monitor — FYP");
    Serial.println("============================================");
    Serial.println("Proceeding to channel scan...");
    Serial.println();
}

// ── Loop — scan all 8 channels continuously ──────────────────────────────────

void loop() {
    Serial.println("---- Scanning all 8 TCA channels ----");

    int good_channels = 0;

    for (uint8_t ch = 0; ch < 8; ch++) {
        tca_select(ch);

        // Scan for any device (report all found addresses)
        bool found_any = false;
        uint8_t found_addrs[10];
        int n_found = 0;

        for (uint8_t addr = 1; addr < 127; addr++) {
            // Skip TCA's own address — it's on the main bus, not behind the mux
            if (addr == TCA_ADDR) continue;
            // Skip battery monitor (0x36) — Adafruit Feather onboard
            if (addr == 0x36) continue;

            Wire.beginTransmission(addr);
            if (Wire.endTransmission() == 0) {
                found_addrs[n_found++] = addr;
                found_any = true;
                if (n_found >= 10) break;
            }
        }

        tca_deselect();

        // Print channel result
        Serial.print("CH"); Serial.print(ch);
        Serial.print(" [SD"); Serial.print(ch);
        Serial.print("/SC"); Serial.print(ch);
        Serial.print("]: ");

        if (!found_any) {
            Serial.println("nothing");
        } else {
            for (int i = 0; i < n_found; i++) {
                Serial.print("0x"); Serial.print(found_addrs[i], HEX);
                // Flag MPU-6050 specifically
                if (found_addrs[i] == 0x68 || found_addrs[i] == 0x69) {
                    Serial.print(" MPU-6050 FOUND ✓");
                    good_channels++;
                }
                if (i < n_found - 1) Serial.print(", ");
            }
            Serial.println();
        }
    }

    Serial.println();
    Serial.print("Result: ");
    Serial.print(good_channels);
    Serial.println(" channel(s) with MPU-6050 detected");
    Serial.println("=====================================");
    Serial.println();

    delay(4000);
}
