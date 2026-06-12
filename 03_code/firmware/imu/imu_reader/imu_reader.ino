/*
 * imu_reader.ino
 * ==============
 * Reads 4× MPU-6050 IMUs via a TCA9548A I2C multiplexer and streams 
 * raw accelerometer + gyroscope counts over USB serial as CSV.
 *
 * Hardware wiring
 * ---------------
 * ESP32-S3 ─── TCA9548A ─── IMU per channel
 *
 * TCA9548A connections:
 *   SDA  → ESP32 SDA pin (GPIO 3 on Adafruit Feather ESP32-S3)
 *   SCL  → ESP32 SCL pin (GPIO 4 on Adafruit Feather ESP32-S3)
 *   VCC  → 3 V
 *   GND  → GND
 *   A0   → GND  }
 *   A1   → GND  } I2C address = 0x70 (all address pins low)
 *   A2   → GND  }
 *
 * MPU-6050 connections (same wiring for all 4, each on a separate mux channel):
 *   VCC  → 3 V  (on mux channel side)
 *   GND  → GND
 *   SDA  → mux channel SDA (SD0 / SD1 / SD2 / SD3)
 *   SCL  → mux channel SCL (SC0 / SC1 / SC2 / SC3)
 *   AD0  → GND   (I2C address = 0x68 on all 4 — safe because mux gates them)
 *   INT  → not connected (polling mode)
 *
 * TCA9548A channel → spinal level mapping (PLTU model):
 *   Channel 6  ->  Pelvis IMU   (reference, worn at iliac crest)
 *   Channel 3  ->  L3 IMU       (lumbar)
 *   Channel 4  ->  T12 IMU      (thoracolumbar junction)
 *   Channel 5  ->  T4 IMU       (upper thoracic)
 *
 * Note: channels 0, 1, and 2 were unreliable on the TCA9548A unit used
 * during bring-up, so this sketch maps the four sensors to 6/3/4/5.
 *
 * MPU-6050 register configuration
 * --------------------------------
 *   ACCEL_CONFIG = 0x00  →  ±2 g range     → 16384 LSB/g
 *   GYRO_CONFIG  = 0x08  ->  +/-500 dps range -> 65.5 LSB/dps
 *   SMPLRT_DIV   = 0x09  →  Sample rate divider = 9 → 100 Hz
 *   (base ODR = 1 kHz, divider = SMPLRT_DIV + 1 = 10 → 100 Hz)
 *   DLPF_CFG = 0x02      →  Bandwidth 94 Hz (accel), 98 Hz (gyro)
 *
 * Serial output format
 * --------------------
 * CSV, one row per sample at 100 Hz:
 *   t_ms, Pelvis_ax, Pelvis_ay, Pelvis_az, Pelvis_gx, Pelvis_gy, Pelvis_gz,
 *         L3_ax, L3_ay, L3_az, L3_gx, L3_gy, L3_gz,
 *         T12_ax, T12_ay, T12_az, T12_gx, T12_gy, T12_gz,
 *         T4_ax, T4_ay, T4_az, T4_gx, T4_gy, T4_gz
 *
 * All values are raw 16-bit signed integers (ADC counts).
 * Use signal_processing/imu/convert.py to convert to physical units.
 *
 * Serial settings
 * ---------------
 *   Baud rate : 115200
 *   Line ending: \r\n
 *
 * Recording
 * ---------
 * Open Arduino Serial Monitor at 115200 baud, or use the Python helper:
 *   python scripts/acquisition/record_imu_serial.py --port COM3 --duration 60 --out data/real/raw/session_001/imu_arduino.csv
 *
 * Dependencies
 * ------------
 *   Wire.h    — built in to Arduino IDE
 *   No external libraries required.
 *
 * Notes
 * -----
 * - The sketch uses blocking I2C reads (no DMA). At 100 kHz I2C with 4 sensors
 *   and 14 bytes per sensor, the I2C overhead is ~2 ms per loop iteration.
 *   Combined with serial print time, actual rate will be 95–100 Hz.
 *   Use t_ms timestamps rather than assuming fixed 10 ms intervals.
 *
 * - The loop uses delayMicroseconds() for pacing, not a hardware timer.
 *   For ±1 ms timing jitter on a loaded ESP32, this is acceptable at 100 Hz.
 *   If sub-ms sync with the Ganglion is needed, trigger both devices from
 *   the same GPIO pulse (future work).
 *
 * - The MPU-6050 FIFO is not used. Data is read synchronously each loop.
 *   Missed reads (if loop takes >10 ms) are flagged as gaps in t_ms.
 */

#include <Wire.h>

// ── I2C configuration ────────────────────────────────────────────────────────

#define SDA_PIN         3       // ESP32 GPIO for SDA (Adafruit Feather ESP32-S3: SDA = GPIO 3)
#define SCL_PIN         4       // ESP32 GPIO for SCL (Adafruit Feather ESP32-S3: SCL = GPIO 4)
#define I2C_FREQ_HZ     100000  // 100 kHz — more reliable on breadboard with long wires

// TCA9548A multiplexer address (A2=A1=A0=GND → 0x70)
#define TCA_ADDR        0x70

// MPU-6050 address (AD0=GND → 0x68, same for all 4)
#define MPU_ADDR        0x68

// ── TCA9548A channel assignments (must match physical wiring) ─────────────────

#define CH_PELVIS       6   // CH0+CH1+CH2 unreliable on this TCA9548A unit
#define CH_L3           3
#define CH_T12          4
#define CH_T4           5

const uint8_t IMU_CHANNELS[4] = { CH_PELVIS, CH_L3, CH_T12, CH_T4 };
const char*   IMU_LABELS[4]   = { "Pelvis", "L3", "T12", "T4" };

// ── MPU-6050 register addresses ───────────────────────────────────────────────

#define MPU_REG_PWR_MGMT_1   0x6B
#define MPU_REG_SMPLRT_DIV   0x19
#define MPU_REG_CONFIG       0x1A    // DLPF config
#define MPU_REG_GYRO_CONFIG  0x1B
#define MPU_REG_ACCEL_CONFIG 0x1C
#define MPU_REG_ACCEL_XOUT_H 0x3B   // first data register (14 bytes total)

// ── Sampling rate ────────────────────────────────────────────────────────────

#define TARGET_HZ       100
#define LOOP_US         (1000000 / TARGET_HZ)   // 10000 µs per loop

// ── Raw data buffer ──────────────────────────────────────────────────────────

struct ImuSample {
    int16_t ax, ay, az;
    int16_t gx, gy, gz;
};

ImuSample samples[4];   // one per IMU
bool       imu_ok[4];   // true if last read succeeded

uint32_t loop_start_us;


// ─────────────────────────────────────────────────────────────────────────────
// TCA9548A mux control
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Select a single TCA9548A channel (0–7). All other channels are disabled.
 * Call before any I2C transaction with the sensor on that channel.
 */
void tca_select(uint8_t channel) {
    if (channel > 7) return;
    Wire.beginTransmission(TCA_ADDR);
    Wire.write(1 << channel);
    Wire.endTransmission();
}

/**
 * Disable all TCA9548A channels (idle state between reads).
 */
void tca_deselect_all() {
    Wire.beginTransmission(TCA_ADDR);
    Wire.write(0x00);
    Wire.endTransmission();
}


// ─────────────────────────────────────────────────────────────────────────────
// MPU-6050 initialisation and read
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Wake the MPU-6050 and configure it for 100 Hz, ±2g, ±500 dps.
 * Call once per IMU during setup, with the correct mux channel selected.
 *
 * Returns true if the device acknowledged on I2C.
 */
bool mpu_init() {
    // Wake from sleep (PWR_MGMT_1: clear SLEEP bit, use internal oscillator)
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(MPU_REG_PWR_MGMT_1);
    Wire.write(0x00);   // SLEEP=0, CYCLE=0, CLKSEL=0 (internal 8 MHz)
    if (Wire.endTransmission() != 0) return false;

    // DLPF: bandwidth 94 Hz accel / 98 Hz gyro (DLPF_CFG = 0x02)
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(MPU_REG_CONFIG);
    Wire.write(0x02);
    Wire.endTransmission();

    // Sample rate divider: 1 kHz / (9+1) = 100 Hz
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(MPU_REG_SMPLRT_DIV);
    Wire.write(0x09);
    Wire.endTransmission();

    // Gyro: ±500 dps (GYRO_FS_SEL = 0b01 → register value 0x08)
    // Increased from ±250 dps to prevent saturation during fast trunk movements
    // Sensitivity: 65.5 LSB/dps
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(MPU_REG_GYRO_CONFIG);
    Wire.write(0x08);
    Wire.endTransmission();

    // Accel: ±2 g (ACCEL_FS_SEL = 0b00)
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(MPU_REG_ACCEL_CONFIG);
    Wire.write(0x00);
    Wire.endTransmission();

    return true;
}


/**
 * Read 14 bytes of accelerometer + temperature + gyroscope data from MPU-6050.
 * Stores the result in *s.
 *
 * Registers 0x3B–0x48 (14 bytes):
 *   ACCEL_XOUT_H/L, ACCEL_YOUT_H/L, ACCEL_ZOUT_H/L  (bytes 0–5)
 *   TEMP_OUT_H/L                                       (bytes 6–7, discarded)
 *   GYRO_XOUT_H/L, GYRO_YOUT_H/L, GYRO_ZOUT_H/L      (bytes 8–13)
 *
 * Returns true if all bytes were received.
 */
bool mpu_read(ImuSample* s) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(MPU_REG_ACCEL_XOUT_H);
    if (Wire.endTransmission(false) != 0) return false;   // false = repeated start

    uint8_t n = Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14);
    if (n < 14) return false;

    s->ax = (int16_t)((Wire.read() << 8) | Wire.read());
    s->ay = (int16_t)((Wire.read() << 8) | Wire.read());
    s->az = (int16_t)((Wire.read() << 8) | Wire.read());
    Wire.read(); Wire.read();   // temperature — discard
    s->gx = (int16_t)((Wire.read() << 8) | Wire.read());
    s->gy = (int16_t)((Wire.read() << 8) | Wire.read());
    s->gz = (int16_t)((Wire.read() << 8) | Wire.read());

    return true;
}


// ─────────────────────────────────────────────────────────────────────────────
// Arduino setup
// ─────────────────────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    delay(500);   // brief pause for USB serial to enumerate

    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(I2C_FREQ_HZ);

    delay(100);   // allow sensors to power up

    Serial.println("# Spinal Movement Risk Monitor — IMU reader v1.0");

    // ── Initialise each IMU via the mux ─────────────────────────────────────
    Serial.println("# Initialising sensors...");

    for (int i = 0; i < 4; i++) {
        tca_select(IMU_CHANNELS[i]);
        delay(20);
        bool ok = mpu_init();
        imu_ok[i] = ok;

        Serial.print("# ");
        Serial.print(IMU_LABELS[i]);
        Serial.print(" (TCA ch ");
        Serial.print(IMU_CHANNELS[i]);
        Serial.print(", I2C 0x68): ");
        Serial.println(ok ? "OK" : "FAILED — check wiring");
    }
    tca_deselect_all();

    // Count how many sensors responded
    int n_ok = 0;
    for (int i = 0; i < 4; i++) if (imu_ok[i]) n_ok++;

    if (n_ok == 0) {
        Serial.println("# FATAL: No IMUs found. Check TCA9548A wiring and I2C pullups.");
        Serial.println("# Halting.");
        while (true) { delay(1000); }
    }
    if (n_ok < 4) {
        Serial.print("# WARNING: Only ");
        Serial.print(n_ok);
        Serial.println(" / 4 IMUs found. Continuing with available sensors.");
        Serial.println("# Missing sensors will output 0,0,0,0,0,0 each row.");
    }

    // ── CSV header ──────────────────────────────────────────────────────────
    Serial.println("# Recording starts now. Press reset to stop.");
    Serial.println(
        "t_ms,"
        "Pelvis_ax,Pelvis_ay,Pelvis_az,Pelvis_gx,Pelvis_gy,Pelvis_gz,"
        "L3_ax,L3_ay,L3_az,L3_gx,L3_gy,L3_gz,"
        "T12_ax,T12_ay,T12_az,T12_gx,T12_gy,T12_gz,"
        "T4_ax,T4_ay,T4_az,T4_gx,T4_gy,T4_gz"
    );

    loop_start_us = micros();
}


// ─────────────────────────────────────────────────────────────────────────────
// Arduino loop — runs at ~100 Hz
// ─────────────────────────────────────────────────────────────────────────────

static uint32_t sample_count = 0;

// CSV header string — re-sent every 1000 samples so Python can sync
// even if it connects mid-stream
static const char CSV_HEADER[] =
    "t_ms,"
    "Pelvis_ax,Pelvis_ay,Pelvis_az,Pelvis_gx,Pelvis_gy,Pelvis_gz,"
    "L3_ax,L3_ay,L3_az,L3_gx,L3_gy,L3_gz,"
    "T12_ax,T12_ay,T12_az,T12_gx,T12_gy,T12_gz,"
    "T4_ax,T4_ay,T4_az,T4_gx,T4_gy,T4_gz";

void loop() {
    uint32_t t_start = micros();

    // ── Re-send CSV header every 1000 samples so Python can sync mid-stream ──
    if (sample_count % 1000 == 0) {
        Serial.println(CSV_HEADER);
    }
    sample_count++;

    // ── Read all 4 IMUs ──────────────────────────────────────────────────────
    for (int i = 0; i < 4; i++) {
        tca_select(IMU_CHANNELS[i]);
        bool ok = mpu_read(&samples[i]);

        if (!ok) {
            // Read failed — attempt one re-init then retry
            // Handles marginal breadboard connections that drop occasionally
            delay(2);
            ok = mpu_init();
            if (ok) ok = mpu_read(&samples[i]);
        }

        if (!ok) {
            samples[i] = {0, 0, 0, 0, 0, 0};
        }
        imu_ok[i] = ok;
    }
    tca_deselect_all();

    // ── Timestamp (ms since boot) ────────────────────────────────────────────
    uint32_t t_ms = millis();

    // ── Serial output — one CSV row ──────────────────────────────────────────
    // Format: t_ms,ax,ay,az,gx,gy,gz (repeated for each IMU)
    Serial.print(t_ms);
    for (int i = 0; i < 4; i++) {
        Serial.print(','); Serial.print(samples[i].ax);
        Serial.print(','); Serial.print(samples[i].ay);
        Serial.print(','); Serial.print(samples[i].az);
        Serial.print(','); Serial.print(samples[i].gx);
        Serial.print(','); Serial.print(samples[i].gy);
        Serial.print(','); Serial.print(samples[i].gz);
    }
    Serial.print("\r\n");

    // ── Rate pacing: wait until LOOP_US has elapsed ──────────────────────────
    uint32_t elapsed = micros() - t_start;
    if (elapsed < LOOP_US) {
        delayMicroseconds(LOOP_US - elapsed);
    }
    // If elapsed > LOOP_US, the loop overran — no sleep, just continue.
    // Gaps will be visible in t_ms timestamps; the converter handles them.
}


/*
 * ── Troubleshooting notes ─────────────────────────────────────────────────────
 *
 * "FAILED — check wiring" during setup:
 *   1. Verify 3V on TCA9548A VCC and all IMU VCC pins
 *   2. Check SDA/SCL are connected to GPIO 3/4 on ESP32-S3
 *   3. Verify I2C pullup resistors (4.7 kΩ to 3 V on SDA and SCL)
 *      — GY-521 breakouts typically have onboard pullups, but long wires
 *      or a belt configuration may need additional ones
 *   4. Confirm AD0 pins on all MPU-6050 breakouts are pulled LOW (GND)
 *   5. Check TCA9548A A0/A1/A2 all grounded → address should be 0x70
 *
 * Serial output looks garbled:
 *   → Set baud rate to 115200 in serial monitor
 *
 * t_ms gaps in output:
 *   → Normal if I2C is slow (belt length, noise). The signal processing
 *     pipeline interpolates timestamps; gaps < 50 ms are handled.
 *   → Persistent gaps > 100 ms: check for I2C clock stretching or
 *     reduce DLPF bandwidth (DLPF_CFG = 0x04 → 21 Hz bandwidth,
 *     slower but more reliable on noisy buses).
 *
 * All IMUs read identical values:
 *   → Mux channel select not working. Verify TCA9548A I2C address (0x70).
 *     Run an I2C scanner sketch to check what addresses respond.
 *
 * Only 1 Hz output instead of 100 Hz:
 *   → Serial.print() is blocking the loop. Increase baud rate to 230400
 *     or 460800 if the USB serial driver supports it.
 */
