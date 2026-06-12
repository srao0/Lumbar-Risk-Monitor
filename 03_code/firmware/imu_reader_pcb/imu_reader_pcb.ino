/*
 * imu_reader_pcb.ino
 * ==================
 * Spinal Movement Risk Monitor — FYP 2025/26
 *
 * PCB variant of imu_reader.ino.
 *
 * Same acquisition logic as the breadboard sketch (4× MPU-6050 read through a
 * TCA9548A I2C multiplexer at 100 Hz), but targeting the custom two-layer PCB
 * instead of the breadboard prototype. Two things change with the board, plus
 * one addition:
 *
 *   1. I2C pins      : breadboard used GPIO3/GPIO4; the PCB routes SDA→GPIO8,
 *                      SCL→GPIO9 (ESP32-S3-WROOM-1, per the PCB schematic and
 *                      report Section 2.5).
 *   2. Mux channels  : breadboard used 6/3/4/5 because channels 0–2 were
 *                      unreliable on that particular TCA9548A breakout. The PCB
 *                      routes the four IMU headers (U11–U14) to sequential
 *                      multiplexer channels 0/1/2/3, so the mapping is clean:
 *                          CH0 → Pelvis   (header U11)
 *                          CH1 → L3       (header U12)
 *                          CH2 → T12      (header U13)
 *                          CH3 → T4       (header U14)
 *                      >>> If your board is silkscreened/wired in a different
 *                          order, edit IMU_CHANNELS below to match. <<<
 *   3. BLE streaming : the board can stream wirelessly so no USB data cable is
 *                      needed during collection. Each 100 Hz sample is sent as a
 *                      compact 52-byte binary packet over a Nordic-UART-style
 *                      notify characteristic. The same data is ALSO mirrored as
 *                      human-readable CSV over USB serial, so the existing
 *                      scripts/acquisition/record_imu_serial.py still works when cabled.
 *
 * Everything downstream is unchanged: the binary packet fields are in the exact
 * same order and units as the CSV columns, so scripts/record_imu_ble.py rebuilds
 * a byte-for-byte compatible imu_arduino.csv for session_converter.py.
 *
 * Hardware wiring (PCB)
 * ---------------------
 * ESP32-S3-WROOM-1 ─── TCA9548A (0x70) ─── 4× MPU-6050 (0x68), one per channel
 *
 *   ESP32-S3 SDA (GPIO8) ─── TCA9548A SDA
 *   ESP32-S3 SCL (GPIO9) ─── TCA9548A SCL
 *   TCA9548A A0/A1/A2 → GND   → address 0x70
 *   TCA9548A RESET#   → VCC via 10 kΩ
 *   SC0/SD0..SC3/SD3  → headers U11..U14 (VCC, GND, SDA, SCL each)
 *
 * Note on pull-ups: the PCB has no external I2C pull-ups and relies on the
 * ESP32-S3 internal pull-ups (~45 kΩ). Wire.begin() enables them by default on
 * the ESP32 Arduino core. This is weaker than the 4.7 kΩ ideal for a 4-device
 * bus but operated reliably at 100 kHz over the short on-board traces. If you
 * see intermittent IMU drop-outs, fit external 4.7 kΩ pull-ups to SDA/SCL.
 *
 * Library dependencies
 * --------------------
 *   Wire.h            — built in to the Arduino IDE
 *   NimBLE-Arduino    — install via Library Manager ("NimBLE-Arduino" by
 *                       h2zero). Lighter and more stable on ESP32-S3 than the
 *                       stock BLEDevice library. Tested with NimBLE-Arduino 1.4.x.
 *
 * Board settings (Arduino IDE)
 * ----------------------------
 *   Board     : "Adafruit Feather ESP32-S3" (or your ESP32-S3 board)
 *   USB CDC On Boot : Enabled   (so Serial works over native USB)
 *   PSRAM     : as per your module (the N8R2 has 2 MB)
 *
 * BLE service / characteristic UUIDs (Nordic UART Service)
 * --------------------------------------------------------
 *   Service : 6E400001-B5A3-F393-E0A9-E50E24DCCA9E
 *   TX (notify, board → host) : 6E400003-B5A3-F393-E0A9-E50E24DCCA9E
 *   Advertised name           : "SpineMonitor"
 *
 * Binary packet format (little-endian, 52 bytes, one per 100 Hz sample)
 * ---------------------------------------------------------------------
 *   offset 0  : uint32  t_ms                  (millis() since boot)
 *   offset 4  : int16   Pelvis_ax, ay, az, gx, gy, gz
 *   offset 16 : int16   L3_ax, ay, az, gx, gy, gz
 *   offset 28 : int16   T12_ax, ay, az, gx, gy, gz
 *   offset 40 : int16   T4_ax, ay, az, gx, gy, gz
 *   (= 4 + 24 * 2 = 52 bytes)
 *
 * MPU-6050 configuration (identical to breadboard sketch)
 * -------------------------------------------------------
 *   ACCEL ±2 g (16384 LSB/g), GYRO ±500 dps (65.5 LSB/dps),
 *   DLPF 94/98 Hz, SMPLRT_DIV = 9 → 100 Hz.
 */

#include <Wire.h>
#include <NimBLEDevice.h>

// ── I2C configuration (PCB) ───────────────────────────────────────────────────

#define SDA_PIN         8       // PCB: SDA on GPIO8 (was GPIO3 on breadboard)
#define SCL_PIN         9       // PCB: SCL on GPIO9 (was GPIO4 on breadboard)
#define I2C_FREQ_HZ     100000  // 100 kHz — reliable over the short on-board traces

#define TCA_ADDR        0x70    // A0=A1=A2=GND
#define MPU_ADDR        0x68    // AD0=GND (mux gates the 4 identical addresses)

// ── TCA9548A channel assignments (PCB: sequential headers U11–U14) ────────────
// Edit these four lines if your board routes the headers in a different order.

#define CH_PELVIS       0   // header U11
#define CH_L3           1   // header U12
#define CH_T12          2   // header U13
#define CH_T4           3   // header U14

const uint8_t IMU_CHANNELS[4] = { CH_PELVIS, CH_L3, CH_T12, CH_T4 };
const char*   IMU_LABELS[4]   = { "Pelvis", "L3", "T12", "T4" };

// ── MPU-6050 register addresses ───────────────────────────────────────────────

#define MPU_REG_PWR_MGMT_1   0x6B
#define MPU_REG_SMPLRT_DIV   0x19
#define MPU_REG_CONFIG       0x1A
#define MPU_REG_GYRO_CONFIG  0x1B
#define MPU_REG_ACCEL_CONFIG 0x1C
#define MPU_REG_ACCEL_XOUT_H 0x3B

// ── Sampling rate ─────────────────────────────────────────────────────────────

#define TARGET_HZ       100
#define LOOP_US         (1000000 / TARGET_HZ)

// ── BLE (Nordic UART Service) ─────────────────────────────────────────────────

#define BLE_DEVICE_NAME   "SpineMonitor"
#define UART_SERVICE_UUID "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define UART_TX_UUID      "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"   // notify, board → host

static NimBLECharacteristic* txChar = nullptr;
static volatile bool bleConnected = false;

// 52-byte binary packet: uint32 t_ms + 24 int16 sensor values
#define PACKET_LEN  52
static uint8_t packet[PACKET_LEN];

class ServerCallbacks : public NimBLEServerCallbacks {
    void onConnect(NimBLEServer* pServer) override {
        bleConnected = true;
        Serial.println("# BLE: central connected");
    }
    void onDisconnect(NimBLEServer* pServer) override {
        bleConnected = false;
        Serial.println("# BLE: central disconnected — re-advertising");
        NimBLEDevice::startAdvertising();
    }
};

// ── Raw data buffer ───────────────────────────────────────────────────────────

struct ImuSample {
    int16_t ax, ay, az;
    int16_t gx, gy, gz;
};

ImuSample samples[4];
bool      imu_ok[4];


// ─────────────────────────────────────────────────────────────────────────────
// TCA9548A mux control
// ─────────────────────────────────────────────────────────────────────────────

void tca_select(uint8_t channel) {
    if (channel > 7) return;
    Wire.beginTransmission(TCA_ADDR);
    Wire.write(1 << channel);
    Wire.endTransmission();
}

void tca_deselect_all() {
    Wire.beginTransmission(TCA_ADDR);
    Wire.write(0x00);
    Wire.endTransmission();
}


// ─────────────────────────────────────────────────────────────────────────────
// MPU-6050 initialisation and read
// ─────────────────────────────────────────────────────────────────────────────

bool mpu_init() {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(MPU_REG_PWR_MGMT_1);
    Wire.write(0x00);   // wake, internal oscillator
    if (Wire.endTransmission() != 0) return false;

    Wire.beginTransmission(MPU_ADDR);
    Wire.write(MPU_REG_CONFIG);
    Wire.write(0x02);   // DLPF 94/98 Hz
    Wire.endTransmission();

    Wire.beginTransmission(MPU_ADDR);
    Wire.write(MPU_REG_SMPLRT_DIV);
    Wire.write(0x09);   // 1 kHz / 10 = 100 Hz
    Wire.endTransmission();

    Wire.beginTransmission(MPU_ADDR);
    Wire.write(MPU_REG_GYRO_CONFIG);
    Wire.write(0x08);   // ±500 dps
    Wire.endTransmission();

    Wire.beginTransmission(MPU_ADDR);
    Wire.write(MPU_REG_ACCEL_CONFIG);
    Wire.write(0x00);   // ±2 g
    Wire.endTransmission();

    return true;
}

bool mpu_read(ImuSample* s) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(MPU_REG_ACCEL_XOUT_H);
    if (Wire.endTransmission(false) != 0) return false;   // repeated start

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
// Packet packing (little-endian, matches scripts/record_imu_ble.py)
// ─────────────────────────────────────────────────────────────────────────────

static inline void put_u32(uint8_t* buf, uint32_t v) {
    buf[0] = (uint8_t)(v       & 0xFF);
    buf[1] = (uint8_t)((v >> 8)  & 0xFF);
    buf[2] = (uint8_t)((v >> 16) & 0xFF);
    buf[3] = (uint8_t)((v >> 24) & 0xFF);
}

static inline void put_i16(uint8_t* buf, int16_t v) {
    buf[0] = (uint8_t)((uint16_t)v       & 0xFF);
    buf[1] = (uint8_t)(((uint16_t)v >> 8) & 0xFF);
}

void build_packet(uint32_t t_ms) {
    put_u32(&packet[0], t_ms);
    int off = 4;
    for (int i = 0; i < 4; i++) {
        put_i16(&packet[off],      samples[i].ax); off += 2;
        put_i16(&packet[off],      samples[i].ay); off += 2;
        put_i16(&packet[off],      samples[i].az); off += 2;
        put_i16(&packet[off],      samples[i].gx); off += 2;
        put_i16(&packet[off],      samples[i].gy); off += 2;
        put_i16(&packet[off],      samples[i].gz); off += 2;
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Arduino setup
// ─────────────────────────────────────────────────────────────────────────────

static const char CSV_HEADER[] =
    "t_ms,"
    "Pelvis_ax,Pelvis_ay,Pelvis_az,Pelvis_gx,Pelvis_gy,Pelvis_gz,"
    "L3_ax,L3_ay,L3_az,L3_gx,L3_gy,L3_gz,"
    "T12_ax,T12_ay,T12_az,T12_gx,T12_gy,T12_gz,"
    "T4_ax,T4_ay,T4_az,T4_gx,T4_gy,T4_gz";

void setup() {
    Serial.begin(115200);
    delay(500);

    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(I2C_FREQ_HZ);
    delay(100);

    Serial.println("# Spinal Movement Risk Monitor — IMU reader (PCB) v1.0");
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

    int n_ok = 0;
    for (int i = 0; i < 4; i++) if (imu_ok[i]) n_ok++;

    if (n_ok == 0) {
        Serial.println("# FATAL: No IMUs found. Check TCA9548A wiring and I2C pull-ups.");
        Serial.println("# Halting.");
        while (true) { delay(1000); }
    }
    if (n_ok < 4) {
        Serial.print("# WARNING: Only ");
        Serial.print(n_ok);
        Serial.println(" / 4 IMUs found. Continuing with available sensors.");
        Serial.println("# Missing sensors will output 0,0,0,0,0,0 each row.");
    }

    // ── BLE init (Nordic UART Service, notify only) ──────────────────────────
    Serial.println("# Starting BLE...");
    NimBLEDevice::init(BLE_DEVICE_NAME);
    NimBLEDevice::setMTU(128);   // allow the 52-byte packet in a single notify

    NimBLEServer* server = NimBLEDevice::createServer();
    server->setCallbacks(new ServerCallbacks());

    NimBLEService* svc = server->createService(UART_SERVICE_UUID);
    txChar = svc->createCharacteristic(UART_TX_UUID, NIMBLE_PROPERTY::NOTIFY);
    svc->start();

    NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
    adv->addServiceUUID(UART_SERVICE_UUID);
    adv->setName(BLE_DEVICE_NAME);
    adv->setScanResponse(true);
    NimBLEDevice::startAdvertising();
    Serial.print("# BLE advertising as \"");
    Serial.print(BLE_DEVICE_NAME);
    Serial.println("\" (Nordic UART Service)");

    // ── CSV header for the USB-serial mirror ─────────────────────────────────
    Serial.println("# Recording starts now. Press reset to stop.");
    Serial.println(CSV_HEADER);
}


// ─────────────────────────────────────────────────────────────────────────────
// Arduino loop — ~100 Hz
// ─────────────────────────────────────────────────────────────────────────────

static uint32_t sample_count = 0;

void loop() {
    uint32_t t_start = micros();

    // Re-send the CSV header every 1000 samples so a serial host can sync mid-stream
    if (sample_count % 1000 == 0) {
        Serial.println(CSV_HEADER);
    }
    sample_count++;

    // ── Read all 4 IMUs ──────────────────────────────────────────────────────
    for (int i = 0; i < 4; i++) {
        tca_select(IMU_CHANNELS[i]);
        bool ok = mpu_read(&samples[i]);
        if (!ok) {
            // One re-init + retry covers an occasional dropped transaction.
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

    uint32_t t_ms = millis();

    // ── BLE: send one 52-byte binary packet ──────────────────────────────────
    if (bleConnected && txChar != nullptr) {
        build_packet(t_ms);
        txChar->setValue(packet, PACKET_LEN);
        txChar->notify();
    }

    // ── USB serial mirror: same CSV row format as imu_reader.ino ──────────────
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

    // ── Rate pacing ──────────────────────────────────────────────────────────
    uint32_t elapsed = micros() - t_start;
    if (elapsed < LOOP_US) {
        delayMicroseconds(LOOP_US - elapsed);
    }
}


/*
 * ── Troubleshooting notes ─────────────────────────────────────────────────────
 *
 * "FAILED — check wiring" during setup:
 *   1. Verify 3.3 V at TCA9548A VCC and each IMU header.
 *   2. Confirm SDA/SCL are on GPIO8/GPIO9 (PCB), not GPIO3/4 (breadboard).
 *   3. TCA9548A address must be 0x70 (A0/A1/A2 to GND).
 *   4. If drop-outs persist, fit external 4.7 kΩ pull-ups to SDA/SCL — the
 *      board relies on the weaker internal ~45 kΩ pull-ups.
 *
 * All IMUs read identical values:
 *   → Mux select not working. Confirm TCA9548A responds at 0x70 and that
 *     CH_PELVIS/L3/T12/T4 match how the four headers (U11–U14) are routed.
 *
 * BLE central won't connect / no data:
 *   1. Confirm the board advertises as "SpineMonitor" (BLE scanner app).
 *   2. The host must subscribe to TX 6E400003-… for notifications.
 *   3. If packets look truncated, the negotiated MTU is too small — this
 *      sketch requests 128; ensure the host accepts ≥ 55.
 *
 * Wireless vs cabled:
 *   → Wireless: run scripts/record_imu_ble.py (no USB data cable needed; the
 *     board can be powered from its battery / USB-C charger).
 *   → Cabled  : the USB-serial CSV mirror is unchanged, so
 *     scripts/acquisition/record_imu_serial.py still works exactly as before.
 */
