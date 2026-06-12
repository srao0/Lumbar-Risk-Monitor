#include <Wire.h>

#define SDA_PIN  3
#define SCL_PIN  4
#define TCA_ADDR 0x70

void tca_select(uint8_t ch) {
    Wire.beginTransmission(TCA_ADDR);
    Wire.write(1 << ch);
    Wire.endTransmission();
}

void tca_deselect() {
    Wire.beginTransmission(TCA_ADDR);
    Wire.write(0x00);
    Wire.endTransmission();
}

void setup() {
    Serial.begin(115200);
    while (!Serial) { delay(10); }
    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(400000);
    Serial.println("TCA9548A channel scanner");
}

void loop() {
    for (uint8_t ch = 0; ch < 4; ch++) {
        tca_select(ch);
        delay(10);

        Serial.print("CH"); Serial.print(ch); Serial.print(": ");
        int found = 0;
        for (uint8_t addr = 1; addr < 127; addr++) {
            Wire.beginTransmission(addr);
            if (Wire.endTransmission() == 0) {
                Serial.print("0x"); Serial.print(addr, HEX); Serial.print(" ");
                found++;
            }
        }
        if (found == 0) Serial.print("nothing");
        Serial.println();
        tca_deselect();
    }
    Serial.println("---");
    delay(3000);
}