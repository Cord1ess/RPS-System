/*
 * RPS v3 - ESP32 reference receiver (Arduino-ESP32 core + ESP32Servo library)
 *
 * Protocol (ASCII over UDP, one message per datagram, port 4210):
 *   PC  -> ESP   P,<seq>,<pose>,<pc_ms>     pose: R, P, S, or N (READY)
 *   PC  -> ESP   L,<seq>,<0|1>              LED off/on (camera-latency test)
 *   ESP -> PC    A,<seq>,<esp_ms>,<reset>   acknowledgement (reset = esp_reset_reason())
 *
 * The PC sends the pose the ROBOT must show (counter logic lives on the PC), immediately on
 * change plus a 100 ms heartbeat. Heartbeats with an unchanged pose are acknowledged but do not
 * touch the servos. If nothing arrives for FAILSAFE_MS, the hand returns to READY.
 *
 * Latency checklist (each item has cost other teams 100+ ms):
 *   - WiFi.setSleep(false): modem sleep delays incoming packets by up to ~100 ms or more.
 *   - Servos jump straight to target (no easing/ramping in firmware).
 *   - Soft-AP mode (default) avoids venue Wi-Fi that blocks client-to-client traffic:
 *     join the "RPS-HAND" network on the laptop, the ESP is 192.168.4.1.
 *   - Power servos from a separate 5-6 V supply with a large capacitor (e.g. 1000 uF) and a
 *     common ground; several servos starting at once can brown out the ESP32 (watch the
 *     reset reason in the ACKs: 9 = ESP_RST_BROWNOUT).
 *   - If ACKs never arrive, check that Windows Firewall allows Python on this network.
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include <ESP32Servo.h>
#include <esp_system.h>

// ------------------------------------------------------------------ network
#define USE_SOFT_AP 1
const char* AP_SSID  = "RPS-HAND";
const char* AP_PASS  = "rpsrobot123";          // at least 8 characters
const char* STA_SSID = "your-wifi";
const char* STA_PASS = "your-password";
IPAddress STA_IP(192, 168, 1, 50), STA_GW(192, 168, 1, 1), STA_MASK(255, 255, 255, 0);
const uint16_t UDP_PORT = 4210;

// ------------------------------------------------------------------ servos
const int NUM_SERVOS = 4;
const int SERVO_PINS[NUM_SERVOS] = {13, 12, 14, 27};
// Angle per servo for each pose, rows in the order R, P, S, N. CALIBRATE ON YOUR HAND.
// Example wiring: servo0 = thumb, servo1 = index+middle, servo2 = ring, servo3 = pinky.
// Tendons pull fingers closed; rubber bands open them.
const int POSE_ANGLES[4][NUM_SERVOS] = {
  /* R */ {160, 160, 160, 160},   // all fingers pulled closed
  /* P */ { 20,  20,  20,  20},   // all released (rubber bands extend)
  /* S */ {160,  20, 160, 160},   // index+middle released, rest pulled
  /* N */ { 90,  90,  90,  90},   // READY: pick from the measured transition-time table
};
const char POSE_CHARS[4] = {'R', 'P', 'S', 'N'};

const int LED_PIN = 2;
const uint32_t FAILSAFE_MS = 2000;
#define DEBUG_SERIAL 0            // Serial printing adds latency; enable only when debugging

Servo servos[NUM_SERVOS];
WiFiUDP udp;
int currentPose = 3;              // start in READY
uint32_t lastPacketMs = 0;
char buf[64];

int poseIndex(char c) {
  for (int i = 0; i < 4; i++) if (POSE_CHARS[i] == c) return i;
  return -1;
}

void applyPose(int idx) {
  for (int s = 0; s < NUM_SERVOS; s++) servos[s].write(POSE_ANGLES[idx][s]);
  currentPose = idx;
}

void sendAck(unsigned long seq) {
  char ack[48];
  int n = snprintf(ack, sizeof(ack), "A,%lu,%lu,%d", seq, (unsigned long)millis(), (int)esp_reset_reason());
  udp.beginPacket(udp.remoteIP(), udp.remotePort());
  udp.write((const uint8_t*)ack, n);
  udp.endPacket();
}

void setup() {
#if DEBUG_SERIAL
  Serial.begin(115200);
#endif
  pinMode(LED_PIN, OUTPUT);
  for (int s = 0; s < NUM_SERVOS; s++) {
    servos[s].setPeriodHertz(50);
    servos[s].attach(SERVO_PINS[s], 500, 2400);
  }
  applyPose(3);

#if USE_SOFT_AP
  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS);
#else
  WiFi.mode(WIFI_STA);
  WiFi.config(STA_IP, STA_GW, STA_MASK);
  WiFi.begin(STA_SSID, STA_PASS);
  while (WiFi.status() != WL_CONNECTED) delay(100);
#endif
  WiFi.setSleep(false);
  udp.begin(UDP_PORT);
#if DEBUG_SERIAL
  Serial.printf("RPS hand ready, IP %s port %u\n",
                USE_SOFT_AP ? WiFi.softAPIP().toString().c_str() : WiFi.localIP().toString().c_str(), UDP_PORT);
#endif
}

void loop() {
  int size = udp.parsePacket();
  while (size > 0) {
    int n = udp.read(buf, sizeof(buf) - 1);
    udp.flush();                    // drop the rest of an oversized packet, or parsePacket() stops returning packets
    buf[n > 0 ? n : 0] = '\0';
    char type = buf[0];
    unsigned long seq = 0;
    if (type == 'P') {
      char pose = 0;
      unsigned long pcMs = 0;
      int idx = -1;
      if (sscanf(buf, "P,%lu,%c,%lu", &seq, &pose, &pcMs) == 3 && (idx = poseIndex(pose)) >= 0) {
        if (idx != currentPose) applyPose(idx);   // servos move before the ACK
        lastPacketMs = millis();
        sendAck(seq);
#if DEBUG_SERIAL
        Serial.printf("seq %lu pose %c\n", seq, pose);
#endif
      }
    } else if (type == 'L') {
      int on = 0;
      if (sscanf(buf, "L,%lu,%d", &seq, &on) == 2 && (on == 0 || on == 1)) {
        digitalWrite(LED_PIN, on ? HIGH : LOW);
        lastPacketMs = millis();
        sendAck(seq);
      }
    }
    size = udp.parsePacket();
  }
  if (currentPose != 3 && lastPacketMs != 0 && millis() - lastPacketMs > FAILSAFE_MS) {
    applyPose(3);   // lost the PC: go to READY
  }
}
