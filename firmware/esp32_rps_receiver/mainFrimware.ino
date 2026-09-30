#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <WiFiManager.h>
#include <Adafruit_PWMServoDriver.h>
#include <Preferences.h>

Preferences preferences;
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

// PCA9685 Pulse Tick Limits for 180° Positional Servos
#define SERVOMIN  102 // ~0.5ms (0 Degrees)
#define SERVOMAX  512 // ~2.5ms (180 Degrees)

// ================================================================
// --- CHANNEL ANGLE MAPPING ---
// Index 0 = Ch 0 (Pinky + Ring): Standard Orientation
// Index 1 = Ch 1 (Index): Standard Orientation
// Index 2 = Ch 2 (Middle + Point): REVERSED ORIENTATION (30° Fold / 120° Extend)
// ================================================================
const int ANGLE_EXTEND[3]   = { 20,  50,  120 }; // Extended Angles
const int ANGLE_FOLD[3]     = { 170, 160, 30  }; // Folded Angles
const int ANGLE_OVERFOLD[3] = { 180, 180, 10  }; // Over-fold for index latching under thumb
// ================================================================

enum RPSState { 
  STATE_UNKNOWN = 0, 
  STATE_ROCK = 1, 
  STATE_PAPER = 2, 
  STATE_SCISSORS = 3, 
  STATE_MIDDLE = 4, 
  STATE_LIKE = 5 
};

struct CommandPayload {
  bool isGestureCommand;
  RPSState gestureState;
  uint8_t channel;
  int angle;
};

QueueHandle_t commandQueue;

const unsigned int localUdpPort = 4210;
WiFiUDP udp;
char packetBuffer[255];
RPSState lastGesture = STATE_UNKNOWN;

inline uint16_t angleToTicks(int angle) {
  angle = constrain(angle, 0, 180);
  return map(angle, 0, 180, SERVOMIN, SERVOMAX);
}

const char* getChannelLabel(uint8_t ch) {
  if (ch == 0) return "Pinky+Ring";
  if (ch == 1) return "Index";
  if (ch == 2) return "Middle+Point [REVERSED]";
  return "Unknown";
}

void setServoAngle(uint8_t channel, int angle) {
  if (channel < 3) {
    pwm.setPWM(channel, 0, angleToTicks(angle));
  }
}

void processServoCommand(const CommandPayload& cmd) {
  if (!cmd.isGestureCommand) {
    setServoAngle(cmd.channel, cmd.angle);
    return;
  }

  RPSState targetState = cmd.gestureState;

  if (targetState != STATE_UNKNOWN && targetState == lastGesture) {
    return;
  }

  if (targetState == STATE_ROCK) {
    setServoAngle(0, ANGLE_FOLD[0]);
    setServoAngle(1, ANGLE_FOLD[1]);
    setServoAngle(2, ANGLE_FOLD[2]);
    lastGesture = STATE_ROCK;
  } 
  else if (targetState == STATE_PAPER) {
    setServoAngle(0, ANGLE_EXTEND[0]);
    setServoAngle(1, ANGLE_EXTEND[1]);
    setServoAngle(2, ANGLE_EXTEND[2]);
    lastGesture = STATE_PAPER;
  } 
  else if (targetState == STATE_SCISSORS) {
    setServoAngle(0, ANGLE_FOLD[0]);
    setServoAngle(1, ANGLE_FOLD[1]);
    setServoAngle(2, ANGLE_EXTEND[2]);
    lastGesture = STATE_SCISSORS;
  }
  else if (targetState == STATE_MIDDLE) {
    // Phase 1: Over-fold index & middle to lock index under thumb
    setServoAngle(0, ANGLE_FOLD[0]);
    setServoAngle(2, ANGLE_OVERFOLD[2]);
    vTaskDelay(pdMS_TO_TICKS(40));
    setServoAngle(1, ANGLE_OVERFOLD[1]);

    vTaskDelay(pdMS_TO_TICKS(40)); // Non-blocking latch delay on Core 1

    // Phase 2: Extend middle finger
    setServoAngle(2, ANGLE_EXTEND[2]);
    lastGesture = STATE_MIDDLE;
  }
  else if (targetState == STATE_LIKE) {
    setServoAngle(0, ANGLE_FOLD[0]);
    setServoAngle(1, ANGLE_EXTEND[1]);
    setServoAngle(2, ANGLE_FOLD[2]);
    lastGesture = STATE_LIKE;
  }
}

// ================================================================
// CORE 1 TASK: Dedicated Hardware Control
// ================================================================
void ServoTask(void *pvParameters) {
  // Initialize I2C and PCA9685 driver safely inside Core 1 thread context
  Wire.begin(21, 22);
  Wire.setClock(400000);
  
  pwm.begin();
  pwm.setPWMFreq(50); // Safe 50 Hz refresh rate matching standard servos
  vTaskDelay(pdMS_TO_TICKS(50));

  // Default Boot state: ROCK
  setServoAngle(0, ANGLE_FOLD[0]);
  setServoAngle(1, ANGLE_FOLD[1]);
  setServoAngle(2, ANGLE_FOLD[2]);
  lastGesture = STATE_ROCK;

  CommandPayload rxCmd;

  for (;;) {
    if (xQueueReceive(commandQueue, &rxCmd, portMAX_DELAY) == pdTRUE) {
      processServoCommand(rxCmd);
    }
  }
}

// ================================================================
// CORE 0 TASK: Dedicated Network Listener
// ================================================================
void NetworkTask(void *pvParameters) {
  WiFiManager wm;
  wm.setConfigPortalTimeout(180);

  if (!wm.autoConnect("ESP32-RPS-Setup")) {
    Serial.println("Failed to connect or portal timed out. Restarting...");
    ESP.restart();
  }

  WiFi.setSleep(false); // Low-latency Wi-Fi mode
  Serial.println("\nWi-Fi Connected Successfully!");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());

  udp.begin(localUdpPort);
  Serial.printf("Listening for UDP commands on port %d\n", localUdpPort);

  for (;;) {
    int packetSize = udp.parsePacket();
    if (packetSize) {
      int len = udp.read(packetBuffer, sizeof(packetBuffer) - 1);
      if (len > 0) packetBuffer[len] = '\0';

      if (strncmp(packetBuffer, "RPS:", 4) == 0) {
        char gesture[16];
        sscanf(packetBuffer + 4, "%s", gesture);

        CommandPayload cmd;
        cmd.isGestureCommand = true;

        if (strcmp(gesture, "ROCK") == 0) cmd.gestureState = STATE_ROCK;
        else if (strcmp(gesture, "PAPER") == 0) cmd.gestureState = STATE_PAPER;
        else if (strcmp(gesture, "SCISSORS") == 0) cmd.gestureState = STATE_SCISSORS;
        else if (strcmp(gesture, "MIDDLE") == 0) cmd.gestureState = STATE_MIDDLE;
        else if (strcmp(gesture, "LIKE") == 0) cmd.gestureState = STATE_LIKE;
        else cmd.gestureState = STATE_UNKNOWN;

        if (cmd.gestureState != STATE_UNKNOWN) {
          xQueueSend(commandQueue, &cmd, 0);
        }
      } 
      else if (strncmp(packetBuffer, "ANGLE:", 6) == 0) {
        int ch, deg;
        if (sscanf(packetBuffer + 6, "%d,%d", &ch, &deg) == 2) {
          if (ch >= 0 && ch < 3) {
            CommandPayload cmd;
            cmd.isGestureCommand = false;
            cmd.channel = ch;
            cmd.angle = deg;

            xQueueSend(commandQueue, &cmd, 0);

            udp.beginPacket(udp.remoteIP(), udp.remotePort());
            udp.printf("CONFIRM_ANGLE:ch=%d(%s),deg=%d", ch, getChannelLabel(ch), deg);
            udp.endPacket();
          }
        }
      }
    }

    vTaskDelay(pdMS_TO_TICKS(1)); // Yield for Core 0 Wi-Fi background stack
  }
}

void setup() {
  Serial.begin(115200);

  commandQueue = xQueueCreate(10, sizeof(CommandPayload));

  if (commandQueue == NULL) {
    Serial.println("Error creating FreeRTOS Queue!");
    while (1);
  }

  // Core 1: Hardware execution task
  xTaskCreatePinnedToCore(
    ServoTask,      
    "ServoTask",    
    4096,           
    NULL,           
    2,              
    NULL,           
    1               
  );

  // Core 0: UDP network listener
  xTaskCreatePinnedToCore(
    NetworkTask,    
    "NetworkTask",  
    8192,           
    NULL,           
    1,              
    NULL,           
    0               
  );
}

void loop() {
  vTaskDelete(NULL); // Hand over control to FreeRTOS tasks
}