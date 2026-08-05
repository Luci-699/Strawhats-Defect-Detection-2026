/*
 * ESP32 Hardware Reject System + 0.96" OLED Display (SSD1306)
 * RVCE Hackathon 2026 — Team Strawhat-Pirates
 * 
 * Communication: Serial @ 115200 baud
 * Command format: "REJECT,STEEL,4" or "PASS,STEEL,0" (or simple "REJECT" / "PASS")
 * 
 * Pinout:
 * - Red LED:   GPIO 14
 * - Buzzer:    GPIO 26
 * - Green LED: GPIO 27
 * - Servo:     GPIO 13
 * - OLED I2C:  SDA=GPIO 21, SCL=GPIO 22
 */

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ESP32Servo.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1
#define SCREEN_ADDRESS 0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

const int RED_LED_PIN   = 14;
const int BUZZER_PIN    = 26;
const int GREEN_LED_PIN = 27;
const int SERVO_PIN     = 13;

Servo rejectServo;
String inputBuffer = "";
unsigned long buzzerOffTime = 0;
bool buzzerActive = false;

void showIdleScreen() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(5, 4);
  display.println("=== STRAWHAT PIRATES ===");
  display.drawLine(0, 15, 128, 15, SSD1306_WHITE);
  
  display.setCursor(20, 24);
  display.setTextSize(2);
  display.println("READY");
  
  display.setTextSize(1);
  display.setCursor(12, 48);
  display.println("Waiting for scan...");
  display.display();
}

void showPassScreen(String material = "STEEL", int defects = 0) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(5, 2);
  display.println("=== STRAWHAT PIRATES ===");
  display.drawLine(0, 13, 128, 13, SSD1306_WHITE);
  
  display.setCursor(5, 18);
  display.print("Material: ");
  display.println(material);
  
  display.setCursor(5, 30);
  display.print("Defects:  ");
  display.println(defects);
  
  display.setTextSize(2);
  display.setCursor(30, 44);
  display.println("PASS");
  display.display();
}

void showRejectScreen(String material = "STEEL", int defects = 1) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(5, 2);
  display.println("=== STRAWHAT PIRATES ===");
  display.drawLine(0, 13, 128, 13, SSD1306_WHITE);
  
  display.setCursor(5, 18);
  display.print("Material: ");
  display.println(material);
  
  display.setCursor(5, 30);
  display.print("Defects:  ");
  display.println(defects);
  
  display.setTextSize(2);
  display.setCursor(20, 44);
  display.println("REJECT!");
  display.display();
}

void resetState() {
  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(GREEN_LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);
  buzzerActive = false;
  rejectServo.write(0); // Servo idle position
  showIdleScreen();
}

void triggerBuzzer(int durationMs) {
  digitalWrite(BUZZER_PIN, HIGH);
  buzzerOffTime = millis() + durationMs;
  buzzerActive = true;
}

void setup() {
  Serial.begin(115200);
  
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(GREEN_LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  
  // Initialize I2C OLED
  Wire.begin(21, 22); // SDA=21, SCL=22
  if(!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println(F("SSD1306 allocation failed"));
  } else {
    showIdleScreen();
  }

  // Servo timers for ESP32
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);
  rejectServo.setPeriodHertz(50);
  rejectServo.attach(SERVO_PIN, 500, 2400);
  
  resetState();
  Serial.println("ESP32_OLED_REJECT_SYSTEM_READY");
}

void loop() {
  // Auto-silence buzzer after duration (non-blocking)
  if (buzzerActive && millis() >= buzzerOffTime) {
    digitalWrite(BUZZER_PIN, LOW);
    buzzerActive = false;
  }

  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      inputBuffer.trim();
      if (inputBuffer.length() > 0) {
        processCommand(inputBuffer);
      }
      inputBuffer = "";
    } else {
      inputBuffer += c;
    }
  }
}

void processCommand(String cmd) {
  cmd.toUpperCase();
  
  // Parse command format: "REJECT,STEEL,4" or "PASS,STEEL,0"
  String action = cmd;
  String material = "STEEL";
  int defects = 0;
  
  int firstComma = cmd.indexOf(',');
  if (firstComma != -1) {
    action = cmd.substring(0, firstComma);
    int secondComma = cmd.indexOf(',', firstComma + 1);
    if (secondComma != -1) {
      material = cmd.substring(firstComma + 1, secondComma);
      defects = cmd.substring(secondComma + 1).toInt();
    } else {
      material = cmd.substring(firstComma + 1);
    }
  }

  if (action == "REJECT") {
    digitalWrite(GREEN_LED_PIN, LOW);
    digitalWrite(RED_LED_PIN, HIGH);
    triggerBuzzer(2000); // Beep for 2 seconds (2000ms) then auto-off
    showRejectScreen(material, defects);
    rejectServo.write(90); // Sweep reject arm 90 degrees
  } 
  else if (action == "PASS") {
    digitalWrite(RED_LED_PIN, LOW);
    digitalWrite(GREEN_LED_PIN, HIGH);
    digitalWrite(BUZZER_PIN, LOW);
    buzzerActive = false;
    showPassScreen(material, defects);
    rejectServo.write(0); // Reset servo to 0
  }
  else if (action == "RESET") {
    resetState();
  }
}
