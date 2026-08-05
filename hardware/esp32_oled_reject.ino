/*
 * ESP32 Hardware Reject System + 0.96" OLED Display (SSD1306)
 * RVCE Hackathon 2026 — Team SafePath
 * 
 * Hardware Required:
 * 1. ESP32 Dev Board
 * 2. 0.96" I2C OLED Display (SSD1306 128x64)
 * 3. Red LED (GPIO 14) + Green LED (GPIO 27)
 * 4. Active/Passive Buzzer (GPIO 26)
 * 5. SG90 Servo Motor (GPIO 13)
 * 
 * I2C OLED Wiring:
 * - GND -> ESP32 GND
 * - VCC -> ESP32 3.3V or 5V
 * - SCL -> ESP32 GPIO 22
 * - SDA -> ESP32 GPIO 21
 * 
 * Arduino IDE Libraries Required:
 * - Adafruit SSD1306
 * - Adafruit GFX Library
 * - ESP32Servo
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

void showIdleScreen() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(10, 5);
  display.println("SAFEPATH INSPECT");
  display.drawLine(0, 16, 128, 16, SSD1306_WHITE);
  
  display.setCursor(20, 28);
  display.setTextSize(2);
  display.println("READY");
  
  display.setTextSize(1);
  display.setCursor(15, 52);
  display.println("Waiting for scan...");
  display.display();
}

void showPassScreen() {
  display.clearDisplay();
  display.fillRect(0, 0, 128, 16, SSD1306_WHITE);
  display.setTextColor(SSD1306_BLACK);
  display.setTextSize(1);
  display.setCursor(12, 4);
  display.println("INSPECTION RESULT");
  
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(2);
  display.setCursor(22, 26);
  display.println("[ PASS ]");
  
  display.setTextSize(1);
  display.setCursor(10, 50);
  display.println("Status: APPROVED");
  display.display();
}

void showRejectScreen() {
  display.clearDisplay();
  display.fillRect(0, 0, 128, 16, SSD1306_WHITE);
  display.setTextColor(SSD1306_BLACK);
  display.setTextSize(1);
  display.setCursor(12, 4);
  display.println("INSPECTION RESULT");
  
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(2);
  display.setCursor(10, 26);
  display.println("[REJECT]");
  
  display.setTextSize(1);
  display.setCursor(5, 50);
  display.println("DEFECT DETECTED!");
  display.display();
}

void resetState() {
  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(GREEN_LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);
  rejectServo.write(0); // Servo idle position
  showIdleScreen();
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
  
  if (cmd == "REJECT") {
    digitalWrite(GREEN_LED_PIN, LOW);
    digitalWrite(RED_LED_PIN, HIGH);
    digitalWrite(BUZZER_PIN, HIGH);
    showRejectScreen();
    rejectServo.write(90); // Sweep reject arm 90 degrees
    delay(500);
    digitalWrite(BUZZER_PIN, LOW); // Silence buzzer after 500ms
  } 
  else if (cmd == "PASS") {
    digitalWrite(RED_LED_PIN, LOW);
    digitalWrite(GREEN_LED_PIN, HIGH);
    showPassScreen();
    rejectServo.write(0);
    delay(1000);
    digitalWrite(GREEN_LED_PIN, LOW);
    showIdleScreen();
  }
  else if (cmd == "RESET") {
    resetState();
  }
}
