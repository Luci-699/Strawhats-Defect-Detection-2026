/*
 * ESP32 / Arduino / STM32 Hardware Reject System Sketch
 * Board: ESP32 Dev Board / Arduino / STM32 (Nucleo/BluePill)
 * 
 * Pinout:
 * - Red LED:   GPIO 14 (Arduino D2)
 * - Buzzer:    GPIO 26 (Arduino D4)
 * - Green LED: GPIO 27 (Arduino D3)
 * - Servo:     GPIO 13 (Arduino D9)
 * 
 * Communication: Serial @ 115200 baud (or 9600 baud)
 * Protocol: "REJECT", "PASS", "RESET", "STATUS:<MAT>,<COUNT>"
 */

#include <ESP32Servo.h>

const int RED_LED_PIN   = 14;
const int BUZZER_PIN    = 26;
const int GREEN_LED_PIN = 27;
const int SERVO_PIN     = 13;

Servo rejectServo;
String inputBuffer = "";

void resetState() {
  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(GREEN_LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);
  rejectServo.write(0); // Servo idle at 0 degrees
}

void setup() {
  Serial.begin(115200);
  
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(GREEN_LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  
  // ESP32 Servo setup
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);
  rejectServo.setPeriodHertz(50);
  rejectServo.attach(SERVO_PIN, 1000, 2000); // Standard SG90 pulse width
  
  resetState();
  Serial.println("STATUS:WAITING_FOR_PC");
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
  if (cmd.startsWith("STATUS:")) return; // Ignore status info line
  
  if (cmd == "REJECT") {
    digitalWrite(GREEN_LED_PIN, LOW);
    digitalWrite(RED_LED_PIN, HIGH);
    digitalWrite(BUZZER_PIN, HIGH);
    rejectServo.write(90); // Sweep reject arm to 90 degrees
    delay(500);            // 500ms buzzer pulse
    digitalWrite(BUZZER_PIN, LOW);
    delay(1500);           // Hold reject arm & Red LED for 1.5s
    resetState();          // Auto-reset back to WAITING FOR PC
    Serial.println("STATUS:WAITING_FOR_PC");
  } 
  else if (cmd == "PASS") {
    digitalWrite(RED_LED_PIN, LOW);
    digitalWrite(GREEN_LED_PIN, HIGH);
    rejectServo.write(0);  // Ensure arm is at 0 degrees
    delay(1500);           // Hold Green LED for 1.5s
    resetState();          // Auto-reset back to WAITING FOR PC
    Serial.println("STATUS:WAITING_FOR_PC");
  }
  else if (cmd == "RESET") {
    resetState();
    Serial.println("STATUS:WAITING_FOR_PC");
  }
}
