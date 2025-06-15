#define BLYNK_TEMPLATE_ID "TMPL6dXyl2CWF"
#define BLYNK_TEMPLATE_NAME "Quickstart Device"
#define BLYNK_AUTH_TOKEN "niTJWQMeIJB_m8hmD9YJuBbcVm8jj5r1"

#include <WiFi.h>
#include <BlynkSimpleEsp32.h>

// WiFi credentials
const char* ssid = "SLT_FIBRE";
const char* password = "abcd1234";

// MQ2 sensor pin
#define MQ2_PIN 36

// Blynk virtual pins
#define VPIN_THRESHOLD    V13  // Slider to set threshold

// Variables
int mq2Value = 0;
int smokeThreshold = 500;  // Default threshold
bool smokeDetected = false;
unsigned long lastReadTime = 0;
const unsigned long readInterval = 1000; // Read sensor every 1 second

BlynkTimer timer;

void setup() {
  Serial.begin(115200);
  
  // Initialize Blynk
  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, password);
  
  // Setup timer for sensor reading
  timer.setInterval(1000L, readMQ2Sensor);
  
  Serial.println("ESP32 MQ2 Smoke Sensor with Blynk initialized");
  Serial.println("Warming up MQ2 sensor... Please wait 20 seconds");
  
  // Initial threshold value to Blynk
  Blynk.virtualWrite(VPIN_THRESHOLD, smokeThreshold);
}

void loop() {
  Blynk.run();
  timer.run();
}

// Function to read MQ2 sensor
void readMQ2Sensor() {
  // Read analog value from MQ2 sensor
  mq2Value = analogRead(MQ2_PIN);
  
  // Check if smoke is detected based on threshold
  smokeDetected = (mq2Value > smokeThreshold);
  
  // Send data to Blynk
  if (smokeDetected) {
    Serial.println("⚠️  SMOKE DETECTED!");
  }
  
  // Print to Serial Monitor
  Serial.printf("MQ2 Value: %d | Threshold: %d | Status: %s\n", 
                mq2Value, smokeThreshold, smokeDetected ? "SMOKE!" : "Normal");
}

// Blynk function to handle threshold slider changes
BLYNK_WRITE(VPIN_THRESHOLD) {
  smokeThreshold = param.asInt();
  Serial.printf("Threshold updated to: %d\n", smokeThreshold);
  
  // Immediately check current reading against new threshold
  readMQ2Sensor();
}

// Blynk function to handle app connection
BLYNK_CONNECTED() {
  Serial.println("Connected to Blynk server");
  // Sync threshold value from app
  Blynk.syncVirtual(VPIN_THRESHOLD);
}

// Function to calibrate sensor (optional - call this when air is clean)
void calibrateSensor() {
  Serial.println("Calibrating MQ2 sensor...");
  int sum = 0;
  for (int i = 0; i < 10; i++) {
    sum += analogRead(MQ2_PIN);
    delay(100);
  }
  int baseline = sum / 10;
  smokeThreshold = baseline + 100; // Set threshold 100 points above baseline
  
  Blynk.virtualWrite(VPIN_THRESHOLD, smokeThreshold);
  Serial.printf("Calibration complete. New threshold: %d\n", smokeThreshold);
}