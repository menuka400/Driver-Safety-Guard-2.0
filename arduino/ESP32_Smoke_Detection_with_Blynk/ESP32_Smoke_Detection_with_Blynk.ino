#define BLYNK_TEMPLATE_ID "TMPL6dXyl2CWF"
#define BLYNK_TEMPLATE_NAME "Quickstart Device"
#define BLYNK_AUTH_TOKEN "niTJWQMeIJB_m8hmD9YJuBbcVm8jj5r1"

#include <WiFi.h>
#include <BlynkSimpleEsp32.h>

// Pin definitions
const int MQ2_PIN = 36;
const int SMOKE_THRESHOLD = 400;

// WiFi credentials
const char* ssid = "SLT_FIBRE";
const char* password = "abcd1234";

// Variables
int smokeLevel = 0;
BlynkTimer timer;

void setup() {
  Serial.begin(115200);
  
  // Initialize WiFi and Blynk
  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, password);
  
  // Setup timer to read smoke sensor every 2 seconds
  timer.setInterval(2000L, readSmokeLevel);
  
  Serial.println("Smoke Detection System Started");
}

void loop() {
  Blynk.run();
  timer.run();
}

void readSmokeLevel() {
  // Read analog value from MQ-2 sensor
  smokeLevel = analogRead(MQ2_PIN);
  
  // Send smoke level to Blynk app (Virtual Pin V3)
  Blynk.virtualWrite(V3, smokeLevel);
  
  // Print detailed info to Serial Monitor
  Serial.println("=== Smoke Sensor Reading ===");
  Serial.print("Raw ADC Value: ");
  Serial.println(smokeLevel);
  Serial.print("Threshold: ");
  Serial.println(SMOKE_THRESHOLD);
  Serial.print("Status: ");
  
  // Check if smoke level exceeds threshold
  if (smokeLevel > SMOKE_THRESHOLD) {
    Serial.println("⚠️  HIGH SMOKE DETECTED!");
    Serial.println("WARNING: Smoke level above threshold!");
    // Send alert notification to Blynk app
    Blynk.logEvent("smoke_alert", "High smoke level detected!");
  } else {
    Serial.println("✅ Normal - No smoke detected");
  }
  
  // Calculate percentage of threshold
  float percentage = (float)smokeLevel / SMOKE_THRESHOLD * 100;
  Serial.print("Threshold Percentage: ");
  Serial.print(percentage, 1);
  Serial.println("%");
  Serial.println("============================");
  Serial.println();
}

// Optional: Handle commands from Blynk app
BLYNK_WRITE(V1) {
  int buttonState = param.asInt();
  if (buttonState == 1) {
    Serial.println("Manual check triggered from app");
    readSmokeLevel();
  }
}