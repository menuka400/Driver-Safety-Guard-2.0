#define BLYNK_TEMPLATE_ID "TMPL6dXyl2CWF"
#define BLYNK_TEMPLATE_NAME "Quickstart Device"
#define BLYNK_AUTH_TOKEN "niTJWQMeIJB_m8hmD9YJuBbcVm8jj5r1"

#include <WiFi.h>
#include <BlynkSimpleEsp32.h>

// Bulb 1 - Distracted
const int LED_DISTRACTED_R = 12;
const int LED_DISTRACTED_G = 14;
const int LED_DISTRACTED_B = 27;

// Bulb 2 - Drowsy
const int LED_DROWSY_R = 26;
const int LED_DROWSY_G = 25;
const int LED_DROWSY_B = 33;

// Bulb 3 - Mobile
const int LED_MOBILE_R = 32;
const int LED_MOBILE_G = 17;
const int LED_MOBILE_B = 34;

// Bulb 4 - Smoking
const int LED_SMOKING_R = 23;
const int LED_SMOKING_G = 22;
const int LED_SMOKING_B = 21;

// Bulb 5 - Idle
const int LED_IDLE_R = 19;
const int LED_IDLE_G = 18;
const int LED_IDLE_B = 5;

const char* ssid = "SLT_FIBRE";
const char* password = "abcd1234";

// PWM channels for each LED
const int DISTRACTED_R_CHANNEL = 0;
const int DISTRACTED_G_CHANNEL = 1;
const int DISTRACTED_B_CHANNEL = 2;
const int DROWSY_R_CHANNEL = 3;
const int DROWSY_G_CHANNEL = 4;
const int DROWSY_B_CHANNEL = 5;
const int MOBILE_R_CHANNEL = 6;
const int MOBILE_G_CHANNEL = 7;
const int MOBILE_B_CHANNEL = 8;
const int SMOKING_R_CHANNEL = 9;
const int SMOKING_G_CHANNEL = 10;
const int SMOKING_B_CHANNEL = 11;
const int IDLE_R_CHANNEL = 12;
const int IDLE_G_CHANNEL = 13;
const int IDLE_B_CHANNEL = 14;

// PWM settings
const int freq = 5000;
const int resolution = 8;

void setup() {
  Serial.begin(115200);
  
  // Configure PWM channels
  configurePWM();
  
  // Connect to WiFi and Blynk
  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, password);
  
  Serial.println("Setup complete. Ready for Blynk control!");
}

void loop() {
  Blynk.run();
}

void configurePWM() {
  // LED 1 - Distracted
  ledcSetup(DISTRACTED_R_CHANNEL, freq, resolution);
  ledcSetup(DISTRACTED_G_CHANNEL, freq, resolution);
  ledcSetup(DISTRACTED_B_CHANNEL, freq, resolution);
  ledcAttachPin(LED_DISTRACTED_R, DISTRACTED_R_CHANNEL);
  ledcAttachPin(LED_DISTRACTED_G, DISTRACTED_G_CHANNEL);
  ledcAttachPin(LED_DISTRACTED_B, DISTRACTED_B_CHANNEL);
  
  // LED 2 - Drowsy
  ledcSetup(DROWSY_R_CHANNEL, freq, resolution);
  ledcSetup(DROWSY_G_CHANNEL, freq, resolution);
  ledcSetup(DROWSY_B_CHANNEL, freq, resolution);
  ledcAttachPin(LED_DROWSY_R, DROWSY_R_CHANNEL);
  ledcAttachPin(LED_DROWSY_G, DROWSY_G_CHANNEL);
  ledcAttachPin(LED_DROWSY_B, DROWSY_B_CHANNEL);
  
  // LED 3 - Mobile
  ledcSetup(MOBILE_R_CHANNEL, freq, resolution);
  ledcSetup(MOBILE_G_CHANNEL, freq, resolution);
  ledcSetup(MOBILE_B_CHANNEL, freq, resolution);
  ledcAttachPin(LED_MOBILE_R, MOBILE_R_CHANNEL);
  ledcAttachPin(LED_MOBILE_G, MOBILE_G_CHANNEL);
  ledcAttachPin(LED_MOBILE_B, MOBILE_B_CHANNEL);
  
  // LED 4 - Smoking
  ledcSetup(SMOKING_R_CHANNEL, freq, resolution);
  ledcSetup(SMOKING_G_CHANNEL, freq, resolution);
  ledcSetup(SMOKING_B_CHANNEL, freq, resolution);
  ledcAttachPin(LED_SMOKING_R, SMOKING_R_CHANNEL);
  ledcAttachPin(LED_SMOKING_G, SMOKING_G_CHANNEL);
  ledcAttachPin(LED_SMOKING_B, SMOKING_B_CHANNEL);
  
  // LED 5 - Idle
  ledcSetup(IDLE_R_CHANNEL, freq, resolution);
  ledcSetup(IDLE_G_CHANNEL, freq, resolution);
  ledcSetup(IDLE_B_CHANNEL, freq, resolution);
  ledcAttachPin(LED_IDLE_R, IDLE_R_CHANNEL);
  ledcAttachPin(LED_IDLE_G, IDLE_G_CHANNEL);
  ledcAttachPin(LED_IDLE_B, IDLE_B_CHANNEL);
}

// LED 1 - Distracted (Virtual Pin V5)
BLYNK_WRITE(V5) {
  int r = param[0].asInt();
  int g = param[1].asInt();
  int b = param[2].asInt();
  
  ledcWrite(DISTRACTED_R_CHANNEL, r);
  ledcWrite(DISTRACTED_G_CHANNEL, g);
  ledcWrite(DISTRACTED_B_CHANNEL, b);
  
  Serial.printf("Distracted LED: R=%d, G=%d, B=%d\n", r, g, b);
}

// LED 2 - Drowsy (Virtual Pin V7)
BLYNK_WRITE(V7) {
  int r = param[0].asInt();
  int g = param[1].asInt();
  int b = param[2].asInt();
  
  ledcWrite(DROWSY_R_CHANNEL, r);
  ledcWrite(DROWSY_G_CHANNEL, g);
  ledcWrite(DROWSY_B_CHANNEL, b);
  
  Serial.printf("Drowsy LED: R=%d, G=%d, B=%d\n", r, g, b);
}

// LED 3 - Mobile (Virtual Pin V8)
BLYNK_WRITE(V8) {
  int r = param[0].asInt();
  int g = param[1].asInt();
  int b = param[2].asInt();
  
  ledcWrite(MOBILE_R_CHANNEL, r);
  ledcWrite(MOBILE_G_CHANNEL, g);
  ledcWrite(MOBILE_B_CHANNEL, b);
  
  Serial.printf("Mobile LED: R=%d, G=%d, B=%d\n", r, g, b);
}

// LED 4 - Smoking (Virtual Pin V9)
BLYNK_WRITE(V9) {
  int r = param[0].asInt();
  int g = param[1].asInt();
  int b = param[2].asInt();
  
  ledcWrite(SMOKING_R_CHANNEL, r);
  ledcWrite(SMOKING_G_CHANNEL, g);
  ledcWrite(SMOKING_B_CHANNEL, b);
  
  Serial.printf("Smoking LED: R=%d, G=%d, B=%d\n", r, g, b);
}

// LED 5 - Idle (Virtual Pin V10)
BLYNK_WRITE(V10) {
  int r = param[0].asInt();
  int g = param[1].asInt();
  int b = param[2].asInt();
  
  ledcWrite(IDLE_R_CHANNEL, r);
  ledcWrite(IDLE_G_CHANNEL, g);
  ledcWrite(IDLE_B_CHANNEL, b);
  
  Serial.printf("Idle LED: R=%d, G=%d, B=%d\n", r, g, b);
}

// Master control for all LEDs (Virtual Pin V11)
BLYNK_WRITE(V11) {
  int r = param[0].asInt();
  int g = param[1].asInt();
  int b = param[2].asInt();
  
  // Set all LEDs to the same color
  ledcWrite(DISTRACTED_R_CHANNEL, r);
  ledcWrite(DISTRACTED_G_CHANNEL, g);
  ledcWrite(DISTRACTED_B_CHANNEL, b);
  
  ledcWrite(DROWSY_R_CHANNEL, r);
  ledcWrite(DROWSY_G_CHANNEL, g);
  ledcWrite(DROWSY_B_CHANNEL, b);
  
  ledcWrite(MOBILE_R_CHANNEL, r);
  ledcWrite(MOBILE_G_CHANNEL, g);
  ledcWrite(MOBILE_B_CHANNEL, b);
  
  ledcWrite(SMOKING_R_CHANNEL, r);
  ledcWrite(SMOKING_G_CHANNEL, g);
  ledcWrite(SMOKING_B_CHANNEL, b);
  
  ledcWrite(IDLE_R_CHANNEL, r);
  ledcWrite(IDLE_G_CHANNEL, g);
  ledcWrite(IDLE_B_CHANNEL, b);
  
  Serial.printf("All LEDs: R=%d, G=%d, B=%d\n", r, g, b);
}