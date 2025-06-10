#define BLYNK_TEMPLATE_ID "TMPL6dXyl2CWF"
#define BLYNK_TEMPLATE_NAME "Quickstart Device"
#define BLYNK_AUTH_TOKEN "niTJWQMeIJB_m8hmD9YJuBbcVm8jj5r1"

#define BLYNK_PRINT Serial

#include <WiFi.h>
#include <WiFiClient.h>
#include <BlynkSimpleEsp32.h>

char ssid[] = "SLT_FIBRE";
char pass[] = "abcd1234";

BlynkTimer timer;
bool systemOnline = true;
int patternStepIndex = 0;

// DISTRACTED LED Pins
const int LED_DISTRACTED_R = 12;
const int LED_DISTRACTED_G = 14;
const int LED_DISTRACTED_B = 27;

// DROWSY LED Pins
const int LED_DROWSY_R = 26;
const int LED_DROWSY_G = 25;
const int LED_DROWSY_B = 33;

// MOBILE LED Pins
const int LED_MOBILE_R = 32;
const int LED_MOBILE_G = 17;
const int LED_MOBILE_B = 34;

// SMOKING LED Pins
const int LED_SMOKING_R = 23;
const int LED_SMOKING_G = 22;
const int LED_SMOKING_B = 21;

// IDLE LED Pins
const int LED_IDLE_R = 19;
const int LED_IDLE_G = 18;
const int LED_IDLE_B = 5;

void setupLEDs() {
  int pins[] = {
    LED_DISTRACTED_R, LED_DISTRACTED_G, LED_DISTRACTED_B,
    LED_DROWSY_R, LED_DROWSY_G, LED_DROWSY_B,
    LED_MOBILE_R, LED_MOBILE_G, LED_MOBILE_B,
    LED_SMOKING_R, LED_SMOKING_G, LED_SMOKING_B,
    LED_IDLE_R, LED_IDLE_G, LED_IDLE_B
  };
  for (int i = 0; i < sizeof(pins) / sizeof(pins[0]); i++) {
    pinMode(pins[i], OUTPUT);
    digitalWrite(pins[i], LOW); // Ensure everything is off at startup
  }
}

void setColor(int rPin, int gPin, int bPin, bool r, bool g, bool b) {
  digitalWrite(rPin, r ? HIGH : LOW);
  digitalWrite(gPin, g ? HIGH : LOW);
  digitalWrite(bPin, b ? HIGH : LOW);
}

void turnOffAllLEDs() {
  int pins[] = {
    LED_DISTRACTED_R, LED_DISTRACTED_G, LED_DISTRACTED_B,
    LED_DROWSY_R, LED_DROWSY_G, LED_DROWSY_B,
    LED_MOBILE_R, LED_MOBILE_G, LED_MOBILE_B,
    LED_SMOKING_R, LED_SMOKING_G, LED_SMOKING_B,
    LED_IDLE_R, LED_IDLE_G, LED_IDLE_B
  };
  for (int i = 0; i < sizeof(pins) / sizeof(pins[0]); i++) {
    digitalWrite(pins[i], LOW);
  }
}

void updatePattern() {
  if (!systemOnline) {
    return;  // Skip lighting logic when OFF
  }

  // Cycle through 4 color combinations
  switch (patternStepIndex) {
    case 0:
      setColor(LED_DISTRACTED_R, LED_DISTRACTED_G, LED_DISTRACTED_B, 1, 0, 0);
      setColor(LED_DROWSY_R, LED_DROWSY_G, LED_DROWSY_B, 0, 1, 0);
      setColor(LED_MOBILE_R, LED_MOBILE_G, LED_MOBILE_B, 0, 0, 1);
      setColor(LED_SMOKING_R, LED_SMOKING_G, LED_SMOKING_B, 1, 1, 0);
      setColor(LED_IDLE_R, LED_IDLE_G, LED_IDLE_B, 1, 0, 1);
      break;

    case 1:
      setColor(LED_DISTRACTED_R, LED_DISTRACTED_G, LED_DISTRACTED_B, 0, 1, 1);
      setColor(LED_DROWSY_R, LED_DROWSY_G, LED_DROWSY_B, 1, 0, 1);
      setColor(LED_MOBILE_R, LED_MOBILE_G, LED_MOBILE_B, 1, 1, 0);
      setColor(LED_SMOKING_R, LED_SMOKING_G, LED_SMOKING_B, 0, 0, 1);
      setColor(LED_IDLE_R, LED_IDLE_G, LED_IDLE_B, 0, 1, 0);
      break;

    case 2:
      setColor(LED_DISTRACTED_R, LED_DISTRACTED_G, LED_DISTRACTED_B, 1, 1, 1);
      setColor(LED_DROWSY_R, LED_DROWSY_G, LED_DROWSY_B, 1, 0, 0);
      setColor(LED_MOBILE_R, LED_MOBILE_G, LED_MOBILE_B, 0, 1, 0);
      setColor(LED_SMOKING_R, LED_SMOKING_G, LED_SMOKING_B, 0, 0, 1);
      setColor(LED_IDLE_R, LED_IDLE_G, LED_IDLE_B, 1, 1, 0);
      break;

    case 3:
      setColor(LED_DISTRACTED_R, LED_DISTRACTED_G, LED_DISTRACTED_B, 0, 0, 1);
      setColor(LED_DROWSY_R, LED_DROWSY_G, LED_DROWSY_B, 0, 1, 0);
      setColor(LED_MOBILE_R, LED_MOBILE_G, LED_MOBILE_B, 1, 0, 0);
      setColor(LED_SMOKING_R, LED_SMOKING_G, LED_SMOKING_B, 1, 0, 1);
      setColor(LED_IDLE_R, LED_IDLE_G, LED_IDLE_B, 0, 1, 1);
      break;
  }

  patternStepIndex = (patternStepIndex + 1) % 4;
}

// ON button (V0)
BLYNK_WRITE(V0) {
  int value = param.asInt();
  if (value == 1 && !systemOnline) {
    systemOnline = true;
    Serial.println("System is ONLINE");
    Blynk.virtualWrite(V2, "ONLINE");
  }
}

// OFF button (V1)
BLYNK_WRITE(V1) {
  int value = param.asInt();
  if (value == 1 && systemOnline) {
    systemOnline = false;
    Serial.println("System is OFFLINE");
    Blynk.virtualWrite(V2, "OFFLINE");
    turnOffAllLEDs(); // Immediately turn off all lights
  }
}

void setup() {
  Serial.begin(115200);
  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass);
  setupLEDs();
  Blynk.virtualWrite(V2, "ONLINE");

  timer.setInterval(100, updatePattern); // Fast pattern change
}

void loop() {
  Blynk.run();
  timer.run();
}