#define BLYNK_TEMPLATE_ID "TMPL6dXyl2CWF"
#define BLYNK_TEMPLATE_NAME "Quickstart Device"
#define BLYNK_AUTH_TOKEN "niTJWQMeIJB_m8hmD9YJuBbcVm8jj5r1"

#include <WiFi.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <BlynkSimpleEsp32.h>

// WiFi credentials
const char* ssid = "SLT_FIBRE";
const char* password = "abcd1234";

// DS18B20 data pin connected to GPIO13
#define ONE_WIRE_BUS 13

// Create OneWire and DallasTemperature objects
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

// Blynk timer
BlynkTimer timer;

void sendTemperature() {
  sensors.requestTemperatures();
  float tempC = sensors.getTempCByIndex(0);

  Serial.print("Temperature: ");
  Serial.print(tempC);
  Serial.println(" °C");

  // Send temperature to Blynk virtual pin V14
  Blynk.virtualWrite(V14, tempC);
}

void setup() {
  Serial.begin(115200);
  
  // Start the temperature sensor
  sensors.begin();

  // Connect to Blynk
  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, password);

  // Send temperature every 2 seconds
  timer.setInterval(2000L, sendTemperature);
}

void loop() {
  Blynk.run();
  timer.run();
}
