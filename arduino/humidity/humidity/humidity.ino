// Tech Trends Shameer

#define BLYNK_TEMPLATE_ID "TMPL6dXyl2CWF"
#define BLYNK_TEMPLATE_NAME "Quickstart Device"
#define BLYNK_AUTH_TOKEN "niTJWQMeIJB_m8hmD9YJuBbcVm8jj5r1"

#define BLYNK_PRINT Serial
#include <WiFi.h>
#include <BlynkSimpleEsp32.h>
#include <DHT.h>

const char* ssid = "SLT_FIBRE";
const char* password = "abcd1234";

BlynkTimer timer;

#define DHTPIN 2        // G2 = GPIO2
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

void sendSensor() {
  float h = dht.readHumidity();

  if (isnan(h)) {
    Serial.println("Failed to read humidity from DHT sensor!");
    return;
  }

  Blynk.virtualWrite(V15, h);  // Send humidity to Virtual Pin V15
  Serial.print("Humidity : ");
  Serial.println(h);
}

void setup() {
  Serial.begin(115200);
  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, password);
  dht.begin();
  timer.setInterval(1000L, sendSensor); // every 1 second
}

void loop() {
  Blynk.run();
  timer.run();
}
