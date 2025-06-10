#define BLYNK_TEMPLATE_ID "TMPL6dXyl2CWF"
#define BLYNK_TEMPLATE_NAME "Quickstart Device"
#define BLYNK_AUTH_TOKEN "niTJWQMeIJB_m8hmD9YJuBbcVm8jj5r1"

#define BLYNK_PRINT Serial

#include <WiFi.h>
#include <WiFiClient.h>
#include <BlynkSimpleEsp32.h>
#include <WebServer.h>
#include <ESPmDNS.h>

// WiFi credentials
const char* ssid = "SLT_FIBRE";
const char* password = "abcd1234";

// Create web server
WebServer server(80);
BlynkTimer timer;

// System state
bool systemOnline = true;

// Pin definitions for RGB LEDs
// Bulb 1 (Distracted)
const int LED_DISTRACTED_R = 12;
const int LED_DISTRACTED_G = 14;
const int LED_DISTRACTED_B = 27;

// Bulb 2 (Drowsy)
const int LED_DROWSY_R = 26;
const int LED_DROWSY_G = 25;
const int LED_DROWSY_B = 33;

// Bulb 3 (Mobile)
const int LED_MOBILE_R = 32;
const int LED_MOBILE_G = 17;
const int LED_MOBILE_B = 34;

// Bulb 4 (Smoking)
const int LED_SMOKING_R = 23;
const int LED_SMOKING_G = 22;
const int LED_SMOKING_B = 21;

// Bulb 5 (Idle)
const int LED_IDLE_R = 19;
const int LED_IDLE_G = 18;
const int LED_IDLE_B = 5;

// Buzzer and Sensor pins
const int BUZZER_PIN = 15;
const int MQ2_PIN = 36;
const int SMOKE_THRESHOLD = 400;

// Timing constants
const unsigned long BUZZER_DURATION = 4000;    // 4 seconds
const unsigned long DETECTION_TIMEOUT = 2000;   // 2 seconds
const unsigned long SENSOR_CHECK_INTERVAL = 500; // 0.5 seconds

// Debug flag
bool DEBUG = true;

void debug(String message) {
    if (DEBUG) {
        Serial.println(message);
    }
}

// Variables to track alert states and timings
struct AlertState {
    bool isActive;
    bool isDetecting;
    unsigned long buzzerStartTime;
    unsigned long lastDetectionTime;
};

AlertState alerts[4] = {
    {false, false, 0, 0},
    {false, false, 0, 0},
    {false, false, 0, 0},
    {false, false, 0, 0}
};

// Function to set RGB LED color
void setRGBColor(int redPin, int greenPin, int bluePin, bool state) {
    if (!systemOnline) {
        digitalWrite(redPin, LOW);
        digitalWrite(greenPin, LOW);
        digitalWrite(bluePin, LOW);
        return;
    }
    digitalWrite(redPin, state ? HIGH : LOW);
    digitalWrite(greenPin, state ? HIGH : LOW);
    digitalWrite(bluePin, state ? HIGH : LOW);
}

// Function to turn off all LEDs
void turnOffAllLEDs() {
    setRGBColor(LED_DISTRACTED_R, LED_DISTRACTED_G, LED_DISTRACTED_B, false);
    setRGBColor(LED_DROWSY_R, LED_DROWSY_G, LED_DROWSY_B, false);
    setRGBColor(LED_MOBILE_R, LED_MOBILE_G, LED_MOBILE_B, false);
    setRGBColor(LED_SMOKING_R, LED_SMOKING_G, LED_SMOKING_B, false);
    setRGBColor(LED_IDLE_R, LED_IDLE_G, LED_IDLE_B, false);
}

// Function to check MQ2 sensor
void checkSmokeSensor() {
    if (!systemOnline) return;
    
    static unsigned long lastCheck = 0;
    unsigned long currentTime = millis();
    
    if (currentTime - lastCheck >= SENSOR_CHECK_INTERVAL) {
        lastCheck = currentTime;
        
        // Read smoke sensor
        int smokeValue = analogRead(MQ2_PIN);
        
        // Print sensor value
        debug("Smoke Sensor Value: " + String(smokeValue));
        
        // Check if smoke is detected
        if (smokeValue > SMOKE_THRESHOLD) {
            debug("⚠️ Smoke Detected! Level: " + String(smokeValue));
            // Directly control smoking LED and buzzer
            setRGBColor(LED_SMOKING_R, LED_SMOKING_G, LED_SMOKING_B, true);
            digitalWrite(BUZZER_PIN, HIGH);
            alerts[3].isActive = true;
            alerts[3].isDetecting = true;
            alerts[3].lastDetectionTime = currentTime;
            alerts[3].buzzerStartTime = currentTime;
        } else {
            debug("✅ Air is Clear.");
            // If no smoke detected for DETECTION_TIMEOUT, turn off the LED
            if (alerts[3].isActive && 
                (currentTime - alerts[3].lastDetectionTime >= DETECTION_TIMEOUT)) {
                setRGBColor(LED_SMOKING_R, LED_SMOKING_G, LED_SMOKING_B, false);
                alerts[3].isActive = false;
                alerts[3].isDetecting = false;
            }
        }
    }
}

void displayConnectionInfo() {
    debug("\n----------------------------------------");
    debug("🌟 CONNECTION STATUS");
    debug("----------------------------------------");
    debug("📱 Connected to WiFi: " + String(ssid));
    debug("📶 Signal Strength: " + String(WiFi.RSSI()) + " dBm");
    debug("🔒 MAC Address: " + WiFi.macAddress());
    debug("----------------------------------------");
    debug("🌐 NETWORK INFORMATION");
    debug("----------------------------------------");
    debug("📍 IP Address: " + WiFi.localIP().toString());
    debug("🎭 Subnet Mask: " + WiFi.subnetMask().toString());
    debug("🚪 Gateway: " + WiFi.gatewayIP().toString());
    debug("🔍 DNS: " + WiFi.dnsIP().toString());
    debug("----------------------------------------");
    debug("🚀 SERVER INFORMATION");
    debug("----------------------------------------");
    debug("📡 HTTP Server: RUNNING");
    debug("🔌 Port: 80");
    debug("🌐 Web Interface: http://" + WiFi.localIP().toString() + "/");
    debug("----------------------------------------\n");
}

void activateAlert(int index) {
    if (!systemOnline) return;
    
    alerts[index].isDetecting = true;
    alerts[index].lastDetectionTime = millis();
    
    if (!alerts[index].isActive) {
        alerts[index].isActive = true;
        alerts[index].buzzerStartTime = millis();
        digitalWrite(BUZZER_PIN, HIGH);
        
        switch (index) {
            case 0: // Distracted
                setRGBColor(LED_DISTRACTED_R, LED_DISTRACTED_G, LED_DISTRACTED_B, true);
                break;
            case 1: // Drowsy
                setRGBColor(LED_DROWSY_R, LED_DROWSY_G, LED_DROWSY_B, true);
                break;
            case 2: // Mobile
                setRGBColor(LED_MOBILE_R, LED_MOBILE_G, LED_MOBILE_B, true);
                break;
            case 3: // Smoking
                setRGBColor(LED_SMOKING_R, LED_SMOKING_G, LED_SMOKING_B, true);
                break;
        }
        debug("🔔 Alert activated: " + String(index));
    }
}

void checkAlertsAndBuzzer() {
    if (!systemOnline) {
        digitalWrite(BUZZER_PIN, LOW);
        return;
    }
    
    unsigned long currentTime = millis();
    bool shouldBuzzerBeOn = false;

    // Check each alert except smoke (index 3) as it's handled separately
    for (int i = 0; i < 3; i++) {
        if (alerts[i].isActive) {
            if (currentTime - alerts[i].buzzerStartTime < BUZZER_DURATION) {
                shouldBuzzerBeOn = true;
            }
            
            if (alerts[i].isDetecting) {
                alerts[i].lastDetectionTime = currentTime;
            } else if (currentTime - alerts[i].lastDetectionTime >= DETECTION_TIMEOUT) {
                alerts[i].isActive = false;
                switch (i) {
                    case 0:
                        setRGBColor(LED_DISTRACTED_R, LED_DISTRACTED_G, LED_DISTRACTED_B, false);
                        break;
                    case 1:
                        setRGBColor(LED_DROWSY_R, LED_DROWSY_G, LED_DROWSY_B, false);
                        break;
                    case 2:
                        setRGBColor(LED_MOBILE_R, LED_MOBILE_G, LED_MOBILE_B, false);
                        break;
                }
                debug("🔕 Alert deactivated (timeout): " + String(i));
            }
        }
        alerts[i].isDetecting = false;
    }

    // Handle smoke alert buzzer separately
    if (alerts[3].isActive && 
        (currentTime - alerts[3].buzzerStartTime < BUZZER_DURATION)) {
        shouldBuzzerBeOn = true;
    }

    digitalWrite(BUZZER_PIN, shouldBuzzerBeOn ? HIGH : LOW);
}

void processCommand(String command) {
    if (!systemOnline) return;
    
    debug("⚡ Processing command: " + command);

    if (command == "DISTRACTED") {
        activateAlert(0);
    }
    else if (command == "DROWSY") {
        activateAlert(1);
    }
    else if (command == "MOBILE") {
        activateAlert(2);
    }
    else if (command == "SMOKING") {
        activateAlert(3);
    }
    else if (command == "STOP") {
        // Turn off all LEDs and buzzer
        for (int i = 0; i < 4; i++) {
            alerts[i].isActive = false;
            alerts[i].isDetecting = false;
        }
        turnOffAllLEDs();
        digitalWrite(BUZZER_PIN, LOW);
        debug("🛑 All alerts stopped");
    }
}

// Blynk ON button (V0)
BLYNK_WRITE(V0) {
    int value = param.asInt();
    if (value == 1 && !systemOnline) {
        systemOnline = true;
        Serial.println("System is ONLINE");
        Blynk.virtualWrite(V2, "ONLINE");
    }
}

// Blynk OFF button (V1)
BLYNK_WRITE(V1) {
    int value = param.asInt();
    if (value == 1 && systemOnline) {
        systemOnline = false;
        Serial.println("System is OFFLINE");
        Blynk.virtualWrite(V2, "OFFLINE");
        turnOffAllLEDs();
        digitalWrite(BUZZER_PIN, LOW);
    }
}

// Handle status request
void handleStatus() {
    String status = "{\"status\":\"ok\",\"uptime\":\"" + String(millis()/1000) + "\"}";
    server.send(200, "application/json", status);
}

// Handle stop request
void handleStop() {
    if (server.method() != HTTP_POST) {
        server.send(405, "text/plain", "Method Not Allowed");
        return;
    }
    processCommand("STOP");
    server.send(200, "text/plain", "Alerts stopped");
}

// Handle trigger request from web server
void handleTrigger() {
    if (server.method() != HTTP_POST) {
        server.send(405, "text/plain", "Method Not Allowed");
        return;
    }

    if (!server.hasArg("type")) {
        server.send(400, "text/plain", "Missing alert type");
        return;
    }

    String alertType = server.arg("type");
    String command;
    
    // Convert alert type to command
    if (alertType == "distracted") command = "DISTRACTED";
    else if (alertType == "drowsy") command = "DROWSY";
    else if (alertType == "phone") command = "MOBILE";
    else if (alertType == "smoking") command = "SMOKING";
    else {
        server.send(400, "text/plain", "Invalid alert type");
        return;
    }

    processCommand(command);
    server.send(200, "text/plain", "Alert triggered: " + alertType);
}

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
        digitalWrite(pins[i], LOW);
    }
}

void setup() {
    Serial.begin(115200);
    
    // Initialize all LED pins
    setupLEDs();
    
    // Initialize buzzer pin
    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(BUZZER_PIN, LOW);
    
    // Connect to WiFi
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi connected");
    
    // Initialize Blynk
    Blynk.begin(BLYNK_AUTH_TOKEN, ssid, password);
    Blynk.virtualWrite(V2, "ONLINE");
    
    // Set up MDNS responder
    if (MDNS.begin("esp32")) {
        Serial.println("MDNS responder started");
    }
    
    // Set up web server routes
    server.on("/status", HTTP_GET, handleStatus);  // Add status endpoint
    server.on("/trigger", HTTP_POST, handleTrigger);
    server.on("/stop", HTTP_POST, handleStop);  // Add stop endpoint
    server.begin();
    
    // Display connection information
    displayConnectionInfo();
    
    // Set up timer for checking alerts
    timer.setInterval(100, checkAlertsAndBuzzer);
    timer.setInterval(500, checkSmokeSensor);
}

void loop() {
    Blynk.run();
    timer.run();
    server.handleClient();
} 