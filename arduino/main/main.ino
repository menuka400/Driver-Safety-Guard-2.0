#define BLYNK_TEMPLATE_ID "TMPL657qAruym"
#define BLYNK_TEMPLATE_NAME "Quickstart Template"
#define BLYNK_AUTH_TOKEN "1NSm1YalmNpNyksHP_E03BfIdCWT9kEw"

#define BLYNK_PRINT Serial

#include <WiFi.h>
#include <WiFiClient.h>
#include <BlynkSimpleEsp32.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// Blynk virtual pins
#define VPIN_THRESHOLD V13  // Slider to set threshold
#define VPIN_TEMPERATURE V14  // Temperature reading

// WiFi credentials
const char* ssid = "SLT_FIBRE";
const char* password = "abcd1234";

// Create web server
WebServer server(80);
BlynkTimer timer;

// System state
bool systemOnline = true;
bool isConnectedToBlynk = false;
bool wifiConnected = false;
bool connectionSoundPlayed = false; // Flag to track if connection sound was played

// Temperature sensor setup
#define ONE_WIRE_BUS 13  // DS18B20 data pin connected to GPIO13
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

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
int smokeThreshold = 400;  // Changed to variable instead of const

// Idle LED brightness control
const int IDLE_GREEN_BRIGHTNESS = 10;

// Timing constants
const unsigned long BUZZER_DURATION = 4000;    // 4 seconds
const unsigned long DETECTION_TIMEOUT = 2000;   // 2 seconds
const unsigned long SENSOR_CHECK_INTERVAL = 500; // 0.5 seconds

// Connection sound timing
const unsigned long CONNECTION_BEEP_DURATION = 150; // 150ms per beep
const unsigned long CONNECTION_BEEP_PAUSE = 100;    // 100ms pause between beeps

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

// Function to play connection success sound
void playConnectionSuccessSound() {
    debug("🎵 Playing connection success sound...");
    
    // Play 3 short beeps to indicate successful connection
    for (int i = 0; i < 3; i++) {
        digitalWrite(BUZZER_PIN, HIGH);
        delay(CONNECTION_BEEP_DURATION);
        digitalWrite(BUZZER_PIN, LOW);
        
        // Don't add pause after the last beep
        if (i < 2) {
            delay(CONNECTION_BEEP_PAUSE);
        }
    }
    
    debug("🎵 Connection success sound completed!");
}

// Function to check if both WiFi and Blynk are connected and play sound if needed
void checkConnectionAndPlaySound() {
    bool currentWifiStatus = (WiFi.status() == WL_CONNECTED);
    bool currentBlynkStatus = isConnectedToBlynk;
    
    // Check if both are connected and sound hasn't been played yet
    if (systemOnline && currentWifiStatus && currentBlynkStatus && !connectionSoundPlayed) {
        playConnectionSuccessSound();
        connectionSoundPlayed = true;
        debug("🎉 Full connection established! WiFi + Blynk connected");
    }
    
    // Reset flag if either connection is lost
    if (!currentWifiStatus || !currentBlynkStatus) {
        connectionSoundPlayed = false;
    }
}

// Function to update Idle LED status
void updateIdleLED() {
    if (WiFi.status() == WL_CONNECTED && isConnectedToBlynk) {
        // Connected to WiFi and Blynk - Green with brightness control
        digitalWrite(LED_IDLE_R, LOW);
        analogWrite(LED_IDLE_G, IDLE_GREEN_BRIGHTNESS);
        digitalWrite(LED_IDLE_B, LOW);
        debug("💚 Idle LED: GREEN (Connected) - Brightness: " + String(IDLE_GREEN_BRIGHTNESS));
    } else {
        // Not connected or connecting - Red
        digitalWrite(LED_IDLE_R, HIGH);
        digitalWrite(LED_IDLE_G, LOW);
        digitalWrite(LED_IDLE_B, LOW);
        debug("❤️ Idle LED: RED (Disconnected/Connecting)");
    }
}

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

// Function to turn off all LEDs (except Idle LED)
void turnOffAllLEDs() {
    setRGBColor(LED_DISTRACTED_R, LED_DISTRACTED_G, LED_DISTRACTED_B, false);
    setRGBColor(LED_DROWSY_R, LED_DROWSY_G, LED_DROWSY_B, false);
    setRGBColor(LED_MOBILE_R, LED_MOBILE_G, LED_MOBILE_B, false);
    setRGBColor(LED_SMOKING_R, LED_SMOKING_G, LED_SMOKING_B, false);
    // Don't turn off Idle LED here - it has its own status function
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
        
        // Send smoke level to Blynk app (Virtual Pin V3)
        if (isConnectedToBlynk) {
            Blynk.virtualWrite(V3, smokeValue);
        }
        
        // Print sensor value
        debug("=== Smoke Sensor Reading ===");
        debug("Raw ADC Value: " + String(smokeValue));
        debug("Current Threshold: " + String(smokeThreshold));
        
        // Check if smoke is detected
        if (smokeValue > smokeThreshold) {
            debug("⚠️ Smoke Detected! Level: " + String(smokeValue));
            // Send alert notification to Blynk app
            if (isConnectedToBlynk) {
                Blynk.logEvent("smoke_alert", "High smoke level detected!");
            }
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
        debug("============================");
    }
}

void displayConnectionInfo() {
    debug("\n----------------------------------------");
    debug("🌟 CONNECTION STATUS");
    debug("----------------------------------------");
    debug("📱 Connected to WiFi: " + String(ssid));
    debug("📶 Signal Strength: " + String(WiFi.RSSI()) + " dBm");
    debug("🔒 MAC Address: " + WiFi.macAddress());
    debug("☁️ Blynk Connected: " + String(isConnectedToBlynk ? "YES" : "NO"));
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
        updateIdleLED(); // Update idle LED when system comes online
        connectionSoundPlayed = false; // Reset flag to allow sound when system comes online
    }
}

// Blynk OFF button (V1)
BLYNK_WRITE(V1) {
    int value = param.asInt();
    if (value == 1 && systemOnline) {
        systemOnline = false;
        Serial.println("System is OFFLINE");
        turnOffAllLEDs();
        digitalWrite(BUZZER_PIN, LOW);
        updateIdleLED(); // Update idle LED when system goes offline
        connectionSoundPlayed = false; // Reset flag when system goes offline
    }
}

// Blynk function to handle threshold slider changes
BLYNK_WRITE(VPIN_THRESHOLD) {
    smokeThreshold = param.asInt();
    debug("Smoke threshold updated to: " + String(smokeThreshold));
}

// Blynk function to handle app connection
BLYNK_CONNECTED() {
    debug("🎉 Connected to Blynk server!");
    isConnectedToBlynk = true;
    updateIdleLED(); // Update LED to green when Blynk connects
    // Sync threshold value from app
    Blynk.syncVirtual(VPIN_THRESHOLD);
    
    // Check if we should play connection sound
    checkConnectionAndPlaySound();
}

// Blynk function to handle app disconnection
BLYNK_DISCONNECTED() {
    debug("⚠️ Disconnected from Blynk server!");
    isConnectedToBlynk = false;
    updateIdleLED(); // Update LED to red when Blynk disconnects
    connectionSoundPlayed = false; // Reset flag when Blynk disconnects
}

// Function to read and send temperature
void sendTemperature() {
    if (!systemOnline) return;
    
    sensors.requestTemperatures();
    float tempC = sensors.getTempCByIndex(0);

    debug("=== Temperature Reading ===");
    debug("Temperature: " + String(tempC) + " °C");
    debug("========================");

    // Send temperature to Blynk virtual pin V14 only if connected
    if (isConnectedToBlynk) {
        Blynk.virtualWrite(VPIN_TEMPERATURE, tempC);
    }
}

// Function to check connection status and update LED
void checkConnectionStatus() {
    static bool lastWiFiStatus = false;
    static bool lastBlynkStatus = false;
    
    bool currentWiFiStatus = (WiFi.status() == WL_CONNECTED);
    bool currentBlynkStatus = Blynk.connected();
    
    // Update Blynk connection status
    if (currentBlynkStatus != isConnectedToBlynk) {
        isConnectedToBlynk = currentBlynkStatus;
        updateIdleLED();
    }
    
    // Check if status changed
    if (currentWiFiStatus != lastWiFiStatus || currentBlynkStatus != lastBlynkStatus) {
        updateIdleLED();
        lastWiFiStatus = currentWiFiStatus;
        lastBlynkStatus = currentBlynkStatus;
    }
    
    // Check for connection sound
    checkConnectionAndPlaySound();
}

// Handle status request
void handleStatus() {
    sensors.requestTemperatures();
    float tempC = sensors.getTempCByIndex(0);
    String status = "{\"status\":\"ok\",\"uptime\":\"" + String(millis()/1000) + 
                   "\",\"temperature\":\"" + String(tempC) + 
                   "\",\"wifi_connected\":\"" + String(WiFi.status() == WL_CONNECTED) +
                   "\",\"blynk_connected\":\"" + String(isConnectedToBlynk) + "\"}";
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
    
    // Set Idle LED to RED initially (not connected)
    updateIdleLED();
    
    // Initialize buzzer pin
    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(BUZZER_PIN, LOW);
    
    // Initialize temperature sensor
    sensors.begin();
    
    debug("🔄 Connecting to WiFi...");
    // Connect to WiFi
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
        updateIdleLED(); // Keep showing red while connecting
    }
    Serial.println("\n✅ WiFi connected");
    wifiConnected = true;
    updateIdleLED(); // Update LED status after WiFi connection
    
    debug("🔄 Connecting to Blynk...");
    // Initialize Blynk
    Blynk.begin(BLYNK_AUTH_TOKEN, ssid, password);
    
    // Set initial threshold value to Blynk
    if (isConnectedToBlynk) {
        Blynk.virtualWrite(VPIN_THRESHOLD, smokeThreshold);
    }
    
    // Set up MDNS responder
    if (MDNS.begin("esp32")) {
        Serial.println("MDNS responder started");
    }
    
    // Set up web server routes
    server.on("/status", HTTP_GET, handleStatus);
    server.on("/trigger", HTTP_POST, handleTrigger);
    server.on("/stop", HTTP_POST, handleStop);
    server.begin();
    
    // Display connection information
    displayConnectionInfo();
    
    // Set up timers
    timer.setInterval(100, checkAlertsAndBuzzer);
    timer.setInterval(500, checkSmokeSensor);
    timer.setInterval(2000, sendTemperature);  // Send temperature every 2 seconds
    timer.setInterval(1000, checkConnectionStatus); // Check connection status every second
}

void loop() {
    Blynk.run();
    timer.run();
    server.handleClient();
}