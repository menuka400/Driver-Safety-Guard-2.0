#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>

// WiFi credentials
const char* ssid = "SLT_FIBRE";      // Your WiFi name
const char* password = "abcd1234";   // Your WiFi password

// Create web server
WebServer server(80);

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

// Bulb 5 (Idle) - Not used, kept off
const int LED_IDLE_R = 19;
const int LED_IDLE_G = 18;
const int LED_IDLE_B = 5;

// Buzzer and Sensor pins
const int BUZZER_PIN = 15;
const int MQ2_PIN = 36;          // Changed to GPIO36 (ADC1_7) for better analog reading
const int SMOKE_THRESHOLD = 400;  // Smoke detection threshold

// Timing constants
const unsigned long BUZZER_DURATION = 4000;    // 4 seconds in milliseconds
const unsigned long DETECTION_TIMEOUT = 2000;   // 2 seconds timeout for detection
const unsigned long SENSOR_CHECK_INTERVAL = 500; // Check sensor every 0.5 seconds

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
    digitalWrite(redPin, state ? HIGH : LOW);
    digitalWrite(greenPin, state ? HIGH : LOW);
    digitalWrite(bluePin, state ? HIGH : LOW);
}

// Function to check MQ2 sensor
void checkSmokeSensor() {
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

// Function to display connection status
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
    alerts[index].isDetecting = true;
    alerts[index].lastDetectionTime = millis();
    
    if (!alerts[index].isActive) {
        alerts[index].isActive = true;
        alerts[index].buzzerStartTime = millis();
        digitalWrite(BUZZER_PIN, HIGH);
        
        // Set the corresponding RGB LED to white (all colors on)
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
    unsigned long currentTime = millis();
    bool shouldBuzzerBeOn = false;

    // Check each alert except smoke (index 3) as it's handled separately
    for (int i = 0; i < 3; i++) {
        // Check if alert is active
        if (alerts[i].isActive) {
            // Check buzzer timing
            if (currentTime - alerts[i].buzzerStartTime < BUZZER_DURATION) {
                shouldBuzzerBeOn = true;
            }
            
            // Check if detection has timed out
            if (alerts[i].isDetecting) {
                alerts[i].lastDetectionTime = currentTime;
            } else if (currentTime - alerts[i].lastDetectionTime >= DETECTION_TIMEOUT) {
                // Turn off the alert if no detection for DETECTION_TIMEOUT
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
        // Reset detection flag for next loop
        alerts[i].isDetecting = false;
    }

    // Handle smoke alert buzzer separately
    if (alerts[3].isActive && 
        (currentTime - alerts[3].buzzerStartTime < BUZZER_DURATION)) {
        shouldBuzzerBeOn = true;
    }

    // Update buzzer state
    digitalWrite(BUZZER_PIN, shouldBuzzerBeOn ? HIGH : LOW);
}

void processCommand(String command) {
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
        setRGBColor(LED_DISTRACTED_R, LED_DISTRACTED_G, LED_DISTRACTED_B, false);
        setRGBColor(LED_DROWSY_R, LED_DROWSY_G, LED_DROWSY_B, false);
        setRGBColor(LED_MOBILE_R, LED_MOBILE_G, LED_MOBILE_B, false);
        setRGBColor(LED_SMOKING_R, LED_SMOKING_G, LED_SMOKING_B, false);
        digitalWrite(BUZZER_PIN, LOW);
        debug("🛑 All alerts stopped");
    }
}

// Handle trigger request
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
    processCommand(alertType);
    server.send(200, "text/plain", "Alert triggered");
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

// Handle status request
void handleStatus() {
    String status = "{\"status\":\"ok\",\"uptime\":\"" + String(millis()/1000) + "\"}";
    server.send(200, "application/json", status);
}

// Handle root page
void handleRoot() {
    String html = "<html><head>";
    html += "<title>Alert System Control Panel</title>";
    html += "<style>";
    html += "body { font-family: Arial, sans-serif; margin: 20px; }";
    html += "h1 { color: #333; }";
    html += ".info { background: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 15px; }";
    html += "</style></head><body>";
    html += "<h1>Alert System Control Panel</h1>";
    html += "<div class='info'>";
    html += "<h2>System Information</h2>";
    html += "<p>IP Address: " + WiFi.localIP().toString() + "</p>";
    html += "<p>WiFi Signal: " + String(WiFi.RSSI()) + " dBm</p>";
    html += "<p>System Uptime: " + String(millis() / 1000) + " seconds</p>";
    html += "</div>";
    html += "<div class='info'>";
    html += "<h2>Active Alerts</h2>";
    html += "<ul>";
    if (alerts[0].isActive) html += "<li>Distracted</li>";
    if (alerts[1].isActive) html += "<li>Drowsy</li>";
    if (alerts[2].isActive) html += "<li>Mobile</li>";
    if (alerts[3].isActive) html += "<li>Smoking</li>";
    html += "</ul>";
    html += "</div>";
    html += "<div class='info'>";
    html += "<h2>API Endpoints</h2>";
    html += "<p>POST /trigger with type parameter</p>";
    html += "<p>POST /stop to stop all alerts</p>";
    html += "</div>";
    html += "</body></html>";
    server.send(200, "text/html", html);
}

void testLEDs() {
    debug("Testing LEDs and buzzer...");
    
    // Test buzzer
    digitalWrite(BUZZER_PIN, HIGH);
    delay(200);
    digitalWrite(BUZZER_PIN, LOW);
    
    // Test each RGB LED individually (excluding 5th bulb)
    int rgbLeds[][3] = {
        {LED_DISTRACTED_R, LED_DISTRACTED_G, LED_DISTRACTED_B},
        {LED_DROWSY_R, LED_DROWSY_G, LED_DROWSY_B},
        {LED_MOBILE_R, LED_MOBILE_G, LED_MOBILE_B},
        {LED_SMOKING_R, LED_SMOKING_G, LED_SMOKING_B}
    };
    
    // Test each LED
    for (int i = 0; i < 4; i++) {
        setRGBColor(rgbLeds[i][0], rgbLeds[i][1], rgbLeds[i][2], true);
        delay(200);
        setRGBColor(rgbLeds[i][0], rgbLeds[i][1], rgbLeds[i][2], false);
        delay(200);
    }
    
    // Flash all LEDs together (excluding 5th bulb)
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            setRGBColor(rgbLeds[j][0], rgbLeds[j][1], rgbLeds[j][2], true);
        }
        delay(200);
        for (int j = 0; j < 4; j++) {
            setRGBColor(rgbLeds[j][0], rgbLeds[j][1], rgbLeds[j][2], false);
        }
        delay(200);
    }
    debug("LED and buzzer test complete");
}

void setup() {
    // Start Serial first for debugging
    Serial.begin(115200);
    delay(1000);  // Give serial connection time to establish
    
    debug("\n\n🚀 Alert System Starting...");
    debug("Initializing pins...");

    // Initialize all RGB LED pins as outputs
    pinMode(LED_DISTRACTED_R, OUTPUT);
    pinMode(LED_DISTRACTED_G, OUTPUT);
    pinMode(LED_DISTRACTED_B, OUTPUT);
    
    pinMode(LED_DROWSY_R, OUTPUT);
    pinMode(LED_DROWSY_G, OUTPUT);
    pinMode(LED_DROWSY_B, OUTPUT);
    
    pinMode(LED_MOBILE_R, OUTPUT);
    pinMode(LED_MOBILE_G, OUTPUT);
    pinMode(LED_MOBILE_B, OUTPUT);
    
    pinMode(LED_SMOKING_R, OUTPUT);
    pinMode(LED_SMOKING_G, OUTPUT);
    pinMode(LED_SMOKING_B, OUTPUT);
    
    pinMode(LED_IDLE_R, OUTPUT);
    pinMode(LED_IDLE_G, OUTPUT);
    pinMode(LED_IDLE_B, OUTPUT);

    // Initialize buzzer pin
    pinMode(BUZZER_PIN, OUTPUT);

    // Initialize MQ2 sensor pin and ADC
    pinMode(MQ2_PIN, INPUT);
    analogSetWidth(12);      // Set ADC resolution to 12 bits (0-4095)
    analogSetAttenuation(ADC_11db);  // Set ADC attenuation for full 3.3V range

    // Initialize all LEDs to OFF
    setRGBColor(LED_DISTRACTED_R, LED_DISTRACTED_G, LED_DISTRACTED_B, false);
    setRGBColor(LED_DROWSY_R, LED_DROWSY_G, LED_DROWSY_B, false);
    setRGBColor(LED_MOBILE_R, LED_MOBILE_G, LED_MOBILE_B, false);
    setRGBColor(LED_SMOKING_R, LED_SMOKING_G, LED_SMOKING_B, false);
    setRGBColor(LED_IDLE_R, LED_IDLE_G, LED_IDLE_B, false);  // Keep 5th bulb off
    digitalWrite(BUZZER_PIN, LOW);

    debug("Pins initialized");
    
    // Test LEDs and buzzer (excluding 5th bulb)
    testLEDs();

    // Print ADC characteristics
    debug("\nADC Configuration:");
    debug("Resolution: 12 bits (0-4095)");
    debug("Attenuation: 11dB (0-3.3V range)");
    debug("Testing MQ2 sensor...");
    for(int i = 0; i < 5; i++) {
        int val = analogRead(MQ2_PIN);
        debug("Test reading " + String(i+1) + ": " + String(val));
        delay(100);
    }

    // Connect to WiFi
    debug("\n📡 Connecting to WiFi...");
    WiFi.begin(ssid, password);
    
    // Wait for WiFi connection
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {  // 15 second timeout
        delay(500);
        Serial.print(".");
        attempts++;
    }
    Serial.println();
    
    if (WiFi.status() == WL_CONNECTED) {
        debug("WiFi Connected Successfully!");
        
        // Display connection information
        displayConnectionInfo();

        // Set up mDNS responder
        if (MDNS.begin("alertsystem")) {
            debug("✅ MDNS responder started");
        }

        // Setup server routes
        server.on("/", HTTP_GET, handleRoot);
        server.on("/trigger", HTTP_POST, handleTrigger);
        server.on("/stop", HTTP_POST, handleStop);
        server.on("/status", HTTP_GET, handleStatus);  // Add status endpoint

        // Start server
        server.begin();
        debug("✅ HTTP server started");
        
        // Print initial sensor reading
        debug("Initial smoke sensor value: " + String(analogRead(MQ2_PIN)));
    } else {
        debug("❌ Failed to connect to WiFi!");
    }
}

void loop() {
    // Handle client requests
    server.handleClient();
    
    // Check smoke sensor
    checkSmokeSensor();
    
    // Check alerts and buzzer states
    checkAlertsAndBuzzer();
    
    // Periodically display connection info (every 30 seconds)
    static unsigned long lastInfoDisplay = 0;
    if (millis() - lastInfoDisplay >= 30000) {  // 30 seconds
        displayConnectionInfo();
        lastInfoDisplay = millis();
    }
    
    // Monitor WiFi connection
    static bool wasConnected = false;
    if (WiFi.status() != WL_CONNECTED) {
        if (wasConnected) {
            debug("❌ WiFi connection lost!");
        }
        wasConnected = false;
    } else if (!wasConnected) {
        wasConnected = true;
        debug("✅ WiFi reconnected!");
        displayConnectionInfo();
    }
}
