#include <WiFi.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>

// Wi-Fi credentials
const char* ssid = "SLT_FIBRE";
const char* password = "abcd1234";

// Pin definitions for Weapon Detection
const int weaponBuzzerPin = 21;    // Shared buzzer on GPIO 21
const int weaponRedPin = 2;        // Changed from 35 to 2 (output capable)
const int weaponGreenPin = 4;      // Changed from 34 to 4 (output capable)
const int weaponBluePin = 14;      // Weapon RGB LED blue pin

// Pin definitions for Violation Detection
const int violationBuzzerPin = 21;  // Shared buzzer on GPIO 21
const int violationRedPin = 25;     // Violation RGB LED red pin
const int violationGreenPin = 33;   // Violation RGB LED green pin
const int violationBluePin = 32;    // Violation RGB LED blue pin

// Startup sound frequencies and durations
const int startupMelody[] = {262, 330, 392, 523}; // C4, E4, G4, C5
const int noteDurations[] = {200, 200, 200, 400}; // Duration in milliseconds

// PWM settings for tone generation
const int buzzerChannel = 0;
const int buzzerFreq = 2000;
const int buzzerResolution = 8;

// Create AsyncWebServer object on port 80
AsyncWebServer server(80);

// Function to play tone using PWM
void playTone(int pin, int frequency, int duration) {
    if (frequency > 0) {
        ledcSetup(buzzerChannel, frequency, buzzerResolution);
        ledcAttachPin(pin, buzzerChannel);
        ledcWrite(buzzerChannel, 128); // 50% duty cycle
        delay(duration);
        ledcWrite(buzzerChannel, 0); // Stop tone
        ledcDetachPin(pin);
    } else {
        delay(duration);
    }
}

// Function to play startup sound
void playStartupSound() {
    for (int i = 0; i < 4; i++) {
        playTone(weaponBuzzerPin, startupMelody[i], noteDurations[i]);
        delay(50); // Small gap between notes
    }
}

// Function to trigger weapon alarm
void triggerWeaponAlarm() {
    // Turn on red LED, turn off others
    digitalWrite(weaponRedPin, HIGH);
    digitalWrite(weaponGreenPin, LOW);
    digitalWrite(weaponBluePin, LOW);
    
    // Sound alarm
    playTone(weaponBuzzerPin, 1000, 1000); // 1kHz for 1 second
    
    delay(1000);  // LED stays on for another second
    
    // Return to green (safe state)
    digitalWrite(weaponRedPin, LOW);
    digitalWrite(weaponGreenPin, HIGH);
}

// Function to trigger violation alarm
void triggerViolationAlarm() {
    // Turn on red LED, turn off others
    digitalWrite(violationRedPin, HIGH);
    digitalWrite(violationGreenPin, LOW);
    digitalWrite(violationBluePin, LOW);
    
    // Sound alarm (higher pitch, shorter duration)
    playTone(violationBuzzerPin, 1500, 500); // 1.5kHz for 0.5 second
    
    delay(1500);  // LED stays on longer
    
    // Return to green (safe state)
    digitalWrite(violationRedPin, LOW);
    digitalWrite(violationGreenPin, HIGH);
}

void setup() {
    Serial.begin(115200);
    Serial.println("Starting Security System...");
    
    // Configure weapon detection pins
    pinMode(weaponBuzzerPin, OUTPUT);
    pinMode(weaponRedPin, OUTPUT);
    pinMode(weaponGreenPin, OUTPUT);
    pinMode(weaponBluePin, OUTPUT);
    
    // Configure violation detection pins
    pinMode(violationBuzzerPin, OUTPUT);
    pinMode(violationRedPin, OUTPUT);
    pinMode(violationGreenPin, OUTPUT);
    pinMode(violationBluePin, OUTPUT);
    
    // Initialize all LEDs to off
    digitalWrite(weaponRedPin, LOW);
    digitalWrite(weaponGreenPin, LOW);
    digitalWrite(weaponBluePin, LOW);
    digitalWrite(violationRedPin, LOW);
    digitalWrite(violationGreenPin, LOW);
    digitalWrite(violationBluePin, LOW);
    
    // Play startup sound
    Serial.println("Playing startup sound...");
    playStartupSound();
    
    // Set initial LED states - Green for both systems (system ready)
    digitalWrite(weaponGreenPin, HIGH);
    digitalWrite(violationGreenPin, HIGH);
    
    // Connect to Wi-Fi with timeout
    Serial.println("Connecting to WiFi...");
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);
    
    int timeout = 0;
    while (WiFi.status() != WL_CONNECTED && timeout < 30) { // Increased timeout
        delay(500);
        Serial.print(".");
        timeout++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nConnected to WiFi");
        Serial.print("IP Address: ");
        Serial.println(WiFi.localIP());
        
        // Flash green LEDs to indicate successful connection
        for (int i = 0; i < 3; i++) {
            digitalWrite(weaponGreenPin, LOW);
            digitalWrite(violationGreenPin, LOW);
            delay(200);
            digitalWrite(weaponGreenPin, HIGH);
            digitalWrite(violationGreenPin, HIGH);
            delay(200);
        }
    } else {
        Serial.println("\nFailed to connect to WiFi. Continuing in offline mode...");
        // Show orange/yellow state (red + green) for offline mode
        digitalWrite(weaponRedPin, HIGH);
        digitalWrite(weaponGreenPin, HIGH);
        digitalWrite(violationRedPin, HIGH);
        digitalWrite(violationGreenPin, HIGH);
    }
    
    // Server endpoints
    server.on("/", HTTP_GET, [](AsyncWebServerRequest *request) {
        String html = "<html><body>";
        html += "<h1>ESP32 Security System</h1>";
        html += "<p>Status: " + String(WiFi.status() == WL_CONNECTED ? "Online" : "Offline") + "</p>";
        html += "<p>IP: " + WiFi.localIP().toString() + "</p>";
        html += "<button onclick=\"fetch('/trigger-weapon')\">Trigger Weapon Alarm</button><br><br>";
        html += "<button onclick=\"fetch('/trigger-violation')\">Trigger Violation Alarm</button><br><br>";
        html += "<button onclick=\"fetch('/ping')\">Test Connection</button>";
        html += "</body></html>";
        request->send(200, "text/html", html);
    });
    
    server.on("/ping", HTTP_GET, [](AsyncWebServerRequest *request) {
        request->send(200, "text/plain", "OK");
        Serial.println("Ping received");
    });
    
    // Endpoint for weapon detection
    server.on("/trigger-weapon", HTTP_GET, [](AsyncWebServerRequest *request) {
        request->send(200, "text/plain", "Weapon alarm triggered");
        Serial.println("Weapon alarm triggered via web");
        triggerWeaponAlarm();
    });
    
    // Endpoint for violation detection
    server.on("/trigger-violation", HTTP_GET, [](AsyncWebServerRequest *request) {
        request->send(200, "text/plain", "Violation alarm triggered");
        Serial.println("Violation alarm triggered via web");
        triggerViolationAlarm();
    });
    
    // Status endpoint
    server.on("/status", HTTP_GET, [](AsyncWebServerRequest *request) {
        String status = "{";
        status += "\"wifi_connected\":" + String(WiFi.status() == WL_CONNECTED ? "true" : "false") + ",";
        status += "\"ip\":\"" + WiFi.localIP().toString() + "\",";
        status += "\"uptime\":" + String(millis()) + ",";
        status += "\"free_heap\":" + String(ESP.getFreeHeap());
        status += "}";
        request->send(200, "application/json", status);
    });
    
    server.begin();
    Serial.println("Web server started");
    Serial.println("System ready!");
}

void loop() {
    static unsigned long lastCheck = 0;
    
    // Check WiFi status every 10 seconds
    if (millis() - lastCheck > 10000) {
        lastCheck = millis();
        
        if (WiFi.status() != WL_CONNECTED) {
            // Show disconnected state on both LED sets (red only)
            digitalWrite(weaponGreenPin, LOW);
            digitalWrite(weaponRedPin, HIGH);
            digitalWrite(violationGreenPin, LOW);
            digitalWrite(violationRedPin, HIGH);
            
            Serial.println("WiFi disconnected, attempting reconnection...");
            WiFi.reconnect();
        } else {
            // Show connected state on both LED sets (green only)
            digitalWrite(weaponGreenPin, HIGH);
            digitalWrite(weaponRedPin, LOW);
            digitalWrite(violationGreenPin, HIGH);
            digitalWrite(violationRedPin, LOW);
        }
    }
    
    // Small delay to prevent watchdog issues
    delay(100);
}