"""
ESP32 Communication Module
Handles communication with ESP32 device for alerts
"""
import requests
import time
import urllib3

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ESP32Communicator:
    def __init__(self, ip="192.168.1.121", port=80):
        """Initialize ESP32 communication"""
        self.esp32_ip = ip
        self.esp32_port = port
        self.esp32_url = f"http://{self.esp32_ip}:{self.esp32_port}"
        self.last_alert_time = 0
        self.alert_cooldown = 2.0  # Seconds between alerts
        self.alert_active = False
        
        print(f"📡 ESP32 URL configured as: {self.esp32_url}")
    
    def send_alert(self, alert_type):
        """Send alert to ESP32 via HTTP"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_alert_time < self.alert_cooldown:
            return False
        
        try:
            # Send POST request to ESP32
            response = requests.post(
                f"{self.esp32_url}/trigger",
                data={"type": alert_type},
                timeout=1.0,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            
            if response.status_code == 200:
                print(f"✅ Alert sent to ESP32: {alert_type}")
                self.alert_active = True
                self.last_alert_time = current_time
                return True
            else:
                print(f"❌ Failed to send alert: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection Error: Could not connect to ESP32 at {self.esp32_url}")
            print("   Make sure the ESP32 is powered on and connected to the network")
            return False
        except Exception as e:
            print(f"❌ Error sending alert to ESP32: {e}")
            return False
    
    def stop_alert(self):
        """Stop active alert on ESP32"""
        if not self.alert_active:
            return True
            
        try:
            response = requests.post(
                f"{self.esp32_url}/stop",
                timeout=1.0,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            
            if response.status_code == 200:
                print("✅ Alert stopped")
                self.alert_active = False
                return True
            else:
                print(f"❌ Failed to stop alert: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error stopping alert: {e}")
            return False
    
    def handle_detection_alerts(self, eye_results, phone_results, gaze_results):
        """Handle alerts based on detection results"""
        if eye_results.get('drowsiness_detected', False):
            self.send_alert("drowsy")
        elif phone_results.get('continuous_detection', False):
            self.send_alert("phone")
        elif (gaze_results.get('direction') in ["left", "right"] and 
              gaze_results.get('duration', 0) >= 3.0):
            self.send_alert("distracted")
        else:
            # If no alerts are active, stop any existing alerts
            self.stop_alert()
    
    def test_connection(self):
        """Test connection to ESP32"""
        try:
            response = requests.get(f"{self.esp32_url}/status", timeout=5, verify=False)
            if response.status_code == 200:
                print("✅ ESP32 connection test successful")
                return True
            else:
                print(f"⚠️ ESP32 connection test failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ ESP32 connection test error: {e}")
            return False
    
    def update_config(self, ip=None, port=None, cooldown=None):
        """Update ESP32 configuration"""
        if ip:
            self.esp32_ip = ip
        if port:
            self.esp32_port = port
        if cooldown:
            self.alert_cooldown = cooldown
            
        self.esp32_url = f"http://{self.esp32_ip}:{self.esp32_port}"
        print(f"📡 ESP32 configuration updated: {self.esp32_url}")
    
    def get_status(self):
        """Get current ESP32 communication status"""
        return {
            "url": self.esp32_url,
            "alert_active": self.alert_active,
            "last_alert_time": self.last_alert_time,
            "alert_cooldown": self.alert_cooldown
        }
