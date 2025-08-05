"""
Configuration Manager
Handles all configuration settings for the tracking system
"""
import json
import os

class ConfigManager:
    def __init__(self):
        self.config = self.load_default_config()
    
    def load_default_config(self):
        """Load default configuration"""
        return {
            # ESP32 Configuration
            "esp32": {
                "ip": "10.27.146.54",
                "port": 80,
                "alert_cooldown": 2.0
            },
            
            # Model Paths
            "models": {
                "yolo_phone": "models/yolo11x.pt",
                "yolo_face": "models/yolov11l-face.pt",
                "hrnetv2": "models/hrnetv2_w32_imagenet_pretrained.pth"
            },
            
            # Detection Thresholds
            "thresholds": {
                "person_confidence": 0.5,
                "phone_confidence": 0.3,
                "face_confidence": 0.4,
                "iou_threshold": 0.2,
                "eye_closed_threshold": 0.25,
                "gaze_threshold": 0.02
            },
            
            # Tracking Parameters
            "tracking": {
                "max_disappeared": 60,
                "max_distance": 200,
                "max_trail_length": 50,
                "feature_dim": 256,
                "max_cosine_distance": 0.3,
                "batch_size": 4
            },
            
            # Timing Parameters
            "timing": {
                "consecutive_time_required": 3.0,
                "gaze_distraction_threshold": 3.0,
                "frames_for_drowsiness": 30,
                "frames_for_removal_announcement": 10,
                "eye_tts_cooldown": 4,
                "gaze_tts_cooldown": 3,
                "phone_tts_cooldown": 5
            },
            
            # Video Settings
            "video": {
                "width": 1280,
                "height": 720,
                "fps": 30
            },
            
            # Feature Flags
            "features": {
                "eye_detection_enabled": True,
                "face_distraction_enabled": True,
                "show_trails": True,
                "use_prediction": True
            }
        }
    
    def load_config(self, config_path):
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
            print(f"✅ Configuration loaded from {config_path}")
        except FileNotFoundError:
            print(f"⚠️ Config file not found: {config_path}, using defaults")
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing config file: {e}")
    
    def save_config(self, config_path):
        """Save current configuration to file"""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"✅ Configuration saved to {config_path}")
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def get(self, key_path, default=None):
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path, value):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def update_threshold(self, threshold_name, value):
        """Update a threshold value"""
        if threshold_name in self.config["thresholds"]:
            self.config["thresholds"][threshold_name] = value
            print(f"Updated {threshold_name} to {value}")
        else:
            print(f"Unknown threshold: {threshold_name}")
    
    def toggle_feature(self, feature_name):
        """Toggle a feature on/off"""
        if feature_name in self.config["features"]:
            self.config["features"][feature_name] = not self.config["features"][feature_name]
            status = "ON" if self.config["features"][feature_name] else "OFF"
            print(f"{feature_name}: {status}")
            return self.config["features"][feature_name]
        else:
            print(f"Unknown feature: {feature_name}")
            return None
