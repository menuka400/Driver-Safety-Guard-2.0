"""
Phone Detection Module
Handles mobile phone detection using YOLO models
"""
import torch
import time
from ultralytics import YOLO

class PhoneDetector:
    def __init__(self, model_path, device='cuda', confidence_threshold=0.3):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.load_model(model_path)
        
        # Phone detection state
        self.phone_detected_state = False
        self.last_phone_tts_time = 0
        self.phone_tts_cooldown = 5
        self.phone_removal_tts_time = 0
        self.phone_removal_announced = False
        self.frames_without_phone = 0
        self.frames_for_removal_announcement = 10
        self.phone_detection_start_time = None  # Track when phone first detected
        self.continuous_phone_threshold = 3.0  # 3 seconds threshold
        self.phone_tts_triggered = False  # Track if TTS has been triggered for current detection
        self.had_previous_detection = False  # Track if we had a previous phone detection
    
    def load_model(self, model_path):
        """Load YOLO model for phone detection"""
        try:
            print("Loading YOLO11x for mobile phone detection...")
            self.model = YOLO(model_path)
            if self.device == 'cuda':
                self.model.to(self.device)
            print(f"✅ Phone detection model loaded on {self.device.upper()}")
        except Exception as e:
            print(f"❌ Error loading phone detection model: {e}")
            raise
    
    def detect(self, frame):
        """
        Detect phones in the frame
        
        Args:
            frame: Input frame
            
        Returns:
            dict: Detection results including boxes and state information
        """
        current_time = time.time()
        
        try:
            with torch.no_grad():
                results = self.model(frame, verbose=False)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Check if it's a mobile phone (class 67 in COCO dataset)
                        if class_id == 67 and confidence >= self.confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence
                            })
            
            # Update detection state
            has_phone_detections = len(detections) > 0
            
            if has_phone_detections:
                # Phone is detected
                self.frames_without_phone = 0
                self.phone_removal_announced = False
                self.had_previous_detection = True
                
                # Start timing for continuous detection
                if not self.phone_detected_state:
                    self.phone_detection_start_time = current_time
                    self.phone_detected_state = True
                
                # Calculate continuous detection duration
                detection_duration = 0
                if self.phone_detection_start_time is not None:
                    detection_duration = current_time - self.phone_detection_start_time
                
                continuous_detection = detection_duration >= self.continuous_phone_threshold
            else:
                # No phone detected
                self.frames_without_phone += 1
                
                if self.phone_detected_state:
                    # Reset detection state
                    self.phone_detected_state = False
                    self.phone_detection_start_time = None
                    self.phone_tts_triggered = False
                
                continuous_detection = False
                detection_duration = 0
            
            # Only indicate removal is ready if we previously had a detection
            removal_ready = (self.had_previous_detection and 
                           self.frames_without_phone >= self.frames_for_removal_announcement and 
                           not self.phone_removal_announced)
            
            if removal_ready:
                # Reset the previous detection flag after announcing removal
                self.had_previous_detection = False
            
            return {
                'detections': detections,
                'phone_detected': has_phone_detections,
                'continuous_detection': continuous_detection,
                'detection_duration': detection_duration,
                'frames_without_phone': self.frames_without_phone,
                'removal_ready': removal_ready
            }
            
        except Exception as e:
            print(f"Error in phone detection: {e}")
            return {
                'detections': [],
                'phone_detected': False,
                'continuous_detection': False,
                'detection_duration': 0,
                'frames_without_phone': self.frames_without_phone,
                'removal_ready': False
            }
    
    def reset_state(self):
        """Reset phone detection state"""
        self.phone_detected_state = False
        self.phone_detection_start_time = None
        self.phone_tts_triggered = False
        self.frames_without_phone = 0
        self.phone_removal_announced = False
        self.had_previous_detection = False
    
    def mark_removal_announced(self):
        """Mark that phone removal has been announced"""
        self.phone_removal_announced = True
        self.phone_removal_tts_time = time.time()
    
    def update_confidence_threshold(self, new_threshold):
        """Update confidence threshold"""
        self.confidence_threshold = max(0.1, min(1.0, new_threshold))
        print(f"Phone detection confidence threshold updated to: {self.confidence_threshold:.2f}")
    
    def get_detection_info(self):
        """Get current detection information"""
        return {
            "phone_detected": self.phone_detected_state,
            "frames_without_phone": self.frames_without_phone,
            "confidence_threshold": self.confidence_threshold
        }
