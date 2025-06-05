"""
Person Detection Module
Handles person detection and confirmation logic
"""
import torch
import numpy as np
import time
import cv2

class PersonDetector:
    def __init__(self, model, device='cuda', confidence_threshold=0.5):
        self.model = model
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Person detection variables
        self.person_first_seen = None
        self.person_confirmed = False
        self.current_person_detections = []
        self.consecutive_time_required = 3.0  # seconds required for person confirmation
        self.target_person_class = 0  # Person class in COCO dataset
    
    def detect_persons_in_frame(self, frame):
        """
        Detect persons in the given frame using YOLO model
        
        Args:
            frame: Input frame from webcam
            
        Returns:
            list: List of person detections with bounding boxes and confidence
        """
        try:
            with torch.no_grad():
                results = self.model(frame, verbose=False, conf=self.confidence_threshold, device=self.device)
            
            person_detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    # Filter for person class (class 0 in COCO)
                    person_indices = np.where(classes == self.target_person_class)[0]
                    
                    for idx in person_indices:
                        x1, y1, x2, y2 = boxes[idx]
                        confidence = confidences[idx]
                        
                        if confidence >= self.confidence_threshold:
                            person_detections.append([x1, y1, x2, y2, confidence])
            
            return person_detections
            
        except Exception as e:
            print(f"Error in person detection: {e}")
            return []

    def update_person_detection_state(self, persons_detected):
        """
        Update the person detection state based on current frame
        
        Args:
            persons_detected: List of person detections in current frame
        """
        current_time = time.time()
        
        if len(persons_detected) > 0:
            # Store current detections
            self.current_person_detections = persons_detected
            
            # If this is the first time seeing a person
            if self.person_first_seen is None:
                self.person_first_seen = current_time
                print(f"👤 Person detected - starting confirmation timer...")
            
            # Check if person has been detected consistently for required time
            elif not self.person_confirmed:
                time_elapsed = current_time - self.person_first_seen
                if time_elapsed >= self.consecutive_time_required:
                    self.person_confirmed = True
                    print(f"✅ Person confirmed after {time_elapsed:.1f} seconds")
                    
        else:
            # No person detected - reset state
            if self.person_first_seen is not None or self.person_confirmed:
                print("❌ Person lost - resetting detection state")
            
            self.person_first_seen = None
            self.person_confirmed = False
            self.current_person_detections = []

    def draw_person_detections(self, frame):
        """
        Draw person detection boxes and status on the frame
        
        Args:
            frame: Input frame
            
        Returns:
            frame: Frame with drawn detections and status
        """
        current_time = time.time()
        
        # Draw person detections if confirmed
        if self.person_confirmed and self.current_person_detections:
            for detection in self.current_person_detections:
                x1, y1, x2, y2, confidence = detection
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw green box for confirmed person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person: {confidence:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        elif self.current_person_detections and not self.person_confirmed:
            # Draw orange box for person being confirmed
            for detection in self.current_person_detections:
                x1, y1, x2, y2, confidence = detection
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                time_remaining = self.consecutive_time_required - (current_time - self.person_first_seen)
                cv2.putText(frame, f"Confirming: {time_remaining:.1f}s", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Add status information at the top of the frame
        if self.person_confirmed:
            status = "Person: CONFIRMED"
            color = (0, 255, 0)
        elif self.person_first_seen is not None:
            time_elapsed = current_time - self.person_first_seen
            status = f"Person: Confirming ({time_elapsed:.1f}s)"
            color = (0, 165, 255)
        else:
            status = "Person: NOT DETECTED"
            color = (0, 0, 255)
        
        # Draw status
        cv2.putText(frame, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return frame

    def reset_person_detection_state(self):
        """Reset person detection state"""
        self.person_first_seen = None
        self.person_confirmed = False
        self.current_person_detections = []
        print("🔄 Person detection state reset")
    
    def is_person_confirmed(self):
        """Check if person is confirmed"""
        return self.person_confirmed
    
    def get_detection_info(self):
        """Get current detection information"""
        current_time = time.time()
        time_elapsed = 0
        if self.person_first_seen:
            time_elapsed = current_time - self.person_first_seen
            
        return {
            "person_confirmed": self.person_confirmed,
            "person_detected": len(self.current_person_detections) > 0,
            "time_elapsed": time_elapsed,
            "time_required": self.consecutive_time_required,
            "confidence_threshold": self.confidence_threshold
        }
