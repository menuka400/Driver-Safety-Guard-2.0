"""
Face Detection Module
Handles face detection using YOLO models
"""
import torch
import numpy as np
from ultralytics import YOLO

class FaceDetector:
    def __init__(self, model_path, device='cuda', confidence_threshold=0.4):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load YOLO model for face detection"""
        try:
            print("Loading YOLOv11l for face detection...")
            self.model = YOLO(model_path)
            if self.device == 'cuda':
                self.model.to(self.device)
            print(f"✅ Face detection model loaded on {self.device.upper()}")
        except Exception as e:
            print(f"❌ Error loading face detection model: {e}")
            raise
    
    def detect_faces_gpu(self, frame):
        """GPU-optimized face detection using YOLOv11l"""
        try:
            with torch.no_grad():
                results = self.model(frame, verbose=False, conf=self.confidence_threshold, device=self.device)
            
            face_detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for i, (box, confidence) in enumerate(zip(boxes, confidences)):
                        if confidence >= self.confidence_threshold:
                            x1, y1, x2, y2 = box
                            face_detections.append([x1, y1, x2, y2, confidence])
            
            return face_detections
            
        except Exception as e:
            print(f"Error in GPU face detection: {e}")
            return []
    
    def update_confidence_threshold(self, new_threshold):
        """Update confidence threshold"""
        self.confidence_threshold = max(0.1, min(1.0, new_threshold))
        print(f"Face detection confidence threshold updated to: {self.confidence_threshold:.2f}")
    
    def get_detection_info(self):
        """Get current detection information"""
        return {
            "confidence_threshold": self.confidence_threshold
        }
