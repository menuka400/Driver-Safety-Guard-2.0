"""
Main Tracker Module
Integrates all detection, tracking, and communication modules
"""
import cv2
import time
import numpy as np
import torch
import os

# Import all our custom modules
from core.gpu_manager import GPUManager
from core.config_manager import ConfigManager
from detection.phone_detector import PhoneDetector
from detection.face_detector import FaceDetector
from detection.person_detector import PersonDetector
from detection.eye_gaze_detector import EyeGazeDetector
from tracking.object_tracker import ObjectTracker
from tracking.feature_extractor import FeatureExtractor
from communication.esp32_communicator import ESP32Communicator
from communication.tts_manager import TTSManager

class FullyGPUOptimizedTracker:
    def __init__(self, yolo_model_path, face_model_path, hrnetv2_model_path=None, config_path=None):
        """
        Fully GPU-Optimized Mobile Phone and Face Tracking System
        
        Args:
            yolo_model_path: Path to YOLO11x model for phone detection
            face_model_path: Path to YOLOv11l face detection model
            hrnetv2_model_path: Path to HRNetV2 model (optional)
            config_path: Path to configuration file (optional)
        """
        print("🚀 Initializing Fully GPU-Optimized Tracker...")
        
        # Initialize configuration manager
        self.config = ConfigManager()
        if config_path:
            self.config.load_config(config_path)
        
        # Initialize GPU manager
        self.gpu_manager = GPUManager()
        
        # Initialize communication components
        esp32_config = self.config.get("esp32", {})
        self.esp32_comm = ESP32Communicator(
            ip=esp32_config.get("ip_address", "192.168.1.121"),
            port=esp32_config.get("port", 80)
        )
        
        self.tts_manager = TTSManager()
        
        # Initialize detection components
        self.phone_detector = PhoneDetector(
            yolo_model_path, 
            self.gpu_manager.device,
            self.config.get("thresholds.phone_confidence")
        )
        
        self.face_detector = FaceDetector(
            face_model_path,
            self.gpu_manager.device,
            self.config.get("thresholds.face_confidence")
        )
        
        # Person detector uses the same model as phone detector (YOLO)
        self.person_detector = PersonDetector(
            self.phone_detector.model,
            self.gpu_manager.device,
            self.config.get("thresholds.person_confidence")
        )
        
        self.eye_gaze_detector = EyeGazeDetector()
        
        # Initialize tracking components
        tracking_config = self.config.get("tracking")
        self.object_tracker = ObjectTracker(
            feature_dim=tracking_config["feature_dim"],
            max_disappeared=tracking_config["max_disappeared"],
            max_distance=tracking_config["max_distance"]
        )
        
        self.feature_extractor = FeatureExtractor(
            device=self.gpu_manager.device,
            feature_dim=tracking_config["feature_dim"],
            hrnetv2_model_path=hrnetv2_model_path
        )
          # Timing and state management
        self.last_eye_tts_time = 0
        self.last_gaze_tts_time = 0
        self.last_phone_tts_time = 0
        
        print("✅ Fully GPU-Optimized Tracker initialized successfully!")
    
    def process_detections_gpu(self, frame, is_flipped=False):
        """Process all detections with GPU acceleration
        
        Args:
            frame: Input video frame
            is_flipped: Whether the frame is horizontally flipped (for webcam mirror mode)
        """
        frame_start = time.time()
        
        # Preprocess frame on GPU
        processed_frame = self.gpu_manager.preprocess_frame_gpu(frame)
        
        # Detect person first (required for other detections)
        person_detections = self.person_detector.detect_persons_in_frame(processed_frame)
        self.person_detector.update_person_detection_state(person_detections)
        
        # Only proceed with other detections if person is confirmed
        phone_detections = []
        face_detections = []
        phone_features = []
        face_features = []
        eye_state = "Unknown"
        left_ear = 0
        right_ear = 0
        drowsiness_detected = False
        gaze_direction = "straight"
        gaze_detected = False
        
        if self.person_detector.is_person_confirmed():
            # Run phone and face detection in parallel
            phone_results = self.phone_detector.detect(processed_frame)
            phone_detections = [det['bbox'] + [det['confidence']] for det in phone_results['detections']]
            face_detections = self.face_detector.detect_faces_gpu(processed_frame)
              # Process eye detection and gaze tracking
            eye_results = self.eye_gaze_detector.detect(processed_frame, is_flipped)
            eye_state = eye_results['eye_state']
            left_ear = eye_results['left_ear']
            right_ear = eye_results['right_ear']
            drowsiness_detected = eye_results['drowsiness_detected']
            gaze_direction = eye_results['gaze_direction']
            gaze_detected = eye_results['gaze_detected']
            
            # Extract features for tracking
            if phone_detections:
                phone_features = self.feature_extractor.extract_features_gpu_batch(
                    processed_frame, phone_detections
                )
            
            if face_detections:
                face_features = self.feature_extractor.extract_features_gpu_batch(
                    processed_frame, face_detections
                )
        
        detection_time = time.time() - frame_start
        
        return {
            'frame': processed_frame,
            'person_detections': person_detections,
            'person_confirmed': self.person_detector.is_person_confirmed(),
            'phone_detections': phone_detections,
            'face_detections': face_detections,
            'phone_features': phone_features,
            'face_features': face_features,
            'eye_state': eye_state,
            'left_ear': left_ear,
            'right_ear': right_ear,
            'drowsiness_detected': drowsiness_detected,
            'gaze_direction': gaze_direction,
            'gaze_detected': gaze_detected,
            'detection_time': detection_time
        }
    
    def process_tracking(self, results):
        """Process object tracking"""
        if not results['person_confirmed']:
            return results
        
        # Track phones
        phone_track_ids = self.object_tracker.track_phones(
            results['phone_detections'], 
            results['phone_features']
        )
        
        # Track faces
        face_track_ids = self.object_tracker.track_faces(
            results['face_detections'],
            results['face_features']
        )
        
        results['phone_track_ids'] = phone_track_ids
        results['face_track_ids'] = face_track_ids
        
        return results
    
    def process_tts_feedback(self, results):
        """Process TTS feedback and ESP32 alerts based on detection results"""
        if not results['person_confirmed']:
            return
        
        # Process eye alerts
        eye_results = {
            'eye_state': results['eye_state'],
            'drowsiness_detected': results['drowsiness_detected'],
            'left_ear': results['left_ear'],
            'right_ear': results['right_ear']
        }
        
        # Process phone alerts
        phone_results = self.phone_detector.detect(results['frame'])
        
        # Process gaze alerts
        gaze_results = {
            'direction': results['gaze_direction'],
            'detected': results['gaze_detected'],
            'duration': self.eye_gaze_detector.gaze_duration_timer if hasattr(self.eye_gaze_detector, 'gaze_duration_timer') else 0,
            'continuous_direction': results.get('continuous_direction', False)
        }
        
        # Handle TTS alerts
        self.tts_manager.handle_eye_alert(eye_results)
        self.tts_manager.handle_phone_alert(phone_results)
        self.tts_manager.handle_gaze_alert(
            gaze_results['direction'], 
            gaze_results['duration'],
            gaze_results['continuous_direction']
        )
        
        # Handle ESP32 alerts
        self.esp32_comm.handle_detection_alerts(eye_results, phone_results, gaze_results)
    
    def draw_enhanced_visualizations(self, frame, results):
        """Draw enhanced visualizations on frame"""
        # Draw person detection status first
        frame = self.person_detector.draw_person_detections(frame)
        
        if not results['person_confirmed']:
            return frame
        
        # Draw phone detections with tracking
        for i, detection in enumerate(results['phone_detections']):
            x1, y1, x2, y2, confidence = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get color for this track
            color_idx = i % len(self.object_tracker.phone_colors)
            color = self.object_tracker.phone_colors[color_idx]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"Phone: {confidence:.2f}"
            if i < len(results.get('phone_track_ids', [])):
                track_id = results['phone_track_ids'][i]
                label = f"Phone {track_id}: {confidence:.2f}"
            
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw face detections with tracking
        for i, detection in enumerate(results['face_detections']):
            x1, y1, x2, y2, confidence = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get color for this track
            color_idx = i % len(self.object_tracker.face_colors)
            color = self.object_tracker.face_colors[color_idx]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"Face: {confidence:.2f}"
            if i < len(results.get('face_track_ids', [])):
                track_id = results['face_track_ids'][i]
                label = f"Face {track_id}: {confidence:.2f}"
            
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw eye state information
        if results['eye_state'] not in ["Error", "Unknown"]:
            eye_text = f"Eyes: {results['eye_state']}"
            if results['drowsiness_detected']:
                eye_text += " (DROWSY!)"
                eye_color = (0, 0, 255)  # Red
            else:
                eye_color = (0, 255, 0) if results['eye_state'] == "Open" else (0, 255, 255)
            
            cv2.putText(frame, eye_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, eye_color, 2)
            
            # Draw EAR values
            ear_text = f"EAR: L={results['left_ear']:.3f} R={results['right_ear']:.3f}"
            cv2.putText(frame, ear_text, (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw gaze information with timing
        if results['gaze_detected'] and self.eye_gaze_detector.face_distraction_enabled:
            gaze_text = f"Gaze: {results['gaze_direction']}"
            gaze_duration = self.eye_gaze_detector.gaze_duration_timer
            
            if gaze_duration > 0:
                gaze_text += f" ({gaze_duration:.1f}s)"
            
            gaze_color = (0, 255, 0) if results['gaze_direction'] == "straight" else (0, 165, 255)
            if gaze_duration >= self.eye_gaze_detector.gaze_distraction_threshold:
                gaze_color = (0, 0, 255)  # Red for distraction
            
            cv2.putText(frame, gaze_text, (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, gaze_color, 2)
        
        return frame
    
    def track_video_full_gpu(self, video_path=None, output_path=None):
        """Main GPU-accelerated tracking function"""
        # Open video capture
        if video_path is None:
            cap = cv2.VideoCapture(0)  # Webcam
        else:
            cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("❌ Error: Could not open video source")
            return
        
        # Set optimal camera properties
        video_config = self.config.get("video")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_config["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_config["height"])
        cap.set(cv2.CAP_PROP_FPS, video_config["fps"])
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or video_config["fps"]
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"🎥 Video Configuration:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Device: {self.gpu_manager.device.upper()}")
        
        # Video writer for output
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        total_detection_time = 0
        
        print("\n🚀 Starting Fully GPU-Accelerated Tracking...")
        print("🎯 Features: Person Detection + Phone Detection + Face Detection + Eye Tracking + Gaze Direction")
        print("\nControls:")
        print("  'q' - Quit")
        print("  'f' - Toggle face gaze detection")
        print("  'e' - Toggle eye detection")
        print("  'g' - Show GPU stats")
        print("  '+' - Increase eye threshold")
        print("  '-' - Decrease eye threshold")
        print("  'r' - Reset all detection states")
        print("  'c' - Clear GPU cache")
        print("  't' - Increase gaze timing threshold")
        print("  'y' - Decrease gaze timing threshold")
        print("  'u' - Increase gaze sensitivity")
        print("  'i' - Decrease gaze sensitivity")
        print("  'p' - Show current gaze debug info")
        print("  's' - Save current configuration")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                  # Mirror frame for better user experience (webcam only)
                is_flipped = video_path is None
                if is_flipped:
                    frame = cv2.flip(frame, 1)
                
                # Process detections
                results = self.process_detections_gpu(frame, is_flipped)
                total_detection_time += results['detection_time']
                
                # Process tracking
                results = self.process_tracking(results)
                
                # Process TTS and ESP32 alerts
                self.process_tts_feedback(results)
                
                # Draw visualizations
                display_frame = self.draw_enhanced_visualizations(frame, results)
                
                # Add performance info
                avg_fps = frame_count / (time.time() - start_time)
                avg_detection_time = total_detection_time / frame_count * 1000
                
                cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (width-150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Det: {avg_detection_time:.1f}ms", (width-150, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow('GPU-Optimized Object Tracking', display_frame)
                
                # Write to output video
                if out:
                    out.write(display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):
                    self.eye_gaze_detector.toggle_gaze_detection()
                elif key == ord('e'):
                    self.eye_gaze_detector.toggle_eye_detection()
                elif key == ord('g'):
                    self.gpu_manager.print_gpu_stats()
                elif key == ord('+'):
                    self.eye_gaze_detector.adjust_eye_threshold(increase=True)
                elif key == ord('-'):
                    self.eye_gaze_detector.adjust_eye_threshold(increase=False)
                elif key == ord('r'):
                    self.reset_all_states()
                elif key == ord('c'):
                    self.gpu_manager.clear_gpu_cache()
                elif key == ord('t'):
                    self.eye_gaze_detector.adjust_gaze_timing_threshold(increase=True)
                elif key == ord('y'):
                    self.eye_gaze_detector.adjust_gaze_timing_threshold(increase=False)
                elif key == ord('u'):
                    self.eye_gaze_detector.adjust_gaze_sensitivity(increase=True)
                elif key == ord('i'):
                    self.eye_gaze_detector.adjust_gaze_sensitivity(increase=False)
                elif key == ord('p'):
                    self.print_gaze_debug_info()
                elif key == ord('s'):
                    self.config.save_config("config/current_config.json")
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        
        except Exception as e:
            print(f"❌ Error during tracking: {e}")
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            # Final statistics
            total_time = time.time() - start_time
            print(f"\n📊 Final Statistics:")
            print(f"   Total frames: {frame_count}")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Average FPS: {frame_count/total_time:.2f}")
            print(f"   Average detection time: {total_detection_time/frame_count*1000:.2f}ms")
    
    def reset_all_states(self):
        """Reset all detection and tracking states"""
        self.person_detector.reset_person_detection_state()
        self.phone_detector.reset_detection_state()
        self.eye_gaze_detector.reset_eye_stats()
        self.object_tracker.reset_tracking()
        print("🔄 All states reset")
    
    def print_gaze_debug_info(self):
        """Print current gaze detection debug information"""
        info = self.eye_gaze_detector.get_detection_info()
        print("\n👀 Gaze Debug Information:")
        print(f"   Current direction: {info['current_gaze_direction']}")
        print(f"   Duration: {info['gaze_duration']:.1f}s")
        print(f"   Threshold: {info['gaze_duration_threshold']:.1f}s")
        print(f"   Sensitivity: {info['gaze_threshold']:.3f}")
        print(f"   Eye detection enabled: {info['eye_detection_enabled']}")
        print(f"   Gaze detection enabled: {info['face_distraction_enabled']}")
        print(f"   Blink count: {info['blink_count']}")
    
    def get_system_status(self):
        """Get comprehensive system status"""
        return {
            "gpu": self.gpu_manager.device,
            "person_detector": self.person_detector.get_detection_info(),
            "phone_detector": self.phone_detector.get_detection_info(),
            "face_detector": self.face_detector.get_detection_info(),
            "eye_gaze_detector": self.eye_gaze_detector.get_detection_info(),
            "object_tracker": self.object_tracker.get_tracking_info(),
            "esp32_comm": self.esp32_comm.get_status(),
            "tts_available": self.tts_manager.is_available(),
            "feature_extractor_available": self.feature_extractor.is_available()
        }
