"""
Eye and Gaze Detection Module
Handles eye state detection and gaze direction tracking using MediaPipe
"""
import cv2
import numpy as np
import mediapipe as mp
import time

class EyeGazeDetector:
    def __init__(self):
        # Eye detection variables
        self.eye_detection_enabled = True
        self.eye_closed_threshold = 0.25
        self.eye_state_history = []
        self.history_length = 5
        self.consecutive_closed_frames = 0
        self.frames_for_drowsiness = 90  # Assuming 30 FPS, this is 3 seconds
        self.last_eye_state = "Open"
        self.blink_count = 0
        self.last_eye_tts_time = 0
        self.eye_tts_cooldown = 4
        self.eye_closure_start_time = None  # Track when eyes first closed
        self.continuous_closure_threshold = 3.0  # 3 seconds threshold
        self.tts_triggered = False  # Track if TTS has been triggered for current closure
        
        # Gaze/Head direction tracking
        self.gaze_threshold = 0.02  # Threshold for direction detection
        self.gaze_direction_start_time = None  # Track when direction change started
        self.continuous_direction_threshold = 3.0  # 3 seconds threshold for head direction
        self.last_gaze_direction = "straight"
        self.gaze_tts_triggered = False  # Track if TTS has been triggered for current direction
        
        # MediaPipe initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,  # Only track one face for better performance
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Eye landmark indices
        self.LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
        
        # Face gaze/distraction detection variables
        self.face_distraction_enabled = True  # Ensure this is enabled by default
        self.gaze_direction = "straight"
        self.gaze_distraction_start_time = None
        self.gaze_distraction_threshold = 3.0  # 3 seconds threshold
        self.gaze_duration_timer = 0
        self.last_gaze_tts_time = 0
        self.gaze_tts_cooldown = 3
        self.tts_triggered_for_current_gaze = False
    
    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio (EAR)"""
        # Vertical distances
        vertical_1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
        vertical_2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
        
        # Horizontal distance
        horizontal = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
        
        # Calculate EAR
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def get_eye_landmarks(self, landmarks, eye_points, frame_width, frame_height):
        """Extract eye landmark coordinates"""
        eye_coords = []
        for point in eye_points:
            x = int(landmarks.landmark[point].x * frame_width)
            y = int(landmarks.landmark[point].y * frame_height)
            eye_coords.append((x, y))
        return eye_coords
    
    def get_gaze_direction(self, landmarks, is_flipped=True):
        """Get gaze direction based on nose and eye positions
        
        Args:
            landmarks: MediaPipe face landmarks
            is_flipped: Whether the frame is horizontally flipped (webcam mirror mode)
        """
        try:
            nose = landmarks[1]  # Nose tip
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            
            dx = left_eye.x - right_eye.x
            dy = left_eye.y - right_eye.y
            center_x = (left_eye.x + right_eye.x) / 2
            
            if nose.x < center_x - self.gaze_threshold:
                # In flipped mode, this corresponds to looking left (user's perspective)
                return "left" if is_flipped else "right"
            elif nose.x > center_x + self.gaze_threshold:
                # In flipped mode, this corresponds to looking right (user's perspective)
                return "right" if is_flipped else "left"
            else:
                return "straight"
        except Exception as e:
            print(f"Error in gaze direction detection: {e}")
            return "unknown"
    
    def track_gaze_duration(self, current_direction):
        """Track how long the user has been looking in the current direction"""
        current_time = time.time()
        
        # Check if direction has changed
        if current_direction != self.gaze_direction:
            self.gaze_direction = current_direction
            self.gaze_distraction_start_time = current_time
            self.gaze_duration_timer = 0
            self.tts_triggered_for_current_gaze = False
        else:
            # Same direction, update timer
            if self.gaze_distraction_start_time:
                self.gaze_duration_timer = current_time - self.gaze_distraction_start_time
        
        return self.gaze_duration_timer
    
    def should_trigger_gaze_tts(self, current_direction, duration):
        """Determine if TTS should be triggered based on gaze direction and duration"""
        # Only trigger for left or right directions
        if current_direction not in ["left", "right"]:
            return False
        
        # Check if duration threshold is met
        if duration < self.gaze_distraction_threshold:
            return False
        
        # Check if TTS already triggered for this gaze session
        if self.tts_triggered_for_current_gaze:
            return False
          # Check cooldown period
        current_time = time.time()
        if current_time - self.last_gaze_tts_time < self.gaze_tts_cooldown:
            return False
        
        return True
    
    def detect(self, frame, is_flipped=True):
        """Detect eye state and calculate metrics
        
        Args:
            frame: Input video frame
            is_flipped: Whether the frame is horizontally flipped (webcam mirror mode)
        """
        if not self.eye_detection_enabled:
            return {
                'eye_state': "Unknown",
                'left_ear': 0.0,
                'right_ear': 0.0,
                'drowsiness_detected': False,
                'closure_duration': 0.0,
                'gaze_direction': "unknown",
                'gaze_detected': False,
                'gaze_duration': 0.0,
                'continuous_direction': False
            }
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            left_ear = 0.0
            right_ear = 0.0
            eye_state = "Open"
            gaze_direction = "unknown"
            gaze_duration = 0.0
            continuous_direction = False
            current_time = time.time()
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    frame_height, frame_width = frame.shape[:2]
                    
                    # Get eye landmarks
                    left_eye_coords = self.get_eye_landmarks(
                        face_landmarks, self.LEFT_EYE_POINTS, frame_width, frame_height
                    )
                    right_eye_coords = self.get_eye_landmarks(
                        face_landmarks, self.RIGHT_EYE_POINTS, frame_width, frame_height
                    )
                    
                    # Calculate EAR for both eyes
                    left_ear = self.calculate_ear(left_eye_coords)
                    right_ear = self.calculate_ear(right_eye_coords)
                    
                    # Determine eye state
                    avg_ear = (left_ear + right_ear) / 2.0
                    eye_state = "Closed" if avg_ear < self.eye_closed_threshold else "Open"
                      # Get gaze direction
                    if self.face_distraction_enabled:
                        gaze_direction = self.get_gaze_direction(face_landmarks.landmark, is_flipped)
                        
                        # Track gaze duration
                        if gaze_direction in ["left", "right"]:
                            if self.gaze_direction_start_time is None or gaze_direction != self.last_gaze_direction:
                                self.gaze_direction_start_time = current_time
                                self.gaze_tts_triggered = False
                            gaze_duration = current_time - self.gaze_direction_start_time
                            continuous_direction = gaze_duration >= self.gaze_distraction_threshold
                        else:
                            self.gaze_direction_start_time = None
                            self.gaze_tts_triggered = False
                            gaze_duration = 0
                            continuous_direction = False
                        
                        self.last_gaze_direction = gaze_direction
                    
                    # Draw eye landmarks if needed
                    for coord in left_eye_coords + right_eye_coords:
                        cv2.circle(frame, coord, 2, (0, 255, 0), -1)
                    
                    break  # Only process first face
            
            # Update eye state history
            self.eye_state_history.append(eye_state)
            if len(self.eye_state_history) > self.history_length:
                self.eye_state_history.pop(0)
            
            # Track continuous eye closure
            closure_duration = 0
            if eye_state == "Closed":
                if self.eye_closure_start_time is None:
                    self.eye_closure_start_time = current_time
                self.consecutive_closed_frames += 1
                closure_duration = current_time - self.eye_closure_start_time
            else:
                self.eye_closure_start_time = None
                self.consecutive_closed_frames = 0
                self.tts_triggered = False
            
            # Get smoothed state
            smoothed_state = self.get_smoothed_eye_state()
            
            self.last_eye_state = eye_state
            
            return {
                'eye_state': smoothed_state,
                'left_ear': left_ear,
                'right_ear': right_ear,
                'drowsiness_detected': closure_duration >= self.continuous_closure_threshold,
                'closure_duration': closure_duration,
                'gaze_direction': gaze_direction,
                'gaze_detected': gaze_direction != "unknown",
                'gaze_duration': gaze_duration,
                'continuous_direction': continuous_direction
            }
            
        except Exception as e:
            print(f"Error in eye state detection: {e}")
            return {
                'eye_state': "Error",
                'left_ear': 0.0,
                'right_ear': 0.0,
                'drowsiness_detected': False,
                'closure_duration': 0.0,
                'gaze_direction': "unknown",
                'gaze_detected': False,
                'gaze_duration': 0.0,
                'continuous_direction': False
            }
    
    def get_smoothed_eye_state(self):
        """Get smoothed eye state based on recent history"""
        if len(self.eye_state_history) == 0:
            return "Open"
        
        closed_count = self.eye_state_history.count("Closed")
        return "Closed" if closed_count > len(self.eye_state_history) // 2 else "Open"
    
    def reset_stats(self):
        """Reset eye detection statistics"""
        self.blink_count = 0
        self.consecutive_closed_frames = 0
        self.eye_state_history = []
        self.eye_closure_start_time = None
        self.tts_triggered = False
    
    def adjust_eye_threshold(self, increase=True):
        """Adjust eye closed threshold"""
        if increase:
            self.eye_closed_threshold = min(0.4, self.eye_closed_threshold + 0.01)
        else:
            self.eye_closed_threshold = max(0.15, self.eye_closed_threshold - 0.01)
        print(f"Eye threshold adjusted to: {self.eye_closed_threshold:.3f}")
    
    def adjust_gaze_timing_threshold(self, increase=True):
        """Adjust the gaze duration threshold"""
        if increase:
            self.gaze_distraction_threshold = min(10.0, self.gaze_distraction_threshold + 0.5)
        else:
            self.gaze_distraction_threshold = max(1.0, self.gaze_distraction_threshold - 0.5)
        print(f"Gaze duration threshold adjusted to: {self.gaze_distraction_threshold:.1f}s")
    
    def adjust_gaze_sensitivity(self, increase=True):
        """Adjust the gaze detection sensitivity"""
        if increase:
            self.gaze_threshold = min(0.05, self.gaze_threshold + 0.005)
        else:
            self.gaze_threshold = max(0.01, self.gaze_threshold - 0.005)
        print(f"Gaze sensitivity threshold adjusted to: {self.gaze_threshold:.3f}")
    
    def toggle_eye_detection(self):
        """Toggle eye detection"""
        self.eye_detection_enabled = not self.eye_detection_enabled
        status = "ON" if self.eye_detection_enabled else "OFF"
        print(f"Eye detection: {status}")
        return status
    
    def toggle_gaze_detection(self):
        """Toggle face gaze detection"""
        self.face_distraction_enabled = not self.face_distraction_enabled
        status = "ON" if self.face_distraction_enabled else "OFF"
        print(f"Face gaze detection: {status}")
        return status
    
    def get_detection_info(self):
        """Get current detection information"""
        return {
            "eye_detection_enabled": self.eye_detection_enabled,
            "face_distraction_enabled": self.face_distraction_enabled,
            "eye_closed_threshold": self.eye_closed_threshold,
            "gaze_threshold": self.gaze_threshold,
            "gaze_duration_threshold": self.gaze_distraction_threshold,
            "blink_count": self.blink_count,
            "consecutive_closed_frames": self.consecutive_closed_frames,
            "current_gaze_direction": self.gaze_direction,
            "gaze_duration": self.gaze_duration_timer
        }
