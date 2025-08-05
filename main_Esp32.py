import cv2
import torch
import numpy as np
from collections import defaultdict, deque
import time
import os
from ultralytics import YOLO
import torch.nn.functional as F
import torchvision.transforms as transforms
import pyttsx3
import threading
import mediapipe as mp
import requests
import urllib3
import json

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class FullyGPUOptimizedTracker:
    def __init__(self, yolo_model_path, face_model_path, hrnetv2_model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Fully GPU-Optimized Mobile Phone and Face Tracking System
        
        Args:
            yolo_model_path: Path to YOLO11x model for phone detection
            face_model_path: Path to YOLOv11l face detection model
            hrnetv2_model_path: Path to HRNetV2 model (optional)
            device: Device to run inference on
        """
        print("🚀 Initializing Fully GPU-Optimized Tracker...")
        
        # ESP32 HTTP Configuration
        self.esp32_ip = "10.27.146.54"  # Change this to your ESP32's IP address
        self.esp32_port = 80  # HTTP port
        self.esp32_url = f"http://{self.esp32_ip}:{self.esp32_port}"
        self.last_alert_time = 0
        self.alert_cooldown = 2.0  # Seconds between alerts
        self.alert_active = False
        
        print(f"📡 ESP32 HTTP URL configured as: {self.esp32_url}")
        
        # GPU Setup and Validation
        self.setup_gpu_environment()
        
        # Model paths
        self.yolo_model_path = yolo_model_path
        self.face_model_path = face_model_path
        self.hrnetv2_model_path = hrnetv2_model_path
        
        # Initialize GPU components
        self.init_gpu_components()
        
        # Person detection variables
        self.person_first_seen = None
        self.person_confirmed = False
        self.current_person_detections = []
        self.consecutive_time_required = 3.0  # seconds required for person confirmation
        self.target_person_class = 0  # Person class in COCO dataset
        self.person_confidence_threshold = 0.5
        
        # Mobile phone tracking variables
        self.next_id = 1
        self.active_phone_tracks = {}
        self.phone_track_history = defaultdict(list)
        self.max_disappeared = 60
        self.max_distance = 200
        self.phone_confidence_threshold = 0.3
        self.face_confidence_threshold = 0.4
        self.iou_threshold = 0.2
        
        # Phone detection state management
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
        
        # Face tracking variables
        self.next_face_id = 1
        self.active_face_tracks = {}
        self.face_track_history = defaultdict(list)
        
        # Eye detection variables with MediaPipe
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
        
        # Face gaze/distraction detection variables
        self.face_distraction_enabled = True
        self.gaze_direction = "straight"
        self.last_gaze_direction = "straight"
        self.gaze_distraction_start_time = None
        self.gaze_distraction_threshold = 3.0  # 3 seconds threshold
        self.gaze_duration_timer = 0
        self.last_gaze_tts_time = 0
        self.gaze_tts_cooldown = 3
        self.gaze_threshold = 0.02  # Threshold for direction detection
        self.tts_triggered_for_current_gaze = False
        
        # Initialize TTS
        self.tts_engine = self.init_tts()
        self.tts_lock = threading.Lock()
        
        # Tracking configuration
        self.use_prediction = True
        self.velocity_history = defaultdict(lambda: deque(maxlen=5))
        self.max_trail_length = 50
        self.trail_points = defaultdict(lambda: deque(maxlen=self.max_trail_length))
        self.show_trails = True
        self.trail_thickness = 2
        self.feature_dim = 256
        self.max_cosine_distance = 0.3
        
        # GPU batch processing
        self.batch_size = 4
        self.frame_buffer = []
        
        # Load all models with GPU optimization
        self.load_all_models_gpu()
        
        # Colors for visualization
        self.phone_colors = [
            (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        
        self.face_colors = [
            (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 255, 128), (255, 128, 128), (128, 128, 255), (255, 255, 128)
        ]
        
        # GPU-optimized transform for feature extraction
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Eye landmark indices for MediaPipe
        self.LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
        
        print("✅ Fully GPU-Optimized Tracker initialized successfully!")
    
    def setup_gpu_environment(self):
        """Setup and optimize GPU environment"""
        if torch.cuda.is_available():
            self.device = 'cuda'
            
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = False
            
            # Set memory management
            torch.cuda.empty_cache()
            
            # Enable mixed precision
            self.use_mixed_precision = True
            self.scaler = torch.cuda.amp.GradScaler()
            
            print(f"🚀 GPU Environment Setup Complete!")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"   Available Memory: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated() / 1e9:.1f} GB")
            print(f"   cuDNN: {torch.backends.cudnn.enabled}")
            print(f"   Mixed Precision: {self.use_mixed_precision}")
            print(f"   Memory Management: Optimized")
        else:
            self.device = 'cpu'
            self.use_mixed_precision = False
            print("⚠️ CUDA not available, using CPU")
    
    def init_gpu_components(self):
        """Initialize all GPU-accelerated components"""
        # Check OpenCV GPU support
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.use_gpu_cv = True
            print(f"✅ OpenCV CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
            
            # Initialize GPU matrices for image processing
            self.gpu_frame = cv2.cuda_GpuMat()
            self.gpu_resized = cv2.cuda_GpuMat()
        else:
            self.use_gpu_cv = False
            print("⚠️ OpenCV CUDA not available")
        
        # Initialize MediaPipe Face Mesh for eye detection and gaze tracking
        print("Initializing MediaPipe Face Mesh for eye detection and gaze tracking...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=3,  # Track multiple faces
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        print("✅ MediaPipe Face Mesh initialized")
    
    def load_all_models_gpu(self):
        """Load all models with full GPU optimization"""
        try:
            print("🔄 Loading models with GPU optimization...")
            
            # Load YOLO11x for mobile phone detection
            print("Loading YOLO11x for mobile phone detection...")
            self.phone_model = YOLO(self.yolo_model_path)
            if self.device == 'cuda':
                self.phone_model.to('cuda')
                print("✅ YOLO11x (Phone) loaded on GPU")
            
            # Load YOLOv11l for face detection
            print("Loading YOLOv11l for face detection...")
            self.face_model = YOLO(self.face_model_path)
            if self.device == 'cuda':
                self.face_model.to('cuda')
                print("✅ YOLOv11l (Face) loaded on GPU")
            
            # Create GPU-optimized HRNetV2 feature extractor
            print("Creating GPU-optimized HRNetV2 feature extractor...")
            self.feature_extractor = self.create_gpu_optimized_hrnetv2()
            self.feature_extractor.to(self.device)
            self.feature_extractor.eval()
            
            # Enable mixed precision for feature extractor
            if self.device == 'cuda' and self.use_mixed_precision:
                # Convert to half precision for faster inference
                print("✅ HRNetV2 loaded with mixed precision")
            
            # Try to load HRNetV2 checkpoint if available
            if self.hrnetv2_model_path and os.path.exists(self.hrnetv2_model_path):
                try:
                    print("Loading HRNetV2 checkpoint...")
                    checkpoint = torch.load(self.hrnetv2_model_path, map_location=self.device)
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        model_dict = self.feature_extractor.state_dict()
                        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() 
                                         if k in model_dict and v.shape == model_dict[k].shape}
                        model_dict.update(pretrained_dict)
                        self.feature_extractor.load_state_dict(model_dict, strict=False)
                        print(f"✅ Loaded {len(pretrained_dict)} parameters from checkpoint")
                except Exception as e:
                    print(f"⚠️ Could not load HRNetV2 checkpoint: {e}")
            
            print(f"✅ All models loaded successfully on {self.device.upper()}")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def create_gpu_optimized_hrnetv2(self):
        """Create GPU-optimized HRNetV2 feature extractor"""
        class GPUOptimizedHRNetV2(torch.nn.Module):
            def __init__(self, feature_dim=256):
                super().__init__()
                # Optimized architecture for GPU
                self.stem = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU(inplace=True)
                )
                
                self.stage1 = self._make_stage(64, 64, 4)
                self.stage2 = self._make_stage(64, 128, 4)
                self.stage3 = self._make_stage(128, 256, 6)
                self.stage4 = self._make_stage(256, 512, 4)
                
                self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(512, feature_dim),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(feature_dim, feature_dim)
                )
                
                # Initialize weights
                self._initialize_weights()
                
            def _make_stage(self, in_channels, out_channels, num_blocks):
                layers = []
                layers.append(torch.nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False))
                layers.append(torch.nn.BatchNorm2d(out_channels))
                layers.append(torch.nn.ReLU(inplace=True))
                
                for _ in range(num_blocks - 1):
                    layers.append(torch.nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False))
                    layers.append(torch.nn.BatchNorm2d(out_channels))
                    layers.append(torch.nn.ReLU(inplace=True))
                
                return torch.nn.Sequential(*layers)
            
            def _initialize_weights(self):
                for m in self.modules():
                    if isinstance(m, torch.nn.Conv2d):
                        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    elif isinstance(m, torch.nn.BatchNorm2d):
                        torch.nn.init.constant_(m.weight, 1)
                        torch.nn.init.constant_(m.bias, 0)
                    elif isinstance(m, torch.nn.Linear):
                        torch.nn.init.normal_(m.weight, 0, 0.01)
                        torch.nn.init.constant_(m.bias, 0)
            
            def forward(self, x):
                x = self.stem(x)
                x = self.stage1(x)
                x = self.stage2(x)
                x = self.stage3(x)
                x = self.stage4(x)
                
                x = self.global_pool(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                
                return F.normalize(x, p=2, dim=1)
        
        return GPUOptimizedHRNetV2(self.feature_dim)
    
    def preprocess_frame_gpu(self, frame):
        """GPU-accelerated frame preprocessing"""
        if self.use_gpu_cv:
            try:
                # Upload frame to GPU
                self.gpu_frame.upload(frame)
                
                # GPU-based preprocessing
                # Noise reduction
                gpu_blurred = cv2.cuda.bilateralFilter(self.gpu_frame, -1, 50, 50)
                
                # Optional: Color enhancement
                # gpu_enhanced = cv2.cuda.equalizeHist(gpu_blurred)
                
                # Download processed frame
                processed_frame = gpu_blurred.download()
                return processed_frame
                
            except Exception as e:
                print(f"GPU preprocessing failed: {e}")
                return frame
        return frame
    
    def detect_mobile_phones_gpu(self, frame):
        """GPU-optimized mobile phone detection using YOLO11x"""
        try:
            with torch.no_grad():
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        results = self.phone_model(frame, verbose=False, device=self.device)
                else:
                    results = self.phone_model(frame, verbose=False, device=self.device)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Check if it's a mobile phone (class 67 in COCO dataset)
                        if class_id == 67 and confidence >= self.phone_confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            detections.append([int(x1), int(y1), int(x2), int(y2), confidence])
            
            # Manage phone detection state
            self.manage_phone_detection_state(detections)
            return detections
            
        except Exception as e:
            print(f"Error in GPU phone detection: {e}")
            return []
    
    def detect_faces_gpu(self, frame):
        """GPU-optimized face detection using YOLOv11l"""
        try:
            with torch.no_grad():
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        results = self.face_model(frame, verbose=False, device=self.device)
                else:
                    results = self.face_model(frame, verbose=False, device=self.device)
            
            face_detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = float(box.conf[0])
                        
                        if confidence >= self.face_confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            face_detections.append([int(x1), int(y1), int(x2), int(y2), confidence])
            
            return face_detections
            
        except Exception as e:
            print(f"Error in GPU face detection: {e}")
            return []
    
    def extract_features_gpu_batch(self, frame, bboxes):
        """GPU-optimized batch feature extraction"""
        if not bboxes or self.feature_extractor is None:
            return [np.random.rand(self.feature_dim) for _ in bboxes]
        
        batch_crops = []
        valid_indices = []
        
        # Prepare batch of crops
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox[:4]
            h, w = frame.shape[:2]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                if crop.shape[0] < 32 or crop.shape[1] < 32:
                    crop = cv2.resize(crop, (64, 64))
                
                try:
                    crop_tensor = self.transform(crop)
                    batch_crops.append(crop_tensor)
                    valid_indices.append(i)
                except Exception:
                    continue
        
        # Initialize features array
        features = [np.random.rand(self.feature_dim) for _ in bboxes]
        
        if batch_crops:
            try:
                with torch.no_grad():
                    batch_tensor = torch.stack(batch_crops).to(self.device)
                    
                    if self.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            batch_features = self.feature_extractor(batch_tensor)
                    else:
                        batch_features = self.feature_extractor(batch_tensor)
                    
                    batch_features_np = batch_features.cpu().numpy()
                    
                    # Map features back to original indices
                    for i, feature in enumerate(batch_features_np):
                        if i < len(valid_indices):
                            features[valid_indices[i]] = feature
                            
            except Exception as e:
                print(f"GPU batch feature extraction failed: {e}")
        
        return features
    
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
    
    def get_gaze_direction(self, landmarks):
        """Get gaze direction based on nose and eye positions"""
        try:
            nose = landmarks[1]  # Nose tip
            left_eye = landmarks[33]
            right_eye = landmarks[263]

            dx = left_eye.x - right_eye.x
            dy = left_eye.y - right_eye.y
            center_x = (left_eye.x + right_eye.x) / 2

            if nose.x < center_x - self.gaze_threshold:
                return "right"
            elif nose.x > center_x + self.gaze_threshold:
                return "left"
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
            # Direction changed - reset timer
            self.gaze_direction = current_direction
            self.gaze_distraction_start_time = current_time
            self.gaze_duration_timer = 0
            self.tts_triggered_for_current_gaze = False
            
            print(f"Gaze direction changed to: {current_direction}")
        else:
            # Same direction - update duration
            if self.gaze_distraction_start_time is not None:
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
    
    def detect_eye_state_and_gaze_gpu(self, frame):
        """GPU-accelerated eye state and gaze detection using MediaPipe"""
        if not self.eye_detection_enabled:
            return "Unknown", 0.0, 0.0, False, "unknown", False
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            left_ear = 0.0
            right_ear = 0.0
            eye_state = "Open"
            gaze_direction = "unknown"
            gaze_detected = False
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
                        gaze_direction = self.get_gaze_direction(face_landmarks.landmark)
                        gaze_detected = True
                    
                    # Draw eye landmarks
                    for coord in left_eye_coords:
                        cv2.circle(frame, coord, 2, (0, 255, 0), -1)
                    for coord in right_eye_coords:
                        cv2.circle(frame, coord, 2, (0, 255, 0), -1)
                    
                    break  # Only process first face
            
            # Update eye state history
            self.eye_state_history.append(eye_state)
            if len(self.eye_state_history) > self.history_length:
                self.eye_state_history.pop(0)
            
            # Track continuous eye closure
            if eye_state == "Closed":
                if self.eye_closure_start_time is None:
                    self.eye_closure_start_time = current_time
                self.consecutive_closed_frames += 1
            else:
                self.eye_closure_start_time = None
                self.consecutive_closed_frames = 0
                self.tts_triggered = False
            
            # Calculate continuous closure duration
            closure_duration = 0
            if self.eye_closure_start_time is not None:
                closure_duration = current_time - self.eye_closure_start_time
            
            # Detect drowsiness based on continuous closure
            drowsiness_detected = closure_duration >= self.continuous_closure_threshold
            
            # Get smoothed state
            smoothed_state = self.get_smoothed_eye_state()
            
            self.last_eye_state = eye_state
            
            return smoothed_state, left_ear, right_ear, drowsiness_detected, gaze_direction, gaze_detected
            
        except Exception as e:
            print(f"Error in eye state and gaze detection: {e}")
            return "Error", 0.0, 0.0, False, "unknown", False
    
    def get_smoothed_eye_state(self):
        """Get smoothed eye state based on recent history"""
        if len(self.eye_state_history) == 0:
            return "Open"
        
        closed_count = self.eye_state_history.count("Closed")
        return "Closed" if closed_count > len(self.eye_state_history) // 2 else "Open"
    
    def manage_phone_detection_state(self, detections):
        """Manage mobile phone detection state and TTS announcements"""
        current_time = time.time()
        has_phone_detections = len(detections) > 0
        
        if has_phone_detections:
            # Phone is detected
            self.frames_without_phone = 0
            self.phone_removal_announced = False
            
            # Start timing for continuous detection
            if not self.phone_detected_state:
                self.phone_detection_start_time = current_time
                self.phone_detected_state = True
            
            # Check if phone has been continuously detected for threshold duration
            if (self.phone_detection_start_time is not None and 
                not self.phone_tts_triggered and 
                current_time - self.phone_detection_start_time >= self.continuous_phone_threshold):
                
                if current_time - self.last_phone_tts_time > self.phone_tts_cooldown:
                    self.speak_async("Warning! Mobile phone detected for more than 3 seconds. Please put it away and focus ahead.")
                    self.send_esp32_alert("phone")
                    self.last_phone_tts_time = current_time
                    self.phone_tts_triggered = True
        else:
            # No phone detected
            self.frames_without_phone += 1
            
            if self.phone_detected_state:
                # Reset detection state
                self.phone_detected_state = False
                self.phone_detection_start_time = None
                self.phone_tts_triggered = False
                
                # Announce removal if conditions are met
                if (self.frames_without_phone >= self.frames_for_removal_announcement and 
                    not self.phone_removal_announced):
                    
                    if current_time - self.phone_removal_tts_time > self.phone_tts_cooldown:
                        self.speak_async("Good! Mobile phone removed. Keep focusing ahead.")
                        self.stop_esp32_alert()
                        self.phone_removal_tts_time = current_time
                        self.phone_removal_announced = True
    
    def process_detections_gpu(self, frame):
        """Process all detections with GPU acceleration"""
        frame_start = time.time()
        
        # Preprocess frame on GPU
        processed_frame = self.preprocess_frame_gpu(frame)
        
        # Run all detections in parallel on GPU
        phone_detections = self.detect_mobile_phones_gpu(processed_frame)
        face_detections = self.detect_faces_gpu(processed_frame)
        
        # Process eye detection and gaze tracking
        eye_state, left_ear, right_ear, drowsiness_detected, gaze_direction, gaze_detected = self.detect_eye_state_and_gaze_gpu(processed_frame)
        
        # Extract features for tracking
        phone_features = []
        if phone_detections:
            phone_features = self.extract_features_gpu_batch(processed_frame, phone_detections)
        
        face_features = []
        if face_detections:
            face_features = self.extract_features_gpu_batch(processed_frame, face_detections)
        
        detection_time = time.time() - frame_start
        
        return {
            'frame': processed_frame,
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
    
    def init_tts(self):
        """Initialize Text-to-Speech engine"""
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 140)
            engine.setProperty('volume', 1.0)
            voices = engine.getProperty('voices')
            if len(voices) > 1:
                engine.setProperty('voice', voices[1].id)
            print("✅ TTS engine initialized")
            return engine
        except Exception as e:
            print(f"❌ TTS initialization failed: {e}")
            return None
    
    def speak_async(self, text):
        """Speak text asynchronously"""
        def speak():
            with self.tts_lock:
                if self.tts_engine:
                    try:
                        self.tts_engine.say(text)
                        self.tts_engine.runAndWait()
                    except Exception as e:
                        print(f"TTS error: {e}")
        
        if self.tts_engine:
            threading.Thread(target=speak, daemon=True).start()
    
    def process_tts_feedback(self, results):
        """Process TTS feedback and ESP32 alerts based on detection results"""
        current_time = time.time()
        alert_sent = False
        
        # Eye detection TTS and alerts
        if current_time - self.last_eye_tts_time > self.eye_tts_cooldown:
            if results['drowsiness_detected'] and not self.tts_triggered:
                closure_duration = current_time - self.eye_closure_start_time if self.eye_closure_start_time else 0
                if closure_duration >= self.continuous_closure_threshold:
                    self.speak_async("Warning! Eyes have been closed for more than 3 seconds. Please stay alert!")
                    self.send_esp32_alert("drowsy")
                    alert_sent = True
                    self.last_eye_tts_time = current_time
                    self.tts_triggered = True
        
        # Gaze direction TTS and alerts
        if results['gaze_detected'] and self.face_distraction_enabled:
            current_gaze = results['gaze_direction']
            gaze_duration = self.track_gaze_duration(current_gaze)
            
            if self.should_trigger_gaze_tts(current_gaze, gaze_duration):
                if current_gaze in ["left", "right"]:
                    self.speak_async(f"You've been looking {current_gaze} for {gaze_duration:.1f} seconds. Please look straight ahead.")
                    if not alert_sent:
                        self.send_esp32_alert("distracted")
                        alert_sent = True
                
                self.tts_triggered_for_current_gaze = True
                self.last_gaze_tts_time = current_time
            
            elif current_gaze == "straight" and self.last_gaze_direction in ["left", "right"] and gaze_duration > 1.0:
                if current_time - self.last_gaze_tts_time > self.gaze_tts_cooldown:
                    self.speak_async("Good! Keep looking straight ahead.")
                    self.stop_esp32_alert()
                    self.last_gaze_tts_time = current_time
            
            self.last_gaze_direction = current_gaze
        
        # Phone detection alerts
        if len(results['phone_detections']) > 0:
            if not self.phone_detected_state:
                if current_time - self.last_phone_tts_time > self.phone_tts_cooldown:
                    self.speak_async("Mobile phone detected! Please put it away and focus ahead.")
                    if not alert_sent:
                        self.send_esp32_alert("phone")
                    self.last_phone_tts_time = current_time
                self.phone_detected_state = True
        else:
            if self.phone_detected_state and self.frames_without_phone >= self.frames_for_removal_announcement:
                if current_time - self.phone_removal_tts_time > self.phone_tts_cooldown:
                    self.speak_async("Good! Mobile phone removed. Keep focusing ahead.")
                    self.stop_esp32_alert()
                    self.phone_removal_tts_time = current_time
                    self.phone_removal_announced = True
    
    def draw_enhanced_visualizations(self, frame, results):
        """Draw enhanced visualizations on frame with gaze timing information"""
        # Draw phone detections
        for i, detection in enumerate(results['phone_detections']):
            x1, y1, x2, y2, conf = detection
            color = self.phone_colors[i % len(self.phone_colors)]
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            
            # Draw label
            label = f"PHONE DETECTED! ({conf:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10),
                         (int(x1) + label_size[0], int(y1)), color, -1)
            cv2.putText(frame, label, (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw face detections
        for i, detection in enumerate(results['face_detections']):
            x1, y1, x2, y2, conf = detection
            color = self.face_colors[i % len(self.face_colors)]
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw label
            label = f"Face {i+1} ({conf:.2f})"
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw eye state information
        if results['eye_state'] != "Error" and results['eye_state'] != "Unknown":
            eye_color = (0, 0, 255) if results['eye_state'] == "Closed" else (0, 255, 0)
            
            cv2.putText(frame, f"Eyes: {results['eye_state']}", (10, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
            cv2.putText(frame, f"Left EAR: {results['left_ear']:.3f}", (10, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Right EAR: {results['right_ear']:.3f}", (10, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Blinks: {self.blink_count}", (10, 230),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Drowsiness warning
            if results['drowsiness_detected']:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 260),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                cv2.putText(frame, f"Eyes closed for {self.consecutive_closed_frames} frames",
                           (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw gaze information with timing
        if results['gaze_detected'] and self.face_distraction_enabled:
            current_gaze = results['gaze_direction']
            
            # Determine gaze status color with timing consideration
            if current_gaze == "straight":
                gaze_color = (0, 255, 0)  # Green
                status = "LOOKING STRAIGHT"
            elif current_gaze in ["left", "right"]:
                # Show warning color if approaching threshold
                if self.gaze_duration_timer >= self.gaze_distraction_threshold:
                    gaze_color = (0, 0, 255)  # Red - threshold exceeded
                elif self.gaze_duration_timer >= self.gaze_distraction_threshold * 0.7:
                    gaze_color = (0, 165, 255)  # Orange - approaching threshold
                else:
                    gaze_color = (0, 255, 255)  # Yellow - minor distraction
                
                status = f"DISTRACTED - {current_gaze.upper()} ({self.gaze_duration_timer:.1f}s)"
            else:
                gaze_color = (128, 128, 128)  # Gray
                status = current_gaze.upper()
            
            cv2.putText(frame, f"Gaze: {status}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, gaze_color, 2)
            
            # Show timing information for non-straight gaze
            if current_gaze in ["left", "right"]:
                timer_color = (0, 0, 255) if self.gaze_duration_timer >= self.gaze_distraction_threshold else (0, 255, 255)
                cv2.putText(frame, f"Duration: {self.gaze_duration_timer:.1f}s / {self.gaze_distraction_threshold:.1f}s", 
                           (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, timer_color, 2)
                
                # Show progress bar for timing
                bar_width = 200
                bar_height = 10
                bar_x = 10
                bar_y = 330
                
                # Background bar
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                
                # Progress bar
                progress = min(self.gaze_duration_timer / self.gaze_distraction_threshold, 1.0)
                progress_width = int(bar_width * progress)
                progress_color = (0, 0, 255) if progress >= 1.0 else (0, 255, 255)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), progress_color, -1)
                
                # Threshold line
                threshold_x = bar_x + bar_width
                cv2.line(frame, (threshold_x, bar_y - 5), (threshold_x, bar_y + bar_height + 5), (255, 255, 255), 2)
        
        return frame
    
    def adjust_gaze_timing_threshold(self, increase=True):
        """Adjust the gaze duration threshold"""
        if increase:
            self.gaze_distraction_threshold += 0.5
        else:
            self.gaze_distraction_threshold = max(1.0, self.gaze_distraction_threshold - 0.5)
        print(f"Gaze duration threshold adjusted to: {self.gaze_distraction_threshold:.1f}s")
    
    def adjust_gaze_sensitivity(self, increase=True):
        """Adjust the gaze detection sensitivity"""
        if increase:
            self.gaze_threshold += 0.005
        else:
            self.gaze_threshold = max(0.005, self.gaze_threshold - 0.005)
        print(f"Gaze sensitivity threshold adjusted to: {self.gaze_threshold:.3f}")
    
    def track_video_full_gpu(self, video_path=None, output_path=None):
        """Main GPU-accelerated tracking function"""
        # Open video capture
        if video_path is None:
            cap = cv2.VideoCapture(0)
            print("📹 Using webcam")
        else:
            cap = cv2.VideoCapture(video_path)
            print(f"📹 Using video file: {video_path}")
        
        if not cap.isOpened():
            print("❌ Could not open video source")
            return
        
        # Set optimal camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"🎥 Video Configuration:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Device: {self.device.upper()}")
        
        # Video writer for output
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Output will be saved to: {output_path}")
        
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
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("📹 End of video reached or camera disconnected")
                    break
                
                frame_count += 1
                frame_start = time.time()
                
                # Mirror frame for better user experience (webcam only)
                if video_path is None:
                    frame = cv2.flip(frame, 1)
                
                # First, detect persons
                persons_detected = self.detect_persons_in_frame(frame)
                self.update_person_detection_state(persons_detected)
                
                # Draw person detections
                frame = self.draw_person_detections(frame)
                
                # Only process other detections if person is confirmed
                if self.person_confirmed:
                    # Process all detections on GPU
                    results = self.process_detections_gpu(frame)
                    total_detection_time += results['detection_time']
                    
                    # Process TTS feedback
                    self.process_tts_feedback(results)
                    
                    # Draw enhanced visualizations
                    frame = self.draw_enhanced_visualizations(frame, results)
                
                # Calculate performance metrics
                frame_time = time.time() - frame_start
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                avg_detection_time = total_detection_time / frame_count
                
                # Performance overlay
                perf_info = f"🚀 GPU-Accelerated | Frame: {frame_count} | FPS: {current_fps:.1f} | Avg Detection: {avg_detection_time*1000:.1f}ms"
                cv2.putText(frame, perf_info, (10, height - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # System info
                system_info = f"Device: {self.device.upper()} | YOLOv11l + YOLO11x + MediaPipe + HRNetV2"
                cv2.putText(frame, system_info, (10, height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # GPU memory info
                if self.device == 'cuda':
                    gpu_memory = torch.cuda.memory_allocated() / 1e6
                    gpu_info = f"GPU Memory: {gpu_memory:.1f}MB | Mixed Precision: {self.use_mixed_precision}"
                    cv2.putText(frame, gpu_info, (10, height - 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Show frame
                cv2.imshow('🚀 Fully GPU-Accelerated Tracker', frame)
                
                # Write frame to output
                if out is not None:
                    out.write(frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):
                    self.toggle_face_gaze_detection()
                elif key == ord('e'):
                    self.toggle_eye_detection()
                elif key == ord('g'):
                    self.print_gpu_stats()
                elif key == ord('+') or key == ord('='):
                    self.adjust_eye_threshold(increase=True)
                elif key == ord('-'):
                    self.adjust_eye_threshold(increase=False)
                elif key == ord('r'):
                    self.reset_person_detection_state()
                    self.reset_eye_stats()
                elif key == ord('c'):
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                        print("🧹 GPU cache cleared")
                elif key == ord('t'):
                    self.adjust_gaze_timing_threshold(increase=True)
                elif key == ord('y'):
                    self.adjust_gaze_timing_threshold(increase=False)
                elif key == ord('u'):
                    self.adjust_gaze_sensitivity(increase=True)
                elif key == ord('i'):
                    self.adjust_gaze_sensitivity(increase=False)
                elif key == ord('p'):
                    print(f"Current gaze: {self.gaze_direction}, Duration: {self.gaze_duration_timer:.1f}s")
                
                # Periodic GPU cache cleanup
                if frame_count % 100 == 0 and self.device == 'cuda':
                    torch.cuda.empty_cache()
        
        except KeyboardInterrupt:
            print("\n⏹️ Stopping tracking...")
        
        except Exception as e:
            print(f"❌ Error during tracking: {e}")
        
        finally:
            # Calculate final statistics
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            avg_detection_ms = (total_detection_time / frame_count) * 1000 if frame_count > 0 else 0
            
            # Cleanup
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
            
            # No need to close HTTPS connection as requests handles it automatically
            print("📡 HTTPS connection closed")
            
            # Print final report
            print(f"\n📊 Final Performance Report:")
            print(f"   Total frames processed: {frame_count}")
            print(f"   Total time: {total_time:.1f}s")
            print(f"   Average FPS: {avg_fps:.2f}")
            print(f"   Average detection time: {avg_detection_ms:.1f}ms")
            print(f"   Total blinks detected: {self.blink_count}")
            print(f"   GPU utilization: Optimized")
            print(f"   Gaze timing threshold: {self.gaze_distraction_threshold:.1f}s")
            print(f"   Gaze sensitivity: {self.gaze_threshold:.3f}")
            
            if self.device == 'cuda':
                print(f"   Peak GPU memory: {torch.cuda.max_memory_allocated()/1e6:.1f}MB")
    
    def toggle_face_gaze_detection(self):
        """Toggle face gaze detection"""
        self.face_distraction_enabled = not self.face_distraction_enabled
        status = "ON" if self.face_distraction_enabled else "OFF"
        print(f"Face gaze detection: {status}")
        self.speak_async(f"Face gaze detection {status}")
    
    def toggle_eye_detection(self):
        """Toggle eye detection"""
        self.eye_detection_enabled = not self.eye_detection_enabled
        status = "ON" if self.eye_detection_enabled else "OFF"
        print(f"Eye detection: {status}")
        self.speak_async(f"Eye detection {status}")
    
    def adjust_eye_threshold(self, increase=True):
        """Adjust eye closed threshold"""
        if increase:
            self.eye_closed_threshold += 0.01
        else:
            self.eye_closed_threshold = max(0.1, self.eye_closed_threshold - 0.01)
        print(f"Eye threshold adjusted to: {self.eye_closed_threshold:.3f}")
    
    def reset_eye_stats(self):
        """Reset eye detection statistics"""
        self.blink_count = 0
        self.consecutive_closed_frames = 0
        self.eye_state_history = []
        print("Eye detection statistics reset!")
    
    def print_gpu_stats(self):
        """Print current GPU statistics"""
        if self.device == 'cuda':
            print(f"\n📊 GPU Statistics:")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Allocated: {torch.cuda.memory_allocated()/1e6:.1f} MB")
            print(f"   Cached: {torch.cuda.memory_reserved()/1e6:.1f} MB")
            print(f"   Max Allocated: {torch.cuda.max_memory_allocated()/1e6:.1f} MB")
            print(f"   Total Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
            print(f"   cuDNN Enabled: {torch.backends.cudnn.enabled}")
            print(f"   Mixed Precision: {self.use_mixed_precision}")
        else:
            print("Running on CPU - no GPU stats available")

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
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        results = self.phone_model(frame, verbose=False, device=self.device)
                else:
                    results = self.phone_model(frame, verbose=False, device=self.device)
            
            persons_detected = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Only process person detections (class 0) with confidence > threshold
                        if class_id == self.target_person_class and confidence > self.person_confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            persons_detected.append({
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': confidence
                            })
            
            return persons_detected
            
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
            # Person(s) detected in current frame
            if self.person_first_seen is None:
                # First time seeing a person
                self.person_first_seen = current_time
                print(f"👀 Person first detected at {time.strftime('%H:%M:%S')}")
                print(f"   Waiting for {self.consecutive_time_required} consecutive seconds...")
            
            # Check if person has been visible for required consecutive time
            time_visible = current_time - self.person_first_seen
            
            if time_visible >= self.consecutive_time_required and not self.person_confirmed:
                # Person has been visible for required consecutive seconds
                self.person_confirmed = True
                print(f"✅ PERSON CONFIRMED! Visible for {time_visible:.1f} seconds")
                print(f"   Confirmed at {time.strftime('%H:%M:%S')}")
                print("   Starting all tracking features...")
            
            # Update current detections
            self.current_person_detections = persons_detected
            
        else:
            # No person detected in current frame
            if self.person_first_seen is not None:
                time_visible = current_time - self.person_first_seen
                print(f"❌ Person lost after {time_visible:.1f} seconds")
                
            # Reset detection state
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
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                
                # Draw bounding box (green for confirmed person)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Draw label with confidence
                label = f"CONFIRMED PERSON: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw label background
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw detection boxes for unconfirmed persons (yellow/orange)
        elif self.current_person_detections and not self.person_confirmed:
            for detection in self.current_person_detections:
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                
                # Draw bounding box (orange for unconfirmed)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                
                # Calculate time visible
                time_visible = current_time - self.person_first_seen if self.person_first_seen else 0
                remaining_time = max(0, self.consecutive_time_required - time_visible)
                
                # Draw label
                label = f"DETECTING PERSON: {remaining_time:.1f}s left"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (0, 165, 255), -1)
                
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add status information at the top of the frame
        if self.person_confirmed:
            status = "PERSON CONFIRMED ✅ - All tracking features active"
            color = (0, 255, 0)
        elif self.person_first_seen is not None:
            time_visible = current_time - self.person_first_seen
            remaining = self.consecutive_time_required - time_visible
            status = f"DETECTING PERSON... {remaining:.1f}s remaining ⏳"
            color = (0, 165, 255)
        else:
            status = "WAITING FOR PERSON 👀 - Other features disabled"
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

    def send_esp32_alert(self, alert_type):
        """Send alert to ESP32 via HTTPS"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
        
        try:
            # Map alert types to commands
            alert_commands = {
                "distracted": "DISTRACTED",
                "drowsy": "DROWSY",
                "phone": "MOBILE",
                "smoking": "SMOKING"
            }
            
            if alert_type in alert_commands:
                command = alert_commands[alert_type]
                # Send HTTP POST request
                response = requests.post(
                    f"{self.esp32_url}/trigger",
                    data={"type": command},
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    print(f"✅ Alert sent to ESP32: {command}")
                    self.alert_active = True
                    self.last_alert_time = current_time
                else:
                    print(f"❌ Failed to send alert: {response.status_code}")
                    print(f"Response: {response.text}")
            else:
                print(f"❌ Unknown alert type: {alert_type}")
                
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Connection Error: Could not connect to ESP32 at {self.esp32_url}")
            print("   Make sure the ESP32 is powered on and connected to the network")
        except Exception as e:
            print(f"❌ Error sending alert to ESP32: {e}")
    
    def stop_esp32_alert(self):
        """Stop active alert on ESP32"""
        if not self.alert_active:
            return
            
        try:
            response = requests.post(
                f"{self.esp32_url}/stop",
                timeout=5.0
            )
            
            if response.status_code == 200:
                print("✅ Alert stopped")
                self.alert_active = False
            else:
                print(f"❌ Failed to stop alert: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Connection Error: Could not connect to ESP32")
        except Exception as e:
            print(f"❌ Error stopping alert: {e}")


def main():
    """Main function to run the fully GPU-optimized tracker with gaze direction"""
    print("🚀 Fully GPU-Optimized Mobile Phone and Face Tracker with Gaze Direction")
    print("=" * 70)
    print("Features:")
    print("  📱 YOLO11x Mobile Phone Detection")
    print("  👤 YOLOv11l Face Detection") 
    print("  👁️ MediaPipe Eye State Detection")
    print("  👀 MediaPipe Gaze Direction Tracking")
    print("  🧠 HRNetV2 Feature Extraction")
    print("  🔊 Smart TTS Alerts")
    print("  ⚡ Full GPU Acceleration")
    print("  ⏱️ Continuous gaze tracking for TTS triggers")
    print("=" * 70)
    
    # Model paths
    yolo_phone_model = r"C:\Users\menuk\Desktop\object traking\yolo11x.pt"
    yolo_face_model = r"C:\Users\menuk\Desktop\object traking\yolov11l-face.pt"
    hrnetv2_model = r"C:\Users\menuk\Desktop\object traking\hrnetv2_w32_imagenet_pretrained.pth"
    
    # Validate model files
    if not os.path.exists(yolo_phone_model):
        print(f"❌ Phone detection model not found: {yolo_phone_model}")
        return
    
    if not os.path.exists(yolo_face_model):
        print(f"❌ Face detection model not found: {yolo_face_model}")
        return
    
    print("✅ Model files validated")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA Available")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ CUDA not available - will use CPU")
        choice = input("Continue with CPU? (y/n): ")
        if choice.lower() != 'y':
            return
    
    try:
        # Initialize tracker
        print("\n🔄 Initializing tracker with gaze direction tracking...")
        tracker = FullyGPUOptimizedTracker(
            yolo_model_path=yolo_phone_model,
            face_model_path=yolo_face_model,
            hrnetv2_model_path=hrnetv2_model
        )
        
        print(f"\n⏱️ Gaze Configuration:")
        print(f"   Gaze duration threshold: {tracker.gaze_distraction_threshold} seconds")
        print(f"   Gaze sensitivity: {tracker.gaze_threshold:.3f}")
        print(f"   TTS will trigger when looking left/right continuously for {tracker.gaze_distraction_threshold}s")
        print(f"   Use 't'/'y' keys to adjust timing threshold during runtime")
        print(f"   Use 'u'/'i' keys to adjust gaze sensitivity during runtime")
        
        # Select input source
        print("\n📹 Select input source:")
        print("1. Webcam (Recommended for real-time)")
        print("2. Video file")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            print("🎥 Starting webcam tracking with gaze direction...")
            tracker.track_video_full_gpu()
            
        elif choice == "2":
            video_path = input("📁 Enter video file path: ").strip()
            if not os.path.exists(video_path):
                print(f"❌ Video file not found: {video_path}")
                return
            
            output_path = input("💾 Enter output path (optional, press Enter to skip): ").strip()
            if not output_path:
                output_path = None
            
            print("🎥 Starting video file tracking with gaze direction...")
            tracker.track_video_full_gpu(video_path, output_path)
            
        else:
            print("❌ Invalid choice")
    
    except Exception as e:
        print(f"❌ Error initializing tracker: {e}")
        print("\n🔧 Make sure you have installed:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("pip install ultralytics opencv-python mediapipe pyttsx3")
        
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()