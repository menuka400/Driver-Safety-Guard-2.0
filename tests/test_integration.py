"""
Integration tests for the object tracking system.
Tests the complete pipeline from detection to tracking.
"""
import unittest
import numpy as np
import cv2
import tempfile
import json
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from unittest.mock import patch, Mock, MagicMock


class TestObjectTrackingIntegration(unittest.TestCase):
    """Integration test cases for the complete tracking system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            "detection": {
                "phone_model_path": "models/yolo11x.pt",
                "face_model_path": "models/yolov11l-face.pt",
                "person_model_path": "models/yolo11x.pt",
                "confidence_threshold": 0.5,
                "phone_confidence": 0.6,
                "face_confidence": 0.4,
                "person_confidence": 0.5
            },
            "tracking": {
                "hrnet_model_path": "models/hrnetv2_w32_imagenet_pretrained.pth",
                "max_disappeared": 30,
                "max_distance": 100,
                "feature_extraction_enabled": True
            },
            "gpu": {
                "enabled": True,
                "device": "auto",
                "memory_fraction": 0.8
            },
            "camera": {
                "width": 640,
                "height": 480,
                "fps": 30
            },
            "esp32": {
                "ip_address": "192.168.1.100",
                "port": 80,
                "timeout": 5
            },
            "tts": {
                "enabled": True,
                "rate": 150,
                "volume": 0.8
            }
        }
        
        # Create test images
        self.test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.test_frame_rgb = cv2.cvtColor(self.test_frame, cv2.COLOR_BGR2RGB)
    
    def create_temp_config(self):
        """Create a temporary configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_config, f)
            return f.name
    
    @patch('src.detection.phone_detector.YOLO')
    @patch('src.detection.face_detector.YOLO') 
    @patch('src.detection.person_detector.YOLO')
    def test_detection_pipeline(self, mock_person_yolo, mock_face_yolo, mock_phone_yolo):
        """Test the complete detection pipeline."""
        from src.detection.phone_detector import PhoneDetector
        from src.detection.face_detector import FaceDetector
        from src.detection.person_detector import PersonDetector
        from src.core.gpu_manager import GPUManager
        
        # Mock YOLO models
        for mock_yolo in [mock_phone_yolo, mock_face_yolo, mock_person_yolo]:
            mock_model = Mock()
            mock_result = Mock()
            mock_result.boxes = Mock()
            mock_result.boxes.xyxy = [[100, 100, 200, 200]]
            mock_result.boxes.conf = [0.8]
            mock_result.boxes.cls = [0]
            mock_model.return_value = [mock_result]
            mock_yolo.return_value = mock_model
        
        # Create GPU manager
        gpu_manager = GPUManager(self.test_config['gpu'])
        
        # Create detectors
        phone_detector = PhoneDetector(self.test_config['detection'], gpu_manager)
        face_detector = FaceDetector(self.test_config['detection'], gpu_manager)
        person_detector = PersonDetector(self.test_config['detection'], gpu_manager)
        
        # Test detection pipeline
        phone_detections = phone_detector.detect(self.test_frame)
        face_detections = face_detector.detect(self.test_frame)
        person_detections = person_detector.detect(self.test_frame)
        
        # Verify results
        self.assertIsInstance(phone_detections, list)
        self.assertIsInstance(face_detections, list)
        self.assertIsInstance(person_detections, list)
    
    @patch('src.communication.esp32_communicator.requests')
    def test_esp32_communication(self, mock_requests):
        """Test ESP32 communication functionality."""
        from src.communication.esp32_communicator import ESP32Communicator
        
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_requests.post.return_value = mock_response
        mock_requests.get.return_value = mock_response
        
        # Create communicator
        communicator = ESP32Communicator(self.test_config['esp32'])
        
        # Test connection check
        self.assertTrue(communicator.check_connection())
        
        # Test sending detection signal
        detection_data = {
            "phone_detected": True,
            "person_count": 1,
            "timestamp": "2024-01-01T12:00:00"
        }
        
        result = communicator.send_detection_signal(detection_data)
        self.assertTrue(result)
    
    @patch('src.communication.tts_manager.pyttsx3')
    def test_tts_functionality(self, mock_pyttsx3):
        """Test text-to-speech functionality."""
        from src.communication.tts_manager import TTSManager
        
        # Mock TTS engine
        mock_engine = Mock()
        mock_pyttsx3.init.return_value = mock_engine
        
        # Create TTS manager
        tts_manager = TTSManager(self.test_config['tts'])
        
        # Test speaking
        tts_manager.speak("Test message")
        
        # Verify TTS calls
        mock_engine.say.assert_called()
        mock_engine.runAndWait.assert_called()
    
    @patch('torch.load')
    @patch('src.tracking.feature_extractor.HRNet')
    def test_feature_extraction(self, mock_hrnet, mock_torch_load):
        """Test feature extraction functionality."""
        from src.tracking.feature_extractor import FeatureExtractor
        from src.core.gpu_manager import GPUManager
        
        # Mock model loading
        mock_model = Mock()
        mock_model.return_value = torch.randn(1, 256)  # Mock feature output
        mock_hrnet.return_value = mock_model
        mock_torch_load.return_value = {}
        
        # Create feature extractor
        gpu_manager = GPUManager(self.test_config['gpu'])
        feature_extractor = FeatureExtractor(self.test_config['tracking'], gpu_manager)
        
        # Test feature extraction
        bboxes = [[100, 100, 200, 200], [300, 300, 400, 400]]
        
        with patch('torch.randn', return_value=torch.randn(2, 256)):
            features = feature_extractor.extract_features(self.test_frame, bboxes)
            self.assertIsInstance(features, np.ndarray)
    
    def test_config_management(self):
        """Test configuration management."""
        from src.core.config_manager import ConfigManager
        
        # Test with dictionary
        config_manager = ConfigManager(self.test_config)
        
        # Test getting values
        self.assertEqual(config_manager.get('detection.confidence_threshold'), 0.5)
        self.assertEqual(config_manager.get('gpu.enabled'), True)
        self.assertEqual(config_manager.get('nonexistent.key', 'default'), 'default')
        
        # Test setting values
        config_manager.set('test.value', 123)
        self.assertEqual(config_manager.get('test.value'), 123)
        
        # Test with file
        config_file = self.create_temp_config()
        try:
            file_config_manager = ConfigManager(config_file)
            self.assertEqual(file_config_manager.get('detection.confidence_threshold'), 0.5)
        finally:
            os.unlink(config_file)
    
    @patch('src.main_tracker.cv2.VideoCapture')
    def test_main_tracker_initialization(self, mock_video_capture):
        """Test main tracker initialization."""
        from src.main_tracker import MainTracker
        
        # Mock camera
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640  # Mock width/height
        mock_video_capture.return_value = mock_cap
        
        # Create config file
        config_file = self.create_temp_config()
        
        try:
            with patch('src.detection.phone_detector.YOLO'), \
                 patch('src.detection.face_detector.YOLO'), \
                 patch('src.detection.person_detector.YOLO'), \
                 patch('torch.load'), \
                 patch('src.tracking.feature_extractor.HRNet'):
                
                # Initialize tracker
                tracker = MainTracker(config_file)
                
                # Verify initialization
                self.assertIsNotNone(tracker.config_manager)
                self.assertIsNotNone(tracker.gpu_manager)
                
        finally:
            os.unlink(config_file)
    
    def test_utility_functions(self):
        """Test utility functions."""
        from utils.video_utils import convert_bbox_format, calculate_iou
        
        # Test bbox conversion
        xyxy_bbox = [100, 100, 200, 200]
        xywh_bbox = convert_bbox_format(xyxy_bbox, 'xyxy', 'xywh')
        self.assertEqual(xywh_bbox, [100, 100, 100, 100])
        
        # Test IoU calculation
        box1 = [100, 100, 200, 200]
        box2 = [150, 150, 250, 250]
        iou = calculate_iou(box1, box2)
        self.assertGreater(iou, 0)
        self.assertLess(iou, 1)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up any temporary files
        pass


if __name__ == '__main__':
    # Add torch import for mocking
    try:
        import torch
    except ImportError:
        print("PyTorch not available, skipping integration tests")
        sys.exit(0)
    
    unittest.main()
