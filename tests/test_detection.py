"""
Unit tests for detection module components.
"""
import unittest
import numpy as np
import cv2
from unittest.mock import patch, Mock, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.detection.phone_detector import PhoneDetector
from src.detection.face_detector import FaceDetector
from src.detection.person_detector import PersonDetector


class TestPhoneDetector(unittest.TestCase):
    """Test cases for PhoneDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'phone_model_path': 'models/yolo11x.pt',
            'confidence_threshold': 0.5,
            'phone_confidence': 0.6
        }
        
        # Create a mock image
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    @patch('src.detection.phone_detector.YOLO')
    def test_phone_detector_init(self, mock_yolo):
        """Test PhoneDetector initialization."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        detector = PhoneDetector(self.config, Mock())
        self.assertIsNotNone(detector.model)
        mock_yolo.assert_called_once()
    
    @patch('src.detection.phone_detector.YOLO')
    def test_detect_phones_success(self, mock_yolo):
        """Test successful phone detection."""
        # Mock YOLO model and results
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = [[100, 100, 200, 200]]
        mock_result.boxes.conf = [0.8]
        mock_result.boxes.cls = [67]  # Phone class in COCO dataset
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        detector = PhoneDetector(self.config, Mock())
        detections = detector.detect(self.test_image)
        
        self.assertIsInstance(detections, list)
        mock_model.assert_called_once()
    
    @patch('src.detection.phone_detector.YOLO')
    def test_detect_phones_no_detections(self, mock_yolo):
        """Test phone detection with no phones found."""
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = None
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        detector = PhoneDetector(self.config, Mock())
        detections = detector.detect(self.test_image)
        
        self.assertEqual(detections, [])


class TestFaceDetector(unittest.TestCase):
    """Test cases for FaceDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'face_model_path': 'models/yolov11l-face.pt',
            'confidence_threshold': 0.5,
            'face_confidence': 0.4
        }
        
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    @patch('src.detection.face_detector.YOLO')
    def test_face_detector_init(self, mock_yolo):
        """Test FaceDetector initialization."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        detector = FaceDetector(self.config, Mock())
        self.assertIsNotNone(detector.model)
        mock_yolo.assert_called_once()
    
    @patch('src.detection.face_detector.YOLO')
    def test_detect_faces_success(self, mock_yolo):
        """Test successful face detection."""
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = [[150, 150, 250, 250]]
        mock_result.boxes.conf = [0.9]
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        detector = FaceDetector(self.config, Mock())
        detections = detector.detect(self.test_image)
        
        self.assertIsInstance(detections, list)
        mock_model.assert_called_once()


class TestPersonDetector(unittest.TestCase):
    """Test cases for PersonDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'person_model_path': 'models/yolo11x.pt',
            'confidence_threshold': 0.5,
            'person_confidence': 0.5
        }
        
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    @patch('src.detection.person_detector.YOLO')
    def test_person_detector_init(self, mock_yolo):
        """Test PersonDetector initialization."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        detector = PersonDetector(self.config, Mock())
        self.assertIsNotNone(detector.model)
        mock_yolo.assert_called_once()
    
    @patch('src.detection.person_detector.YOLO')
    def test_detect_persons_success(self, mock_yolo):
        """Test successful person detection."""
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = [[50, 50, 300, 400]]
        mock_result.boxes.conf = [0.85]
        mock_result.boxes.cls = [0]  # Person class in COCO dataset
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        detector = PersonDetector(self.config, Mock())
        detections = detector.detect(self.test_image)
        
        self.assertIsInstance(detections, list)
        mock_model.assert_called_once()
    
    def test_is_phone_near_person(self):
        """Test phone-person proximity checking."""
        detector = PersonDetector(self.config, Mock())
        
        # Test case: phone inside person bounding box
        person_box = [50, 50, 300, 400]
        phone_box = [100, 100, 150, 150]
        
        result = detector._is_phone_near_person(phone_box, person_box)
        self.assertTrue(result)
        
        # Test case: phone far from person
        phone_box_far = [500, 500, 550, 550]
        result = detector._is_phone_near_person(phone_box_far, person_box)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
