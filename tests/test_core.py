"""
Unit tests for the core module components.
"""
import unittest
import os
import json
import tempfile
from unittest.mock import patch, Mock
import sys
import torch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.config_manager import ConfigManager
from src.core.gpu_manager import GPUManager


class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            "detection": {
                "confidence_threshold": 0.5,
                "phone_confidence": 0.6
            },
            "gpu": {
                "enabled": True,
                "device": "auto"
            }
        }
    
    def test_load_config_from_dict(self):
        """Test loading configuration from dictionary."""
        manager = ConfigManager(self.test_config)
        self.assertEqual(manager.get('detection.confidence_threshold'), 0.5)
        self.assertEqual(manager.get('gpu.enabled'), True)
    
    def test_load_config_from_file(self):
        """Test loading configuration from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_config, f)
            temp_path = f.name
        
        try:
            manager = ConfigManager(temp_path)
            self.assertEqual(manager.get('detection.confidence_threshold'), 0.5)
            self.assertEqual(manager.get('gpu.enabled'), True)
        finally:
            os.unlink(temp_path)
    
    def test_get_with_default(self):
        """Test getting configuration values with defaults."""
        manager = ConfigManager(self.test_config)
        self.assertEqual(manager.get('nonexistent.key', 'default'), 'default')
        self.assertEqual(manager.get('detection.confidence_threshold', 0.8), 0.5)
    
    def test_set_config_value(self):
        """Test setting configuration values."""
        manager = ConfigManager(self.test_config)
        manager.set('new.key', 'new_value')
        self.assertEqual(manager.get('new.key'), 'new_value')
    
    def test_invalid_config_file(self):
        """Test handling of invalid configuration file."""
        with self.assertRaises(Exception):
            ConfigManager('nonexistent_file.json')


class TestGPUManager(unittest.TestCase):
    """Test cases for GPUManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'enabled': True,
            'device': 'auto',
            'memory_fraction': 0.8,
            'benchmark': True
        }
    
    @patch('torch.cuda.is_available')
    def test_gpu_manager_init_cuda_available(self, mock_cuda_available):
        """Test GPU manager initialization when CUDA is available."""
        mock_cuda_available.return_value = True
        
        with patch('torch.cuda.get_device_count', return_value=1):
            manager = GPUManager(self.config)
            self.assertTrue(manager.cuda_available)
            self.assertEqual(manager.device_count, 1)
    
    @patch('torch.cuda.is_available')
    def test_gpu_manager_init_cuda_not_available(self, mock_cuda_available):
        """Test GPU manager initialization when CUDA is not available."""
        mock_cuda_available.return_value = False
        
        manager = GPUManager(self.config)
        self.assertFalse(manager.cuda_available)
        self.assertEqual(manager.device_count, 0)
    
    @patch('torch.cuda.is_available')
    def test_get_device_auto(self, mock_cuda_available):
        """Test automatic device selection."""
        mock_cuda_available.return_value = True
        
        with patch('torch.cuda.get_device_count', return_value=1):
            manager = GPUManager(self.config)
            device = manager.get_device()
            self.assertTrue(device.type == 'cuda' or device.type == 'cpu')
    
    def test_get_device_cpu_only(self):
        """Test CPU-only device selection."""
        config = self.config.copy()
        config['enabled'] = False
        
        manager = GPUManager(config)
        device = manager.get_device()
        self.assertEqual(device.type, 'cpu')
    
    @patch('torch.cuda.is_available')
    def test_optimize_memory_cuda(self, mock_cuda_available):
        """Test memory optimization for CUDA."""
        mock_cuda_available.return_value = True
        
        with patch('torch.cuda.empty_cache') as mock_empty_cache:
            manager = GPUManager(self.config)
            manager.optimize_memory()
            mock_empty_cache.assert_called_once()


if __name__ == '__main__':
    unittest.main()
