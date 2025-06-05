"""
Object Tracking System

A professional, GPU-optimized object tracking system with multi-modal detection capabilities.
Features phone detection, face detection, eye tracking, person detection, and ESP32 integration.
"""

__version__ = "1.0.0"
__author__ = "Object Tracking Team"
__email__ = "contact@objecttracking.com"
__description__ = "Advanced GPU-optimized object tracking system with multi-modal detection"

# Import main classes for easy access
from .main_tracker import FullyGPUOptimizedTracker
from .core.config_manager import ConfigManager
from .core.gpu_manager import GPUManager

__all__ = [
    "FullyGPUOptimizedTracker",
    "ConfigManager", 
    "GPUManager",
    "__version__",
    "__author__",
    "__email__",
    "__description__"
]
