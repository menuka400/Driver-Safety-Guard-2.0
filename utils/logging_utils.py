"""
Logging configuration utility for the object tracking project.
Sets up consistent logging across all modules.
"""
import logging
import logging.handlers
import os
from pathlib import Path
from datetime import datetime


def setup_logging(config=None, log_level="INFO"):
    """
    Set up logging configuration for the application.
    
    Args:
        config (dict, optional): Logging configuration
        log_level (str): Default log level
    """
    # Default configuration
    default_config = {
        'level': log_level,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': 'logs/tracker.log',
        'max_bytes': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5
    }
    
    # Update with provided config
    if config:
        default_config.update(config)
    
    # Create logs directory
    log_file = Path(default_config['file_path'])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up logging level
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    level = log_level_map.get(default_config['level'].upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(default_config['format'])
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        default_config['file_path'],
        maxBytes=default_config['max_bytes'],
        backupCount=default_config['backup_count']
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")
    logger.info(f"Log level: {default_config['level']}")
    logger.info(f"Log file: {default_config['file_path']}")


def get_logger(name):
    """
    Get a logger instance for a specific module.
    
    Args:
        name (str): Logger name (typically __name__)
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


class PerformanceLogger:
    """Context manager for performance logging."""
    
    def __init__(self, operation_name, logger=None):
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        duration_ms = duration.total_seconds() * 1000
        
        if exc_type is None:
            self.logger.debug(f"Completed: {self.operation_name} in {duration_ms:.2f}ms")
        else:
            self.logger.error(f"Failed: {self.operation_name} after {duration_ms:.2f}ms - {exc_val}")


def log_system_info():
    """Log system information for debugging purposes."""
    logger = logging.getLogger(__name__)
    
    try:
        import torch
        import cv2
        import platform
        import psutil
        
        logger.info("=" * 50)
        logger.info("SYSTEM INFORMATION")
        logger.info("=" * 50)
        
        # Python and OS info
        logger.info(f"Python version: {platform.python_version()}")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Architecture: {platform.architecture()[0]}")
        
        # Hardware info
        logger.info(f"CPU cores: {psutil.cpu_count()}")
        logger.info(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        # PyTorch info
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # OpenCV info
        logger.info(f"OpenCV version: {cv2.__version__}")
        
        logger.info("=" * 50)
        
    except ImportError as e:
        logger.warning(f"Could not import required modules for system info: {e}")
    except Exception as e:
        logger.error(f"Error gathering system info: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Test logging setup
    setup_logging(log_level="DEBUG")
    
    logger = get_logger(__name__)
    logger.info("Testing logging system")
    
    # Test performance logger
    with PerformanceLogger("test operation", logger):
        import time
        time.sleep(0.1)
    
    # Test system info logging
    log_system_info()
    
    logger.info("Logging test completed")
