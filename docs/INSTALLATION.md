# Installation Guide

## System Requirements

### Hardware Requirements

**Minimum Requirements:**
- CPU: Intel i5 or AMD Ryzen 5 (4+ cores)
- RAM: 8 GB
- Storage: 5 GB free space
- Camera: USB webcam or integrated camera

**Recommended Requirements:**
- CPU: Intel i7 or AMD Ryzen 7 (8+ cores)
- RAM: 16 GB
- GPU: NVIDIA RTX 3060 or better (8+ GB VRAM)
- Storage: 10 GB free space (SSD preferred)
- Camera: High-resolution USB camera (1080p+)

### Software Requirements

- **Operating System**: Windows 10/11, Linux (Ubuntu 18.04+), or macOS 10.15+
- **Python**: 3.8 or newer
- **CUDA**: 11.8+ (for GPU acceleration)
- **Git**: For cloning the repository

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/object-tracking.git
cd object-tracking
```

### 2. Create Virtual Environment

#### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Linux/macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

#### Option A: Install from requirements.txt
```bash
pip install -r requirements.txt
```

#### Option B: Manual Installation
```bash
# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python numpy Pillow
pip install mediapipe scipy scikit-learn
pip install requests urllib3 pyttsx3

# Development dependencies (optional)
pip install pytest pytest-cov black flake8
```

### 4. Download Model Files

#### Option A: Using Download Utility
```bash
python utils/download_models.py
```

#### Option B: Manual Download
Download the following model files to the `models/` directory:

1. **YOLO11x Model** (`yolo11x.pt`)
   - Download from: [Ultralytics Releases](https://github.com/ultralytics/ultralytics/releases)
   - Size: ~136 MB

2. **YOLOv11l-Face Model** (`yolov11l-face.pt`)
   - Download from: [YOLOv5-Face Releases](https://github.com/deepcam-cn/yolov5-face/releases)
   - Size: ~47 MB

3. **HRNetV2 Model** (`hrnetv2_w32_imagenet_pretrained.pth`)
   - Download from: [HRNet Repository](https://github.com/HRNet/HRNet-Image-Classification)
   - Size: ~124 MB

### 5. Configure Settings

#### Create Configuration File
Copy the default configuration and customize it:

```bash
cp config/default_config.json config/my_config.json
```

#### Edit Configuration
Modify `config/my_config.json` to match your setup:

```json
{
    "camera": {
        "device_id": 0,
        "width": 640,
        "height": 480,
        "fps": 30
    },
    "esp32": {
        "ip_address": "YOUR_ESP32_IP",
        "port": 80
    },
    "gpu": {
        "enabled": true,
        "device": "auto"
    }
}
```

### 6. Test Installation

Run the test suite to verify installation:

```bash
python -m pytest tests/ -v
```

Run a quick system check:

```bash
python -c "
import torch
import cv2
import ultralytics
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'OpenCV: {cv2.__version__}')
print('Installation successful!')
"
```

## ESP32 Setup (Optional)

If you plan to use ESP32 integration:

### 1. Hardware Setup
- Connect ESP32 to your network
- Note the assigned IP address
- Upload the Arduino sketch from `arduino/smoke_detector/`

### 2. Network Configuration
- Ensure ESP32 and computer are on the same network
- Update the ESP32 IP address in your configuration file
- Test connectivity: `ping YOUR_ESP32_IP`

## Troubleshooting

### Common Issues

#### 1. CUDA Installation Issues

**Problem**: PyTorch not detecting CUDA
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solution**:
```bash
# Uninstall existing PyTorch
pip uninstall torch torchvision

# Install CUDA-compatible version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Model Download Failures

**Problem**: Models fail to download
**Solution**:
- Check internet connection
- Download models manually from provided links
- Place files in `models/` directory

#### 3. Camera Access Issues

**Problem**: Cannot access camera
```
cv2.error: (-215:Assertion failed) !_src.empty()
```

**Solution**:
- Check camera permissions
- Try different camera indices (0, 1, 2...)
- Update camera drivers
- Test with other applications

#### 4. Memory Issues

**Problem**: Out of memory errors
**Solution**:
- Reduce batch size in configuration
- Lower camera resolution
- Enable GPU memory optimization
- Close other applications

#### 5. Import Errors

**Problem**: Module import failures
```
ModuleNotFoundError: No module named 'src'
```

**Solution**:
- Ensure you're running from the project root directory
- Activate the virtual environment
- Check PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"`

### Performance Optimization

#### For GPU Users
```json
{
    "gpu": {
        "enabled": true,
        "device": "cuda:0",
        "memory_fraction": 0.8,
        "benchmark": true
    }
}
```

#### For CPU-Only Users
```json
{
    "gpu": {
        "enabled": false
    },
    "camera": {
        "width": 480,
        "height": 360,
        "fps": 15
    }
}
```

## Verification

After installation, verify everything works:

### 1. Run Basic Test
```bash
python main.py --config config/default_config.json --test
```

### 2. Check System Information
```bash
python utils/logging_utils.py
```

### 3. Test Components
```bash
python -m pytest tests/test_detection.py -v
python -m pytest tests/test_core.py -v
```

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [API Documentation](API.md)
3. Check system requirements
4. Look for similar issues in the project repository
5. Create a detailed issue report including:
   - Operating system and version
   - Python version
   - Error messages
   - Steps to reproduce

## Next Steps

After successful installation:

1. Read the [Usage Guide](USAGE.md)
2. Review the [API Documentation](API.md)
3. Explore configuration options
4. Run your first tracking session
