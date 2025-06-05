# API Documentation

## Core Modules

### ConfigManager

The `ConfigManager` class handles configuration loading and management for the object tracking system.

#### Constructor

```python
ConfigManager(config_source)
```

**Parameters:**
- `config_source` (str or dict): Path to JSON configuration file or configuration dictionary

#### Methods

##### get(key, default=None)

Retrieve a configuration value using dot notation.

**Parameters:**
- `key` (str): Configuration key in dot notation (e.g., "detection.confidence_threshold")
- `default` (any, optional): Default value if key not found

**Returns:**
- Configuration value or default

**Example:**
```python
config = ConfigManager("config/default_config.json")
threshold = config.get("detection.confidence_threshold", 0.5)
```

##### set(key, value)

Set a configuration value using dot notation.

**Parameters:**
- `key` (str): Configuration key in dot notation
- `value` (any): Value to set

**Example:**
```python
config.set("gpu.enabled", True)
```

### GPUManager

The `GPUManager` class handles GPU detection, optimization, and device management.

#### Constructor

```python
GPUManager(gpu_config)
```

**Parameters:**
- `gpu_config` (dict): GPU configuration parameters

#### Methods

##### get_device()

Get the optimal compute device (GPU or CPU).

**Returns:**
- `torch.device`: Optimal device for computations

##### optimize_memory()

Optimize GPU memory usage by clearing cache.

##### setup_mixed_precision()

Set up automatic mixed precision for faster training/inference.

**Returns:**
- `torch.cuda.amp.GradScaler`: Gradient scaler for mixed precision

## Detection Modules

### PhoneDetector

Detects mobile phones in images using YOLO11x model.

#### Constructor

```python
PhoneDetector(detection_config, gpu_manager)
```

**Parameters:**
- `detection_config` (dict): Detection configuration
- `gpu_manager` (GPUManager): GPU manager instance

#### Methods

##### detect(image)

Detect phones in an image.

**Parameters:**
- `image` (np.ndarray): Input image in BGR format

**Returns:**
- `list`: List of detection dictionaries with keys:
  - `bbox`: Bounding box coordinates [x1, y1, x2, y2]
  - `confidence`: Detection confidence score
  - `class_id`: Class ID (67 for phone in COCO dataset)

### FaceDetector

Detects human faces using YOLOv11l-face model.

#### Constructor

```python
FaceDetector(detection_config, gpu_manager)
```

#### Methods

##### detect(image)

Detect faces in an image.

**Parameters:**
- `image` (np.ndarray): Input image in BGR format

**Returns:**
- `list`: List of face detection dictionaries

### PersonDetector

Detects persons and analyzes phone usage patterns.

#### Constructor

```python
PersonDetector(detection_config, gpu_manager)
```

#### Methods

##### detect(image)

Detect persons in an image.

**Returns:**
- `list`: List of person detection dictionaries

##### confirm_phone_usage(person_detections, phone_detections, face_detections)

Analyze if detected persons are using phones.

**Parameters:**
- `person_detections` (list): Person detection results
- `phone_detections` (list): Phone detection results  
- `face_detections` (list): Face detection results

**Returns:**
- `list`: List of confirmed phone usage detections

### EyeGazeDetector

Performs eye state detection and gaze tracking using MediaPipe.

#### Constructor

```python
EyeGazeDetector(detection_config)
```

#### Methods

##### detect_eye_state(image, face_bbox)

Detect eye state (open/closed) within a face region.

**Parameters:**
- `image` (np.ndarray): Input image
- `face_bbox` (list): Face bounding box [x1, y1, x2, y2]

**Returns:**
- `dict`: Eye state information with keys:
  - `left_eye_open`: Boolean for left eye state
  - `right_eye_open`: Boolean for right eye state
  - `gaze_direction`: Estimated gaze direction

##### track_gaze(image, face_landmarks)

Track gaze direction from face landmarks.

**Parameters:**
- `image` (np.ndarray): Input image
- `face_landmarks`: MediaPipe face landmarks

**Returns:**
- `dict`: Gaze tracking results

## Tracking Modules

### ObjectTracker

Handles multi-object tracking across video frames.

#### Constructor

```python
ObjectTracker(tracking_config, feature_extractor)
```

**Parameters:**
- `tracking_config` (dict): Tracking configuration
- `feature_extractor` (FeatureExtractor): Feature extraction instance

#### Methods

##### update(detections, image)

Update tracker with new detections.

**Parameters:**
- `detections` (list): Current frame detections
- `image` (np.ndarray): Current frame image

**Returns:**
- `list`: Updated tracking results with track IDs

##### get_active_tracks()

Get currently active tracking targets.

**Returns:**
- `dict`: Active tracks with track IDs as keys

### FeatureExtractor

Extracts visual features using HRNetV2 for tracking.

#### Constructor

```python
FeatureExtractor(model_config, gpu_manager)
```

#### Methods

##### extract_features(image, bboxes)

Extract features from detected objects.

**Parameters:**
- `image` (np.ndarray): Input image
- `bboxes` (list): Bounding boxes for feature extraction

**Returns:**
- `np.ndarray`: Extracted feature vectors

## Communication Modules

### ESP32Communicator

Handles HTTP communication with ESP32 devices.

#### Constructor

```python
ESP32Communicator(esp32_config)
```

#### Methods

##### send_detection_signal(detection_data)

Send detection results to ESP32.

**Parameters:**
- `detection_data` (dict): Detection information to send

**Returns:**
- `bool`: Success status

##### check_connection()

Check ESP32 connectivity.

**Returns:**
- `bool`: Connection status

### TTSManager

Manages text-to-speech functionality.

#### Constructor

```python
TTSManager(tts_config)
```

#### Methods

##### speak(text, priority="normal")

Convert text to speech.

**Parameters:**
- `text` (str): Text to speak
- `priority` (str): Speech priority ("low", "normal", "high")

##### set_voice_properties(rate, volume, voice_index)

Configure voice properties.

**Parameters:**
- `rate` (int): Speech rate
- `volume` (float): Volume level (0.0-1.0)
- `voice_index` (int): Voice selection index

## Main Tracker

### MainTracker

Main integration class that coordinates all tracking components.

#### Constructor

```python
MainTracker(config_path)
```

**Parameters:**
- `config_path` (str): Path to configuration file

#### Methods

##### run()

Start the main tracking loop.

##### process_frame(frame)

Process a single video frame.

**Parameters:**
- `frame` (np.ndarray): Input video frame

**Returns:**
- `dict`: Processing results

##### stop()

Stop the tracking system gracefully.

## Configuration Schema

### Detection Configuration

```json
{
    "detection": {
        "phone_model_path": "models/yolo11x.pt",
        "face_model_path": "models/yolov11l-face.pt",
        "person_model_path": "models/yolo11x.pt",
        "confidence_threshold": 0.5,
        "nms_threshold": 0.5,
        "phone_confidence": 0.6,
        "face_confidence": 0.4,
        "person_confidence": 0.5
    }
}
```

### GPU Configuration

```json
{
    "gpu": {
        "enabled": true,
        "device": "auto",
        "memory_fraction": 0.8,
        "benchmark": true
    }
}
```

### Tracking Configuration

```json
{
    "tracking": {
        "hrnet_model_path": "models/hrnetv2_w32_imagenet_pretrained.pth",
        "max_disappeared": 30,
        "max_distance": 100,
        "feature_extraction_enabled": true
    }
}
```

## Error Handling

All modules implement comprehensive error handling:

- **Model Loading Errors**: Graceful fallback when models are missing
- **GPU Errors**: Automatic fallback to CPU when GPU is unavailable
- **Network Errors**: Retry mechanisms for ESP32 communication
- **Camera Errors**: Proper resource cleanup on camera failures

## Performance Optimization

The system includes several performance optimizations:

- **GPU Memory Management**: Automatic memory optimization
- **Model Caching**: Efficient model loading and reuse
- **Frame Processing**: Optimized image processing pipelines
- **Mixed Precision**: Automatic mixed precision support for compatible GPUs
