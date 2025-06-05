"""
Main Entry Point for GPU-Optimized Object Tracking System
Enhanced modular structure with professional organization
"""
import os
import sys
import torch
import traceback

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.main_tracker import FullyGPUOptimizedTracker

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
    print("  📡 ESP32 Communication")
    print("  ⏱️ Continuous gaze tracking for TTS triggers")
    print("  🏗️ Professional Modular Architecture")
    print("=" * 70)
    
    # Model paths (updated for new structure)
    yolo_phone_model = os.path.join("models", "yolo11x.pt")
    yolo_face_model = os.path.join("models", "yolov11l-face.pt")
    hrnetv2_model = os.path.join("models", "hrnetv2_w32_imagenet_pretrained.pth")
    
    # Validate model files
    if not os.path.exists(yolo_phone_model):
        print(f"❌ Phone detection model not found: {yolo_phone_model}")
        print("   Please ensure model files are in the 'models' directory")
        return
    
    if not os.path.exists(yolo_face_model):
        print(f"❌ Face detection model not found: {yolo_face_model}")
        print("   Please ensure model files are in the 'models' directory")
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
        # Initialize tracker with modular architecture
        print("\n🔄 Initializing modular tracker with gaze direction tracking...")
        tracker = FullyGPUOptimizedTracker(
            yolo_model_path=yolo_phone_model,
            face_model_path=yolo_face_model,
            hrnetv2_model_path=hrnetv2_model,
            config_path="config/default_config.json"  # Optional config file
        )
        
        print(f"\n⏱️ Gaze Configuration:")
        print(f"   Gaze duration threshold: {tracker.eye_gaze_detector.gaze_distraction_threshold} seconds")
        print(f"   Gaze sensitivity: {tracker.eye_gaze_detector.gaze_threshold:.3f}")
        print(f"   TTS will trigger when looking left/right continuously for {tracker.eye_gaze_detector.gaze_distraction_threshold}s")
        print(f"   Use 't'/'y' keys to adjust timing threshold during runtime")
        print(f"   Use 'u'/'i' keys to adjust gaze sensitivity during runtime")
        
        # Test ESP32 connection
        print(f"\n📡 Testing ESP32 connection...")
        esp32_status = tracker.esp32_comm.test_connection()
        if not esp32_status:
            print("⚠️ ESP32 not reachable - alerts will be logged only")
        
        # Show system status
        print("\n📊 System Status:")
        status = tracker.get_system_status()
        print(f"   GPU: {status['gpu'].upper()}")
        print(f"   Person Detection: Ready")
        print(f"   Phone Detection: Ready") 
        print(f"   Face Detection: Ready")
        print(f"   Eye/Gaze Detection: {'Enabled' if status['eye_gaze_detector']['eye_detection_enabled'] else 'Disabled'}")
        print(f"   Feature Extractor: {'Available' if status['feature_extractor_available'] else 'Not Available'}")
        print(f"   TTS: {'Available' if status['tts_available'] else 'Not Available'}")
        print(f"   ESP32: {'Connected' if esp32_status else 'Disconnected'}")
        
        # Select input source
        print("\n📹 Select input source:")
        print("1. Webcam (Recommended for real-time)")
        print("2. Video file")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            print("\n🎥 Starting webcam tracking...")
            output_choice = input("Save output video? (y/n): ").strip().lower()
            output_path = "data/webcam_output.mp4" if output_choice == 'y' else None
            tracker.track_video_full_gpu(video_path=None, output_path=output_path)
            
        elif choice == "2":
            video_path = input("Enter video file path: ").strip()
            if not os.path.exists(video_path):
                print(f"❌ Video file not found: {video_path}")
                return
            
            print(f"\n🎥 Starting video file tracking: {video_path}")
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = f"data/{base_name}_tracked.mp4"
            tracker.track_video_full_gpu(video_path=video_path, output_path=output_path)
            
        else:
            print("❌ Invalid choice")
            return
    
    except Exception as e:
        print(f"❌ Error initializing tracker: {e}")
        print("\n🔧 Make sure you have installed required dependencies:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("pip install ultralytics opencv-python mediapipe pyttsx3 requests")
        traceback.print_exc()

def create_default_config():
    """Create default configuration file if it doesn't exist"""
    config_dir = "config"
    config_file = os.path.join(config_dir, "default_config.json")
    
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    
    if not os.path.exists(config_file):
        from src.core.config_manager import ConfigManager
        config_manager = ConfigManager()
        config_manager.save_config(config_file)
        print(f"✅ Created default configuration: {config_file}")

def print_project_structure():
    """Print the project structure"""
    print("\n📁 Project Structure:")
    print("object-tracking/")
    print("├── main.py                    # Main entry point")
    print("├── main_Esp32.py             # Original monolithic file (deprecated)")
    print("├── src/                      # Source code modules")
    print("│   ├── core/                 # Core system components")
    print("│   │   ├── gpu_manager.py    # GPU management and optimization")
    print("│   │   └── config_manager.py # Configuration management")
    print("│   ├── detection/            # Detection modules")
    print("│   │   ├── phone_detector.py # Phone detection with YOLO")
    print("│   │   ├── face_detector.py  # Face detection with YOLO")
    print("│   │   ├── person_detector.py# Person detection and confirmation")
    print("│   │   └── eye_gaze_detector.py # Eye state and gaze tracking")
    print("│   ├── tracking/             # Tracking and feature extraction")
    print("│   │   ├── object_tracker.py # Multi-object tracking")
    print("│   │   └── feature_extractor.py # HRNetV2 feature extraction")
    print("│   ├── communication/        # Communication modules")
    print("│   │   ├── esp32_communicator.py # ESP32 communication")
    print("│   │   └── tts_manager.py    # Text-to-speech management")
    print("│   └── main_tracker.py       # Main tracker integration")
    print("├── models/                   # AI model files")
    print("│   ├── yolo11x.pt           # Phone detection model")
    print("│   ├── yolov11l-face.pt     # Face detection model")
    print("│   └── hrnetv2_w32_imagenet_pretrained.pth # Feature extraction")
    print("├── config/                   # Configuration files")
    print("├── data/                     # Data files and outputs")
    print("├── arduino/                  # ESP32/Arduino code")
    print("│   └── smoke_detector/       # ESP32 project")
    print("├── docs/                     # Documentation")
    print("├── tests/                    # Unit tests")
    print("└── utils/                    # Utility scripts")

if __name__ == "__main__":
    # Print project structure for reference
    print_project_structure()
    
    # Create default config if needed
    create_default_config()
    
    # Run main application
    main()
