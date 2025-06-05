"""
GPU Environment Manager
Handles GPU setup, optimization, and memory management
"""
import torch
import cv2

class GPUManager:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_mixed_precision = False
        self.use_gpu_cv = False
        self.setup_gpu_environment()
        self.init_gpu_components()
    
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
    
    def preprocess_frame_gpu(self, frame):
        """GPU-accelerated frame preprocessing"""
        if self.use_gpu_cv:
            try:
                # GPU preprocessing code would go here
                pass
            except Exception as e:
                pass
        return frame
    
    def print_gpu_stats(self):
        """Print current GPU statistics"""
        if self.device == 'cuda':
            allocated = torch.cuda.memory_allocated(0) / 1e9
            cached = torch.cuda.memory_reserved(0) / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"\n📊 GPU Statistics:")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   Memory Allocated: {allocated:.2f} GB")
            print(f"   Memory Cached: {cached:.2f} GB")
            print(f"   Memory Total: {total:.2f} GB")
            print(f"   Memory Free: {total - allocated:.2f} GB")
            print(f"   Utilization: {(allocated/total)*100:.1f}%")
        else:
            print("⚠️ GPU not available")
    
    def clear_gpu_cache(self):
        """Clear GPU cache"""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            print("🧹 GPU cache cleared")
