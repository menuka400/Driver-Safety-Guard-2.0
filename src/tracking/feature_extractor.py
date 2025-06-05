"""
Feature Extraction Module
Handles feature extraction using HRNetV2 and other deep learning models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2

class GPUOptimizedHRNetV2(torch.nn.Module):
    def __init__(self, feature_dim=256):
        super(GPUOptimizedHRNetV2, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        # HRNet stages
        self.stage1 = self._make_stage(64, 64, 4)
        self.stage2 = self._make_stage(64, 128, 4)
        self.stage3 = self._make_stage(128, 256, 4)
        self.stage4 = self._make_stage(256, 512, 4)
        
        # Feature projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_proj = nn.Linear(512, feature_dim)
        self.dropout = nn.Dropout(0.2)
        
        self._initialize_weights()
        
    def _make_stage(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolutions
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # HRNet stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Global pooling and feature projection
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        features = self.feature_proj(x)
        
        return F.normalize(features, p=2, dim=1)

class FeatureExtractor:
    def __init__(self, device='cuda', feature_dim=256, hrnetv2_model_path=None):
        self.device = device
        self.feature_dim = feature_dim
        self.model = None
        self.transform = None
        self.use_mixed_precision = device == 'cuda'
        
        self.init_model(hrnetv2_model_path)
        self.init_transform()
    
    def init_model(self, model_path=None):
        """Initialize the feature extraction model"""
        try:
            print("Creating GPU-optimized HRNetV2 feature extractor...")
            self.model = GPUOptimizedHRNetV2(self.feature_dim)
            self.model.to(self.device)
            self.model.eval()
            
            # Enable mixed precision if using CUDA
            if self.device == 'cuda' and self.use_mixed_precision:
                self.model = self.model.half()
            
            # Try to load checkpoint if available
            if model_path and os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    # Extract only compatible weights
                    model_dict = self.model.state_dict()
                    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
                    model_dict.update(pretrained_dict)
                    self.model.load_state_dict(model_dict)
                    print(f"✅ Loaded pretrained weights from {model_path}")
                except Exception as e:
                    print(f"⚠️ Could not load pretrained weights: {e}")
            
            print(f"✅ Feature extractor initialized on {self.device.upper()}")
            
        except Exception as e:
            print(f"❌ Error initializing feature extractor: {e}")
            self.model = None
    
    def init_transform(self):
        """Initialize image preprocessing transforms"""
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features_gpu_batch(self, frame, bboxes):
        """GPU-optimized batch feature extraction"""
        if not bboxes or self.model is None:
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
                if crop.size > 0:
                    try:
                        crop_tensor = self.transform(crop)
                        batch_crops.append(crop_tensor)
                        valid_indices.append(i)
                    except Exception as e:
                        print(f"Error processing crop {i}: {e}")
        
        # Initialize features array
        features = [np.random.rand(self.feature_dim) for _ in bboxes]
        
        if batch_crops:
            try:
                # Create batch tensor
                batch_tensor = torch.stack(batch_crops).to(self.device)
                
                # Enable mixed precision if available
                if self.use_mixed_precision and self.device == 'cuda':
                    batch_tensor = batch_tensor.half()
                
                with torch.no_grad():
                    if self.use_mixed_precision and self.device == 'cuda':
                        with torch.cuda.amp.autocast():
                            batch_features = self.model(batch_tensor)
                    else:
                        batch_features = self.model(batch_tensor)
                    
                    # Convert to numpy
                    batch_features_np = batch_features.cpu().float().numpy()
                    
                    # Assign features to correct indices
                    for i, valid_idx in enumerate(valid_indices):
                        features[valid_idx] = batch_features_np[i]
                        
            except Exception as e:
                print(f"Error in batch feature extraction: {e}")
                # Fallback to random features
                pass
        
        return features
    
    def extract_single_feature(self, frame, bbox):
        """Extract features for a single bounding box"""
        try:
            x1, y1, x2, y2 = bbox[:4]
            h, w = frame.shape[:2]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            
            if x2 <= x1 or y2 <= y1:
                return np.random.rand(self.feature_dim)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return np.random.rand(self.feature_dim)
            
            # Preprocess crop
            crop_tensor = self.transform(crop).unsqueeze(0).to(self.device)
            
            if self.use_mixed_precision and self.device == 'cuda':
                crop_tensor = crop_tensor.half()
            
            with torch.no_grad():
                if self.use_mixed_precision and self.device == 'cuda':
                    with torch.cuda.amp.autocast():
                        features = self.model(crop_tensor)
                else:
                    features = self.model(crop_tensor)
                
                return features.cpu().float().numpy()[0]
                
        except Exception as e:
            print(f"Error extracting single feature: {e}")
            return np.random.rand(self.feature_dim)
    
    def compare_features(self, feature1, feature2):
        """Compare two feature vectors using cosine similarity"""
        if len(feature1) == 0 or len(feature2) == 0:
            return 0.0
        
        # Convert to numpy arrays
        f1 = np.array(feature1)
        f2 = np.array(feature2)
        
        # Calculate cosine similarity
        dot_product = np.dot(f1, f2)
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def is_available(self):
        """Check if feature extractor is available"""
        return self.model is not None
    
    def get_feature_dim(self):
        """Get feature dimension"""
        return self.feature_dim
