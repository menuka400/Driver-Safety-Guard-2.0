"""
Model download utility for object tracking project.
Downloads required model files if they don't exist.
"""
import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
import hashlib


def download_file(url, filepath, expected_hash=None):
    """
    Download a file from URL with progress bar and hash verification.
    
    Args:
        url (str): URL to download from
        filepath (str): Local path to save the file
        expected_hash (str, optional): Expected SHA256 hash for verification
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if os.path.exists(filepath):
        print(f"File already exists: {filepath}")
        return True
    
    print(f"Downloading {os.path.basename(filepath)}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Progress") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Verify hash if provided
        if expected_hash:
            if verify_file_hash(filepath, expected_hash):
                print(f"✓ Hash verification passed for {os.path.basename(filepath)}")
            else:
                print(f"✗ Hash verification failed for {os.path.basename(filepath)}")
                os.remove(filepath)
                return False
        
        print(f"✓ Successfully downloaded: {os.path.basename(filepath)}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download {os.path.basename(filepath)}: {str(e)}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False


def verify_file_hash(filepath, expected_hash):
    """
    Verify file SHA256 hash.
    
    Args:
        filepath (str): Path to the file
        expected_hash (str): Expected SHA256 hash
        
    Returns:
        bool: True if hash matches, False otherwise
    """
    sha256_hash = hashlib.sha256()
    
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest() == expected_hash


def download_models():
    """Download all required model files."""
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / "models"
    
    # Model URLs (these would be real URLs in production)
    models = {
        "yolo11x.pt": {
            "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11x.pt",
            "hash": None  # Add actual hash in production
        },
        "yolov11l-face.pt": {
            "url": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l-face.pt",
            "hash": None  # Add actual hash in production
        },
        "hrnetv2_w32_imagenet_pretrained.pth": {
            "url": "https://download.pytorch.org/models/hrnet_w32-36af842e.pth",
            "hash": None  # Add actual hash in production
        }
    }
    
    success_count = 0
    total_count = len(models)
    
    for model_name, model_info in models.items():
        model_path = models_dir / model_name
        
        if download_file(model_info["url"], str(model_path), model_info["hash"]):
            success_count += 1
        else:
            print(f"Failed to download {model_name}")
    
    print(f"\nDownload Summary: {success_count}/{total_count} models downloaded successfully")
    
    if success_count == total_count:
        print("✓ All models are ready!")
        return True
    else:
        print("✗ Some models failed to download. Please check your internet connection.")
        return False


if __name__ == "__main__":
    print("Object Tracking Project - Model Downloader")
    print("=" * 50)
    
    if download_models():
        print("\nSetup complete! You can now run the object tracker.")
        sys.exit(0)
    else:
        print("\nSetup incomplete. Please resolve download issues and try again.")
        sys.exit(1)
