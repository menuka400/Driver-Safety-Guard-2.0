"""
Setup script for Object Tracking Project.
Enables pip installation and development setup.
"""
from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    """Read README.md file for long description."""
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Advanced object tracking system with phone detection, face detection, and ESP32 integration."

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt."""
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "ultralytics>=8.0.0",
            "opencv-python>=4.8.0",
            "numpy>=1.24.0",
            "Pillow>=9.5.0",
            "mediapipe>=0.10.0",
            "scipy>=1.10.0",
            "scikit-learn>=1.3.0",
            "requests>=2.31.0",
            "pyttsx3>=2.90",
        ]

# Get version from __init__.py
def get_version():
    """Get version from src/__init__.py."""
    version_file = os.path.join("src", "__init__.py")
    if os.path.exists(version_file):
        with open(version_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"\'')
    return "1.0.0"

setup(
    name="object-tracking-system",
    version=get_version(),
    author="Object Tracking Team",
    author_email="contact@objecttracking.com",
    description="Advanced GPU-optimized object tracking system with multi-modal detection",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/object-tracking",
    project_urls={
        "Bug Reports": "https://github.com/your-username/object-tracking/issues",
        "Source": "https://github.com/your-username/object-tracking",
        "Documentation": "https://github.com/your-username/object-tracking/docs",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Video :: Capture",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
            "torchvision[cuda]>=0.15.0",
        ],
        "visualization": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "object-tracker=main:main",
            "download-models=utils.download_models:main",
            "setup-logging=utils.logging_utils:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.txt", "*.md"],
    },
    data_files=[
        ("config", ["config/default_config.json", "config/production_config.json"]),
        ("docs", ["docs/API.md", "docs/INSTALLATION.md"]),
    ],
    zip_safe=False,
    keywords=[
        "object-tracking", "computer-vision", "yolo", "face-detection", 
        "phone-detection", "esp32", "pytorch", "opencv", "mediapipe",
        "real-time", "gpu-acceleration", "machine-learning", "ai"
    ],
    platforms=["any"],
    license="MIT",
)

# Post-installation setup
def post_install():
    """Run post-installation setup tasks."""
    import subprocess
    import sys
    
    print("Running post-installation setup...")
    
    # Create necessary directories
    directories = ["logs", "output", "temp"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Check for model files
    models_dir = "models"
    required_models = [
        "yolo11x.pt",
        "yolov11l-face.pt", 
        "hrnetv2_w32_imagenet_pretrained.pth"
    ]
    
    missing_models = []
    for model in required_models:
        model_path = os.path.join(models_dir, model)
        if not os.path.exists(model_path):
            missing_models.append(model)
    
    if missing_models:
        print(f"Warning: Missing model files: {missing_models}")
        print("Run 'download-models' command to download required models.")
    else:
        print("All required model files are present.")
    
    print("Post-installation setup completed!")

if __name__ == "__main__":
    # Run post-install if this script is executed directly
    if len(sys.argv) > 1 and sys.argv[1] == "develop":
        post_install()
