# Development Scripts and Commands

## PowerShell Scripts for Windows Development

### Setup Development Environment
```powershell
# setup_dev.ps1
Write-Host "Setting up Object Tracking development environment..." -ForegroundColor Green

# Create virtual environment
if (!(Test-Path "venv")) {
    python -m venv venv
    Write-Host "Created virtual environment" -ForegroundColor Yellow
}

# Activate virtual environment
& .\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Create necessary directories
$directories = @("logs", "output", "temp", "data/output")
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force
        Write-Host "Created directory: $dir" -ForegroundColor Yellow
    }
}

Write-Host "Development environment setup complete!" -ForegroundColor Green
```

### Run Tests
```powershell
# run_tests.ps1
Write-Host "Running Object Tracking test suite..." -ForegroundColor Green

# Activate virtual environment
& .\venv\Scripts\Activate.ps1

# Run all tests with coverage
python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Run specific test modules
Write-Host "Running core tests..." -ForegroundColor Yellow
python -m pytest tests/test_core.py -v

Write-Host "Running detection tests..." -ForegroundColor Yellow
python -m pytest tests/test_detection.py -v

Write-Host "Running integration tests..." -ForegroundColor Yellow
python -m pytest tests/test_integration.py -v

Write-Host "Test execution complete!" -ForegroundColor Green
```

### Code Quality Checks
```powershell
# code_quality.ps1
Write-Host "Running code quality checks..." -ForegroundColor Green

# Activate virtual environment
& .\venv\Scripts\Activate.ps1

# Format code with black
Write-Host "Formatting code with black..." -ForegroundColor Yellow
python -m black src/ tests/ utils/ --line-length 88

# Run flake8 linting
Write-Host "Running flake8 linting..." -ForegroundColor Yellow
python -m flake8 src/ tests/ utils/ --max-line-length 88 --ignore E203,W503

# Type checking (if mypy is installed)
if (Get-Command mypy -ErrorAction SilentlyContinue) {
    Write-Host "Running type checking..." -ForegroundColor Yellow
    python -m mypy src/ --ignore-missing-imports
}

Write-Host "Code quality checks complete!" -ForegroundColor Green
```

### Download Models
```powershell
# download_models.ps1
Write-Host "Downloading required model files..." -ForegroundColor Green

# Activate virtual environment
& .\venv\Scripts\Activate.ps1

# Run model download utility
python utils/download_models.py

Write-Host "Model download complete!" -ForegroundColor Green
```

### Clean Project
```powershell
# clean.ps1
Write-Host "Cleaning project files..." -ForegroundColor Green

# Remove Python cache files
Get-ChildItem -Path . -Recurse -Name "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Path . -Recurse -Name "*.pyc" | Remove-Item -Force
Get-ChildItem -Path . -Recurse -Name "*.pyo" | Remove-Item -Force

# Remove test artifacts
if (Test-Path "htmlcov") { Remove-Item -Recurse -Force "htmlcov" }
if (Test-Path ".coverage") { Remove-Item -Force ".coverage" }
if (Test-Path ".pytest_cache") { Remove-Item -Recurse -Force ".pytest_cache" }

# Remove temporary files
if (Test-Path "temp") { Remove-Item -Recurse -Force "temp/*" }
if (Test-Path "logs") { Remove-Item -Force "logs/*.log*" }

Write-Host "Project cleaning complete!" -ForegroundColor Green
```

### Run Object Tracker
```powershell
# run_tracker.ps1
param(
    [string]$Config = "config/default_config.json",
    [switch]$Debug = $false
)

Write-Host "Starting Object Tracker..." -ForegroundColor Green

# Activate virtual environment
& .\venv\Scripts\Activate.ps1

# Set debug logging if requested
if ($Debug) {
    $env:PYTHONPATH = "$PWD/src"
    Write-Host "Debug mode enabled" -ForegroundColor Yellow
}

# Run the tracker
python main.py --config $Config

Write-Host "Object Tracker stopped." -ForegroundColor Green
```

## Bash Scripts for Linux/macOS Development

### Setup Development Environment
```bash
#!/bin/bash
# setup_dev.sh

echo "Setting up Object Tracking development environment..."

# Create virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Created virtual environment"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Create necessary directories
for dir in logs output temp data/output; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "Created directory: $dir"
    fi
done

echo "Development environment setup complete!"
```

### Run Tests
```bash
#!/bin/bash
# run_tests.sh

echo "Running Object Tracking test suite..."

# Activate virtual environment
source venv/bin/activate

# Run all tests with coverage
python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

echo "Test execution complete!"
```

### Download Models
```bash
#!/bin/bash
# download_models.sh

echo "Downloading required model files..."

# Activate virtual environment
source venv/bin/activate

# Run model download utility
python utils/download_models.py

echo "Model download complete!"
```

## Common Development Tasks

### 1. Initial Setup
```powershell
# Clone repository
git clone https://github.com/your-username/object-tracking.git
cd object-tracking

# Run setup script
.\scripts\setup_dev.ps1
```

### 2. Daily Development Workflow
```powershell
# Activate environment
& .\venv\Scripts\Activate.ps1

# Pull latest changes
git pull origin main

# Run tests to ensure everything works
.\scripts\run_tests.ps1

# Make your changes...

# Check code quality
.\scripts\code_quality.ps1

# Run tests again
.\scripts\run_tests.ps1

# Commit changes
git add .
git commit -m "Your commit message"
git push origin your-branch
```

### 3. Testing Different Configurations
```powershell
# Test with default config
python main.py --config config/default_config.json

# Test with production config
python main.py --config config/production_config.json

# Test with custom config
python main.py --config config/my_custom_config.json
```

### 4. Performance Profiling
```powershell
# Install profiling tools
pip install line_profiler memory_profiler

# Profile memory usage
python -m memory_profiler main.py

# Profile line-by-line execution
kernprof -l -v main.py
```

### 5. Building Documentation
```powershell
# Install documentation tools
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs
sphinx-build -b html . _build
```

### 6. Creating Release Package
```powershell
# Clean project
.\scripts\clean.ps1

# Run full test suite
.\scripts\run_tests.ps1

# Build distribution packages
python setup.py sdist bdist_wheel

# Upload to PyPI (if configured)
pip install twine
twine upload dist/*
```

## Environment Variables

Set these environment variables for development:

```powershell
# Development mode
$env:OBJECT_TRACKER_DEBUG = "1"

# Custom model path
$env:OBJECT_TRACKER_MODELS_PATH = "path/to/models"

# Custom config path
$env:OBJECT_TRACKER_CONFIG = "path/to/config.json"

# Disable GPU for testing
$env:OBJECT_TRACKER_NO_GPU = "1"
```

## IDE Integration

### VS Code Settings
Add to `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./venv/Scripts/python.exe",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
```

### VS Code Tasks
Add to `.vscode/tasks.json`:
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Run Tests",
            "type": "shell",
            "command": "python",
            "args": ["-m", "pytest", "tests/", "-v"],
            "group": "test"
        },
        {
            "label": "Run Tracker",
            "type": "shell",
            "command": "python",
            "args": ["main.py", "--config", "config/default_config.json"],
            "group": "build"
        }
    ]
}
```
