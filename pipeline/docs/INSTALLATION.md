# Installation Guide

Detailed instructions for installing and setting up the Arthropod Classification Pipeline.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Python Installation](#python-installation)
3. [Repository Setup](#repository-setup)
4. [Dependencies](#dependencies)
5. [GPU Setup (Optional)](#gpu-setup-optional)
6. [Database Setup (Optional)](#database-setup-optional)
7. [Verification](#verification)
8. [Troubleshooting](#troubleshooting)

## System Requirements

### Operating System

- **Linux** (Ubuntu 20.04+, recommended)
- **Windows** (10/11)
- **macOS** (10.15+)

### Hardware

**Minimum:**
- 4-core CPU
- 8 GB RAM
- 50 GB free disk space

**Recommended:**
- 8+ core CPU
- 16+ GB RAM
- NVIDIA GPU with 8+ GB VRAM
- 500 GB+ free disk space (for large datasets)

## Python Installation

### Required Version

Python **3.8 or higher** is required.

### Check Current Version

```bash
python --version
# or
python3 --version
```

### Install Python

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.8 python3.8-venv python3-pip
```

**Windows:**
Download from [python.org](https://www.python.org/downloads/) and install.

**macOS:**
```bash
brew install python@3.8
```

## Repository Setup

### 1. Clone Repository

```bash
git clone https://github.com/PhilMeyer1/arthropod-classification-pipeline.git
cd arthropod-classification-pipeline
```

### 2. Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

## Dependencies

### Install Required Packages

```bash
pip install -r requirements.txt
```

This installs:
- `ultralytics` - YOLO models
- `torch` - PyTorch (CPU version)
- `opencv-python` - Image processing
- `pandas` - Data manipulation
- `openpyxl` - Excel export
- `matplotlib` - Plotting
- `pyyaml` - Configuration
- And more...

### Verify Installation

```bash
python -c "import ultralytics; print(ultralytics.__version__)"
python -c "import torch; print(torch.__version__)"
python -c "import cv2; print(cv2.__version__)"
```

## GPU Setup (Optional)

GPU acceleration significantly speeds up training and inference.

### 1. Check GPU Availability

**NVIDIA GPUs only** (AMD/Apple not supported by PyTorch CUDA).

```bash
# Check if you have NVIDIA GPU
nvidia-smi
```

### 2. Install CUDA Toolkit

**Ubuntu:**
```bash
# CUDA 11.8 (recommended for PyTorch 2.0+)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

**Windows:**
Download CUDA Toolkit from [NVIDIA website](https://developer.nvidia.com/cuda-downloads).

### 3. Install PyTorch with CUDA

Uninstall CPU-only PyTorch:
```bash
pip uninstall torch torchvision
```

Install GPU version:
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Verify GPU Support

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Should output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3080
```

## Database Setup (Optional)

The pipeline can run **without** a database using file-based storage. Database is optional for advanced users.

### SQLite (Default, No Setup Needed)

SQLite is included with Python. Just set in `config/default_config.yaml`:

```yaml
database:
  use_database: true
  url: "sqlite:///arthropod_pipeline.db"
```

### PostgreSQL (Advanced)

For multi-user setups:

**1. Install PostgreSQL:**
```bash
# Ubuntu
sudo apt install postgresql postgresql-contrib

# macOS
brew install postgresql
```

**2. Create Database:**
```bash
sudo -u postgres psql
CREATE DATABASE arthropod_pipeline;
CREATE USER your_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE arthropod_pipeline TO your_user;
\q
```

**3. Install Python Driver:**
```bash
pip install psycopg2-binary
```

**4. Configure:**
```yaml
database:
  use_database: true
  url: "postgresql://your_user:your_password@localhost/arthropod_pipeline"
```

**5. Initialize:**
```bash
python scripts/01_setup_database.py
```

## Verification

### 1. Test Imports

```bash
python -c "from src.image_processing import SpecimenDetector; print('✓ Image processing OK')"
python -c "from src.classification import TaxonomyHierarchy; print('✓ Classification OK')"
python -c "from src.export import ExcelExporter; print('✓ Export OK')"
```

### 2. Create Test Images

```bash
python scripts/create_test_images.py
```

Should create synthetic test images in `data/examples/`.

### 3. Run Example

```bash
python scripts/02_process_images.py \
    --input data/examples/composite_S001_1mm.png \
    --sample-id S001 \
    --size-fraction 1 \
    --output-dir ./test_output
```

Check `./test_output/S001/1/` for extracted specimens.

### 4. Check GPU (if installed)

```bash
python -c "from src.utils.validation import check_gpu_available; print(f'GPU: {check_gpu_available()}')"
```

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'ultralytics'`

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### CUDA Errors

**Problem:** `RuntimeError: CUDA out of memory`

**Solution:**
- Reduce batch size in config
- Use smaller YOLO model
- Process fewer images at once
- Use CPU instead: `--device cpu`

**Problem:** `CUDA not available` despite having GPU

**Solution:**
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### OpenCV Errors

**Problem:** `ImportError: libGL.so.1: cannot open shared object file`

**Solution (Linux):**
```bash
sudo apt install libgl1-mesa-glx
```

### Permission Errors

**Problem:** `PermissionError: [Errno 13] Permission denied`

**Solution:**
```bash
# Ensure output directories are writable
chmod -R u+w ./output ./data

# Or use different output directory
python scripts/02_process_images.py --output-dir ~/arthropod_output
```

### Slow Performance

**Problem:** Processing is very slow

**Solutions:**
1. **Use GPU:** Follow GPU setup above
2. **Increase workers:** Edit `config/default_config.yaml`:
   ```yaml
   classification:
     training:
       workers: 16  # Increase
   ```
3. **Use faster model:**
   ```yaml
   classification:
     training:
       base_model: "yolo11n-cls.pt"  # Nano (fastest)
   ```

### Windows-Specific Issues

**Problem:** Scripts don't run on Windows

**Solution:**
Use Python explicitly:
```cmd
python scripts\02_process_images.py --help
```

**Problem:** Path errors with spaces

**Solution:**
Use quotes:
```cmd
python scripts\02_process_images.py --input "C:\Users\My Name\image.png"
```

## Getting Help

If you encounter issues not covered here:

1. **Check logs:** `./logs/pipeline.log`
2. **Search Issues:** https://github.com/PhilMeyer1/arthropod-classification-pipeline/issues
3. **Ask for Help:** Open a new issue with:
   - Python version (`python --version`)
   - OS (`uname -a` or Windows version)
   - Error message
   - Full command you ran

## Next Steps

After successful installation:

1. **Read the main README:** [`../README.md`](../README.md)
2. **Try the example workflow:** [`TESTING.md`](TESTING.md)
3. **Configure for your data:** Edit `config/default_config.yaml`
4. **Process your images:** Start with `scripts/02_process_images.py`

---

**Installation complete!** 🎉

You're ready to process arthropod specimens.
