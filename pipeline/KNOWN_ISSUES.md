# Known Issues

This document lists known issues, limitations, and workarounds for the Arthropod Classification Pipeline.

**Last Updated**: 2025-10-27
**Version**: 1.0.0

---

## Critical Issues

### None Currently

All critical issues have been resolved in v1.0.0.

---

## Minor Issues

### 1. Unicode Display in Windows Console
**Status**: ⚠️ Known limitation
**Severity**: Low (cosmetic only)
**Affected**: Windows CMD users

**Description:**
Checkmark characters (✓, ✗) and other Unicode symbols cannot be displayed in Windows Command Prompt with default CP1252 encoding, resulting in logging errors.

**Impact:**
- Console shows UnicodeEncodeError messages
- Log file output is correct
- Functionality is not affected

**Workaround:**
Use one of these alternatives:
```bash
# Option 1: Use Windows Terminal (recommended)
# Download from Microsoft Store

# Option 2: Use PowerShell 7+
pwsh

# Option 3: Redirect output to file
python scripts/01_setup_database.py > output.log 2>&1
```

**Technical Details:**
- Python logging tries to write UTF-8 to CP1252 console
- File logging works correctly (UTF-8 encoding)
- No data loss or functionality impairment

---

### 2. Database Recreation Required After Schema Updates
**Status**: ⚠️ By design
**Severity**: Low
**Affected**: Users updating from earlier versions

**Description:**
SQLite does not support easy schema migrations. If the database schema is updated, the database must be recreated.

**Impact:**
- Cannot update existing database in-place
- Must export data before schema update

**Workaround:**
```bash
# Before updating code:
python scripts/export_database.py --output backup.json

# After updating code:
rm data/arthropod_pipeline.db
python scripts/01_setup_database.py
python scripts/import_database.py --input backup.json
```

**Note:** For production use with large datasets, consider migrating to PostgreSQL which supports Alembic migrations.

---

## Limitations

### 1. Single-User Database Access
**Status**: By design
**Severity**: Low

**Description:**
SQLite does not support concurrent writes. Only one process can write to the database at a time.

**Impact:**
- Cannot run multiple training/classification processes simultaneously
- Not suitable for multi-user environments

**Workaround:**
- Run pipeline steps sequentially
- For multi-user scenarios, migrate to PostgreSQL

**Code Changes Required for PostgreSQL:**
```python
# config/default_config.yaml
database:
  url: "postgresql://user:pass@localhost/arthropod_db"
```

---

### 2. Memory Usage with Large Batches
**Status**: By design
**Severity**: Low

**Description:**
Large batch sizes (>500) may cause memory issues on systems with limited RAM.

**Impact:**
- Out of memory errors on systems with <8 GB RAM
- Slower processing if batch size is too small

**Recommended Settings:**
```yaml
# For 8 GB RAM
batch_size: 200  # Default, Methods-compliant

# For 16+ GB RAM
batch_size: 500

# For 4-6 GB RAM
batch_size: 50-100
```

---

### 3. GPU Memory Overflow
**Status**: By design
**Severity**: Low

**Description:**
Training with large image sizes (>640px) on GPUs with <6 GB VRAM may cause CUDA out-of-memory errors.

**Impact:**
- Training fails with CUDA OOM
- Must use CPU mode or smaller images

**Workaround:**
```bash
# Option 1: Use CPU
python scripts/03_train_models.py --device cpu

# Option 2: Reduce image size
python scripts/03_train_models.py --image-size 416

# Option 3: Reduce batch size
python scripts/03_train_models.py --batch-size 4
```

---

## Platform-Specific Issues

### Windows

#### Long Path Names
**Issue**: Windows has 260-character path limit
**Impact**: Deep nested directories may fail
**Workaround**: Keep project path short (e.g., `C:\arthropod-pipeline\`)

#### File Permissions
**Issue**: Antivirus may block model downloads
**Impact**: YOLO model download failures
**Workaround**: Add project folder to antivirus exceptions

### macOS

#### M1/M2 GPU Support
**Issue**: Limited YOLO support for Apple Silicon GPU
**Impact**: May fall back to CPU
**Workaround**: Use `--device cpu` explicitly

### Linux

#### CUDA Version Mismatch
**Issue**: PyTorch CUDA version must match system CUDA
**Impact**: GPU not detected
**Workaround**: Install correct PyTorch version:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Performance Notes

### Expected Processing Times

**Standard Laptop** (i7-9700K, RTX 2070, 16GB RAM):
- Database setup: <1 minute
- Process 1 composite image (150 specimens): 1-2 minutes
- Train 1 model (50 epochs, 100 images): 3-5 minutes
- Classify 150 specimens: 30-60 seconds

**Server** (Xeon, RTX 3090, 64GB RAM):
- Database setup: <30 seconds
- Process 1 composite image: 30-60 seconds
- Train 1 model: 1-2 minutes
- Classify 150 specimens: 10-20 seconds

**CPU-Only** (i7-9700K, 16GB RAM):
- Database setup: <1 minute
- Process 1 composite image: 2-3 minutes
- Train 1 model: 15-20 minutes ⚠️
- Classify 150 specimens: 2-3 minutes

---

## Compatibility

### Tested Configurations

✅ **Fully Tested:**
- Windows 10/11 + Python 3.8-3.11 + CUDA 11.8
- Ubuntu 20.04/22.04 + Python 3.8-3.11 + CUDA 11.8
- macOS 12+ + Python 3.8-3.11 (CPU only)

⚠️ **Partially Tested:**
- Windows 11 + Python 3.12 (some dependency issues)
- Apple Silicon M1/M2 (limited GPU support)
- CUDA 12.x (works but not extensively tested)

❌ **Not Supported:**
- Python < 3.8 (missing features)
- Python 3.13+ (dependency compatibility issues)
- Windows 7 (EOL)

---

## Debugging Tips

### Issue: "Database is locked"
**Cause**: Another process is accessing the database
**Solution:**
```bash
# Check for running processes
ps aux | grep python

# Kill if necessary
killall python

# Or use absolute path
rm -f data/arthropod_pipeline.db-journal
```

### Issue: "No module named 'ultralytics'"
**Cause**: Dependencies not installed
**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: "CUDA out of memory"
**Cause**: GPU memory exhausted
**Solution:**
```bash
# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"

# Use smaller batch size
python scripts/03_train_models.py --batch-size 4
```

### Issue: "Cannot find YOLO model"
**Cause**: Base model not downloaded
**Solution:**
```bash
# Ultralytics will auto-download on first use
# Or manually download:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11m-cls.pt
mv yolo11m-cls.pt models/
```

---

## Reporting Issues

If you encounter an issue not listed here:

1. Check [TEST_REPORT.md](TEST_REPORT.md) for test status
2. Check GitHub Issues: https://github.com/PhilMeyer1/arthropod-pipeline/issues
3. Provide the following information:
   - Operating system and version
   - Python version (`python --version`)
   - Full error message
   - Steps to reproduce
   - Relevant log files (`logs/pipeline.log`)

---

## Future Improvements

### Planned
- [ ] Alembic database migrations
- [ ] Multi-process support
- [ ] Better progress bars
- [ ] Resume capability for interrupted training
- [ ] Docker container for reproducibility

### Under Consideration
- [ ] PostgreSQL backend option
- [ ] Web UI for result visualization
- [ ] Automatic hyperparameter tuning
- [ ] Model compression for faster inference

---

**Last Reviewed**: 2025-10-27
**Next Review**: Before v2.0.0 release
