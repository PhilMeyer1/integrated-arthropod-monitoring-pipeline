# Contributing to Arthropod Classification Pipeline

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:

1. **Clear title** describing the problem
2. **Steps to reproduce** the bug
3. **Expected behavior** vs actual behavior
4. **Environment details:**
   - Python version (`python --version`)
   - OS (`uname -a` or Windows version)
   - Relevant package versions (`pip list | grep ultralytics`)
5. **Error messages** or logs
6. **Code sample** if applicable

### Suggesting Enhancements

For feature requests:

1. Check if the feature already exists
2. Explain the use case
3. Describe the proposed solution
4. Consider backwards compatibility

### Pull Requests

1. **Fork** the repository
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make changes** following the code style below
4. **Test** your changes
5. **Commit** with clear messages:
   ```bash
   git commit -m "Add feature: description"
   ```
6. **Push** to your fork:
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request** on GitHub

## Code Style

### Python Code

Follow **PEP 8** with these specifics:

```python
# Use type hints
def process_image(
    image_path: Path,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Process image with detection.

    Args:
        image_path: Path to image
        threshold: Confidence threshold

    Returns:
        Dictionary with results

    Example:
        >>> result = process_image(Path('image.png'), threshold=0.7)
    """
    pass

# Use descriptive variable names
specimen_count = len(detections)  # Good
n = len(detections)  # Bad

# Keep functions focused
def detect_specimens(image_path):  # Good: one task
    pass

def detect_and_classify_and_export(image_path):  # Bad: too many tasks
    pass
```

### Docstrings

Use **Google-style docstrings**:

```python
def function(arg1: str, arg2: int) -> bool:
    """
    Short description (one line).

    Longer description if needed (optional).

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When something is wrong

    Example:
        >>> function('test', 42)
        True
    """
    pass
```

### Imports

Organize imports:

```python
# Standard library
import os
from pathlib import Path
from typing import Dict, List

# Third-party
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Local
from src.utils.logging_config import get_logger
from src.config import config
```

### Code Formatting

Use **Black** for formatting:

```bash
pip install black
black src/ scripts/ tests/
```

Use **flake8** for linting:

```bash
pip install flake8
flake8 src/ scripts/ tests/
```

## Project Structure

When adding new code:

```
src/
├── your_module/           # New module
│   ├── __init__.py        # Export main classes
│   ├── core.py            # Main functionality
│   ├── utils.py           # Helper functions
│   └── README.md          # Module documentation
scripts/
└── XX_your_script.py      # Executable script
tests/
└── test_your_module.py    # Unit tests
docs/
└── YOUR_MODULE.md         # Additional docs
```

## Testing

### Writing Tests

Use **pytest**:

```python
# tests/test_detection.py
import pytest
from pathlib import Path
from src.image_processing import SpecimenDetector

def test_detection_basic():
    """Test basic detection functionality."""
    detector = SpecimenDetector(size_fraction='1')
    detections = detector.detect_specimens(Path('test.png'))

    assert len(detections) > 0
    assert all('confidence' in d for d in detections)

def test_detection_invalid_image():
    """Test detection with invalid image."""
    detector = SpecimenDetector(size_fraction='1')

    with pytest.raises(FileNotFoundError):
        detector.detect_specimens(Path('nonexistent.png'))
```

### Running Tests

```bash
# Install pytest
pip install pytest

# Run all tests
pytest

# Run specific test file
pytest tests/test_detection.py

# Run with coverage
pip install pytest-cov
pytest --cov=src tests/
```

## Documentation

### Module Documentation

Each module needs a `README.md`:

```markdown
# Module Name

Brief description.

## Quick Start

\`\`\`python
from src.your_module import YourClass

obj = YourClass()
obj.do_something()
\`\`\`

## API Reference

### YourClass

...
```

### Updating Main README

If your changes affect the main README:

1. Update **Overview** if adding major features
2. Update **Quick Start** if changing API
3. Update **Configuration** if adding config options
4. Add to **Troubleshooting** if addressing common issues

## Commit Messages

Use clear, descriptive commit messages:

```bash
# Good
git commit -m "Add hierarchical threshold optimization"
git commit -m "Fix: Handle empty detection results"
git commit -m "Docs: Update installation guide for Windows"

# Bad
git commit -m "Update stuff"
git commit -m "Fix bug"
git commit -m "Changes"
```

Format:
```
<type>: <subject>

<body (optional)>

<footer (optional)>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance (dependencies, build, etc.)

## Versioning

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New features (backwards-compatible)
- **PATCH**: Bug fixes (backwards-compatible)

Example: `1.2.3`
- `1`: Major version
- `2`: Minor version
- `3`: Patch version

## Release Process

1. Update version in relevant files
2. Update `CHANGELOG.md`
3. Create git tag: `git tag v1.2.3`
4. Push tag: `git push origin v1.2.3`
5. Create GitHub release
6. Update Zenodo (if applicable)

## Code Review

All contributions go through code review:

1. **Functionality**: Does it work as intended?
2. **Code quality**: Is it readable and maintainable?
3. **Tests**: Are there tests? Do they pass?
4. **Documentation**: Is it documented?
5. **Style**: Does it follow the code style?

Reviews may request changes. Don't take it personally - it's about making the code better!

## Questions?

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Open an Issue
- **Security issues**: Email the maintainers privately
- **Other**: See README for contact info

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🎉
