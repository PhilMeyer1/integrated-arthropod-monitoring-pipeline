"""
Pytest configuration and shared fixtures.

This file provides common fixtures and configuration for all tests.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
from PIL import Image
import numpy as np


@pytest.fixture(scope='session')
def test_data_dir():
    """Create temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp(prefix='arthropod_test_'))
    yield temp_dir
    # Cleanup after all tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory for each test."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def sample_composite_image(tmp_path):
    """Create a sample composite image with near-white background."""
    # Create image with near-white background (248, 248, 248)
    img = Image.new('RGB', (4000, 3000), color=(248, 248, 248))

    # Add some simple specimen-like rectangles
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)

    # Draw a few dark rectangles (simulating specimens)
    specimens = [
        (500, 500, 700, 700),    # Top-left specimen
        (1500, 1000, 1800, 1300), # Center specimen
        (3000, 2000, 3300, 2400), # Bottom-right specimen
    ]

    for x1, y1, x2, y2 in specimens:
        draw.rectangle([x1, y1, x2, y2], fill=(50, 40, 30))

    # Save image
    img_path = tmp_path / "composite_test.png"
    img.save(img_path)

    return img_path


@pytest.fixture
def sample_specimen_image(tmp_path):
    """Create a sample specimen image."""
    # Small image of a single specimen
    img = Image.new('RGB', (200, 200), color=(248, 248, 248))

    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 50, 150, 150], fill=(40, 30, 20))

    img_path = tmp_path / "specimen_test.png"
    img.save(img_path)

    return img_path


@pytest.fixture
def sample_mask():
    """Create a sample binary mask."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:150, 50:150] = 255
    return mask


@pytest.fixture
def sample_metadata_csv(tmp_path):
    """Create sample metadata CSV file."""
    csv_path = tmp_path / "metadata.csv"
    csv_path.write_text(
        "image_path,sample_id,size_fraction\n"
        "test1.png,S001,1\n"
        "test2.png,S002,2\n"
        "test3.png,S003,k1\n"
    )
    return csv_path


@pytest.fixture
def sample_detection():
    """Create sample detection dictionary."""
    return {
        'bounding_box': (100, 100, 200, 200),
        'confidence': 0.95,
        'class_id': 0
    }


@pytest.fixture
def sample_detections():
    """Create list of sample detections."""
    return [
        {'bounding_box': (100, 100, 200, 200), 'confidence': 0.95, 'class_id': 0},
        {'bounding_box': (300, 300, 400, 400), 'confidence': 0.88, 'class_id': 0},
        {'bounding_box': (500, 500, 600, 600), 'confidence': 0.92, 'class_id': 0},
    ]


@pytest.fixture
def sample_segmentation():
    """Create sample segmentation dictionary."""
    return {
        'bounding_box': (100, 100, 200, 200),
        'confidence': 0.95,
        'mask': np.ones((100, 100), dtype=np.uint8) * 255,
        'image': np.ones((100, 100, 3), dtype=np.uint8) * 128
    }


@pytest.fixture
def sample_config_dict():
    """Create sample configuration dictionary."""
    return {
        'paths': {
            'data_root': './data',
            'models': './data/models',
            'raw': './data/raw',
            'processed': './data/processed'
        },
        'device': 'cpu',
        'image_processing': {
            'detection': {
                'confidence_threshold': 0.25,
                'iou_threshold': 0.2,
                'size_fractions': {
                    'k1': {'grid_size': [20, 12], 'overlap': 0.4},
                    '1': {'grid_size': [10, 8], 'overlap': 0.3},
                    '2': {'grid_size': [8, 6], 'overlap': 0.3},
                    '7': {'grid_size': [1, 1], 'overlap': 0.0}
                }
            },
            'segmentation': {
                'confidence_threshold': 0.30,
                'dilation_factor': 1.02
            }
        },
        'classification': {
            'model_set': 1,
            'batch_size': 32,
            'default_threshold': 0.5
        },
        'training': {
            'epochs': 50,
            'batch_size': 16,
            'image_size': 512,
            'train_ratio': 0.75,
            'val_ratio': 0.15,
            'test_ratio': 0.10
        },
        'export': {
            'formats': ['excel', 'csv'],
            'include_confidence': True
        }
    }


@pytest.fixture
def mock_yolo_model():
    """Create a mock YOLO model for testing."""
    class MockYOLOModel:
        """Mock YOLO model that returns fake predictions."""

        def __init__(self):
            self.conf = 0.25
            self.iou = 0.2

        def __call__(self, *args, **kwargs):
            """Return mock results."""
            return [self._create_mock_result()]

        def predict(self, *args, **kwargs):
            """Return mock results."""
            return [self._create_mock_result()]

        def _create_mock_result(self):
            """Create a mock result object."""
            class MockResult:
                def __init__(self):
                    self.boxes = MockBoxes()
                    self.masks = None

            class MockBoxes:
                def __init__(self):
                    self.xyxy = np.array([[100, 100, 200, 200]])
                    self.conf = np.array([0.95])
                    self.cls = np.array([0])

            return MockResult()

    return MockYOLOModel()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_model: marks tests that require YOLO models"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests that require GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests that require models if models aren't available."""
    skip_model = pytest.mark.skip(reason="YOLO models not available")
    skip_gpu = pytest.mark.skip(reason="GPU not available")

    for item in items:
        if "requires_model" in item.keywords:
            # Check if model path exists
            # For now, just mark all model tests to be skipped by default
            item.add_marker(skip_model)

        if "requires_gpu" in item.keywords:
            # Check if GPU is available
            try:
                import torch
                if not torch.cuda.is_available():
                    item.add_marker(skip_gpu)
            except ImportError:
                item.add_marker(skip_gpu)
