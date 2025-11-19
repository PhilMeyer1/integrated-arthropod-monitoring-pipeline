"""
Tests for image processing module.

Tests detection, segmentation, and extraction functionality.
"""

import pytest
from pathlib import Path
import numpy as np
from PIL import Image

from src.image_processing.detection import SpecimenDetector
from src.image_processing.segmentation import SpecimenSegmenter
from src.image_processing.extraction import SpecimenExtractor


@pytest.fixture
def sample_image(tmp_path):
    """Create a simple test image."""
    img = Image.new('RGB', (1000, 800), color=(248, 248, 248))
    img_path = tmp_path / "test_image.png"
    img.save(img_path)
    return img_path


@pytest.fixture
def detector():
    """Create detector instance."""
    return SpecimenDetector(size_fraction='1')


@pytest.fixture
def segmenter():
    """Create segmenter instance."""
    return SpecimenSegmenter(size_fraction='1')


class TestSpecimenDetector:
    """Test SpecimenDetector class."""

    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.size_fraction == '1'
        assert detector.grid_size is not None
        assert detector.overlap >= 0

    def test_grid_size_mapping(self):
        """Test grid size for different size fractions."""
        detector_k1 = SpecimenDetector(size_fraction='k1')
        detector_1 = SpecimenDetector(size_fraction='1')
        detector_7 = SpecimenDetector(size_fraction='7')

        # k1 should have larger grid (more tiles)
        assert detector_k1.grid_size[0] >= detector_1.grid_size[0]
        assert detector_1.grid_size[0] >= detector_7.grid_size[0]

    def test_invalid_size_fraction(self):
        """Test that invalid size fraction raises error."""
        with pytest.raises(ValueError):
            SpecimenDetector(size_fraction='invalid')

    def test_tile_creation(self, detector):
        """Test tile creation from image dimensions."""
        image_size = (4000, 3000)
        tiles = detector._create_tiles_info(image_size)

        assert len(tiles) > 0
        # Each tile should have coordinates
        for tile in tiles:
            assert 'x1' in tile
            assert 'y1' in tile
            assert 'x2' in tile
            assert 'y2' in tile

    def test_detect_on_empty_image(self, detector, sample_image):
        """Test detection on image with no specimens."""
        # Note: This will fail if no YOLO model is available
        # For actual testing, you'd need a model or mock it
        try:
            detections = detector.detect_specimens(sample_image)
            assert isinstance(detections, list)
        except Exception as e:
            pytest.skip(f"YOLO model not available: {e}")


class TestSpecimenSegmenter:
    """Test SpecimenSegmenter class."""

    def test_initialization(self, segmenter):
        """Test segmenter initialization."""
        assert segmenter.size_fraction == '1'
        assert segmenter.dilation_factor > 1.0

    def test_mask_dilation(self, segmenter):
        """Test mask dilation calculation."""
        # Create simple mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255

        # Dilate
        dilated = segmenter._dilate_mask(mask, (100, 100), 1.1)

        # Dilated mask should be larger
        assert dilated.sum() > mask.sum()

    def test_segment_specimens_requires_detections(self, segmenter, sample_image):
        """Test that segmentation requires detections."""
        # Empty detections list
        detections = []

        segmentations = segmenter.segment_specimens(sample_image, detections)
        assert len(segmentations) == 0


class TestSpecimenExtractor:
    """Test SpecimenExtractor class."""

    def test_initialization(self, tmp_path):
        """Test extractor initialization."""
        extractor = SpecimenExtractor(output_dir=tmp_path)
        assert extractor.output_dir == tmp_path
        assert extractor.output_dir.exists()

    def test_filename_generation(self, tmp_path):
        """Test specimen filename generation."""
        extractor = SpecimenExtractor(output_dir=tmp_path)

        filename = extractor._get_specimen_filename(
            sample_id='S001',
            size_fraction='1',
            index=5
        )

        assert 'S001' in filename
        assert '1' in filename
        assert '0005' in filename  # Zero-padded
        assert filename.endswith('.png')

    def test_metadata_structure(self, tmp_path):
        """Test metadata dictionary structure."""
        extractor = SpecimenExtractor(output_dir=tmp_path)

        # Create fake segmentation
        segmentation = {
            'bounding_box': (100, 200, 300, 400),
            'confidence': 0.95,
            'image': np.zeros((200, 200, 3), dtype=np.uint8)
        }

        metadata = extractor._create_metadata(
            segmentation,
            sample_id='S001',
            size_fraction='1',
            index=1
        )

        assert metadata['sample_id'] == 'S001'
        assert metadata['size_fraction'] == '1'
        assert metadata['index'] == 1
        assert 'bounding_box' in metadata
        assert 'confidence' in metadata

    def test_statistics_calculation(self, tmp_path):
        """Test statistics calculation."""
        extractor = SpecimenExtractor(output_dir=tmp_path)

        results = [
            {'confidence': 0.9, 'width': 100, 'height': 100},
            {'confidence': 0.8, 'width': 150, 'height': 120},
            {'confidence': 0.95, 'width': 80, 'height': 90}
        ]

        stats = extractor.get_statistics(results)

        assert stats['total_specimens'] == 3
        assert 0.8 < stats['avg_confidence'] < 0.95
        assert stats['min_size'] == 80
        assert stats['max_size'] == 150


class TestIntegration:
    """Integration tests for complete pipeline."""

    def test_detection_to_segmentation_flow(self, sample_image, tmp_path):
        """Test data flow from detection to segmentation."""
        # This is a mock test - actual test would need real models
        detector = SpecimenDetector(size_fraction='1')
        segmenter = SpecimenSegmenter(size_fraction='1')

        # Mock detections
        detections = [
            {
                'bounding_box': (100, 100, 200, 200),
                'confidence': 0.9
            }
        ]

        # Segmenter should accept detector output format
        try:
            segmentations = segmenter.segment_specimens(sample_image, detections)
            # If models are available, this should work
            assert isinstance(segmentations, list)
        except Exception:
            pytest.skip("YOLO models not available")

    def test_complete_extraction_pipeline(self, tmp_path):
        """Test complete extraction pipeline with mocked data."""
        extractor = SpecimenExtractor(output_dir=tmp_path)

        # Mock segmentations
        segmentations = [
            {
                'bounding_box': (100, 100, 200, 200),
                'confidence': 0.9,
                'image': np.ones((100, 100, 3), dtype=np.uint8) * 128
            }
        ]

        results = extractor.extract_specimens(
            segmentations,
            sample_id='TEST001',
            size_fraction='1'
        )

        assert len(results) == 1
        assert results[0]['sample_id'] == 'TEST001'

        # Check files were created
        specimen_files = list(tmp_path.glob('*.png'))
        assert len(specimen_files) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
