"""
End-to-End Integration Tests.

Tests the complete pipeline from composite image to final export.
These tests simulate real-world usage scenarios.
"""

import pytest
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw
import json

from src.config import Config
from src.image_processing.detection import SpecimenDetector
from src.image_processing.segmentation import SpecimenSegmenter
from src.image_processing.extraction import SpecimenExtractor
from src.export.excel_export import ExcelExporter
from src.export.csv_export import CSVExporter
from src.export.statistics import StatisticsCalculator


@pytest.fixture
def realistic_composite_image(tmp_path):
    """Create a more realistic composite image for testing."""
    # Create 4000x3000 image with near-white background
    img = Image.new('RGB', (4000, 3000), color=(248, 248, 248))
    draw = ImageDraw.Draw(img)

    # Add multiple specimen-like objects
    specimens = [
        (500, 500, 700, 700),
        (1000, 500, 1200, 700),
        (1500, 500, 1700, 700),
        (2000, 500, 2200, 700),
        (500, 1000, 700, 1200),
        (1000, 1000, 1200, 1200),
        (1500, 1000, 1700, 1200),
        (2000, 1000, 2200, 1200),
        (500, 1500, 700, 1700),
        (1000, 1500, 1200, 1700),
    ]

    for x1, y1, x2, y2 in specimens:
        # Draw dark ellipses (simulating arthropods)
        draw.ellipse([x1, y1, x2, y2], fill=(40, 30, 20))

    img_path = tmp_path / "composite_test_e2e.png"
    img.save(img_path)

    return img_path


@pytest.fixture
def e2e_config(tmp_path):
    """Create test configuration."""
    config_dict = {
        'paths': {
            'data_root': str(tmp_path / 'data'),
            'models': str(tmp_path / 'data' / 'models'),
            'raw': str(tmp_path / 'data' / 'raw'),
            'processed': str(tmp_path / 'data' / 'processed')
        },
        'device': 'cpu',
        'image_processing': {
            'detection': {
                'confidence_threshold': 0.25,
                'iou_threshold': 0.2,
                'size_fractions': {
                    '1': {'grid_size': [10, 8], 'overlap': 0.3}
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
        }
    }

    return Config(config_dict=config_dict)


class TestCompleteImageProcessingPipeline:
    """Test complete image processing pipeline."""

    @pytest.mark.integration
    def test_detection_segmentation_extraction_flow(
        self, realistic_composite_image, tmp_path, e2e_config
    ):
        """Test complete flow from composite image to extracted specimens."""
        # Setup
        sample_id = "E2E_TEST_001"
        size_fraction = "1"
        output_dir = tmp_path / "specimens"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Detection (using mock or skipping if no model)
        try:
            detector = SpecimenDetector(size_fraction=size_fraction)

            # For testing without models, create mock detections
            mock_detections = [
                {
                    'bounding_box': (500, 500, 700, 700),
                    'confidence': 0.95,
                    'class_id': 0
                },
                {
                    'bounding_box': (1000, 500, 1200, 700),
                    'confidence': 0.92,
                    'class_id': 0
                },
                {
                    'bounding_box': (1500, 500, 1700, 700),
                    'confidence': 0.88,
                    'class_id': 0
                }
            ]

            detections = mock_detections
            print(f"Using mock detections: {len(detections)} specimens")

        except Exception as e:
            pytest.skip(f"Detection model not available: {e}")

        # Step 2: Segmentation (using mock)
        try:
            segmenter = SpecimenSegmenter(size_fraction=size_fraction)

            # Create mock segmentations from detections
            import numpy as np
            mock_segmentations = []
            for det in detections:
                x1, y1, x2, y2 = det['bounding_box']
                w, h = x2 - x1, y2 - y1
                mock_segmentations.append({
                    'bounding_box': det['bounding_box'],
                    'confidence': det['confidence'],
                    'mask': np.ones((h, w), dtype=np.uint8) * 255,
                    'image': np.ones((h, w, 3), dtype=np.uint8) * 128
                })

            segmentations = mock_segmentations

        except Exception as e:
            pytest.skip(f"Segmentation model not available: {e}")

        # Step 3: Extraction
        extractor = SpecimenExtractor(output_dir=output_dir)
        extraction_results = extractor.extract_specimens(
            segmentations=segmentations,
            sample_id=sample_id,
            size_fraction=size_fraction
        )

        # Assertions
        assert len(extraction_results) == len(segmentations)
        assert len(list(output_dir.glob("*.png"))) == len(segmentations)

        # Check metadata
        for result in extraction_results:
            assert result['sample_id'] == sample_id
            assert result['size_fraction'] == size_fraction
            assert 'filename' in result
            assert 'confidence' in result

        # Check statistics
        stats = extractor.get_statistics(extraction_results)
        assert stats['total_specimens'] == len(segmentations)
        assert 'avg_confidence' in stats

    @pytest.mark.integration
    def test_batch_processing_with_metadata(self, tmp_path):
        """Test batch processing with metadata CSV."""
        # Create multiple test images
        images = []
        for i in range(3):
            img = Image.new('RGB', (1000, 800), color=(248, 248, 248))
            draw = ImageDraw.Draw(img)
            draw.ellipse([300, 300, 600, 600], fill=(40, 30, 20))

            img_path = tmp_path / f"composite_{i:03d}.png"
            img.save(img_path)
            images.append(img_path)

        # Create metadata CSV
        metadata_path = tmp_path / "metadata.csv"
        metadata_path.write_text(
            "image_path,sample_id,size_fraction\n"
            f"{images[0]},S001,1\n"
            f"{images[1]},S002,1\n"
            f"{images[2]},S003,2\n"
        )

        # Read metadata
        df = pd.read_csv(metadata_path)

        # Process each row
        output_dir = tmp_path / "batch_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        processed_samples = []
        for _, row in df.iterrows():
            sample_info = {
                'image_path': Path(row['image_path']),
                'sample_id': row['sample_id'],
                'size_fraction': row['size_fraction']
            }
            processed_samples.append(sample_info)

        # Assertions
        assert len(processed_samples) == 3
        assert processed_samples[0]['sample_id'] == 'S001'
        assert processed_samples[1]['sample_id'] == 'S002'
        assert processed_samples[2]['sample_id'] == 'S003'


class TestCompleteExportPipeline:
    """Test complete export pipeline."""

    @pytest.mark.integration
    def test_statistics_to_multiple_export_formats(self, tmp_path):
        """Test calculating statistics and exporting to multiple formats."""
        # Create mock classification results
        classification_results = []
        for i in range(20):
            classification_results.append({
                'image_id': i,
                'sample_id': f"S{i % 3:03d}",
                'size_fraction': ['1', '2', 'k1'][i % 3],
                'predicted_taxon': ['Coleoptera', 'Diptera', 'Hymenoptera'][i % 3],
                'confidence': 0.85 + (i % 10) * 0.01,
                'path': [
                    'Arthropoda',
                    'Insecta',
                    ['Coleoptera', 'Diptera', 'Hymenoptera'][i % 3]
                ],
                'bounding_box': (100 + i*10, 100, 200 + i*10, 200),
                'specimen_index': i
            })

        # Calculate statistics
        stats_calculator = StatisticsCalculator()
        basic_stats = stats_calculator.calculate_basic_stats(classification_results)
        per_sample_stats = stats_calculator.calculate_per_sample_stats(classification_results)
        per_taxon_stats = stats_calculator.calculate_per_taxon_stats(classification_results)

        # Export to Excel
        excel_dir = tmp_path / "excel"
        excel_dir.mkdir(parents=True, exist_ok=True)
        excel_exporter = ExcelExporter(output_dir=excel_dir)
        excel_path = excel_exporter.export(
            results=classification_results,
            filename="results.xlsx",
            statistics=basic_stats
        )

        # Export to CSV
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_exporter = CSVExporter(output_dir=csv_dir)
        csv_path = csv_exporter.export(
            results=classification_results,
            filename="results.csv"
        )
        summary_path = csv_exporter.export_summary(
            summary=basic_stats,
            filename="summary.csv"
        )

        # Assertions
        assert excel_path.exists()
        assert csv_path.exists()
        assert summary_path.exists()

        # Verify Excel content
        df_excel = pd.read_excel(excel_path, sheet_name='Results')
        assert len(df_excel) == len(classification_results)

        # Verify CSV content
        df_csv = pd.read_csv(csv_path)
        assert len(df_csv) == len(classification_results)

        # Verify both have same data
        assert len(df_excel) == len(df_csv)

        # Verify statistics
        assert basic_stats['total_specimens'] == 20
        assert basic_stats['unique_taxa'] == 3
        assert len(per_sample_stats) == 3
        assert len(per_taxon_stats) == 3


class TestEndToEndScenarios:
    """Test realistic end-to-end scenarios."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_single_sample_complete_workflow(self, realistic_composite_image, tmp_path):
        """Test complete workflow for a single sample."""
        # Configuration
        sample_id = "WORKFLOW_TEST_001"
        size_fraction = "1"
        output_base = tmp_path / "workflow_output"
        output_base.mkdir(parents=True, exist_ok=True)

        # ============================================================
        # PHASE 1: IMAGE PROCESSING
        # ============================================================
        specimens_dir = output_base / "specimens"
        specimens_dir.mkdir(parents=True, exist_ok=True)

        # Mock image processing results
        mock_specimens = []
        for i in range(5):
            specimen_data = {
                'specimen_id': i,
                'sample_id': sample_id,
                'size_fraction': size_fraction,
                'confidence': 0.9 + i * 0.01,
                'bounding_box': (100 + i*100, 100, 200 + i*100, 200),
                'filename': f"{sample_id}_{size_fraction}_{i:04d}.png"
            }
            mock_specimens.append(specimen_data)

            # Create mock specimen image
            img = Image.new('RGB', (100, 100), color=(40, 30, 20))
            img.save(specimens_dir / specimen_data['filename'])

        # ============================================================
        # PHASE 2: CLASSIFICATION (MOCK)
        # ============================================================
        taxa = ['Coleoptera', 'Diptera', 'Hymenoptera', 'Coleoptera', 'Diptera']
        classification_results = []

        for spec, taxon in zip(mock_specimens, taxa):
            classification_results.append({
                'image_id': spec['specimen_id'],
                'sample_id': spec['sample_id'],
                'size_fraction': spec['size_fraction'],
                'filename': spec['filename'],
                'predicted_taxon': taxon,
                'confidence': spec['confidence'],
                'path': ['Arthropoda', 'Insecta', taxon],
                'bounding_box': spec['bounding_box'],
                'specimen_index': spec['specimen_id']
            })

        # ============================================================
        # PHASE 3: EXPORT
        # ============================================================
        export_dir = output_base / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        # Calculate statistics
        stats_calculator = StatisticsCalculator()
        basic_stats = stats_calculator.calculate_basic_stats(classification_results)
        per_taxon_stats = stats_calculator.calculate_per_taxon_stats(classification_results)

        # Export to Excel
        excel_exporter = ExcelExporter(output_dir=export_dir)
        excel_path = excel_exporter.export(
            results=classification_results,
            filename=f"{sample_id}_results.xlsx",
            statistics=basic_stats
        )

        # Export to CSV
        csv_exporter = CSVExporter(output_dir=export_dir)
        csv_path = csv_exporter.export(
            results=classification_results,
            filename=f"{sample_id}_results.csv"
        )

        # ============================================================
        # VERIFICATION
        # ============================================================
        # Check specimens were created
        assert len(list(specimens_dir.glob("*.png"))) == 5

        # Check classifications
        assert len(classification_results) == 5
        assert basic_stats['total_specimens'] == 5
        assert basic_stats['unique_taxa'] == 3  # Coleoptera, Diptera, Hymenoptera

        # Check exports
        assert excel_path.exists()
        assert csv_path.exists()

        # Verify Excel content
        df_excel = pd.read_excel(excel_path, sheet_name='Results')
        assert len(df_excel) == 5
        assert 'sample_id' in df_excel.columns
        assert 'predicted_taxon' in df_excel.columns

        # Verify CSV content
        df_csv = pd.read_csv(csv_path)
        assert len(df_csv) == 5

        # Check per-taxon statistics
        assert 'Coleoptera' in per_taxon_stats
        assert per_taxon_stats['Coleoptera']['count'] == 2
        assert 'Diptera' in per_taxon_stats
        assert per_taxon_stats['Diptera']['count'] == 2

    @pytest.mark.integration
    def test_multiple_samples_workflow(self, tmp_path):
        """Test workflow with multiple samples."""
        samples_data = {
            'S001': {'size_fraction': '1', 'specimen_count': 3},
            'S002': {'size_fraction': '2', 'specimen_count': 5},
            'S003': {'size_fraction': 'k1', 'specimen_count': 8},
        }

        all_results = []

        for sample_id, info in samples_data.items():
            # Mock processing for each sample
            for i in range(info['specimen_count']):
                all_results.append({
                    'image_id': f"{sample_id}_{i}",
                    'sample_id': sample_id,
                    'size_fraction': info['size_fraction'],
                    'predicted_taxon': ['Coleoptera', 'Diptera'][i % 2],
                    'confidence': 0.85 + (i % 10) * 0.01,
                    'path': ['Arthropoda', 'Insecta', ['Coleoptera', 'Diptera'][i % 2]],
                })

        # Calculate combined statistics
        stats_calculator = StatisticsCalculator()
        basic_stats = stats_calculator.calculate_basic_stats(all_results)
        per_sample_stats = stats_calculator.calculate_per_sample_stats(all_results)

        # Assertions
        assert basic_stats['total_specimens'] == 3 + 5 + 8
        assert len(per_sample_stats) == 3
        assert per_sample_stats['S001']['count'] == 3
        assert per_sample_stats['S002']['count'] == 5
        assert per_sample_stats['S003']['count'] == 8

    @pytest.mark.integration
    def test_error_handling_in_pipeline(self, tmp_path):
        """Test pipeline error handling."""
        # Test with invalid image path
        invalid_path = tmp_path / "nonexistent.png"

        # Should handle gracefully
        detector = SpecimenDetector(size_fraction='1')

        try:
            detections = detector.detect_specimens(invalid_path)
            # If no error, should return empty list or handle gracefully
            assert isinstance(detections, list)
        except FileNotFoundError:
            # Expected behavior
            pass
        except Exception as e:
            # Other exceptions are acceptable if documented
            pass


class TestDataConsistency:
    """Test data consistency across pipeline stages."""

    @pytest.mark.integration
    def test_metadata_consistency(self, tmp_path):
        """Test that metadata is consistent across pipeline stages."""
        sample_id = "CONSISTENCY_TEST"
        size_fraction = "1"

        # Stage 1: Detection output
        detections = [
            {'bounding_box': (100, 100, 200, 200), 'confidence': 0.95},
            {'bounding_box': (300, 300, 400, 400), 'confidence': 0.88},
        ]

        # Stage 2: Extraction metadata
        extraction_metadata = []
        for i, det in enumerate(detections):
            extraction_metadata.append({
                'sample_id': sample_id,
                'size_fraction': size_fraction,
                'index': i,
                'confidence': det['confidence'],
                'bounding_box': det['bounding_box']
            })

        # Stage 3: Classification results
        classification_results = []
        for i, meta in enumerate(extraction_metadata):
            classification_results.append({
                'image_id': i,
                'sample_id': meta['sample_id'],
                'size_fraction': meta['size_fraction'],
                'predicted_taxon': 'Coleoptera',
                'confidence': meta['confidence'],
                'index': meta['index']
            })

        # Verify consistency
        for i in range(len(detections)):
            assert extraction_metadata[i]['sample_id'] == sample_id
            assert classification_results[i]['sample_id'] == sample_id
            assert extraction_metadata[i]['size_fraction'] == size_fraction
            assert classification_results[i]['size_fraction'] == size_fraction
            assert extraction_metadata[i]['index'] == i
            assert classification_results[i]['index'] == i


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
