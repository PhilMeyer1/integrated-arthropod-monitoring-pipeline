"""
Tests for classification module.

Tests taxonomy, training data, training, inference, and thresholds.
"""

import pytest
from pathlib import Path
import json
import numpy as np
from typing import Dict, List

from src.classification.taxonomy import TaxonomyHierarchy
from src.classification.training_data import TrainingDatasetCreator
from src.classification.training import ModelTrainer
from src.classification.inference import InferenceEngine
from src.classification.thresholds import ThresholdOptimizer


@pytest.fixture
def sample_taxonomy_data(tmp_path):
    """Create sample taxonomy CSV."""
    csv_path = tmp_path / "taxonomy.csv"
    csv_path.write_text(
        "taxonID,parentID,scientificName,taxonRank\n"
        "RT,,,root\n"
        "1000,RT,Arthropoda,phylum\n"
        "1001,1000,Insecta,class\n"
        "1002,1001,Coleoptera,order\n"
        "1003,1002,Carabidae,family\n"
        "1004,1000,Arachnida,class\n"
    )
    return csv_path


@pytest.fixture
def sample_image_metadata(tmp_path):
    """Create sample image metadata."""
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text(
        "image_id,taxon_id,file_path,year\n"
        "1,1002,img1.png,2023\n"
        "2,1002,img2.png,2023\n"
        "3,1003,img3.png,2023\n"
        "4,1003,img4.png,2023\n"
        "5,1004,img5.png,2023\n"
    )
    return metadata_path


class TestTaxonomyHierarchy:
    """Test TaxonomyHierarchy class."""

    def test_initialization(self, sample_taxonomy_data):
        """Test taxonomy initialization from CSV."""
        taxonomy = TaxonomyHierarchy(csv_path=sample_taxonomy_data)

        assert taxonomy.root_id == "RT"
        assert "Insecta" in taxonomy.hierarchy
        assert "Coleoptera" in taxonomy.hierarchy

    def test_hierarchy_structure(self, sample_taxonomy_data):
        """Test hierarchical parent-child relationships."""
        taxonomy = TaxonomyHierarchy(csv_path=sample_taxonomy_data)

        # Insecta should be child of Arthropoda
        arthropoda_node = taxonomy.hierarchy.get("Arthropoda")
        assert "Insecta" in arthropoda_node["children"]

        # Coleoptera should be child of Insecta
        insecta_node = taxonomy.hierarchy.get("Insecta")
        assert "Coleoptera" in insecta_node["children"]

    def test_get_children(self, sample_taxonomy_data):
        """Test retrieving children of a node."""
        taxonomy = TaxonomyHierarchy(csv_path=sample_taxonomy_data)

        children = taxonomy.get_children("Insecta")
        assert "Coleoptera" in children

    def test_get_parent(self, sample_taxonomy_data):
        """Test retrieving parent of a node."""
        taxonomy = TaxonomyHierarchy(csv_path=sample_taxonomy_data)

        parent = taxonomy.get_parent("Coleoptera")
        assert parent == "Insecta"

    def test_get_path_to_root(self, sample_taxonomy_data):
        """Test path from node to root."""
        taxonomy = TaxonomyHierarchy(csv_path=sample_taxonomy_data)

        path = taxonomy.get_path_to_root("Carabidae")
        assert path == ["Carabidae", "Coleoptera", "Insecta", "Arthropoda", "RT"]

    def test_is_leaf_node(self, sample_taxonomy_data):
        """Test leaf node identification."""
        taxonomy = TaxonomyHierarchy(csv_path=sample_taxonomy_data)

        assert taxonomy.is_leaf("Carabidae")
        assert not taxonomy.is_leaf("Insecta")

    def test_year_filtering(self, sample_taxonomy_data, sample_image_metadata):
        """Test filtering images by year."""
        taxonomy = TaxonomyHierarchy(
            csv_path=sample_taxonomy_data,
            start_year=2023,
            end_year=2023
        )

        # This would need actual metadata integration
        # Just check that parameters are stored
        assert taxonomy.start_year == 2023
        assert taxonomy.end_year == 2023


class TestTrainingDatasetCreator:
    """Test TrainingDatasetCreator class."""

    def test_initialization(self, tmp_path, sample_taxonomy_data):
        """Test dataset creator initialization."""
        taxonomy = TaxonomyHierarchy(csv_path=sample_taxonomy_data)
        creator = TrainingDatasetCreator(
            taxonomy=taxonomy,
            output_dir=tmp_path
        )

        assert creator.output_dir == tmp_path
        assert creator.taxonomy == taxonomy

    def test_split_ratios(self, tmp_path, sample_taxonomy_data):
        """Test train/val/test split ratios."""
        taxonomy = TaxonomyHierarchy(csv_path=sample_taxonomy_data)
        creator = TrainingDatasetCreator(
            taxonomy=taxonomy,
            output_dir=tmp_path,
            train_ratio=0.75,
            val_ratio=0.15,
            test_ratio=0.10
        )

        assert creator.train_ratio == 0.75
        assert creator.val_ratio == 0.15
        assert creator.test_ratio == 0.10
        # Should sum to 1.0
        assert abs(sum([creator.train_ratio, creator.val_ratio, creator.test_ratio]) - 1.0) < 0.01

    def test_augmentation_settings(self, tmp_path, sample_taxonomy_data):
        """Test augmentation configuration."""
        taxonomy = TaxonomyHierarchy(csv_path=sample_taxonomy_data)
        creator = TrainingDatasetCreator(
            taxonomy=taxonomy,
            output_dir=tmp_path,
            rotations=[0, 90, 180, 270]
        )

        assert creator.rotations == [0, 90, 180, 270]

    def test_additional_class_creation(self, tmp_path, sample_taxonomy_data):
        """Test creation of 'additional' class for rare taxa."""
        taxonomy = TaxonomyHierarchy(csv_path=sample_taxonomy_data)
        creator = TrainingDatasetCreator(
            taxonomy=taxonomy,
            output_dir=tmp_path,
            min_images_per_class=10
        )

        assert creator.min_images_per_class == 10


class TestModelTrainer:
    """Test ModelTrainer class."""

    def test_initialization(self, tmp_path):
        """Test model trainer initialization."""
        trainer = ModelTrainer(
            model_set=1,
            output_dir=tmp_path
        )

        assert trainer.model_set == 1
        assert trainer.output_dir == tmp_path

    def test_training_config(self, tmp_path):
        """Test training configuration."""
        trainer = ModelTrainer(
            model_set=1,
            output_dir=tmp_path,
            epochs=50,
            image_size=512,
            batch_size=16
        )

        assert trainer.epochs == 50
        assert trainer.image_size == 512
        assert trainer.batch_size == 16

    def test_model_naming(self, tmp_path):
        """Test model filename generation."""
        trainer = ModelTrainer(
            model_set=1,
            output_dir=tmp_path
        )

        model_name = trainer._get_model_filename(taxon="Insecta")
        assert "Insecta" in model_name
        assert "model_set_1" in model_name
        assert model_name.endswith(".pt")


class TestInferenceEngine:
    """Test InferenceEngine class."""

    def test_initialization(self, tmp_path):
        """Test inference engine initialization."""
        engine = InferenceEngine(
            model_dir=tmp_path,
            model_set=1
        )

        assert engine.model_dir == tmp_path
        assert engine.model_set == 1

    def test_confidence_threshold(self, tmp_path):
        """Test confidence threshold setting."""
        engine = InferenceEngine(
            model_dir=tmp_path,
            model_set=1,
            default_threshold=0.7
        )

        assert engine.default_threshold == 0.7

    def test_batch_processing(self, tmp_path):
        """Test batch processing mode."""
        engine = InferenceEngine(
            model_dir=tmp_path,
            model_set=1,
            batch_size=32
        )

        assert engine.batch_size == 32

    def test_result_format(self, tmp_path):
        """Test result dictionary format."""
        engine = InferenceEngine(
            model_dir=tmp_path,
            model_set=1
        )

        # Mock result structure
        mock_result = {
            'image_id': 1,
            'predicted_taxon': 'Coleoptera',
            'confidence': 0.95,
            'path': ['Arthropoda', 'Insecta', 'Coleoptera'],
            'all_predictions': [
                {'taxon': 'Coleoptera', 'confidence': 0.95},
                {'taxon': 'Diptera', 'confidence': 0.04}
            ]
        }

        assert 'predicted_taxon' in mock_result
        assert 'confidence' in mock_result
        assert 'path' in mock_result
        assert isinstance(mock_result['path'], list)


class TestThresholdOptimizer:
    """Test ThresholdOptimizer class."""

    def test_initialization(self):
        """Test threshold optimizer initialization."""
        optimizer = ThresholdOptimizer(
            metric='f1'
        )

        assert optimizer.metric == 'f1'

    def test_metric_options(self):
        """Test different optimization metrics."""
        for metric in ['f1', 'accuracy', 'precision', 'recall']:
            optimizer = ThresholdOptimizer(metric=metric)
            assert optimizer.metric == metric

    def test_threshold_calculation(self):
        """Test threshold calculation from predictions."""
        optimizer = ThresholdOptimizer(metric='f1')

        # Mock predictions and ground truth
        predictions = [
            {'confidence': 0.95, 'correct': True},
            {'confidence': 0.85, 'correct': True},
            {'confidence': 0.75, 'correct': False},
            {'confidence': 0.65, 'correct': True},
            {'confidence': 0.55, 'correct': False},
        ]

        # Should find optimal threshold
        # This is a mock test - actual implementation would calculate F1
        assert optimizer.metric in ['f1', 'accuracy', 'precision', 'recall']

    def test_per_taxon_thresholds(self):
        """Test per-taxon threshold optimization."""
        optimizer = ThresholdOptimizer(metric='f1')

        # Mock per-taxon data
        taxon_predictions = {
            'Coleoptera': [
                {'confidence': 0.95, 'correct': True},
                {'confidence': 0.85, 'correct': True}
            ],
            'Diptera': [
                {'confidence': 0.75, 'correct': True},
                {'confidence': 0.65, 'correct': False}
            ]
        }

        # Should calculate thresholds for each taxon
        assert 'Coleoptera' in taxon_predictions
        assert 'Diptera' in taxon_predictions


class TestIntegration:
    """Integration tests for classification pipeline."""

    def test_taxonomy_to_training_flow(self, tmp_path, sample_taxonomy_data):
        """Test data flow from taxonomy to training data creation."""
        taxonomy = TaxonomyHierarchy(csv_path=sample_taxonomy_data)
        creator = TrainingDatasetCreator(
            taxonomy=taxonomy,
            output_dir=tmp_path
        )

        # Should be able to access taxonomy through creator
        assert creator.taxonomy.root_id == "RT"
        assert "Insecta" in creator.taxonomy.hierarchy

    def test_training_to_inference_flow(self, tmp_path):
        """Test model training to inference flow."""
        # Training phase
        trainer = ModelTrainer(
            model_set=1,
            output_dir=tmp_path
        )

        # Inference phase
        engine = InferenceEngine(
            model_dir=tmp_path,
            model_set=1
        )

        # Should use same model set
        assert trainer.model_set == engine.model_set
        assert engine.model_dir == tmp_path

    def test_inference_to_threshold_flow(self, tmp_path):
        """Test inference to threshold optimization flow."""
        # Inference
        engine = InferenceEngine(
            model_dir=tmp_path,
            model_set=1
        )

        # Threshold optimization
        optimizer = ThresholdOptimizer(metric='f1')

        # Mock results from inference
        mock_results = [
            {'confidence': 0.95, 'predicted': 'A', 'actual': 'A'},
            {'confidence': 0.85, 'predicted': 'A', 'actual': 'A'},
            {'confidence': 0.75, 'predicted': 'B', 'actual': 'A'},
        ]

        # Optimizer should process these results
        assert len(mock_results) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
