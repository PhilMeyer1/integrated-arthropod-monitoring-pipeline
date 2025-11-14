"""
Hierarchical classification module for arthropod taxonomy.

This module implements a hierarchical YOLO-based classification system
that follows the taxonomic hierarchy (Phylum → Class → Order → Family → etc.).

Modules:
    taxonomy: Taxonomy hierarchy building and management
    training_data: Training dataset creation with augmentation
    training: YOLO model training for each taxonomic level
    inference: Hierarchical inference engine
    thresholds: Confidence threshold optimization

Example:
    >>> from src.classification import (
    ...     TaxonomyHierarchy,
    ...     TrainingDatasetCreator,
    ...     ModelTrainer,
    ...     InferenceEngine,
    ...     ThresholdOptimizer
    ... )
    >>>
    >>> # Build hierarchy
    >>> hierarchy = TaxonomyHierarchy(start_year=2020)
    >>> tree = hierarchy.build()
    >>>
    >>> # Create training data
    >>> creator = TrainingDatasetCreator(Path('./training'))
    >>> creator.generate_training_data(top_taxon_id='RT')
    >>>
    >>> # Train models
    >>> trainer = ModelTrainer(
    ...     trainingsets_path=Path('./training'),
    ...     models_path=Path('./models')
    ... )
    >>> trainer.train_all_models()
    >>>
    >>> # Run inference
    >>> engine = InferenceEngine(set_number=1)
    >>> results = engine.run_inference()
    >>>
    >>> # Optimize thresholds
    >>> optimizer = ThresholdOptimizer()
    >>> optimizer.load_results(set_number=1)
    >>> thresholds = optimizer.optimize_thresholds()
"""

from src.classification.taxonomy import TaxonomyHierarchy
from src.classification.training_data import (
    TrainingDatasetCreator,
    apply_transformations,
    process_image_for_training
)
from src.classification.training import ModelTrainer
from src.classification.inference import InferenceEngine
from src.classification.thresholds import ThresholdOptimizer

__all__ = [
    'TaxonomyHierarchy',
    'TrainingDatasetCreator',
    'ModelTrainer',
    'InferenceEngine',
    'ThresholdOptimizer',
    'apply_transformations',
    'process_image_for_training',
]
