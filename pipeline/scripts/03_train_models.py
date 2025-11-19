#!/usr/bin/env python3
"""
Train Hierarchical Classification Models

This script trains YOLO classification models for hierarchical taxonomic
classification of arthropod specimens using the SQLite database.

It builds the taxonomy hierarchy, creates training datasets, and trains
models for each taxonomic level.

Usage:
    # Basic usage (uses default config)
    python scripts/03_train_models.py

    # Custom parameters
    python scripts/03_train_models.py \
        --set-number 1 \
        --min-images 7 \
        --epochs 50 \
        --device cuda

    # Generate datasets only (no training)
    python scripts/03_train_models.py --skip-training

Methods correspondence:
    - Training Set Generation (72%/18%/10% split, rotation augmentation)
    - Model Training (YOLOv11, 512x512, HSV augmentation)
"""

import argparse
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.utils import get_data_manager
from src.classification.taxonomy import TaxonomyHierarchy
from src.classification.training_data import TrainingDatasetCreator
from src.classification.training import ModelTrainer
from src.config import config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train hierarchical classification models from SQLite database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model versioning
    parser.add_argument(
        '--set-number',
        type=int,
        default=None,
        help='Model set number for version tracking (auto-increments if not specified)'
    )

    # Training parameters
    parser.add_argument(
        '--min-images',
        type=int,
        default=7,
        help='Minimum images required per taxon to train model (default: 7)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (Methods: 50)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=-1,
        help='Training batch size (-1 for auto, default: -1)'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=512,
        help='Image size for training in pixels (Methods: 512)'
    )

    # Data split parameters (from Methods)
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.72,
        help='Training set ratio (Methods: 72%%)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.18,
        help='Validation set ratio (Methods: 18%%)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.10,
        help='Test set ratio (Methods: 10%%)'
    )

    # Augmentation parameters
    parser.add_argument(
        '--rotations',
        nargs='+',
        type=int,
        default=[0, 90, 180, 270],
        help='Rotation angles for augmentation (Methods: 90°, 180°, 270°)'
    )

    # Time filtering
    parser.add_argument(
        '--start-year',
        type=int,
        help='Filter images from this year onwards'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        help='Filter images up to this year'
    )

    # Device selection
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for training (default: cuda)'
    )

    # Base model
    parser.add_argument(
        '--base-model',
        type=str,
        default='yolo11m-cls.pt',
        help='Base YOLO model (Methods: yolo11m-cls.pt)'
    )

    # Paths
    parser.add_argument(
        '--db-path',
        type=Path,
        default=Path('data/arthropod_pipeline.db'),
        help='Path to SQLite database (default: data/arthropod_pipeline.db)'
    )
    parser.add_argument(
        '--training-data-dir',
        type=Path,
        default=Path('./data/training_datasets'),
        help='Directory to save training datasets'
    )
    parser.add_argument(
        '--models-dir',
        type=Path,
        default=Path('./data/models/classification'),
        help='Directory to save trained models'
    )

    # Processing options
    parser.add_argument(
        '--skip-training-data',
        action='store_true',
        help='Skip training data generation (use existing datasets)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training (only generate datasets)'
    )
    parser.add_argument(
        '--delete-datasets-after',
        action='store_true',
        help='Delete training datasets after successful training (saves disk space)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    # Check database exists
    if not args.db_path.exists():
        logger.error(f"Database not found: {args.db_path}")
        logger.error("Run scripts/01_setup_database.py first")
        sys.exit(1)

    # Check ratios sum to 1.0
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 0.01:
        logger.error(f"Train/val/test ratios must sum to 1.0, got {ratio_sum}")
        sys.exit(1)

    # Create output directories
    args.training_data_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Arguments validated:")
    logger.info(f"  Database: {args.db_path}")
    logger.info(f"  Training data: {args.training_data_dir}")
    logger.info(f"  Models output: {args.models_dir}")
    logger.info(f"  Device: {args.device}")
    logger.info("")


def build_taxonomy(data_manager, args):
    """Build taxonomy hierarchy from database."""
    logger.info("Building taxonomy hierarchy from database...")

    taxonomy_builder = TaxonomyHierarchy(
        data_manager=data_manager,
        start_year=args.start_year,
        end_year=args.end_year
    )

    # Build the actual hierarchy
    hierarchy = taxonomy_builder.build()

    logger.info(f"Taxonomy hierarchy built: {len(hierarchy)} taxa")
    logger.info("")
    return taxonomy_builder, hierarchy


def create_training_datasets(data_manager, taxonomy_builder, hierarchy, args):
    """
    Create training datasets using the original GUI workflow.

    This uses TrainingDatasetCreator.generate_training_data() which automatically
    creates datasets for all qualifying taxa in the hierarchy (same as GUI).
    """
    logger.info("Creating training datasets...")
    logger.info(f"  Min images per taxon: {args.min_images}")
    logger.info(f"  Train/Val/Test split: {args.train_ratio:.0%}/{args.val_ratio:.0%}/{args.test_ratio:.0%}")
    logger.info("")

    # Find root taxon (taxon without parent)
    root_taxon = None
    for taxon_id, taxon_data in hierarchy.items():
        if taxon_data.get('parent_id') is None:
            root_taxon = taxon_id
            break

    if not root_taxon:
        logger.error("Could not find root taxon in hierarchy")
        return []

    logger.info(f"Root taxon: {root_taxon}")
    logger.info("")

    # Create TrainingDatasetCreator
    creator = TrainingDatasetCreator(
        base_directory=args.training_data_dir,
        data_manager=data_manager,
        excluded_ids=None,  # Will be tracked by generate_training_data()
        start_year=args.start_year,
        end_year=args.end_year
    )

    # Use the original GUI workflow: generate_training_data()
    # This automatically creates datasets for ALL qualifying taxa
    try:
        logger.info("Generating training datasets (this may take a few minutes)...")
        creator.generate_training_data(
            top_taxon_id=root_taxon,
            min_images_per_class=args.min_images,
            additional_class_train_images=10,  # From GUI defaults
            additional_class_val_images=2,     # From GUI defaults
            num_processes=4,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio
        )
        logger.info("Training data generation complete")
        logger.info("")
    except Exception as e:
        logger.error(f"Failed to generate training data: {e}")
        if args.verbose:
            logger.exception(e)
        return []

    # Scan created directories to build datasets_created list
    datasets_created = []

    if not args.training_data_dir.exists():
        logger.warning(f"Training data directory not found: {args.training_data_dir}")
        return []

    # Each subdirectory in training_data_dir is a dataset for a taxon
    for dataset_dir in args.training_data_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        taxon_id = dataset_dir.name

        # Count images in train/val directories
        train_dir = dataset_dir / 'train'
        val_dir = dataset_dir / 'val'

        if not train_dir.exists() or not val_dir.exists():
            continue

        # Count classes (subdirectories in train/)
        train_classes = [d for d in train_dir.iterdir() if d.is_dir()]
        num_classes = len(train_classes)

        # Count images
        train_count = sum(len(list(class_dir.glob('*.png'))) for class_dir in train_classes)

        val_classes = [d for d in val_dir.iterdir() if d.is_dir()]
        val_count = sum(len(list(class_dir.glob('*.png'))) for class_dir in val_classes)

        datasets_created.append({
            'taxon_id': taxon_id,
            'dataset_path': dataset_dir,
            'num_classes': num_classes,
            'train_count': train_count,
            'val_count': val_count,
            'test_count': 0  # Test images are excluded, not in separate directory
        })

        logger.info(f"Dataset found for {taxon_id}:")
        logger.info(f"  Path: {dataset_dir}")
        logger.info(f"  Classes: {num_classes}")
        logger.info(f"  Train/Val: {train_count}/{val_count}")
        logger.info("")

    logger.info(f"Total training datasets created: {len(datasets_created)}")
    logger.info("")

    return datasets_created


def train_models(data_manager, datasets_created, args):
    """Train YOLO models for all datasets."""
    logger.info("="*70)
    logger.info("TRAINING MODELS")
    logger.info("="*70)
    logger.info(f"Model set number: {args.set_number}")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Image size: {args.image_size}")
    logger.info(f"Device: {args.device}")
    logger.info("")

    trainer = ModelTrainer(
        trainingsets_path=args.training_data_dir,
        models_path=args.models_dir,
        data_manager=data_manager
    )

    # Override config with command line args
    config.set('classification.training.epochs', args.epochs)
    config.set('classification.training.batch_size', args.batch_size)
    config.set('classification.training.image_size', args.image_size)
    config.set('classification.device', args.device)  # Fixed: was classification.training.device
    config.set('classification.training.base_model', args.base_model)

    trained_models = {}

    for i, dataset in enumerate(datasets_created, 1):
        taxon_id = dataset['taxon_id']

        logger.info(f"[{i}/{len(datasets_created)}] Training model for: {taxon_id}")
        logger.info(f"  Classes: {dataset['num_classes']}")
        logger.info(f"  Training images: {dataset['train_count']}")
        logger.info(f"  Validation images: {dataset['val_count']}")
        logger.info("")

        try:
            model_path = trainer.train_model(
                taxon_id=taxon_id,
                set_number=args.set_number,
                base_model_path=Path(args.base_model),
                delete_dataset_after=args.delete_datasets_after
            )

            if model_path:
                trained_models[taxon_id] = model_path
                logger.info(f"  Model saved: {model_path}")
                logger.info("")
            else:
                logger.error(f"  Training failed for {taxon_id}")
                logger.info("")

        except Exception as e:
            logger.error(f"  Training failed for {taxon_id}: {e}")
            if args.verbose:
                logger.exception(e)
            logger.info("")
            continue

    logger.info("="*70)
    logger.info(f"TRAINING COMPLETE: {len(trained_models)}/{len(datasets_created)} models")
    logger.info("="*70)
    logger.info("")

    return trained_models


def save_training_summary(datasets_created, trained_models, args):
    """Save training summary to JSON."""
    summary = {
        'set_number': args.set_number,
        'database': str(args.db_path),
        'training_parameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'image_size': args.image_size,
            'base_model': args.base_model,
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio,
            'test_ratio': args.test_ratio,
            'rotations': args.rotations,
            'min_images': args.min_images,
            'device': args.device
        },
        'datasets': [
            {
                'taxon_id': d['taxon_id'],
                'num_classes': d['num_classes'],
                'train_count': d['train_count'],
                'val_count': d['val_count'],
                'test_count': d['test_count'],
                'dataset_path': str(d['dataset_path'])
            }
            for d in datasets_created
        ],
        'trained_models': [
            {
                'taxon_id': taxon,
                'model_path': str(path)
            }
            for taxon, path in trained_models.items()
        ],
        'summary': {
            'total_datasets_created': len(datasets_created),
            'total_models_trained': len(trained_models),
            'success_rate': f"{len(trained_models)/len(datasets_created)*100:.1f}%" if datasets_created else "N/A"
        }
    }

    summary_file = args.models_dir / f'training_summary_set{args.set_number}.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Training summary saved: {summary_file}")
    logger.info("")

    return summary


def main():
    """Main execution function."""
    args = parse_arguments()

    logger.info("="*70)
    logger.info("HIERARCHICAL MODEL TRAINING")
    logger.info("="*70)
    logger.info("")

    # Validate arguments
    validate_arguments(args)

    # Initialize database connection
    logger.info("Connecting to database...")
    db_url = f"sqlite:///{args.db_path}"
    data_manager = get_data_manager(use_database=True)
    data_manager.db_url = db_url
    data_manager.engine = None  # Force reconnect with new URL
    data_manager.__init__(use_database=True, database_url=db_url)
    logger.info(f"Connected to: {args.db_path}")
    logger.info("")

    # Auto-increment set number if not specified
    if args.set_number is None:
        max_set = data_manager.get_max_set_number()
        args.set_number = (max_set or 0) + 1
        logger.info(f"Auto-assigned set number: {args.set_number}")
        logger.info("")

    # Build taxonomy hierarchy
    taxonomy_builder, hierarchy = build_taxonomy(data_manager, args)

    # Create training datasets
    if args.skip_training_data:
        logger.info("Skipping training data generation (--skip-training-data)")
        logger.info("Assuming datasets already exist in: {args.training_data_dir}")
        logger.info("")
        datasets_created = []
    else:
        datasets_created = create_training_datasets(data_manager, taxonomy_builder, hierarchy, args)

    # Train models
    if args.skip_training:
        logger.info("Skipping model training (--skip-training)")
        logger.info("")
        trained_models = {}
    else:
        if not datasets_created:
            logger.error("No datasets available for training")
            logger.error("Either create datasets first or use existing ones with --skip-training-data")
            data_manager.close()
            sys.exit(1)

        trained_models = train_models(data_manager, datasets_created, args)

    # Save summary
    if datasets_created or trained_models:
        summary = save_training_summary(datasets_created, trained_models, args)

        logger.info("="*70)
        logger.info("TRAINING SESSION SUMMARY")
        logger.info("="*70)
        logger.info(f"Model set: {args.set_number}")
        logger.info(f"Datasets created: {len(datasets_created)}")
        logger.info(f"Models trained: {len(trained_models)}")
        logger.info(f"Models directory: {args.models_dir}")
        logger.info("="*70)
        logger.info("")

    # Commit changes to database
    if data_manager.use_database:
        data_manager.session.commit()
        logger.info("Database changes committed")

    # Close database
    data_manager.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())
