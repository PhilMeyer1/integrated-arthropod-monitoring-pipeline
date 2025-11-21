#!/usr/bin/env python3
"""
Run Hierarchical Classification on Specimens

This script performs hierarchical taxonomic classification on arthropod
specimens using trained YOLO models from the SQLite database.

It iteratively applies models from top-level (e.g., Arthropoda) down to
the most specific taxonomic level, recording confidence scores at each step.

Usage:
    # Classify all specimens with set 1 models
    python scripts/04_classify.py --set-number 1

    # Classify specific range of image IDs
    python scripts/04_classify.py \
        --set-number 1 \
        --start-id 1 \
        --end-id 100

    # Classify specific images
    python scripts/04_classify.py \
        --set-number 1 \
        --image-ids 1 2 3 4 5

    # Use custom batch size
    python scripts/04_classify.py \
        --set-number 1 \
        --batch-size 100

Methods correspondence:
    - Hierarchical Inference (Methods section: Classification Workflow)
    - Batch size: 200 (Methods)
    - Confidence tracking at each taxonomic level
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.utils import get_data_manager
from src.classification.inference import InferenceEngine
from src.config import config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run hierarchical classification on arthropod specimens',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model selection
    parser.add_argument(
        '--set-number',
        type=int,
        required=True,
        help='Model set number to use for classification'
    )

    # Image selection (mutually exclusive)
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument(
        '--image-ids',
        type=int,
        nargs='+',
        help='Specific image IDs to classify (space-separated)'
    )
    image_group.add_argument(
        '--start-id',
        type=int,
        help='Start from this image ID (use with --end-id)'
    )

    parser.add_argument(
        '--end-id',
        type=int,
        help='End at this image ID (use with --start-id)'
    )

    # Processing parameters
    parser.add_argument(
        '--batch-size',
        type=int,
        default=200,
        help='Batch size for inference (Methods: 200)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for inference (default: cuda)'
    )

    # Database path
    parser.add_argument(
        '--db-path',
        type=Path,
        default=Path('data/arthropod_pipeline.db'),
        help='Path to SQLite database (default: data/arthropod_pipeline.db)'
    )

    # Models path
    parser.add_argument(
        '--models-path',
        type=Path,
        default=Path('data/models/classification'),
        help='Directory with trained models'
    )

    # Output options
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1000,
        help='Save results every N images (default: 1000)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Validate range selection
    if args.start_id is not None or args.end_id is not None:
        if args.start_id is None or args.end_id is None:
            parser.error("--start-id and --end-id must be used together")
        if args.start_id > args.end_id:
            parser.error("--start-id must be <= --end-id")

    return args


def validate_arguments(args, data_manager):
    """Validate command line arguments."""
    # Check database exists
    if not args.db_path.exists():
        logger.error(f"Database not found: {args.db_path}")
        logger.error("Run scripts/01_setup_database.py first")
        sys.exit(1)

    # Check models directory exists
    if not args.models_path.exists():
        logger.error(f"Models directory not found: {args.models_path}")
        logger.error(f"Train models first with: python scripts/03_train_models.py --set-number {args.set_number}")
        sys.exit(1)

    # Check if models exist for this set
    models = data_manager.get_models_by_set(args.set_number)
    if not models:
        logger.error(f"No models found for set number {args.set_number}")
        logger.error(f"Available sets: {data_manager.get_max_set_number()}")
        sys.exit(1)

    logger.info("Arguments validated:")
    logger.info(f"  Database: {args.db_path}")
    logger.info(f"  Models set: {args.set_number}")
    logger.info(f"  Models found: {len(models)}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Device: {args.device}")
    logger.info("")


def get_images_to_classify(args, data_manager) -> List[int]:
    """
    Get list of image IDs to classify based on arguments.

    Returns:
        List of image IDs
    """
    if args.image_ids:
        # User specified exact IDs
        image_ids = args.image_ids
        logger.info(f"Classifying {len(image_ids)} specified images")

    elif args.start_id is not None and args.end_id is not None:
        # User specified range
        image_ids = list(range(args.start_id, args.end_id + 1))
        logger.info(f"Classifying images {args.start_id} to {args.end_id} ({len(image_ids)} images)")

    else:
        # Classify all images
        image_ids = data_manager.get_all_image_ids()
        logger.info(f"Classifying all images in database ({len(image_ids)} images)")

    if not image_ids:
        logger.error("No images found to classify")
        sys.exit(1)

    logger.info("")
    return image_ids


def run_classification(args, data_manager, image_ids):
    """
    Run hierarchical classification using InferenceEngine.

    Args:
        args: Command line arguments
        data_manager: Database manager
        image_ids: List of image IDs to classify

    Returns:
        List of inference results
    """
    logger.info("="*70)
    logger.info("STARTING HIERARCHICAL CLASSIFICATION")
    logger.info("="*70)
    logger.info(f"Model set: {args.set_number}")
    logger.info(f"Images to classify: {len(image_ids)}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Device: {args.device}")
    logger.info("")

    # Initialize inference engine
    engine = InferenceEngine(
        set_number=args.set_number,
        data_manager=data_manager,
        models_path=args.models_path
    )

    # Override config with command line args
    config.set('classification.inference.batch_size', args.batch_size)
    config.set('classification.inference.device', args.device)

    # Run inference
    results = engine.run_inference(
        image_ids=image_ids,
        batch_size=args.batch_size
    )

    logger.info("="*70)
    logger.info("CLASSIFICATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Total inferences: {len(results)}")
    logger.info("")

    return results


def print_classification_summary(results, data_manager):
    """
    Print summary statistics of classification results.

    Args:
        results: List of inference result dictionaries
        data_manager: Database manager
    """
    if not results:
        logger.warning("No results to summarize")
        return

    # Count predictions by taxon
    taxon_counts = {}
    for result in results:
        taxon = result.get('predicted_taxon_id')
        if taxon:
            taxon_counts[taxon] = taxon_counts.get(taxon, 0) + 1

    # Count unique images
    unique_images = len(set(r['image_id'] for r in results))

    # Calculate average confidence
    confidences = [r['confidence'] for r in results if r.get('confidence') is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    # Count debris
    debris_count = sum(1 for r in results if r.get('predicted_taxon_id') == 'Debris')

    logger.info("="*70)
    logger.info("CLASSIFICATION SUMMARY")
    logger.info("="*70)
    logger.info(f"Total inferences: {len(results)}")
    logger.info(f"Unique images: {unique_images}")
    logger.info(f"Average confidence: {avg_confidence:.3f}")
    logger.info(f"Debris detected: {debris_count}")
    logger.info("")

    logger.info("Top predicted taxa:")
    sorted_taxa = sorted(taxon_counts.items(), key=lambda x: x[1], reverse=True)
    for taxon, count in sorted_taxa[:10]:
        percentage = count / len(results) * 100
        logger.info(f"  {taxon}: {count} ({percentage:.1f}%)")

    logger.info("="*70)
    logger.info("")


def export_results_summary(results, args):
    """
    Export classification results summary to JSON.

    Args:
        results: List of inference result dictionaries
        args: Command line arguments
    """
    import json

    # Count predictions by taxon
    taxon_counts = {}
    for result in results:
        taxon = result.get('predicted_taxon_id')
        if taxon:
            taxon_counts[taxon] = taxon_counts.get(taxon, 0) + 1

    summary = {
        'set_number': args.set_number,
        'total_inferences': len(results),
        'unique_images': len(set(r['image_id'] for r in results)),
        'batch_size': args.batch_size,
        'device': args.device,
        'taxon_counts': taxon_counts,
        'avg_confidence': sum(r['confidence'] for r in results if r.get('confidence')) / len(results) if results else 0
    }

    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f'classification_summary_set{args.set_number}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Classification summary exported to: {output_file}")
    logger.info("")


def main():
    """Main execution function."""
    args = parse_arguments()

    logger.info("="*70)
    logger.info("HIERARCHICAL CLASSIFICATION PIPELINE")
    logger.info("="*70)
    logger.info("")

    # Initialize database connection
    logger.info("Connecting to database...")
    db_url = f"sqlite:///{args.db_path}"
    data_manager = get_data_manager(use_database=True)
    data_manager.db_url = db_url
    data_manager.engine = None  # Force reconnect with new URL
    data_manager.__init__(use_database=True, database_url=db_url)
    logger.info(f"Connected to: {args.db_path}")
    logger.info("")

    # Validate arguments
    validate_arguments(args, data_manager)

    # Get images to classify
    image_ids = get_images_to_classify(args, data_manager)

    # Run classification
    results = run_classification(args, data_manager, image_ids)

    # Print summary
    print_classification_summary(results, data_manager)

    # Export summary
    export_results_summary(results, args)

    # Close database
    data_manager.close()

    logger.info("Classification pipeline completed successfully!")
    logger.info("")

    return 0


if __name__ == '__main__':
    sys.exit(main())
