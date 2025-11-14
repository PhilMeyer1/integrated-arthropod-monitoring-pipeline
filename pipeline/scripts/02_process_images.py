#!/usr/bin/env python3
"""
Script 02: Process Images - Detection, Segmentation, Extraction

This script processes composite images to detect and extract individual specimens.
Corresponds to Methods section "Image Analysis and Specimen Extraction".

Usage:
    python scripts/02_process_images.py \\
        --input composite_image.png \\
        --sample-id S001 \\
        --size-fraction 1 \\
        --output-dir ./output/specimens

    # Process from metadata CSV
    python scripts/02_process_images.py \\
        --metadata metadata.csv \\
        --output-dir ./output/specimens

    # Batch process directory
    python scripts/02_process_images.py \\
        --input-dir ./composites \\
        --output-dir ./output/specimens \\
        --pattern "*.png"
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.image_processing.extraction import SpecimenExtractor
from src.utils.logging_config import setup_logging, LogContext, ProgressLogger
from src.utils.validation import (
    validate_file_exists,
    validate_directory_exists,
    validate_image_file,
    validate_size_fraction,
    validate_sample_id
)
from src.config import config


def process_single_image(
    image_path: Path,
    sample_id: str,
    size_fraction: str,
    output_dir: Path,
    logger
) -> dict:
    """
    Process a single composite image.

    Args:
        image_path: Path to composite image
        sample_id: Sample identifier
        size_fraction: Size fraction code
        output_dir: Output directory
        logger: Logger instance

    Returns:
        Result dictionary with metadata
    """
    # Validate inputs
    validate_image_file(image_path)
    validate_sample_id(sample_id)
    validate_size_fraction(size_fraction)

    logger.info(f"Processing {image_path.name}")
    logger.info(f"  Sample ID: {sample_id}")
    logger.info(f"  Size fraction: {size_fraction}")

    # Create extractor
    sample_output_dir = output_dir / sample_id / size_fraction
    extractor = SpecimenExtractor(output_dir=sample_output_dir)

    # Run complete pipeline
    with LogContext(f"Extraction for {sample_id}/{size_fraction}", logger):
        results, metadata = extractor.extract_from_composite(
            composite_image_path=image_path,
            sample_id=sample_id,
            size_fraction=size_fraction,
            verbose=False  # We handle logging ourselves
        )

    # Save metadata
    extractor.save_metadata(results)

    # Get statistics
    stats = extractor.get_statistics(results)

    logger.info(f"  Extracted: {len(results)} specimens")
    logger.info(f"  Average confidence: {stats['avg_confidence']:.3f}")
    logger.info(f"  Output: {sample_output_dir}")

    return {
        'sample_id': sample_id,
        'size_fraction': size_fraction,
        'composite_image': str(image_path),
        'num_extracted': len(results),
        'avg_confidence': stats['avg_confidence'],
        'output_dir': str(sample_output_dir)
    }


def process_from_metadata(
    metadata_file: Path,
    output_dir: Path,
    logger
) -> pd.DataFrame:
    """
    Process images from metadata CSV.

    Expected CSV columns:
    - image_path or file_path: Path to composite image
    - sample_id: Sample identifier
    - size_fraction: Size fraction code

    Args:
        metadata_file: Path to metadata CSV
        output_dir: Output directory
        logger: Logger instance

    Returns:
        DataFrame with processing results
    """
    validate_file_exists(metadata_file, "Metadata file")

    logger.info(f"Loading metadata from {metadata_file}")
    metadata = pd.read_csv(metadata_file)

    required_cols = ['sample_id', 'size_fraction']
    path_col = 'image_path' if 'image_path' in metadata.columns else 'file_path'

    if path_col not in metadata.columns:
        raise ValueError(f"Metadata CSV must have '{path_col}' column")

    for col in required_cols:
        if col not in metadata.columns:
            raise ValueError(f"Metadata CSV must have '{col}' column")

    logger.info(f"Found {len(metadata)} images to process")

    # Process each row
    results = []
    progress = ProgressLogger(total=len(metadata), name='Processing images', logger=logger)

    for idx, row in metadata.iterrows():
        try:
            result = process_single_image(
                image_path=Path(row[path_col]),
                sample_id=row['sample_id'],
                size_fraction=row['size_fraction'],
                output_dir=output_dir,
                logger=logger
            )
            results.append(result)

        except Exception as e:
            logger.error(f"Failed to process row {idx}: {e}")
            results.append({
                'sample_id': row.get('sample_id', 'unknown'),
                'size_fraction': row.get('size_fraction', 'unknown'),
                'composite_image': row.get(path_col, 'unknown'),
                'num_extracted': 0,
                'error': str(e)
            })

        progress.update()

    progress.finish()

    return pd.DataFrame(results)


def process_directory(
    input_dir: Path,
    output_dir: Path,
    pattern: str,
    sample_id_pattern: Optional[str],
    size_fraction: Optional[str],
    logger
) -> pd.DataFrame:
    """
    Process all images in directory.

    Args:
        input_dir: Directory with composite images
        output_dir: Output directory
        pattern: File pattern (e.g., "*.png")
        sample_id_pattern: Pattern to extract sample_id from filename
        size_fraction: Size fraction (or None to extract from filename)
        logger: Logger instance

    Returns:
        DataFrame with processing results
    """
    validate_directory_exists(input_dir)

    image_paths = sorted(input_dir.glob(pattern))

    if len(image_paths) == 0:
        raise ValueError(f"No images found matching '{pattern}' in {input_dir}")

    logger.info(f"Found {len(image_paths)} images in {input_dir}")

    results = []
    progress = ProgressLogger(total=len(image_paths), name='Processing directory', logger=logger)

    for image_path in image_paths:
        try:
            # Extract sample_id from filename
            # Expected format: {sample_id}_{size_fraction}_...
            parts = image_path.stem.split('_')

            if sample_id_pattern:
                # Custom pattern extraction (simplified)
                extracted_sample_id = parts[0]
            else:
                extracted_sample_id = parts[0] if len(parts) > 0 else image_path.stem

            # Extract size fraction
            if size_fraction is None:
                # Try to extract from filename
                # Look for common patterns: k1, 1, 2, 7, etc.
                extracted_size_fraction = None
                for part in parts:
                    if part in ['k1', '1', '1.5', '2', '3', '5', '7', 'A']:
                        extracted_size_fraction = part
                        break

                if extracted_size_fraction is None:
                    logger.warning(f"Could not extract size_fraction from {image_path.name}, skipping")
                    continue
            else:
                extracted_size_fraction = size_fraction

            result = process_single_image(
                image_path=image_path,
                sample_id=extracted_sample_id,
                size_fraction=extracted_size_fraction,
                output_dir=output_dir,
                logger=logger
            )
            results.append(result)

        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {e}")

        progress.update()

    progress.finish()

    return pd.DataFrame(results)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Process composite images: detection, segmentation, extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input',
        type=Path,
        help='Single composite image to process'
    )
    input_group.add_argument(
        '--metadata',
        type=Path,
        help='CSV file with image paths and metadata'
    )
    input_group.add_argument(
        '--input-dir',
        type=Path,
        help='Directory with composite images'
    )

    # Required for single image mode
    parser.add_argument(
        '--sample-id',
        type=str,
        help='Sample identifier (required with --input)'
    )
    parser.add_argument(
        '--size-fraction',
        type=str,
        help='Size fraction code: k1, 1, 2, 7, etc. (required with --input)'
    )

    # Optional for directory mode
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.png',
        help='File pattern for --input-dir (default: *.png)'
    )

    # Output
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for extracted specimens'
    )

    # Optional
    parser.add_argument(
        '--log-file',
        type=Path,
        help='Log file path (default: from config)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logging(
        name='process_images',
        log_file=args.log_file,
        level=log_level
    )

    logger.info("=" * 70)
    logger.info("SCRIPT 02: PROCESS IMAGES")
    logger.info("=" * 70)

    # Validate output directory
    validate_directory_exists(args.output_dir, create=True)

    try:
        # Process based on input mode
        if args.input:
            # Single image mode
            if not args.sample_id or not args.size_fraction:
                parser.error("--sample-id and --size-fraction required with --input")

            result = process_single_image(
                image_path=args.input,
                sample_id=args.sample_id,
                size_fraction=args.size_fraction,
                output_dir=args.output_dir,
                logger=logger
            )

            # Save result
            result_df = pd.DataFrame([result])
            result_file = args.output_dir / 'processing_results.csv'
            result_df.to_csv(result_file, index=False)
            logger.info(f"Results saved to {result_file}")

        elif args.metadata:
            # Metadata mode
            results_df = process_from_metadata(
                metadata_file=args.metadata,
                output_dir=args.output_dir,
                logger=logger
            )

            # Save results
            result_file = args.output_dir / 'processing_results.csv'
            results_df.to_csv(result_file, index=False)
            logger.info(f"Results saved to {result_file}")

            # Summary
            logger.info("=" * 70)
            logger.info("SUMMARY")
            logger.info("=" * 70)
            logger.info(f"Total images processed: {len(results_df)}")
            logger.info(f"Total specimens extracted: {results_df['num_extracted'].sum()}")
            logger.info(f"Average specimens per image: {results_df['num_extracted'].mean():.1f}")

        elif args.input_dir:
            # Directory mode
            results_df = process_directory(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                pattern=args.pattern,
                sample_id_pattern=None,
                size_fraction=args.size_fraction,
                logger=logger
            )

            # Save results
            result_file = args.output_dir / 'processing_results.csv'
            results_df.to_csv(result_file, index=False)
            logger.info(f"Results saved to {result_file}")

            # Summary
            logger.info("=" * 70)
            logger.info("SUMMARY")
            logger.info("=" * 70)
            logger.info(f"Total images processed: {len(results_df)}")
            logger.info(f"Total specimens extracted: {results_df['num_extracted'].sum()}")

        logger.info("=" * 70)
        logger.info("Processing complete!")
        logger.info("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
