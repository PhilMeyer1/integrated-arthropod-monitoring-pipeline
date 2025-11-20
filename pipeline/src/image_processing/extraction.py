"""
Specimen Extraction Module

Extracts individual specimen images from segmentations and saves them as PNG files
with standardized naming and near-white backgrounds.

Corresponds to Methods section: "Image Analysis and Specimen Extraction - Extraction"
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from src.config import config


class SpecimenExtractor:
    """
    Extracts individual specimen images using segmentation masks.

    Saves specimens as PNG files with:
    - Near-white background (RGB: 248, 248, 248) as specified in Methods
    - Standardized naming: {sample_id}_{size_fraction}_{index:04d}.png
    - Metadata tracking (bounding box, confidence, etc.)

    Attributes:
        output_dir: Directory to save extracted specimens
        background_color: RGB(A) background color
    """

    def __init__(
        self,
        output_dir: Path,
        background_color: Optional[Tuple[int, int, int, int]] = None
    ):
        """
        Initialize specimen extractor.

        Args:
            output_dir: Directory to save extracted specimen images
            background_color: RGBA background color. If None, uses config or default.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if background_color is None:
            # Try to get from config, fallback to (248, 248, 248, 255)
            bg_rgb = config.get('image_processing.extraction.background_color', [248, 248, 248])
            if len(bg_rgb) == 3:
                background_color = tuple(bg_rgb) + (255,)  # Add alpha
            else:
                background_color = tuple(bg_rgb)

        self.background_color = background_color

        # Output format (PNG recommended for transparency)
        self.output_format = config.get('image_processing.extraction.output_format', 'png')

    def extract_specimens(
        self,
        segmentations: List[Dict],
        sample_id: str,
        size_fraction: str,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Extract and save individual specimen images.

        Args:
            segmentations: List of segmentation dicts from SpecimenSegmenter
            sample_id: Sample identifier for naming (e.g., 'S001')
            size_fraction: Size fraction for naming (e.g., '1')
            verbose: Print progress messages

        Returns:
            List of extraction results with keys:
                - file_path: Path to saved PNG file
                - bounding_box: (x1, y1, x2, y2) in original image
                - confidence: Segmentation confidence
                - width: Image width in pixels
                - height: Image height in pixels
                - created_at: Timestamp

        Example:
            >>> extractor = SpecimenExtractor(output_dir='./output/specimens')
            >>> results = extractor.extract_specimens(
            ...     segmentations, sample_id='S001', size_fraction='1'
            ... )
            >>> print(f"Extracted {len(results)} specimens to {extractor.output_dir}")
        """
        if verbose:
            print(f"Extracting {len(segmentations)} specimens...")
            print(f"Output directory: {self.output_dir}")

        results = []

        for i, segmentation in enumerate(segmentations):
            # Get segmented image (already has near-white background from segmenter)
            segmented_image = segmentation.get('segmented_image')

            if segmented_image is None:
                if verbose:
                    print(f"Warning: Segmentation {i} has no image, skipping")
                continue

            # Generate filename: {sample_id}_{size_fraction}_{index:04d}.png
            filename = f"{sample_id}_{size_fraction}_{i + 1:04d}.{self.output_format}"
            file_path = self.output_dir / filename

            # Save image (write directly - segmented_image is already in correct format)
            success = cv2.imwrite(str(file_path), segmented_image)

            if not success:
                if verbose:
                    print(f"Warning: Failed to save {filename}")
                continue

            # Record result
            results.append({
                'file_path': str(file_path),
                'bounding_box': tuple(segmentation['box']),
                'confidence': segmentation['confidence'],
                'width': segmented_image.shape[1],
                'height': segmented_image.shape[0],
                'created_at': datetime.now().isoformat()
            })

            if verbose and (i + 1) % 100 == 0:
                print(f"  Extracted {i + 1}/{len(segmentations)} specimens...")

        if verbose:
            print(f"Successfully extracted {len(results)} specimens")

        return results

    def extract_from_composite(
        self,
        composite_image_path: Path,
        sample_id: str,
        size_fraction: str,
        detector_params: Optional[Dict] = None,
        segmenter_params: Optional[Dict] = None,
        verbose: bool = True
    ) -> Tuple[List[Dict], Dict]:
        """
        Complete extraction pipeline: detect → segment → extract.

        Convenience method that runs the full pipeline on a composite image.

        Args:
            composite_image_path: Path to composite image
            sample_id: Sample identifier
            size_fraction: Size fraction (k1, 1, 2, 7, etc.)
            detector_params: Optional parameters for SpecimenDetector
            segmenter_params: Optional parameters for SpecimenSegmenter
            verbose: Print progress

        Returns:
            Tuple of (extraction_results, pipeline_metadata)
                - extraction_results: List of extracted specimen dicts
                - pipeline_metadata: Dict with detection/segmentation stats

        Example:
            >>> extractor = SpecimenExtractor(output_dir='./output')
            >>> results, meta = extractor.extract_from_composite(
            ...     'composite.png', 'S001', '1'
            ... )
            >>> print(f"Detected: {meta['num_detections']}")
            >>> print(f"Segmented: {meta['num_segmentations']}")
            >>> print(f"Extracted: {len(results)}")
        """
        from src.image_processing.detection import SpecimenDetector
        from src.image_processing.segmentation import SpecimenSegmenter

        if verbose:
            print("=" * 60)
            print(f"Running extraction pipeline on {composite_image_path.name}")
            print(f"Sample ID: {sample_id}, Size fraction: {size_fraction}")
            print("=" * 60)

        # Step 1: Detection
        if verbose:
            print("\n[1/3] Running detection...")

        detector = SpecimenDetector(size_fraction=size_fraction, **(detector_params or {}))
        detections = detector.detect_specimens(composite_image_path, verbose=verbose)

        # Step 2: Segmentation
        if verbose:
            print(f"\n[2/3] Running segmentation on {len(detections)} detections...")

        segmenter = SpecimenSegmenter(size_fraction=size_fraction, **(segmenter_params or {}))
        segmentations = segmenter.segment_specimens(composite_image_path, detections, verbose=verbose)

        # Step 3: Extraction
        if verbose:
            print(f"\n[3/3] Extracting {len(segmentations)} specimens...")

        results = self.extract_specimens(segmentations, sample_id, size_fraction, verbose=verbose)

        # Metadata
        metadata = {
            'composite_image': str(composite_image_path),
            'sample_id': sample_id,
            'size_fraction': size_fraction,
            'num_detections': len(detections),
            'num_segmentations': len(segmentations),
            'num_extracted': len(results),
            'output_dir': str(self.output_dir),
            'timestamp': datetime.now().isoformat()
        }

        if verbose:
            print("\n" + "=" * 60)
            print("Pipeline complete!")
            print(f"  Detections:    {metadata['num_detections']}")
            print(f"  Segmentations: {metadata['num_segmentations']}")
            print(f"  Extracted:     {metadata['num_extracted']}")
            print(f"  Output:        {self.output_dir}")
            print("=" * 60)

        return results, metadata

    def save_metadata(self, results: List[Dict], output_path: Optional[Path] = None):
        """
        Save extraction metadata to CSV file.

        Args:
            results: List of extraction result dicts
            output_path: Path to save CSV. If None, saves to output_dir/metadata.csv
        """
        import pandas as pd

        if output_path is None:
            output_path = self.output_dir / "metadata.csv"

        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)

        print(f"Metadata saved to {output_path}")

    def get_statistics(self, results: List[Dict]) -> Dict:
        """
        Calculate extraction statistics.

        Args:
            results: List of extraction result dicts

        Returns:
            Statistics dictionary with:
                - num_specimens: Total specimens
                - avg_width: Average width in pixels
                - avg_height: Average height in pixels
                - avg_confidence: Average confidence score
                - total_size_bytes: Total file size
        """
        if len(results) == 0:
            return {
                'num_specimens': 0,
                'avg_width': 0,
                'avg_height': 0,
                'avg_confidence': 0,
                'total_size_bytes': 0
            }

        widths = [r['width'] for r in results]
        heights = [r['height'] for r in results]
        confidences = [r['confidence'] for r in results]

        # Calculate file sizes
        total_size = 0
        for r in results:
            try:
                total_size += Path(r['file_path']).stat().st_size
            except:
                pass

        return {
            'num_specimens': len(results),
            'avg_width': np.mean(widths),
            'avg_height': np.mean(heights),
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024)
        }
