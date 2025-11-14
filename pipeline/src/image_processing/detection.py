"""
Specimen Detection Module

Detects arthropod specimens in composite images using tiled YOLO object detection
with adaptive grid sizes and Non-Maximum Suppression (NMS).

Corresponds to Methods section: "Image Analysis and Specimen Extraction - Tiled Detection"
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO

from src.config import config


class SpecimenDetector:
    """
    Detects arthropod specimens using tiled object detection.

    The detector divides large composite images into overlapping tiles,
    runs YOLO detection on each tile, and applies global NMS to remove duplicates.

    Attributes:
        model: YOLO detection model
        size_fraction: Size category (k1, 1, 2, 7, etc.)
        grid_size: Tuple of (rows, cols) for tiling
        overlap: Overlap ratio (0.0-1.0)
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
    """

    def __init__(
        self,
        size_fraction: str = "1",
        model_path: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ):
        """
        Initialize specimen detector.

        Args:
            size_fraction: Size category (k1, 1, 2, 7, etc.)
            model_path: Path to YOLO detection model. If None, uses config.
            conf_threshold: Confidence threshold. If None, uses config.
            iou_threshold: IoU threshold for NMS. If None, uses config.
        """
        self.size_fraction = size_fraction

        # Load model path from config if not provided
        if model_path is None:
            model_path = config.get(f'image_processing.size_fractions.{size_fraction}.detection_model')
            if model_path is None:
                model_path = config.get('image_processing.detection.model_path')

        self.model = YOLO(model_path)

        # Load parameters from config
        size_config = config.get(f'image_processing.size_fractions.{size_fraction}', {})
        self.grid_size = size_config.get('grid_size', [1, 1])
        self.overlap = size_config.get('overlap', 0.0)

        self.conf_threshold = conf_threshold or config.get('image_processing.detection.confidence_threshold', 0.25)
        self.iou_threshold = iou_threshold or config.get('image_processing.detection.iou_threshold', 0.2)

        # Expansion ratio for bounding boxes (makes boxes slightly larger to capture full specimen)
        self.expansion_ratio = 0.2

        # Border margin ratio (boxes near tile borders are filtered out)
        self.border_margin_ratio = 0.01

    def detect_specimens(self, image_path: Path, verbose: bool = True) -> List[Dict]:
        """
        Detect specimens in composite image.

        Args:
            image_path: Path to composite image
            verbose: Print progress messages

        Returns:
            List of detection dictionaries with keys:
                - bounding_box: (x1, y1, x2, y2) in image coordinates
                - confidence: Detection confidence score
                - tile_position: Original tile position (debugging)

        Example:
            >>> detector = SpecimenDetector(size_fraction='1')
            >>> detections = detector.detect_specimens('composite.png')
            >>> print(f"Found {len(detections)} specimens")
        """
        # Load image
        image = Image.open(image_path)
        Image.MAX_IMAGE_PIXELS = None  # Allow very large images

        if verbose:
            print(f"Loaded image: {image.size[0]}x{image.size[1]} pixels")
            print(f"Size fraction: {self.size_fraction}")
            print(f"Grid size: {self.grid_size}, Overlap: {self.overlap}")

        # Create tiles
        tiles = self._create_tiles(image)

        if verbose:
            print(f"Created {len(tiles)} tiles")

        # Detect in each tile
        all_detections = []
        for i, tile_info in enumerate(tiles):
            if verbose and (i + 1) % 10 == 0:
                print(f"Processing tile {i + 1}/{len(tiles)}...")

            tile_detections = self._detect_in_tile(tile_info, image.size)
            all_detections.extend(tile_detections)

        if verbose:
            print(f"Found {len(all_detections)} detections before NMS")

        # Apply global NMS
        filtered_detections = self._apply_nms(all_detections)

        if verbose:
            print(f"Found {len(filtered_detections)} detections after NMS")

        return filtered_detections

    def _create_tiles(self, image: Image.Image) -> List[Dict]:
        """
        Divide image into overlapping tiles.

        Args:
            image: PIL Image

        Returns:
            List of tile dictionaries with:
                - tile: PIL Image (cropped tile)
                - position: (left, top) position in original image
                - coordinates: (left, top, right, bottom) in original image
        """
        width, height = image.size
        rows, cols = self.grid_size

        tile_width = width / cols
        tile_height = height / rows

        overlap_width = tile_width * self.overlap
        overlap_height = tile_height * self.overlap

        tiles = []

        for i in range(rows):
            for j in range(cols):
                left = int(j * tile_width)
                top = int(i * tile_height)
                right = int(left + tile_width)
                bottom = int(top + tile_height)

                # Add overlap (except for last column/row)
                if j < cols - 1:
                    right = min(int(right + overlap_width), width)
                if i < rows - 1:
                    bottom = min(int(bottom + overlap_height), height)

                tile = image.crop((left, top, right, bottom))

                tiles.append({
                    'tile': tile,
                    'position': (left, top),
                    'coordinates': (left, top, right, bottom)
                })

        return tiles

    def _detect_in_tile(self, tile_info: Dict, image_size: Tuple[int, int]) -> List[Dict]:
        """
        Run YOLO detection on a single tile.

        Args:
            tile_info: Tile dictionary from _create_tiles
            image_size: (width, height) of original image

        Returns:
            List of detections in global coordinates
        """
        tile = tile_info['tile']
        tile_position = tile_info['position']
        tile_coords = tile_info['coordinates']

        # Convert to numpy array for YOLO
        tile_array = np.array(tile)

        # Run YOLO detection
        results = self.model(tile_array, conf=self.conf_threshold, verbose=False)

        detections = []
        margin_x, margin_y = self._calculate_border_margins(tile)

        for result in results:
            if not hasattr(result.boxes, 'xyxy') or len(result.boxes.xyxy) == 0:
                continue

            for box, cls, confidence in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                class_name = self.model.names[int(cls)]

                # Filter by class (typically "Area" or "Object")
                if class_name not in ["Area", "Object"]:
                    continue

                # Box coordinates in tile space
                x1, y1, x2, y2 = map(int, box.cpu().numpy())

                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                # Skip boxes near tile borders (unless at image edge)
                if self._is_near_tile_border(
                    x1, y1, x2, y2,
                    tile.width, tile.height,
                    margin_x, margin_y,
                    tile_position,
                    image_size
                ):
                    continue

                # Expand box slightly
                x1, y1, x2, y2 = self._expand_bounding_box(x1, y1, x2, y2)

                # Ensure within tile bounds
                x1, y1, x2, y2 = self._ensure_within_bounds(x1, y1, x2, y2, tile.width, tile.height)

                # Convert to global coordinates
                abs_x1 = x1 + tile_position[0]
                abs_y1 = y1 + tile_position[1]
                abs_x2 = x2 + tile_position[0]
                abs_y2 = y2 + tile_position[1]

                # Ensure within image bounds
                abs_x1, abs_y1, abs_x2, abs_y2 = self._ensure_within_bounds(
                    abs_x1, abs_y1, abs_x2, abs_y2, image_size[0], image_size[1]
                )

                detections.append({
                    'bounding_box': (abs_x1, abs_y1, abs_x2, abs_y2),
                    'confidence': float(confidence.cpu().numpy()),
                    'tile_position': tile_position  # For debugging
                })

        return detections

    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply Non-Maximum Suppression (NMS) globally across all detections.

        Uses area-based sorting and IoU threshold to remove duplicate detections
        from overlapping tiles.

        Args:
            detections: List of detection dictionaries

        Returns:
            Filtered list of detections
        """
        if len(detections) == 0:
            return []

        boxes = np.array([d['bounding_box'] for d in detections])
        confidences = np.array([d['confidence'] for d in detections])

        # Calculate areas (prefer larger detections)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Sort by area (largest first)
        order = areas.argsort()[::-1]

        keep_indices = []

        while len(order) > 0:
            # Keep the largest remaining box
            i = order[0]
            keep_indices.append(i)

            # Calculate IoU with remaining boxes
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)

            # Keep boxes with IoU below threshold
            remaining = np.where(iou <= self.iou_threshold)[0]
            order = order[remaining + 1]

            # Also remove fully contained boxes
            to_remove = []
            for j in order:
                if self._is_fully_contained(boxes[j], boxes[i]):
                    to_remove.append(j)

            order = np.array([j for j in order if j not in to_remove])

        return [detections[i] for i in keep_indices]

    def _is_fully_contained(self, inner_box: np.ndarray, outer_box: np.ndarray) -> bool:
        """Check if inner_box is fully contained within outer_box."""
        return (
            inner_box[0] >= outer_box[0] and
            inner_box[1] >= outer_box[1] and
            inner_box[2] <= outer_box[2] and
            inner_box[3] <= outer_box[3]
        )

    def _calculate_border_margins(self, tile: Image.Image) -> Tuple[int, int]:
        """Calculate border margins for filtering edge detections."""
        margin_x = int(tile.width * self.border_margin_ratio)
        margin_y = int(tile.height * self.border_margin_ratio)
        return margin_x, margin_y

    def _is_near_tile_border(
        self,
        x1: int, y1: int, x2: int, y2: int,
        tile_width: int, tile_height: int,
        margin_x: int, margin_y: int,
        tile_position: Tuple[int, int],
        image_size: Tuple[int, int]
    ) -> bool:
        """
        Check if bounding box is near tile border.

        Boxes near internal tile borders are filtered out (likely duplicates from overlap).
        Boxes at image edges are kept.
        """
        tile_x, tile_y = tile_position
        image_width, image_height = image_size

        # Check if tile is at image edge
        at_left_edge = (tile_x == 0)
        at_right_edge = (tile_x + tile_width >= image_width)
        at_top_edge = (tile_y == 0)
        at_bottom_edge = (tile_y + tile_height >= image_height)

        # Box near left tile border (but not image edge)
        if not at_left_edge and x1 < margin_x:
            return True

        # Box near top tile border (but not image edge)
        if not at_top_edge and y1 < margin_y:
            return True

        # Box near right tile border (but not image edge)
        if not at_right_edge and x2 > tile_width - margin_x:
            return True

        # Box near bottom tile border (but not image edge)
        if not at_bottom_edge and y2 > tile_height - margin_y:
            return True

        return False

    def _expand_bounding_box(
        self,
        x1: int, y1: int, x2: int, y2: int
    ) -> Tuple[int, int, int, int]:
        """
        Expand bounding box by expansion_ratio to capture full specimen.

        Args:
            x1, y1, x2, y2: Original box coordinates

        Returns:
            Expanded box coordinates
        """
        width = x2 - x1
        height = y2 - y1

        x1 = int(x1 - self.expansion_ratio * width)
        y1 = int(y1 - self.expansion_ratio * height)
        x2 = int(x2 + self.expansion_ratio * width)
        y2 = int(y2 + self.expansion_ratio * height)

        return x1, y1, x2, y2

    def _ensure_within_bounds(
        self,
        x1: int, y1: int, x2: int, y2: int,
        max_width: int, max_height: int
    ) -> Tuple[int, int, int, int]:
        """Clip bounding box to image/tile bounds."""
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(max_width, x2)
        y2 = min(max_height, y2)

        return x1, y1, x2, y2
