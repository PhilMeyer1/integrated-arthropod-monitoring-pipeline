"""
Specimen Segmentation Module

Segments individual arthropod specimens from detected bounding boxes using YOLO
segmentation models with mask dilation and NMS.

Corresponds to Methods section: "Image Analysis and Specimen Extraction - Segmentation"
"""

import numpy as np
import cv2
import torch
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO

from src.config import config


class SpecimenSegmenter:
    """
    Segments individual specimens within detected bounding boxes.

    Uses YOLO segmentation models to generate precise masks for each specimen,
    then dilates masks to capture fine morphological details.

    Attributes:
        model: YOLO segmentation model
        size_fraction: Size category
        dilation_factor: Base factor for mask dilation
        background_color: RGBA color for masked background
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for second NMS step
    """

    def __init__(
        self,
        size_fraction: str = "1",
        model_path: Optional[str] = None,
        dilation_factor: int = 10,
        background_color: Tuple[int, int, int, int] = (248, 248, 248, 255),
        conf_threshold: float = 0.05,
        iou_threshold: float = 0.2
    ):
        """
        Initialize specimen segmenter.

        Args:
            size_fraction: Size category (k1, 1, 2, 7, etc.)
            model_path: Path to YOLO segmentation model. If None, uses config.
            dilation_factor: Base dilation factor (scaled by image size)
            background_color: RGBA background color for extracted specimens
            conf_threshold: Confidence threshold for segmentation
            iou_threshold: IoU threshold for NMS
        """
        self.size_fraction = size_fraction

        # Load model path from config if not provided
        if model_path is None:
            model_path = config.get(f'image_processing.size_fractions.{size_fraction}.segmentation_model')
            if model_path is None:
                model_path = config.get('image_processing.segmentation.model_path')

        self.model = YOLO(model_path)

        self.dilation_factor = dilation_factor
        self.background_color = background_color
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Weighting for NMS (balance confidence vs. size)
        self.size_weight = 0.6

    def segment_specimens(
        self,
        image_path: Path,
        detections: List[Dict],
        verbose: bool = True
    ) -> List[Dict]:
        """
        Segment specimens within detected bounding boxes.

        Args:
            image_path: Path to composite image
            detections: List of detection dicts from SpecimenDetector
            verbose: Print progress messages

        Returns:
            List of segmentation dictionaries with keys:
                - box: (x1, y1, x2, y2) bounding box in image coordinates
                - confidence: Segmentation confidence score
                - mask: Dilated polygon mask (N, 2) array
                - segmented_image: RGBA image with near-white background

        Example:
            >>> segmenter = SpecimenSegmenter(size_fraction='1')
            >>> segmentations = segmenter.segment_specimens(
            ...     'composite.png', detections
            ... )
            >>> print(f"Segmented {len(segmentations)} specimens")
        """
        # Load image
        image = Image.open(image_path)
        image_array = np.array(image)

        # Convert to RGB if needed
        if image_array.shape[2] == 4:  # RGBA
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        elif image_array.shape[2] == 3:  # Already RGB or BGR
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        if verbose:
            print(f"Processing {len(detections)} detections for segmentation...")

        # Segment each detection
        provisional_segmentations = []

        for i, detection in enumerate(detections):
            box = detection['bounding_box']
            x1, y1, x2, y2 = map(int, box)

            # Crop tile to bounding box
            tile = image_array[y1:y2, x1:x2]

            if tile.size == 0:
                continue

            # Run YOLO segmentation on tile
            results = self.model(tile, iou=0.1, conf=self.conf_threshold, verbose=False)

            if not results:
                continue

            for result in results:
                if result.masks is None or not result.masks.xy:
                    continue

                # Process each detected object with mask
                for mask, res_box, cls, conf in zip(
                    result.masks.xy,
                    result.boxes.xyxy,
                    result.boxes.cls,
                    result.boxes.conf
                ):
                    class_name = self.model.names[int(cls)]

                    # Filter by class (typically "Object")
                    if class_name != "Object":
                        continue

                    # Dilate mask to capture fine details
                    dilated_mask = self._dilate_mask(
                        np.array(mask),
                        tile.shape,
                        self.dilation_factor
                    )

                    if dilated_mask is None:
                        continue

                    # Create segmented image with background
                    segmented_image = self._create_segmented_image(tile, dilated_mask)

                    if segmented_image is None:
                        continue

                    # Convert box to global coordinates
                    global_box = res_box.cpu().numpy() + np.array([x1, y1, x1, y1])

                    provisional_segmentations.append({
                        'box': global_box,
                        'confidence': float(conf.cpu().numpy()),
                        'mask': dilated_mask,
                        'segmented_image': segmented_image
                    })

        if verbose:
            print(f"Found {len(provisional_segmentations)} segmentations before NMS")

        # Apply second NMS to remove overlapping segmentations
        final_segmentations = self._apply_nms(provisional_segmentations)

        if verbose:
            print(f"Found {len(final_segmentations)} segmentations after NMS")

        return final_segmentations

    def _dilate_mask(
        self,
        mask: np.ndarray,
        image_shape: Tuple[int, int, int],
        base_dilation_factor: int
    ) -> Optional[np.ndarray]:
        """
        Dilate segmentation mask to capture fine morphological details.

        The dilation factor is scaled by image size to maintain consistent
        dilation across different image sizes.

        Args:
            mask: Polygon mask as (N, 2) array of xy coordinates
            image_shape: (height, width, channels) of tile image
            base_dilation_factor: Base dilation factor

        Returns:
            Dilated mask as (N, 2) array, or None if invalid
        """
        # Validate mask
        if mask is None or len(mask) < 3:
            return None

        if mask.ndim != 2 or mask.shape[1] != 2:
            return None

        # Clip mask to image bounds
        x_max, y_max = image_shape[1] - 1, image_shape[0] - 1
        mask = np.clip(mask, [0, 0], [x_max, y_max])

        # Create binary mask image
        mask_image = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)

        try:
            cv2.fillPoly(mask_image, [mask.astype(np.int32)], 255)
        except cv2.error as e:
            print(f"Error filling polygon: {e}")
            return None

        # Calculate dynamic dilation factor based on image size
        scaling_factor = (image_shape[1] + image_shape[0]) / 2
        dynamic_dilation_factor = max(1, int(base_dilation_factor * (scaling_factor / 1000)))

        # Dilate mask
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask_image = cv2.dilate(mask_image, kernel, iterations=dynamic_dilation_factor)

        # Extract contour from dilated mask
        contours, _ = cv2.findContours(dilated_mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        # Take largest contour
        dilated_mask = max(contours, key=cv2.contourArea)

        if len(dilated_mask) < 3:
            return None

        return dilated_mask.reshape(-1, 2)

    def _create_segmented_image(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Create segmented image with near-white background.

        Applies the mask to extract the specimen and places it on a
        near-white (RGB: 248, 248, 248) background as specified in Methods.

        Args:
            image: RGB tile image
            mask: Polygon mask (N, 2) array

        Returns:
            RGBA image with specimen on near-white background, or None if error
        """
        if mask is None:
            return None

        # Create binary mask image
        mask_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        try:
            cv2.fillPoly(mask_image, [mask.astype(np.int32)], 255)
        except cv2.error as e:
            print(f"Error creating segmented image: {e}")
            return None

        # Convert image to RGBA
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

        # Apply mask
        segmented_object = cv2.bitwise_and(image, image, mask=mask_image)

        # Crop to bounding box of mask
        x_min = int(np.min(mask[:, 0]))
        x_max = int(np.max(mask[:, 0]))
        y_min = int(np.min(mask[:, 1]))
        y_max = int(np.max(mask[:, 1]))

        cropped_segmented_object = segmented_object[y_min:y_max, x_min:x_max]

        # Create background image with near-white color
        blank_image = np.zeros((y_max - y_min, x_max - x_min, 4), dtype=np.uint8)
        blank_image[:, :] = self.background_color

        # Combine: place specimen on background
        combined_image = blank_image.copy()
        alpha_mask = (cropped_segmented_object[:, :, 3] != 0)
        combined_image[alpha_mask] = cropped_segmented_object[alpha_mask]

        return combined_image

    def _apply_nms(self, segmentations: List[Dict]) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove overlapping segmentations.

        Uses weighted score combining confidence and box size to prioritize
        larger, more confident detections.

        Args:
            segmentations: List of segmentation dictionaries

        Returns:
            Filtered list of segmentations
        """
        if len(segmentations) == 0:
            return []

        boxes = np.array([s['box'] for s in segmentations])
        scores = np.array([s['confidence'] for s in segmentations])

        # Convert torch tensors if needed
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()

        # Calculate box areas
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # Weighted score: combine confidence and size
        weighted_scores = scores * (1 - self.size_weight) + areas * self.size_weight

        # Sort by weighted score
        order = weighted_scores.argsort()[::-1]

        keep_indices = []

        while order.size > 0:
            # Keep highest scoring box
            i = order[0]
            keep_indices.append(i)

            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)

            # Keep boxes with IoU below threshold
            remaining = np.where(iou <= self.iou_threshold)[0]
            order = order[remaining + 1]

        return [segmentations[i] for i in keep_indices]
