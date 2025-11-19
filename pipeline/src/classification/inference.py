"""
Hierarchical inference engine for taxonomic classification.

This module implements hierarchical classification by recursively applying
YOLO models from top-level (e.g., Arthropoda) down to the most specific
taxonomic level possible.

Corresponds to Methods section "Classification Workflow".
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import io

from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

from src.config import config
from src.utils.logging_config import get_logger, ProgressLogger

logger = get_logger(__name__)


class InferenceEngine:
    """
    Hierarchical inference engine for specimen classification.

    This engine:
    1. Starts with top-level model (e.g., Arthropoda/Insecta)
    2. For each specimen, gets classification
    3. If child models exist, applies next level model
    4. Repeats until reaching leaf level or debris
    5. Records confidence at each level

    Example:
        >>> engine = InferenceEngine(set_number=1)
        >>> results = engine.run_inference(
        ...     image_ids=[1, 2, 3],
        ...     batch_size=32
        ... )
        >>> for result in results:
        ...     print(f"Image {result['image_id']}: {result['final_taxon']}")
    """

    def __init__(
        self,
        set_number: int,
        data_manager=None,
        models_path: Optional[Path] = None
    ):
        """
        Initialize inference engine.

        Args:
            set_number: Model set number to use
            data_manager: DataManager for database/file access
            models_path: Directory with trained models (default: from config)
        """
        self.set_number = set_number
        self.data_manager = data_manager

        if models_path is None:
            models_path = Path(config.get('paths.models', './models'))

        self.models_path = Path(models_path)

        # Load excluded IDs (test set)
        self.excluded_ids = self.load_excluded_ids()

        # Storage for results
        self.inference_results = []

        logger.info(
            f"InferenceEngine initialized: set={set_number}, "
            f"excluded={len(self.excluded_ids)} images"
        )

    def load_excluded_ids(self) -> Set[int]:
        """
        Load excluded image IDs (test set) for this model set.

        Returns:
            Set of image IDs that were excluded from training
        """
        if self.data_manager is None:
            return set()

        excluded = self.data_manager.get_excluded_ids(self.set_number)
        logger.info(f"Loaded {len(excluded)} excluded image IDs")

        return excluded

    def get_models_info(self) -> List[Dict]:
        """
        Get all models for this set number.

        Returns:
            List of model dictionaries with:
            - model_id: Model identifier
            - taxon_id: Taxon this model classifies
            - parent_id: Parent taxon
            - model_path: Path to .pt file

        Example:
            >>> models = engine.get_models_info()
            >>> for model in models:
            ...     print(f"{model['taxon_id']}: {model['model_path']}")
        """
        if self.data_manager is None:
            # Load from files
            models = []
            for model_file in self.models_path.glob("*.pt"):
                metadata_file = model_file.with_suffix('.json')
                if metadata_file.exists():
                    import json
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    if metadata.get('set_number') == self.set_number:
                        models.append({
                            'model_id': model_file.stem,
                            'taxon_id': metadata['taxon_id'],
                            'parent_id': None,  # Would need separate taxonomy file
                            'model_path': str(model_file)
                        })

            return models

        else:
            # Query database
            return self.data_manager.get_models_by_set(self.set_number)

    def find_top_model(self, models_info: List[Dict]) -> Optional[Dict]:
        """
        Find the top-level model (root of hierarchy).

        The top model is one whose parent_id is not in the set of taxa
        that have models, or is None.

        Args:
            models_info: List of model dictionaries

        Returns:
            Top model dictionary, or None if not found

        Example:
            >>> models = engine.get_models_info()
            >>> top = engine.find_top_model(models)
            >>> print(f"Top model: {top['taxon_id']}")
        """
        taxon_ids = {model['taxon_id'] for model in models_info}

        top_level_models = [
            model for model in models_info
            if model['parent_id'] not in taxon_ids or model['parent_id'] is None
        ]

        if top_level_models:
            logger.info(f"Found top model: {top_level_models[0]['taxon_id']}")
            return top_level_models[0]
        else:
            logger.error("No top-level model found")
            return None

    def get_image_path(self, image_id: int) -> Optional[Path]:
        """
        Get file path for specimen image.

        Args:
            image_id: Image identifier

        Returns:
            Path to image file
        """
        if self.data_manager is None:
            logger.warning(f"No data manager, cannot get path for image {image_id}")
            return None

        return self.data_manager.get_image_path(image_id)

    def load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Load image for YOLO inference.

        Args:
            image_path: Path to image file

        Returns:
            Image as numpy array (BGR format for OpenCV)
        """
        try:
            with open(image_path, 'rb') as f:
                img = Image.open(io.BytesIO(f.read()))

            # Convert to BGR (YOLO expects BGR from cv2)
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            return img_cv

        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None

    def run_model_inference(
        self,
        model: YOLO,
        image_paths: List[Path]
    ) -> List[Tuple[Optional[str], Optional[float]]]:
        """
        Run YOLO inference on batch of images.

        Args:
            model: Loaded YOLO model
            image_paths: List of image paths

        Returns:
            List of (predicted_class, confidence) tuples

        Example:
            >>> model = YOLO('models/1.pt')
            >>> predictions = engine.run_model_inference(
            ...     model,
            ...     [Path('img1.png'), Path('img2.png')]
            ... )
        """
        try:
            # Run inference
            results = model(image_paths)

            predictions = []
            for result in results:
                if hasattr(result, 'probs') and hasattr(result.probs, 'top1'):
                    class_index = result.probs.top1
                    predicted_class = result.names.get(class_index, None)
                    confidence = float(result.probs.top1conf.cpu().item())

                    predictions.append((predicted_class, confidence))
                else:
                    predictions.append((None, None))

            return predictions

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return [(None, None)] * len(image_paths)

    def get_correct_taxon_at_rank(
        self,
        actual_taxon_id: str,
        predicted_rank: str
    ) -> Optional[str]:
        """
        Get the correct taxon at a specific rank for comparison.

        When a model predicts at Order level, we need to compare with
        the Order-level ancestor of the actual determination.

        Args:
            actual_taxon_id: Actual (manual) determination
            predicted_rank: Rank the model predicts at

        Returns:
            Taxon ID at the predicted rank, or None

        Example:
            >>> # Actual = "Carabidae" (Family)
            >>> # Predicted rank = "Order"
            >>> correct = engine.get_correct_taxon_at_rank("Carabidae", "Order")
            >>> # Returns "Coleoptera"
        """
        if self.data_manager is None:
            return None

        # Traverse up the hierarchy until we find matching rank
        current_taxon = self.data_manager.get_taxon(actual_taxon_id)

        while current_taxon:
            if current_taxon.get('rank') == predicted_rank:
                return current_taxon['taxon_id']

            # Move to parent
            parent_id = current_taxon.get('parent_id')
            if not parent_id:
                break

            current_taxon = self.data_manager.get_taxon(parent_id)

        return None

    def add_inference_result(
        self,
        image_id: int,
        model_taxon_id: str,
        predicted_taxon_id: Optional[str],
        correct_taxon_id: Optional[str],
        confidence: float,
        is_excluded: bool
    ):
        """
        Add inference result to results list.

        Args:
            image_id: Specimen image ID
            model_taxon_id: Which model was used
            predicted_taxon_id: What the model predicted
            correct_taxon_id: Correct taxon at this rank
            confidence: Prediction confidence
            is_excluded: Whether this image was in test set
        """
        # Special handling for debris
        if predicted_taxon_id == "Debris":
            # Mark image as debris in database
            if self.data_manager:
                self.data_manager.mark_as_debris(image_id)
            logger.info(f"Image {image_id} classified as Debris")

        self.inference_results.append({
            'image_id': image_id,
            'model_taxon_id': model_taxon_id,
            'predicted_taxon_id': predicted_taxon_id,
            'correct_taxon_id': correct_taxon_id,
            'confidence': confidence,
            'is_excluded': is_excluded,
            'set_number': self.set_number
        })

    def save_inference_results(self):
        """
        Save accumulated inference results to database or file.
        """
        if not self.inference_results:
            logger.warning("No inference results to save")
            return

        if self.data_manager is None:
            # Save to JSON file
            output_file = Path(f"./inference_results_set{self.set_number}.json")
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.inference_results, f, indent=2)

            logger.info(f"Saved {len(self.inference_results)} results to {output_file}")

        else:
            # Save to database
            self.data_manager.save_inference_results(self.inference_results)
            logger.info(f"Saved {len(self.inference_results)} results to database")

        # Clear results
        self.inference_results = []

    def run_inference(
        self,
        image_ids: Optional[List[int]] = None,
        start_id: Optional[int] = None,
        end_id: Optional[int] = None,
        batch_size: int = 200
    ) -> List[Dict]:
        """
        Run hierarchical inference on specimens.

        This is the main entry point. It:
        1. Loads all models for this set
        2. Starts with top model
        3. Iteratively applies child models
        4. Saves results periodically

        Args:
            image_ids: Specific image IDs to process (or None for all)
            start_id: Process images from this ID (inclusive)
            end_id: Process images up to this ID (inclusive)
            batch_size: Number of images per batch

        Returns:
            List of inference result dictionaries

        Example:
            >>> results = engine.run_inference(
            ...     image_ids=[1, 2, 3, 4, 5],
            ...     batch_size=32
            ... )
        """
        logger.info("=" * 70)
        logger.info("STARTING HIERARCHICAL INFERENCE")
        logger.info("=" * 70)

        # Step 1: Get image IDs to process
        if image_ids is not None:
            ids_to_process = image_ids
        elif start_id is not None and end_id is not None:
            ids_to_process = list(range(start_id, end_id + 1))
        else:
            # Get all images
            if self.data_manager:
                ids_to_process = self.data_manager.get_all_image_ids()
            else:
                logger.error("No image IDs specified and no data manager")
                return []

        logger.info(f"Processing {len(ids_to_process)} images")

        # Step 2: Load all models
        models_info = self.get_models_info()
        if not models_info:
            logger.error("No models found for this set")
            return []

        logger.info(f"Loaded {len(models_info)} models")

        # Step 3: Find top model
        top_model_info = self.find_top_model(models_info)
        if not top_model_info:
            logger.error("No top model found")
            return []

        top_taxon_id = top_model_info['taxon_id']

        # Step 4: Initialize all images with top taxon
        image_determinations = {
            image_id: top_taxon_id
            for image_id in ids_to_process
        }

        logger.info(f"Initialized with top taxon: {top_taxon_id}")

        # Step 5: Create model dictionary for quick lookup
        model_dict = {info['taxon_id']: info for info in models_info}

        # Step 6: Track processed (image, taxon) pairs
        processed = set()

        # Step 7: Iterative inference loop
        iteration = 0
        while True:
            iteration += 1
            logger.info(f"\n{'='*70}")
            logger.info(f"ITERATION {iteration}")
            logger.info(f"{'='*70}")

            changes_made = False
            next_iteration_taxa = {}

            # Collect images for each model
            for model_taxon, model_info in model_dict.items():
                # Find images currently at this taxon level
                current_image_ids = [
                    img_id
                    for img_id, determination in image_determinations.items()
                    if determination == model_taxon and
                       (img_id, determination) not in processed
                ]

                if current_image_ids:
                    next_iteration_taxa[model_taxon] = current_image_ids

            if not next_iteration_taxa:
                logger.info("No more images to process, ending inference loop")
                break

            logger.info(
                f"Processing {len(next_iteration_taxa)} taxa with images"
            )

            # Process each taxon's model
            for model_taxon, current_image_ids in next_iteration_taxa.items():
                model_info = model_dict[model_taxon]
                model_path = Path(model_info['model_path'])

                logger.info(
                    f"\nProcessing {model_taxon}: "
                    f"{len(current_image_ids)} images"
                )

                # Load YOLO model
                try:
                    yolo_model = YOLO(str(model_path))
                    logger.info(f"Loaded model from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load model: {e}")
                    continue

                # Process in batches
                progress = ProgressLogger(
                    total=len(current_image_ids),
                    name=f'{model_taxon} inference',
                    log_interval=50,
                    logger=logger
                )

                for i in range(0, len(current_image_ids), batch_size):
                    batch_ids = current_image_ids[i:i + batch_size]

                    # Get image paths
                    batch_paths = []
                    valid_ids = []

                    for img_id in batch_ids:
                        path = self.get_image_path(img_id)
                        if path and path.exists():
                            batch_paths.append(str(path))
                            valid_ids.append(img_id)

                    if not batch_paths:
                        continue

                    # Run inference
                    try:
                        predictions = self.run_model_inference(
                            yolo_model,
                            batch_paths
                        )

                        # Process results
                        for img_id, (predicted_class, confidence) in zip(valid_ids, predictions):
                            if predicted_class is None:
                                continue

                            # Get actual taxon for comparison
                            actual_taxon = None
                            if self.data_manager:
                                actual_taxon = self.data_manager.get_actual_determination(img_id)

                            # Get correct taxon at predicted rank
                            correct_taxon = None
                            if actual_taxon and self.data_manager:
                                predicted_taxon_info = self.data_manager.get_taxon(predicted_class)
                                if predicted_taxon_info:
                                    predicted_rank = predicted_taxon_info.get('rank')
                                    correct_taxon = self.get_correct_taxon_at_rank(
                                        actual_taxon,
                                        predicted_rank
                                    )

                            # Check if excluded
                            is_excluded = img_id in self.excluded_ids

                            # Add result
                            self.add_inference_result(
                                image_id=img_id,
                                model_taxon_id=model_taxon,
                                predicted_taxon_id=predicted_class,
                                correct_taxon_id=correct_taxon,
                                confidence=confidence,
                                is_excluded=is_excluded
                            )

                            # Update determination for next iteration
                            image_determinations[img_id] = predicted_class
                            changes_made = True

                            # Mark as processed
                            processed.add((img_id, model_taxon))

                        progress.update(len(valid_ids))

                    except Exception as e:
                        logger.error(f"Batch inference failed: {e}")
                        continue

                progress.finish()

                # Save results periodically
                self.save_inference_results()

                # Unload model
                del yolo_model
                logger.info(f"Unloaded model for {model_taxon}")

            if not changes_made:
                logger.info("No changes made in this iteration, ending loop")
                break

        # Final save
        self.save_inference_results()

        logger.info("=" * 70)
        logger.info("HIERARCHICAL INFERENCE COMPLETE")
        logger.info("=" * 70)

        return self.inference_results
