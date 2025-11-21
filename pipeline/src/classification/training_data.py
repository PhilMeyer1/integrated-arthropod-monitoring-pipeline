"""
Training dataset creation for hierarchical YOLO classification.

This module creates training/validation/test splits for each taxonomic
level in the hierarchy. It handles data augmentation, additional classes,
and debris detection.

Corresponds to Methods section "Model Training and Validation".
"""

from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import random
import json
import shutil
from functools import partial
import multiprocessing

from PIL import Image
import cv2
import numpy as np

from src.classification.taxonomy import TaxonomyHierarchy
from src.utils.logging_config import get_logger
from src.config import config

# Allow very large composite images (typical: 42-48 MB, high resolution)
Image.MAX_IMAGE_PIXELS = None

logger = get_logger(__name__)


def apply_transformations(image: Image.Image) -> List[Image.Image]:
    """
    Apply rotation transformations for data augmentation.

    Creates 4 versions: original + 90°, 180°, 270° rotations.
    Only applied to training data, not validation.

    Args:
        image: PIL Image to transform

    Returns:
        List of 4 transformed images

    Example:
        >>> img = Image.open('specimen.png')
        >>> transformed = apply_transformations(img)
        >>> len(transformed)
        4
    """
    # Convert to OpenCV format
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Apply rotations
    transformations = [
        image_np,  # Original
        cv2.rotate(image_np, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(image_np, cv2.ROTATE_180),
        cv2.rotate(image_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]

    # Convert back to PIL
    transformations = [
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for img in transformations
    ]

    return transformations


def process_image_for_training(
    image_path: Path,
    class_images_directory: Path,
    dataset_type: str,
    class_id: str,
    apply_augmentation: bool = True
):
    """
    Copy and optionally augment image for training dataset.

    Args:
        image_path: Source specimen image
        class_images_directory: Destination directory for this class
        dataset_type: 'train' or 'val'
        class_id: Class identifier (taxon ID)
        apply_augmentation: Whether to apply rotations (only for training)

    Example:
        >>> process_image_for_training(
        ...     Path('specimens/S001_1_0001.png'),
        ...     Path('training/Insecta/train/Coleoptera'),
        ...     'train',
        ...     'Coleoptera'
        ... )
    """
    if not image_path.exists():
        logger.warning(f"Image not found: {image_path}")
        return

    # Load original image
    original_image = Image.open(image_path)

    # Save original
    destination_path = class_images_directory / image_path.name
    original_image.save(destination_path, format='PNG')

    # Apply transformations only for training (not val/test) and not for special classes
    if (dataset_type == 'train' and
        apply_augmentation and
        class_id not in ['additional_class', 'Debris']):

        transformed_images = apply_transformations(original_image)

        # Save rotated versions (skip index 0 = original)
        for i, transformed_image in enumerate(transformed_images[1:], start=1):
            transformed_name = f'{image_path.stem}_rot{i * 90}.png'
            transformed_path = class_images_directory / transformed_name
            transformed_image.save(transformed_path, format='PNG')


class TrainingDatasetCreator:
    """
    Create training datasets for hierarchical YOLO classification.

    This class:
    1. Splits data into train/val/test sets
    2. Creates directory structure for each taxonomic level
    3. Identifies suitable taxa for training (minimum image requirement)
    4. Creates "additional class" for taxa with too few specimens
    5. Adds debris class for non-specimen images
    6. Applies data augmentation

    Example:
        >>> creator = TrainingDatasetCreator(
        ...     base_directory=Path('./training_data'),
        ...     start_year=2020,
        ...     end_year=2023
        ... )
        >>> creator.generate_training_data(
        ...     top_taxon_id='RT',
        ...     min_images_per_class=7
        ... )
    """

    def __init__(
        self,
        base_directory: Path,
        data_manager=None,
        excluded_ids: Optional[Set[int]] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None
    ):
        """
        Initialize training dataset creator.

        Args:
            base_directory: Root directory for training datasets
            data_manager: DataManager instance for database/file access
            excluded_ids: Image IDs to exclude (e.g., test set)
            start_year: Filter specimens from this year onwards
            end_year: Filter specimens up to this year
        """
        self.base_directory = Path(base_directory)
        self.data_manager = data_manager
        self.excluded_ids = excluded_ids or set()
        self.start_year = start_year
        self.end_year = end_year

        logger.info(
            f"TrainingDatasetCreator initialized: "
            f"excluded={len(self.excluded_ids)}, "
            f"years={start_year}-{end_year}"
        )

    def get_total_images_for_taxon(
        self,
        hierarchy: Dict,
        taxon_id: str
    ) -> int:
        """
        Calculate total images for taxon including descendants.

        Args:
            hierarchy: Taxonomy tree
            taxon_id: Taxon to count

        Returns:
            Total number of images
        """
        if taxon_id not in hierarchy:
            return 0

        total = len(hierarchy[taxon_id]['images'])

        for child_id in hierarchy[taxon_id]['children']:
            total += self.get_total_images_for_taxon(hierarchy, child_id)

        return total

    def find_suitable_taxa_for_training(
        self,
        hierarchy: Dict,
        min_images: int
    ) -> List[str]:
        """
        Find taxa suitable for creating training models.

        A taxon is suitable if:
        - It has at least 1 child
        - At least 1 child has >= min_images specimens

        Args:
            hierarchy: Taxonomy tree
            min_images: Minimum images required per class

        Returns:
            List of taxon IDs suitable for training

        Example:
            >>> suitable = creator.find_suitable_taxa_for_training(tree, 7)
            >>> print(f"Can create {len(suitable)} models")
        """
        suitable_taxa = []

        for taxon_id, data in hierarchy.items():
            # Must have at least one child
            if len(data['children']) < 1:
                continue

            # Count children with enough images
            children_with_enough_images = [
                child_id
                for child_id in data['children']
                if self.get_total_images_for_taxon(hierarchy, child_id) >= min_images
            ]

            # At least one child must have enough images
            if len(children_with_enough_images) >= 1:
                suitable_taxa.append(taxon_id)

        logger.info(
            f"Found {len(suitable_taxa)} taxa suitable for training "
            f"(min_images={min_images})"
        )

        return suitable_taxa

    def create_train_val_split(
        self,
        images: List[int],
        train_ratio: float = 0.8
    ) -> Tuple[List[int], List[int]]:
        """
        Split images into train and validation sets.

        Args:
            images: List of image IDs
            train_ratio: Fraction for training (default: 0.8 = 80%)

        Returns:
            Tuple of (train_images, val_images)

        Note:
            If only 1 image, puts it in training set only.
        """
        if len(images) == 1:
            return images, []

        # Shuffle and split
        images_shuffled = images.copy()
        random.shuffle(images_shuffled)

        split_index = int(len(images_shuffled) * train_ratio)
        train_images = images_shuffled[:split_index]
        val_images = images_shuffled[split_index:]

        return train_images, val_images

    def build_trainingset_structure(
        self,
        hierarchy: Dict,
        suitable_taxa: List[str],
        min_images: int,
        train_ratio: float = 0.8
    ) -> Dict:
        """
        Build training dataset structure for all suitable taxa.

        Creates a nested dictionary defining which images go to which
        train/val splits for each taxon and its children.

        Args:
            hierarchy: Taxonomy tree
            suitable_taxa: Taxa to create datasets for
            min_images: Minimum images per class
            train_ratio: Train/val split ratio

        Returns:
            Dictionary with training structure

        Structure:
            {
                'Insecta': {
                    'Coleoptera': {
                        'descendants': {'Coleoptera', 'Carabidae', ...},
                        'images': {1, 2, 3, ...},
                        'train_images': [1, 2, ...],
                        'val_images': [3, 4, ...]
                    },
                    'Diptera': {...},
                    ...
                },
                ...
            }
        """
        trainingset_structure = {}

        def add_descendants_and_images(
            taxon_id: str,
            descendants: Set[str],
            images: Set[int]
        ):
            """Recursively collect all descendants and their images."""
            descendants.add(taxon_id)
            images.update(hierarchy[taxon_id]['images'])

            for child_id in hierarchy[taxon_id]['children']:
                add_descendants_and_images(child_id, descendants, images)

        # Process each suitable taxon
        for taxon_id in suitable_taxa:
            class_structure = {}

            # Process each child as a separate class
            for child_id in hierarchy[taxon_id]['children']:
                descendants = set()
                images = set()

                # Collect all descendants and images
                add_descendants_and_images(child_id, descendants, images)

                # Convert to list and check minimum
                images_list = sorted(list(images))

                if len(images_list) >= min_images:
                    # Create train/val split
                    train_images, val_images = self.create_train_val_split(
                        images_list,
                        train_ratio
                    )

                    class_structure[child_id] = {
                        'descendants': descendants,
                        'images': images,
                        'train_images': train_images,
                        'val_images': val_images
                    }

            trainingset_structure[taxon_id] = class_structure

        logger.info(f"Built training structure for {len(trainingset_structure)} taxa")
        return trainingset_structure

    def build_additional_class(
        self,
        taxon_id: str,
        hierarchy: Dict,
        trainingset_structure: Dict,
        num_train_images: int,
        num_val_images: int,
        min_images: int
    ):
        """
        Build "additional class" for taxa with too few specimens.

        This creates a catch-all class for taxa that don't meet the
        minimum image requirement, allowing the model to learn they
        exist but aren't classified further.

        Args:
            taxon_id: Parent taxon to create additional class for
            hierarchy: Taxonomy tree
            trainingset_structure: Training structure to modify
            num_train_images: Training images per independent taxon
            num_val_images: Validation images per independent taxon
            min_images: Minimum threshold

        Example:
            If Insecta has children [Coleoptera: 100 imgs, Diptera: 90 imgs,
            Hemiptera: 5 imgs], and min_images=7, then Hemiptera goes to
            additional_class.
        """
        # Find taxa connected to this taxon (this taxon + descendants + parent)
        connected_taxa = set()

        def add_connected_taxa(tid: str):
            """Recursively add all descendants."""
            connected_taxa.add(tid)
            for child_id in hierarchy[tid]['children']:
                add_connected_taxa(child_id)

        add_connected_taxa(taxon_id)

        # Add parent
        for key, value in hierarchy.items():
            if taxon_id in value['children']:
                connected_taxa.add(key)

        # Find independent taxa (not connected)
        independent_taxa = set(hierarchy.keys()) - connected_taxa

        # Also add children with too few images
        for child_id in hierarchy[taxon_id]['children']:
            if self.get_total_images_for_taxon(hierarchy, child_id) < min_images:
                independent_taxa.add(child_id)

        # Collect images from independent taxa
        train_additional_images = []
        val_additional_images = []

        for independent_taxon_id in independent_taxa:
            # Get images, excluding test set
            images = [
                img for img in hierarchy[independent_taxon_id]['images']
                if img not in self.excluded_ids
            ]

            random.shuffle(images)

            train_count = 0
            val_count = 0

            # Distribute images maintaining 5:1 train/val ratio
            for image in images:
                if train_count < num_train_images and val_count < num_val_images:
                    # Prefer training until we have 5x more than val
                    if train_count < 5 * val_count or val_count == num_val_images:
                        train_additional_images.append(image)
                        train_count += 1
                    else:
                        val_additional_images.append(image)
                        val_count += 1
                elif train_count < num_train_images:
                    train_additional_images.append(image)
                    train_count += 1
                elif val_count < num_val_images:
                    val_additional_images.append(image)
                    val_count += 1

        # Add to structure if we have images
        if train_additional_images or val_additional_images:
            if taxon_id not in trainingset_structure:
                trainingset_structure[taxon_id] = {}

            trainingset_structure[taxon_id]['additional_class'] = {
                'train_images': train_additional_images,
                'val_images': val_additional_images
            }

            logger.info(
                f"Additional class for {taxon_id}: "
                f"train={len(train_additional_images)}, "
                f"val={len(val_additional_images)}"
            )
        else:
            logger.info(f"No images for additional_class in {taxon_id}")

    def add_debris_class(
        self,
        top_taxon_id: str,
        train_ratio: float = 0.8
    ):
        """
        Add debris class to top-level taxon.

        Debris are non-specimen images (dust, plant material, etc.)
        that should be filtered out during inference.

        Args:
            top_taxon_id: Taxon to add debris class to (usually 'RT')
            train_ratio: Train/val split ratio

        Example:
            >>> creator.add_debris_class('RT')
            # Creates RT/train/Debris and RT/val/Debris directories
        """
        if self.data_manager is None:
            logger.warning("No data manager, skipping debris class")
            return

        # Get debris images
        debris_images = self.data_manager.get_debris_images()

        if not debris_images:
            logger.info("No debris images found")
            return

        # Shuffle and split
        random.shuffle(debris_images)
        split_index = int(len(debris_images) * train_ratio)

        train_images = debris_images[:split_index]
        val_images = debris_images[split_index:]

        # Create directories
        train_dir = self.base_directory / str(top_taxon_id) / 'train' / 'Debris'
        val_dir = self.base_directory / str(top_taxon_id) / 'val' / 'Debris'

        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        # Process training images (with augmentation)
        for image_id in train_images:
            image_path = self.data_manager.get_image_path(image_id)
            if image_path and image_path.exists():
                process_image_for_training(
                    image_path,
                    train_dir,
                    'train',
                    'Debris',
                    apply_augmentation=True
                )

        # Process validation images (no augmentation)
        for image_id in val_images:
            image_path = self.data_manager.get_image_path(image_id)
            if image_path and image_path.exists():
                process_image_for_training(
                    image_path,
                    val_dir,
                    'val',
                    'Debris',
                    apply_augmentation=False
                )

        logger.info(
            f"Debris class created: train={len(train_images)}, "
            f"val={len(val_images)}"
        )

    def create_dataset_directories(
        self,
        trainingset_structure: Dict,
        num_processes: int = 4
    ):
        """
        Create physical directory structure and copy images.

        Args:
            trainingset_structure: Training structure from build_trainingset_structure()
            num_processes: Number of parallel processes for image copying

        Example:
            >>> creator.create_dataset_directories(structure, num_processes=8)
        """
        # Use multiprocessing pool
        pool = multiprocessing.Pool(processes=num_processes)

        for taxon_id, classes in trainingset_structure.items():
            for dataset_type in ['train', 'val']:
                for class_id, details in classes.items():
                    # Get image IDs for this split
                    image_ids = details.get(f'{dataset_type}_images', [])

                    if not image_ids:
                        continue

                    # Create class directory
                    class_dir = (
                        self.base_directory /
                        str(taxon_id) /
                        dataset_type /
                        str(class_id)
                    )
                    class_dir.mkdir(parents=True, exist_ok=True)

                    # Get image paths
                    image_paths = [
                        self.data_manager.get_image_path(img_id)
                        for img_id in image_ids
                    ]

                    # Filter valid paths
                    valid_paths = [p for p in image_paths if p and p.exists()]

                    # Process images in parallel
                    process_func = partial(
                        process_image_for_training,
                        class_images_directory=class_dir,
                        dataset_type=dataset_type,
                        class_id=class_id,
                        apply_augmentation=(class_id not in ['additional_class', 'Debris'])
                    )

                    pool.map(process_func, valid_paths)

        pool.close()
        pool.join()

        logger.info("Dataset directories created")

    def save_counts(self, taxon_id: str, classes: Dict):
        """
        Save image counts to JSON file.

        Creates counts.json with statistics for each model.

        Args:
            taxon_id: Taxon this model is for
            classes: Class structure from trainingset_structure

        Example counts.json:
            {
                "model_taxon_id": "Insecta",
                "train_count": 1234,
                "val_count": 308,
                "class_counts": {
                    "Coleoptera": {
                        "taxon": "Coleoptera",
                        "train_taxon_count": 567,
                        "val_taxon_count": 142
                    },
                    ...
                }
            }
        """
        counts = {
            'model_taxon_id': taxon_id,
            'train_count': 0,
            'val_count': 0,
            'class_counts': {}
        }

        for class_id, details in classes.items():
            # Skip special classes
            if class_id in ['additional_class', 'Debris']:
                continue

            train_images = details.get('train_images', [])
            val_images = details.get('val_images', [])

            train_count = len(train_images)
            val_count = len(val_images)

            counts['train_count'] += train_count
            counts['val_count'] += val_count

            counts['class_counts'][class_id] = {
                'taxon': class_id,
                'train_taxon_count': train_count,
                'val_taxon_count': val_count
            }

        # Save to file
        counts_file = self.base_directory / str(taxon_id) / 'counts.json'
        counts_file.parent.mkdir(parents=True, exist_ok=True)

        with open(counts_file, 'w', encoding='utf-8') as f:
            json.dump(counts, f, indent=4)

        logger.info(
            f"Counts saved for {taxon_id}: "
            f"train={counts['train_count']}, val={counts['val_count']}"
        )

    def print_structure(self, trainingset_structure: Dict):
        """
        Print training structure to console (for debugging).

        Args:
            trainingset_structure: Training structure
        """
        for taxon_id, classes in trainingset_structure.items():
            print(f"\nTraining dataset for taxon {taxon_id}:")

            for class_id, details in classes.items():
                if class_id == 'additional_class':
                    print(
                        f"  Additional class: "
                        f"train={len(details['train_images'])}, "
                        f"val={len(details['val_images'])}"
                    )
                else:
                    print(f"  Class {class_id}:")
                    print(f"    Descendants: {len(details['descendants'])}")
                    print(f"    Train: {len(details['train_images'])}")
                    print(f"    Val: {len(details['val_images'])}")

    def generate_training_data(
        self,
        top_taxon_id: str = 'RT',
        min_images_per_class: int = 7,
        additional_class_train_images: int = 10,
        additional_class_val_images: int = 2,
        num_processes: int = 4,
        test_ratio: float = 0.10,
        val_ratio: float = 0.15
    ):
        """
        Generate complete training dataset structure.

        This is the main entry point that orchestrates all steps:
        1. Build taxonomy hierarchy
        2. Create train/val/test split
        3. Find suitable taxa
        4. Build training structure
        5. Add debris class
        6. Add additional classes
        7. Create directories and copy images
        8. Save counts

        Args:
            top_taxon_id: Root taxon (usually 'RT' for Arthropoda)
            min_images_per_class: Minimum specimens per class
            additional_class_train_images: Training images per taxon for additional class
            additional_class_val_images: Val images per taxon for additional class
            num_processes: Parallel processes for image copying
            test_ratio: Fraction of images for test set (excluded)
            val_ratio: Fraction of remaining images for validation

        Example:
            >>> creator = TrainingDatasetCreator(Path('./training'))
            >>> creator.generate_training_data(
            ...     top_taxon_id='RT',
            ...     min_images_per_class=7
            ... )
        """
        logger.info("=" * 70)
        logger.info("GENERATING TRAINING DATA")
        logger.info("=" * 70)

        # Step 1: Build taxonomy hierarchy
        logger.info("Step 1: Building taxonomy hierarchy")
        hierarchy_builder = TaxonomyHierarchy(
            data_manager=self.data_manager,
            start_year=self.start_year,
            end_year=self.end_year
        )
        hierarchy = hierarchy_builder.build()

        # Step 2: Create test/train/val split
        logger.info("Step 2: Creating train/val/test split")
        new_excluded = set()

        for taxon_id, taxon_data in hierarchy.items():
            images = taxon_data['images']
            if not images:
                continue

            random.shuffle(images)
            total = len(images)

            # Calculate split sizes
            test_count = int(round(total * test_ratio))
            val_count = int(round(total * val_ratio))

            # Split
            test_images = images[:test_count]
            val_images = images[test_count:test_count + val_count]
            train_images = images[test_count + val_count:]

            # Add test images to excluded
            new_excluded.update(test_images)

            # Update hierarchy with only train+val
            taxon_data['images'] = train_images + val_images

            logger.debug(
                f"{taxon_id}: Total={total}, Test={len(test_images)}, "
                f"Val={len(val_images)}, Train={len(train_images)}"
            )

        self.excluded_ids = new_excluded
        logger.info(f"Excluded {len(self.excluded_ids)} images for testing")

        # Remove excluded from all taxa
        for taxon_data in hierarchy.values():
            taxon_data['images'] = [
                img_id for img_id in taxon_data['images']
                if img_id not in self.excluded_ids
            ]

        # Step 3: Filter to top taxon
        logger.info(f"Step 3: Filtering hierarchy to {top_taxon_id}")
        filtered_hierarchy = hierarchy_builder.filter_hierarchy_by_taxon(
            hierarchy,
            top_taxon_id
        )

        # Step 4: Find suitable taxa
        logger.info("Step 4: Finding suitable taxa for training")
        suitable_taxa = self.find_suitable_taxa_for_training(
            filtered_hierarchy,
            min_images_per_class
        )

        # Step 5: Build training structure
        logger.info("Step 5: Building training structure")
        train_ratio = 1.0 - val_ratio  # Remaining after test
        trainingset_structure = self.build_trainingset_structure(
            filtered_hierarchy,
            suitable_taxa,
            min_images_per_class,
            train_ratio
        )

        # Step 6: Add debris class
        logger.info("Step 6: Adding debris class")
        self.add_debris_class(top_taxon_id)

        # Step 7: Add additional classes
        logger.info("Step 7: Adding additional classes")
        for taxon_id in suitable_taxa:
            self.build_additional_class(
                taxon_id,
                hierarchy,
                trainingset_structure,
                additional_class_train_images,
                additional_class_val_images,
                min_images_per_class
            )

        # Step 8: Print structure
        logger.info("Step 8: Training structure summary")
        self.print_structure(trainingset_structure)

        # Step 9: Create directories and copy images
        logger.info("Step 9: Creating dataset directories")
        self.create_dataset_directories(trainingset_structure, num_processes)

        # Step 10: Save counts
        logger.info("Step 10: Saving counts")
        for taxon_id, classes in trainingset_structure.items():
            self.save_counts(taxon_id, classes)

        logger.info("=" * 70)
        logger.info("TRAINING DATA GENERATION COMPLETE")
        logger.info("=" * 70)

        return trainingset_structure, self.excluded_ids
