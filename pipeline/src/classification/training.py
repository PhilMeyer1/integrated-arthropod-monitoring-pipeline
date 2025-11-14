"""
YOLO model training for hierarchical classification.

This module handles training YOLO classification models for each taxonomic
level, managing model versions (sets), and saving trained models with metadata.

Corresponds to Methods section "Model Training and Validation".
"""

from pathlib import Path
from typing import Optional, Dict
import json
import shutil

from ultralytics import YOLO

from src.config import config
from src.utils.logging_config import get_logger, LogContext

logger = get_logger(__name__)


class ModelTrainer:
    """
    Train YOLO classification models for taxonomic hierarchy.

    This class trains one model for each taxonomic level, where each
    model classifies specimens into child taxa. All models in a training
    run share a set_number for version tracking.

    Example:
        >>> trainer = ModelTrainer(
        ...     trainingsets_path=Path('./training_data'),
        ...     models_path=Path('./models')
        ... )
        >>> trainer.train_all_models(set_number=1)
    """

    def __init__(
        self,
        trainingsets_path: Path,
        models_path: Path,
        data_manager=None
    ):
        """
        Initialize model trainer.

        Args:
            trainingsets_path: Directory with training datasets
            models_path: Directory to save trained models
            data_manager: DataManager for saving model metadata
        """
        self.trainingsets_path = Path(trainingsets_path)
        self.models_path = Path(models_path)
        self.data_manager = data_manager

        # Create models directory
        self.models_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"ModelTrainer initialized: "
            f"training={trainingsets_path}, models={models_path}"
        )

    def get_next_set_number(self) -> int:
        """
        Get next available set number for model versioning.

        Set numbers group related models trained together. This allows
        tracking different training runs and comparing model versions.

        Returns:
            Next set number (1 if no models exist)

        Example:
            >>> set_num = trainer.get_next_set_number()
            >>> print(f"Training set {set_num}")
        """
        if self.data_manager is None:
            # Without database, scan model files
            existing_models = list(self.models_path.glob("*.pt"))
            if not existing_models:
                return 1

            # Extract set numbers from metadata files
            max_set = 0
            for model_file in existing_models:
                metadata_file = model_file.with_suffix('.json')
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        set_num = metadata.get('set_number', 0)
                        max_set = max(max_set, set_num)

            return max_set + 1

        else:
            # Query database
            max_set = self.data_manager.get_max_set_number()
            return (max_set or 0) + 1

    def get_training_config(self) -> Dict:
        """
        Get YOLO training configuration from config file.

        Returns:
            Dictionary with training parameters

        Example:
            >>> cfg = trainer.get_training_config()
            >>> print(f"Training for {cfg['epochs']} epochs")
        """
        # Get from config or use defaults
        return {
            'epochs': config.get('classification.training.epochs', 50),
            'imgsz': config.get('classification.training.image_size', 512),
            'batch': config.get('classification.training.batch_size', -1),  # Auto
            'device': config.get('classification.device', 'cuda'),
            'workers': config.get('classification.training.workers', 10),
            'patience': config.get('classification.training.patience', 10),
            'save_period': config.get('classification.training.save_period', -1),

            # Augmentation settings (minimal for specimens)
            'mosaic': 0.0,
            'close_mosaic': 0,
            'hsv_h': 0.025,  # Slight hue variation
            'hsv_s': 0.3,    # Slight saturation variation
            'hsv_v': 0.3,    # Slight value variation
            'degrees': 0.0,  # No rotation (already applied)
            'translate': 0.0,  # No translation
            'scale': 0.0,  # No scaling
            'shear': 0.0,  # No shearing
            'flipud': 0.0,  # No vertical flip
            'fliplr': 0.0,  # No horizontal flip
            'mixup': 0.0,  # No mixup
            'bgr': 0,  # No BGR conversion
            'erasing': 0,  # No random erasing
            'auto_augment': None,  # No auto augmentation
        }

    def load_counts_file(self, folder_path: Path) -> Optional[Dict]:
        """
        Load counts.json from training folder.

        Args:
            folder_path: Path to training dataset folder

        Returns:
            Dictionary with counts, or None if not found
        """
        counts_file = folder_path / 'counts.json'

        if not counts_file.exists():
            logger.warning(f"No counts.json found in {folder_path}")
            return None

        with open(counts_file, 'r', encoding='utf-8') as f:
            counts = json.load(f)

        logger.info(f"Loaded counts from {counts_file}")
        return counts

    def delete_training_dataset(self, folder_path: Path):
        """
        Delete training dataset folder after successful training.

        This saves disk space by removing the dataset once the model
        is trained and saved.

        Args:
            folder_path: Path to training dataset folder

        Example:
            >>> trainer.delete_training_dataset(Path('./training_data/Insecta'))
        """
        try:
            shutil.rmtree(folder_path)
            logger.info(f"Deleted training dataset: {folder_path}")
        except OSError as e:
            logger.error(f"Failed to delete training dataset {folder_path}: {e}")

    def train_model(
        self,
        taxon_id: str,
        set_number: int,
        base_model_path: Optional[Path] = None,
        delete_dataset_after: bool = False
    ) -> Optional[Path]:
        """
        Train a single YOLO model for a taxonomic level.

        Args:
            taxon_id: Taxon this model is for (folder name)
            set_number: Model set number for versioning
            base_model_path: Path to base YOLO model (default: from config)
            delete_dataset_after: Whether to delete training data after training

        Returns:
            Path to trained model, or None if training failed

        Example:
            >>> model_path = trainer.train_model('Insecta', set_number=1)
            >>> print(f"Model saved to {model_path}")
        """
        # Get training folder
        folder_path = self.trainingsets_path / str(taxon_id)

        if not folder_path.is_dir():
            logger.error(f"Training folder not found: {folder_path}")
            return None

        # Check for train and val directories
        train_path = folder_path / "train"
        val_path = folder_path / "val"

        if not train_path.is_dir() or not val_path.is_dir():
            logger.error(f"Train or val directory missing in {folder_path}")
            return None

        # Get base model
        if base_model_path is None:
            base_model_path = Path(
                config.get(
                    'classification.training.base_model',
                    'yolo11m-cls.pt'
                )
            )

        logger.info(f"Training model for {taxon_id} (set {set_number})")
        logger.info(f"Base model: {base_model_path}")

        # Load model
        model = YOLO(str(base_model_path))

        # Get training config
        train_config = self.get_training_config()

        # Train
        with LogContext(f"Training {taxon_id}", logger):
            try:
                results = model.train(
                    data=str(folder_path),
                    **train_config
                )

                # Get save directory
                save_dir = model.trainer.save_dir
                trained_model_path = Path(save_dir) / 'weights' / 'best.pt'

                if not trained_model_path.exists():
                    logger.error(f"Trained model not found at {trained_model_path}")
                    return None

                # Load counts
                counts = self.load_counts_file(folder_path)

                # Save model metadata
                model_id = self.save_model_metadata(
                    taxon_id=taxon_id,
                    set_number=set_number,
                    counts=counts
                )

                # Determine final model path
                if model_id:
                    final_model_path = self.models_path / f"{model_id}.pt"
                else:
                    final_model_path = self.models_path / f"{taxon_id}_set{set_number}.pt"

                # Move model to models directory
                shutil.move(str(trained_model_path), str(final_model_path))
                logger.info(f"Model saved to {final_model_path}")

                # Update metadata with final path
                if model_id:
                    self.update_model_path(model_id, final_model_path)

                # Delete training dataset if requested
                if delete_dataset_after:
                    self.delete_training_dataset(folder_path)

                return final_model_path

            except Exception as e:
                logger.error(f"Training failed for {taxon_id}: {e}", exc_info=True)
                return None

    def save_model_metadata(
        self,
        taxon_id: str,
        set_number: int,
        counts: Optional[Dict] = None
    ) -> Optional[int]:
        """
        Save model metadata to database or file.

        Args:
            taxon_id: Taxon this model is for
            set_number: Model set number
            counts: Training/validation counts

        Returns:
            Model ID if using database, None otherwise
        """
        metadata = {
            'taxon_id': taxon_id,
            'set_number': set_number,
            'train_count': counts.get('train_count', 0) if counts else 0,
            'val_count': counts.get('val_count', 0) if counts else 0,
            'class_counts': counts.get('class_counts', {}) if counts else {}
        }

        if self.data_manager is None:
            # Save to JSON file
            metadata_file = self.models_path / f"{taxon_id}_set{set_number}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Metadata saved to {metadata_file}")
            return None

        else:
            # Save to database
            model_id = self.data_manager.save_model(
                taxon_id=taxon_id,
                set_number=set_number,
                train_count=metadata['train_count'],
                val_count=metadata['val_count']
            )

            # Save class counts
            if counts and 'class_counts' in counts:
                for class_id, class_data in counts['class_counts'].items():
                    self.data_manager.save_model_taxon_count(
                        model_id=model_id,
                        taxon=class_data['taxon'],
                        train_count=class_data.get('train_taxon_count', 0),
                        val_count=class_data.get('val_taxon_count', 0),
                        set_number=set_number
                    )

            logger.info(f"Metadata saved to database (model_id={model_id})")
            return model_id

    def update_model_path(self, model_id: int, model_path: Path):
        """
        Update model path in database.

        Args:
            model_id: Model ID
            model_path: Path to trained model file
        """
        if self.data_manager:
            self.data_manager.update_model_path(model_id, str(model_path))
            logger.info(f"Updated model path for ID {model_id}")

    def train_all_models(
        self,
        set_number: Optional[int] = None,
        base_model_path: Optional[Path] = None,
        delete_datasets_after: bool = False
    ) -> Dict[str, Path]:
        """
        Train all models in the training sets directory.

        Iterates through all subdirectories in trainingsets_path and
        trains a model for each.

        Args:
            set_number: Model set number (auto-increments if None)
            base_model_path: Path to base YOLO model
            delete_datasets_after: Delete training data after successful training

        Returns:
            Dictionary mapping taxon_id to trained model path

        Example:
            >>> trained_models = trainer.train_all_models()
            >>> print(f"Trained {len(trained_models)} models")
        """
        logger.info("=" * 70)
        logger.info("TRAINING ALL MODELS")
        logger.info("=" * 70)

        # Get set number
        if set_number is None:
            set_number = self.get_next_set_number()
            logger.info(f"Using set number: {set_number}")

        # Find all training folders
        training_folders = [
            d for d in self.trainingsets_path.iterdir()
            if d.is_dir() and (d / 'train').exists()
        ]

        if not training_folders:
            logger.warning(f"No training folders found in {self.trainingsets_path}")
            return {}

        logger.info(f"Found {len(training_folders)} training folders")

        # Train each model
        trained_models = {}

        for folder in training_folders:
            taxon_id = folder.name

            model_path = self.train_model(
                taxon_id=taxon_id,
                set_number=set_number,
                base_model_path=base_model_path,
                delete_dataset_after=delete_datasets_after
            )

            if model_path:
                trained_models[taxon_id] = model_path
            else:
                logger.error(f"Failed to train model for {taxon_id}")

        logger.info("=" * 70)
        logger.info(f"TRAINING COMPLETE: {len(trained_models)}/{len(training_folders)} models")
        logger.info("=" * 70)

        return trained_models

    def export_model(
        self,
        model_path: Path,
        format: str = 'onnx',
        **kwargs
    ) -> Optional[Path]:
        """
        Export trained model to different format.

        Useful for deployment or optimization.

        Args:
            model_path: Path to trained .pt model
            format: Export format ('onnx', 'torchscript', 'tflite', etc.)
            **kwargs: Additional export arguments

        Returns:
            Path to exported model

        Example:
            >>> exported = trainer.export_model(
            ...     Path('./models/1.pt'),
            ...     format='onnx'
            ... )
        """
        logger.info(f"Exporting {model_path} to {format}")

        try:
            model = YOLO(str(model_path))
            export_path = model.export(format=format, **kwargs)
            logger.info(f"Exported to {export_path}")
            return Path(export_path)

        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            return None

    def validate_model(
        self,
        model_path: Path,
        data_path: Path
    ) -> Dict:
        """
        Run validation on trained model.

        Args:
            model_path: Path to trained model
            data_path: Path to validation dataset

        Returns:
            Dictionary with validation metrics

        Example:
            >>> metrics = trainer.validate_model(
            ...     Path('./models/1.pt'),
            ...     Path('./training_data/Insecta')
            ... )
            >>> print(f"Accuracy: {metrics['top1']:.3f}")
        """
        logger.info(f"Validating {model_path}")

        try:
            model = YOLO(str(model_path))
            results = model.val(data=str(data_path))

            metrics = {
                'top1': results.top1,
                'top5': results.top5,
            }

            logger.info(f"Validation results: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Validation failed: {e}", exc_info=True)
            return {}
