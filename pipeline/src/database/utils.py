"""
Database utilities and abstraction layer.

Provides a unified interface for both database and file-based storage.
"""

from typing import Union, Optional, List, Dict, Any
from pathlib import Path
import pandas as pd

from src.config import config
from src.database.models import (
    Base, Project, Location, SamplingRound, SampleAssignment,
    CompositeImage, SingleImage, Taxa, Model, InferenceResult,
    ModelTaxonCount, ExcludedIdsForTraining,
    get_session as get_db_session, init_database
)
from src.utils.file_utils import FileBasedDataManager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


class DataManager:
    """
    Unified data management interface.

    Automatically uses either database or file-based storage
    based on configuration.

    Example:
        >>> from src.database.utils import get_data_manager
        >>> dm = get_data_manager()
        >>> dm.save_sample_metadata([{'sample_id': 'S001', ...}])
        >>> images = dm.get_image_paths(sample_id='S001')
    """

    def __init__(self, use_database: bool = None, database_url: str = None, data_root: Path = None):
        """
        Initialize data manager.

        Args:
            use_database: Whether to use database. If None, reads from config.
            database_url: Database URL (if using database)
            data_root: Data root directory (if using file-based)
        """
        if use_database is None:
            use_database = config.get('database.use_database', False)

        self.use_database = use_database

        if self.use_database:
            # Database mode
            if database_url is None:
                database_url = config.get('database.url', 'sqlite:///arthropod_pipeline.db')

            self.db_url = database_url
            self.engine = create_engine(self.db_url, echo=config.get('database.echo', False))
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            self.session = self.Session()
            self.backend = "database"
        else:
            # File-based mode
            if data_root is None:
                data_root = Path(config.get('paths.data_root', './data'))

            self.file_manager = FileBasedDataManager(data_root)
            self.backend = "file"

    def __repr__(self):
        return f"DataManager(backend={self.backend})"

    # --- Sample Metadata ---

    def save_sample_metadata(self, metadata: List[Dict]):
        """Save sample metadata."""
        if self.use_database:
            # Database implementation
            for item in metadata:
                sample = SampleAssignment(
                    sample_id=item.get('sample_id'),
                    size_fraction=item.get('size_fraction'),
                    tray_size=item.get('tray_size'),
                    weight=item.get('weight_mg')
                )
                self.session.add(sample)
            self.session.commit()
        else:
            # File-based implementation
            self.file_manager.save_sample_metadata(metadata)

    def load_sample_metadata(self) -> Union[pd.DataFrame, List[Any]]:
        """Load all sample metadata."""
        if self.use_database:
            samples = self.session.query(SampleAssignment).all()
            return samples
        else:
            return self.file_manager.load_sample_metadata()

    def get_sample_by_id(self, sample_id: str) -> Optional[Union[SampleAssignment, Dict]]:
        """Get sample by ID."""
        if self.use_database:
            return self.session.query(SampleAssignment).filter(
                SampleAssignment.sample_id == sample_id
            ).first()
        else:
            return self.file_manager.get_sample_by_id(sample_id)

    # --- Images ---

    def save_single_images(self, images: List[Dict]):
        """Save extracted specimen images."""
        if self.use_database:
            for item in images:
                img = SingleImage(
                    composite_image_id=item.get('composite_id'),
                    work_images_path=item.get('file_path'),
                    bbox_x1=item.get('bbox_x1'),
                    bbox_y1=item.get('bbox_y1'),
                    bbox_x2=item.get('bbox_x2'),
                    bbox_y2=item.get('bbox_y2'),
                    detection_confidence=item.get('detection_confidence')
                )
                self.session.add(img)
            self.session.commit()
        else:
            self.file_manager.save_single_images(images)

    def get_image_paths(self, sample_id: Optional[str] = None) -> List[Path]:
        """Get image file paths, optionally filtered by sample."""
        if self.use_database:
            query = self.session.query(SingleImage)
            if sample_id:
                query = query.join(SampleAssignment).filter(
                    SampleAssignment.sample_id == sample_id
                )
            images = query.all()
            return [Path(img.work_images_path) for img in images if img.work_images_path]
        else:
            return self.file_manager.get_image_paths(sample_id)

    # --- Classifications ---

    def save_classifications(self, results: List[Dict]):
        """Save classification results."""
        if self.use_database:
            for item in results:
                result = InferenceResult(
                    image_id=item.get('image_id'),
                    predicted_taxon=item.get('predicted_taxon'),
                    confidence=item.get('confidence')
                )
                self.session.add(result)
            self.session.commit()
        else:
            self.file_manager.save_classifications(results)

    def load_classifications(self) -> Union[pd.DataFrame, List[Any]]:
        """Load all classification results."""
        if self.use_database:
            return self.session.query(InferenceResult).all()
        else:
            return self.file_manager.load_classifications()

    # --- Models ---

    def register_model(self, taxon: str, model_path: str, metadata: Optional[Dict] = None):
        """Register a trained model."""
        if self.use_database:
            model = Model(
                taxon_id=taxon,
                model_path=model_path,
                model_type=metadata.get('model_type', 'classification') if metadata else 'classification',
                accuracy=metadata.get('accuracy') if metadata else None,
                num_training_images=metadata.get('num_train') if metadata else None
            )
            self.session.add(model)
            self.session.commit()
        else:
            self.file_manager.register_model(taxon, model_path, metadata)

    def get_model_path(self, taxon: str) -> Optional[Path]:
        """Get model path for a taxon."""
        if self.use_database:
            model = self.session.query(Model).filter(Model.taxon_id == taxon).first()
            if model:
                return Path(model.model_path)
            return None
        else:
            return self.file_manager.get_model_path(taxon)

    def get_model_registry(self) -> Dict[str, str]:
        """Get all registered models."""
        if self.use_database:
            models = self.session.query(Model).all()
            return {m.taxon_id: m.model_path for m in models}
        else:
            registry = self.file_manager.get_model_registry()
            return {taxon: info['model_path'] for taxon, info in registry.items()}

    # --- Taxonomy ---

    def get_specimens(self, taxon_id: Optional[str] = None,
                     start_year: Optional[int] = None,
                     end_year: Optional[int] = None) -> List:
        """
        Get specimen images filtered by taxon and/or year.

        Args:
            taxon_id: Filter by taxon (manual determination)
            start_year: Filter from this year onwards
            end_year: Filter up to this year

        Returns:
            List of SingleImage records or dictionaries
        """
        if self.use_database:
            query = self.session.query(SingleImage)

            if taxon_id:
                query = query.filter(SingleImage.manual_determination == taxon_id)

            if start_year or end_year:
                # Join with SampleAssignment -> SamplingRound to filter by year
                query = query.join(SampleAssignment,
                                  SingleImage.sample_assignment_id == SampleAssignment.id)
                query = query.join(SamplingRound,
                                  SampleAssignment.sampling_round_id == SamplingRound.id)

                if start_year:
                    query = query.filter(SamplingRound.year >= start_year)
                if end_year:
                    query = query.filter(SamplingRound.year <= end_year)

            return query.all()
        else:
            # File-based mode would need custom filtering
            return self.file_manager.get_specimens(taxon_id, start_year, end_year)

    def get_taxon(self, taxon_id: str) -> Optional[Dict]:
        """Get taxon information by ID.

        Returns:
            Dictionary with taxon information, or None if not found.
            Always returns a dict for consistent API (not ORM object).
        """
        if self.use_database:
            taxa = self.session.query(Taxa).filter(Taxa.taxon_id == taxon_id).first()
            if taxa:
                # Convert ORM object to dict for consistent API
                return {
                    'taxon_id': taxa.taxon_id,
                    'name': taxa.name,
                    'rank': taxa.rank,
                    'parent_id': taxa.parent_id,
                    'col_id': taxa.col_id,
                    'col_name': taxa.col_name
                }
            return None
        else:
            return self.file_manager.get_taxon(taxon_id)

    def get_parent_taxon(self, taxon_id: str) -> Optional[str]:
        """Get parent taxon ID."""
        if self.use_database:
            taxon = self.session.query(Taxa).filter(Taxa.taxon_id == taxon_id).first()
            return taxon.parent_id if taxon else None
        else:
            taxon = self.file_manager.get_taxon(taxon_id)
            return taxon.get('parent_id') if taxon else None

    def get_image_path(self, image_id: int) -> Optional[Path]:
        """Get file path for a single image by ID."""
        if self.use_database:
            img = self.session.query(SingleImage).filter(SingleImage.id == image_id).first()
            return Path(img.work_images_path) if img and img.work_images_path else None
        else:
            return self.file_manager.get_image_path(image_id)

    def get_debris_images(self) -> List[int]:
        """Get list of image IDs marked as debris."""
        if self.use_database:
            images = self.session.query(SingleImage.id).filter(
                SingleImage.is_debris == True
            ).all()
            return [img.id for img in images]
        else:
            return self.file_manager.get_debris_images()

    # --- Model Management ---

    def get_max_set_number(self) -> Optional[int]:
        """Get maximum set number from trained models."""
        if self.use_database:
            from sqlalchemy import func
            max_set = self.session.query(func.max(Model.set_number)).scalar()
            return max_set
        else:
            return self.file_manager.get_max_set_number()

    def save_model(self, taxon_id: str, set_number: int,
                   train_count: int = 0, val_count: int = 0) -> Optional[int]:
        """Save model metadata and return model ID."""
        if self.use_database:
            model = Model(
                taxon_id=taxon_id,
                model_path='',  # Will be updated later
                set_number=set_number,
                num_training_images=train_count,
                num_validation_images=val_count
            )
            self.session.add(model)
            self.session.flush()  # Get ID without committing
            return model.id
        else:
            return None

    def save_model_taxon_count(self, model_id: int, taxon: str,
                               train_count: int, val_count: int,
                               set_number: int):
        """Save per-taxon counts for a model."""
        if self.use_database:
            count_entry = ModelTaxonCount(
                model_id=model_id,
                taxon_id=taxon,
                num_train_images=train_count,
                num_val_images=val_count
            )
            self.session.add(count_entry)
            self.session.flush()

    def update_model_path(self, model_id: int, model_path: str):
        """Update the file path for a trained model."""
        if self.use_database:
            model = self.session.query(Model).filter(Model.id == model_id).first()
            if model:
                model.model_path = model_path
                self.session.flush()

    def get_models_by_set(self, set_number: int) -> List[Dict]:
        """Get all models for a specific set number with parent information."""
        if self.use_database:
            models = self.session.query(Model).filter(
                Model.set_number == set_number
            ).all()

            result = []
            for model in models:
                # Get parent taxon
                taxon = self.session.query(Taxa).filter(
                    Taxa.taxon_id == model.taxon_id
                ).first()

                result.append({
                    'model_id': model.id,
                    'taxon_id': model.taxon_id,
                    'parent_id': taxon.parent_id if taxon else None,
                    'model_path': model.model_path,
                    'set_number': model.set_number
                })

            return result
        else:
            return []

    def get_excluded_ids(self, set_number: int) -> List[int]:
        """Get list of image IDs excluded from training for a specific set."""
        if self.use_database:
            excluded = self.session.query(ExcludedIdsForTraining.image_id).filter(
                ExcludedIdsForTraining.set_number == set_number
            ).all()
            return [e.image_id for e in excluded]
        else:
            return []

    def get_all_image_ids(self) -> List[int]:
        """Get all valid image IDs in the database."""
        if self.use_database:
            images = self.session.query(SingleImage.id).filter(
                SingleImage.work_images_path.isnot(None)
            ).all()
            return [img.id for img in images]
        else:
            return []

    def get_actual_determination(self, image_id: int) -> Optional[str]:
        """Get manual determination (actual taxon) for an image."""
        if self.use_database:
            img = self.session.query(SingleImage).filter(
                SingleImage.id == image_id
            ).first()

            if img and img.manual_determination:
                return str(img.manual_determination)
            return None
        else:
            return None

    def mark_as_debris(self, image_id: int):
        """Mark an image as debris."""
        if self.use_database:
            img = self.session.query(SingleImage).filter(
                SingleImage.id == image_id
            ).first()

            if img:
                img.is_debris = True
                self.session.flush()

    def save_inference_results(self, results: List[Dict]):
        """
        Save inference results to database.

        Args:
            results: List of inference result dictionaries with keys:
                - image_id
                - model_taxon_id
                - predicted_taxon_id
                - correct_taxon_id
                - confidence
                - is_excluded
                - set_number
        """
        if self.use_database:
            for result in results:
                # Find model ID
                model = self.session.query(Model).filter(
                    Model.taxon_id == result['model_taxon_id'],
                    Model.set_number == result['set_number']
                ).first()

                model_id = model.id if model else None

                # Create inference result entry
                inference_result = InferenceResult(
                    image_id=result['image_id'],
                    model_id=model_id,
                    predicted_taxon=result.get('predicted_taxon_id'),
                    confidence=result.get('confidence'),
                    actual_taxon=result.get('correct_taxon_id')
                )

                self.session.add(inference_result)

            self.session.commit()

    # --- Cleanup ---

    def close(self):
        """Close database connection if applicable."""
        if self.use_database and hasattr(self, 'session'):
            self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Factory function
def get_data_manager(use_database: bool = None) -> DataManager:
    """
    Get data manager instance.

    Args:
        use_database: Override config setting

    Returns:
        DataManager instance

    Example:
        >>> dm = get_data_manager()
        >>> dm.save_sample_metadata([...])
    """
    return DataManager(use_database=use_database)


# Convenience function for database-only operations
def get_database_session() -> Optional[Session]:
    """
    Get SQLAlchemy session (database mode only).

    Returns:
        Session if database mode enabled, None otherwise
    """
    if config.get('database.use_database', False):
        database_url = config.get('database.url')
        return get_db_session(database_url)
    return None
