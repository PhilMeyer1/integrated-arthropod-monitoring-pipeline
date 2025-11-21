"""
File-based data management utilities.

Alternative to database for small-scale processing and better reproducibility.
Uses CSV/JSON files for metadata storage.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any
import json
from datetime import datetime


class FileBasedDataManager:
    """
    File-based data management system (alternative to database).

    Stores metadata in CSV files instead of SQL database.
    Suitable for small to medium datasets and ensures reproducibility.

    Directory structure:
        data_root/
        ├── metadata.csv              # Main metadata file
        ├── composite_images.csv      # Composite image info
        ├── single_images.csv         # Extracted specimen info
        ├── classifications.csv       # Classification results
        └── models/
            └── model_registry.json   # Model paths and metadata
    """

    def __init__(self, data_root: Path):
        """
        Initialize file-based data manager.

        Args:
            data_root: Root directory for data files
        """
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)

        # File paths
        self.metadata_file = self.data_root / "metadata.csv"
        self.composite_images_file = self.data_root / "composite_images.csv"
        self.single_images_file = self.data_root / "single_images.csv"
        self.classifications_file = self.data_root / "classifications.csv"
        self.model_registry_file = self.data_root / "models" / "model_registry.json"

        # Initialize files if they don't exist
        self._initialize_files()

    def _initialize_files(self):
        """Create empty data files if they don't exist."""
        # Metadata file
        if not self.metadata_file.exists():
            df = pd.DataFrame(columns=[
                'sample_id', 'location', 'date', 'size_fraction',
                'tray_size', 'weight_mg', 'notes'
            ])
            df.to_csv(self.metadata_file, index=False)

        # Composite images file
        if not self.composite_images_file.exists():
            df = pd.DataFrame(columns=[
                'composite_id', 'sample_id', 'file_path', 'width', 'height',
                'created_at', 'processed'
            ])
            df.to_csv(self.composite_images_file, index=False)

        # Single images file
        if not self.single_images_file.exists():
            df = pd.DataFrame(columns=[
                'image_id', 'composite_id', 'sample_id', 'file_path',
                'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
                'detection_confidence', 'created_at'
            ])
            df.to_csv(self.single_images_file, index=False)

        # Classifications file
        if not self.classifications_file.exists():
            df = pd.DataFrame(columns=[
                'image_id', 'predicted_taxon', 'confidence',
                'classification_history', 'timestamp'
            ])
            df.to_csv(self.classifications_file, index=False)

        # Model registry
        if not self.model_registry_file.exists():
            self.model_registry_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.model_registry_file, 'w') as f:
                json.dump({}, f, indent=2)

    # --- Sample Metadata ---

    def save_sample_metadata(self, metadata: List[Dict]):
        """
        Save sample metadata.

        Args:
            metadata: List of metadata dictionaries with keys:
                - sample_id
                - location
                - date
                - size_fraction
                - tray_size
                - weight_mg (optional)
                - notes (optional)
        """
        df = pd.DataFrame(metadata)
        df.to_csv(self.metadata_file, index=False)

    def load_sample_metadata(self) -> pd.DataFrame:
        """Load sample metadata."""
        if self.metadata_file.exists():
            return pd.read_csv(self.metadata_file)
        return pd.DataFrame()

    def get_sample_by_id(self, sample_id: str) -> Optional[Dict]:
        """Get sample metadata by ID."""
        df = self.load_sample_metadata()
        result = df[df['sample_id'] == sample_id]
        if not result.empty:
            return result.iloc[0].to_dict()
        return None

    # --- Composite Images ---

    def save_composite_image(self, composite_data: Dict):
        """
        Save composite image metadata.

        Args:
            composite_data: Dictionary with keys:
                - composite_id
                - sample_id
                - file_path
                - width, height
                - created_at
                - processed
        """
        df = self.load_composite_images()

        # Append new row
        df = pd.concat([df, pd.DataFrame([composite_data])], ignore_index=True)
        df.to_csv(self.composite_images_file, index=False)

    def load_composite_images(self) -> pd.DataFrame:
        """Load all composite image metadata."""
        if self.composite_images_file.exists():
            return pd.read_csv(self.composite_images_file)
        return pd.DataFrame()

    def get_composite_by_sample(self, sample_id: str) -> List[Dict]:
        """Get all composite images for a sample."""
        df = self.load_composite_images()
        results = df[df['sample_id'] == sample_id]
        return results.to_dict('records')

    # --- Single Images (Extracted Specimens) ---

    def save_single_images(self, images: List[Dict]):
        """
        Save extracted specimen image metadata.

        Args:
            images: List of image dictionaries with keys:
                - image_id
                - composite_id
                - sample_id
                - file_path
                - bbox_x1, bbox_y1, bbox_x2, bbox_y2
                - detection_confidence
                - created_at
        """
        df = self.load_single_images()

        new_df = pd.DataFrame(images)
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(self.single_images_file, index=False)

    def load_single_images(self) -> pd.DataFrame:
        """Load all single image metadata."""
        if self.single_images_file.exists():
            return pd.read_csv(self.single_images_file)
        return pd.DataFrame()

    def get_images_by_sample(self, sample_id: str) -> pd.DataFrame:
        """Get all extracted images for a sample."""
        df = self.load_single_images()
        return df[df['sample_id'] == sample_id]

    def get_image_paths(self, sample_id: Optional[str] = None) -> List[Path]:
        """
        Get file paths for images.

        Args:
            sample_id: Filter by sample ID. If None, returns all images.

        Returns:
            List of image file paths
        """
        df = self.load_single_images()

        if sample_id:
            df = df[df['sample_id'] == sample_id]

        return [Path(p) for p in df['file_path'].tolist()]

    # --- Classifications ---

    def save_classifications(self, results: List[Dict]):
        """
        Save classification results.

        Args:
            results: List of classification dictionaries with keys:
                - image_id
                - predicted_taxon
                - confidence
                - classification_history (JSON string)
                - timestamp
        """
        df = pd.DataFrame(results)
        df.to_csv(self.classifications_file, index=False)

    def load_classifications(self) -> pd.DataFrame:
        """Load classification results."""
        if self.classifications_file.exists():
            return pd.read_csv(self.classifications_file)
        return pd.DataFrame()

    def get_classification_by_image(self, image_id: int) -> Optional[Dict]:
        """Get classification for a specific image."""
        df = self.load_classifications()
        result = df[df['image_id'] == image_id]
        if not result.empty:
            return result.iloc[0].to_dict()
        return None

    # --- Model Registry ---

    def register_model(self, taxon: str, model_path: str, metadata: Optional[Dict] = None):
        """
        Register a trained model in the registry.

        Args:
            taxon: Taxon name this model classifies
            model_path: Path to model file
            metadata: Optional metadata (training date, accuracy, etc.)
        """
        with open(self.model_registry_file, 'r') as f:
            registry = json.load(f)

        registry[taxon] = {
            'model_path': str(model_path),
            'registered_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        with open(self.model_registry_file, 'w') as f:
            json.dump(registry, f, indent=2)

    def get_model_registry(self) -> Dict[str, Dict]:
        """Get full model registry."""
        if self.model_registry_file.exists():
            with open(self.model_registry_file, 'r') as f:
                return json.load(f)
        return {}

    def get_model_path(self, taxon: str) -> Optional[Path]:
        """Get model path for a specific taxon."""
        registry = self.get_model_registry()
        if taxon in registry:
            return Path(registry[taxon]['model_path'])
        return None

    # --- Export ---

    def export_to_excel(self, output_path: Path):
        """
        Export all data to Excel file with multiple sheets.

        Args:
            output_path: Path to output Excel file
        """
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            self.load_sample_metadata().to_excel(writer, sheet_name='Samples', index=False)
            self.load_composite_images().to_excel(writer, sheet_name='Composite Images', index=False)
            self.load_single_images().to_excel(writer, sheet_name='Specimens', index=False)
            self.load_classifications().to_excel(writer, sheet_name='Classifications', index=False)

        print(f"Data exported to {output_path}")

    def export_to_json(self, output_path: Path):
        """
        Export all data to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        data = {
            'samples': self.load_sample_metadata().to_dict('records'),
            'composite_images': self.load_composite_images().to_dict('records'),
            'single_images': self.load_single_images().to_dict('records'),
            'classifications': self.load_classifications().to_dict('records'),
            'model_registry': self.get_model_registry()
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Data exported to {output_path}")

    # --- Statistics ---

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'num_samples': len(self.load_sample_metadata()),
            'num_composite_images': len(self.load_composite_images()),
            'num_single_images': len(self.load_single_images()),
            'num_classified_images': len(self.load_classifications()),
            'num_registered_models': len(self.get_model_registry())
        }

    def print_summary(self):
        """Print data summary."""
        stats = self.get_statistics()
        print("=" * 50)
        print("File-Based Data Manager Summary")
        print("=" * 50)
        print(f"Samples:            {stats['num_samples']}")
        print(f"Composite Images:   {stats['num_composite_images']}")
        print(f"Single Images:      {stats['num_single_images']}")
        print(f"Classifications:    {stats['num_classified_images']}")
        print(f"Registered Models:  {stats['num_registered_models']}")
        print("=" * 50)
