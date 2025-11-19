"""
CSV export functionality for classification results.

This module exports results and metadata to CSV format for easy
analysis in R, Python, or other tools.

Corresponds to Methods section "Data Export and Analysis".
"""

from pathlib import Path
from typing import List, Optional
import csv

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class CSVExporter:
    """
    Export classification results and metadata to CSV files.

    Creates simple, analysis-ready CSV files that can be used in
    R, Python pandas, or other statistical tools.

    Example:
        >>> exporter = CSVExporter()
        >>> exporter.export_results(
        ...     results_df,
        ...     Path('./results.csv')
        ... )
    """

    def __init__(self, data_manager=None):
        """
        Initialize CSV exporter.

        Args:
            data_manager: DataManager for database access
        """
        self.data_manager = data_manager

    def export_results(
        self,
        results_df: pd.DataFrame,
        output_path: Path,
        delimiter: str = ',',
        include_index: bool = False
    ):
        """
        Export classification results to CSV.

        Args:
            results_df: DataFrame with results
            output_path: Path to save CSV
            delimiter: CSV delimiter (default: comma)
            include_index: Include DataFrame index

        Example:
            >>> exporter.export_results(df, Path('./results.csv'))
        """
        logger.info(f"Exporting {len(results_df)} results to {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_df.to_csv(
            output_path,
            sep=delimiter,
            index=include_index,
            encoding='utf-8'
        )

        logger.info(f"Exported to {output_path}")

    def export_statistics(
        self,
        results_df: pd.DataFrame,
        output_path: Path
    ):
        """
        Export summary statistics to CSV.

        Args:
            results_df: Classification results
            output_path: Path to save statistics CSV

        Example:
            >>> exporter.export_statistics(df, Path('./stats.csv'))
        """
        logger.info(f"Exporting statistics to {output_path}")

        # Overall statistics
        total = len(results_df)
        correct = (
            results_df['predicted_taxon_id'] == results_df['correct_taxon_id']
        ).sum()
        accuracy = correct / total if total > 0 else 0

        # Per-taxon statistics
        taxon_stats = []

        for model_taxon in results_df['model_taxon_id'].unique():
            taxon_df = results_df[results_df['model_taxon_id'] == model_taxon]

            taxon_total = len(taxon_df)
            taxon_correct = (
                taxon_df['predicted_taxon_id'] == taxon_df['correct_taxon_id']
            ).sum()
            taxon_accuracy = taxon_correct / taxon_total if taxon_total > 0 else 0

            taxon_stats.append({
                'model_taxon': model_taxon,
                'n_predictions': taxon_total,
                'n_correct': taxon_correct,
                'accuracy': taxon_accuracy,
                'avg_confidence': taxon_df['confidence'].mean(),
                'median_confidence': taxon_df['confidence'].median()
            })

        stats_df = pd.DataFrame(taxon_stats)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(output_path, index=False, encoding='utf-8')

        logger.info(f"Exported statistics to {output_path}")

    def export_confusion_matrix_data(
        self,
        results_df: pd.DataFrame,
        output_path: Path,
        model_taxon: Optional[str] = None
    ):
        """
        Export confusion matrix data to CSV.

        Args:
            results_df: Classification results
            output_path: Path to save CSV
            model_taxon: Specific taxon (or None for all)

        Example:
            >>> exporter.export_confusion_matrix_data(
            ...     df,
            ...     Path('./confusion.csv'),
            ...     model_taxon='Insecta'
            ... )
        """
        logger.info(f"Exporting confusion matrix data to {output_path}")

        # Filter to specific taxon if requested
        if model_taxon:
            results_df = results_df[results_df['model_taxon_id'] == model_taxon]

        # Create confusion matrix
        confusion_data = []

        for _, row in results_df.iterrows():
            confusion_data.append({
                'model_taxon': row['model_taxon_id'],
                'predicted': row['predicted_taxon_id'],
                'actual': row['correct_taxon_id'],
                'confidence': row['confidence'],
                'is_test_set': row.get('is_excluded', False)
            })

        confusion_df = pd.DataFrame(confusion_data)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        confusion_df.to_csv(output_path, index=False, encoding='utf-8')

        logger.info(f"Exported confusion matrix data to {output_path}")

    def export_hierarchical_path(
        self,
        results_df: pd.DataFrame,
        output_path: Path
    ):
        """
        Export complete hierarchical classification path for each specimen.

        Creates a CSV where each row shows the complete path through
        the hierarchy for one specimen.

        Args:
            results_df: Classification results
            output_path: Path to save CSV

        Example:
            >>> exporter.export_hierarchical_path(df, Path('./paths.csv'))
        """
        if self.data_manager is None:
            logger.error("Data manager required for hierarchical paths")
            return

        logger.info(f"Exporting hierarchical paths to {output_path}")

        # Group by image_id to get complete classification path
        paths_data = []

        for image_id in results_df['image_id'].unique():
            image_results = results_df[results_df['image_id'] == image_id]

            # Sort by hierarchy depth (assuming higher-level taxa are processed first)
            image_results = image_results.sort_values('model_taxon_id')

            # Build path
            path = []
            confidences = []

            for _, row in image_results.iterrows():
                path.append(row['predicted_taxon_id'])
                confidences.append(row['confidence'])

            # Get final (most specific) determination
            final_taxon = path[-1] if path else None
            final_confidence = confidences[-1] if confidences else None

            # Get actual determination
            actual = image_results.iloc[0]['correct_taxon_id'] if len(image_results) > 0 else None

            paths_data.append({
                'image_id': image_id,
                'classification_path': ' > '.join(path),
                'final_taxon': final_taxon,
                'final_confidence': final_confidence,
                'actual_taxon': actual,
                'is_correct': final_taxon == actual,
                'path_length': len(path)
            })

        paths_df = pd.DataFrame(paths_data)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        paths_df.to_csv(output_path, index=False, encoding='utf-8')

        logger.info(f"Exported {len(paths_df)} hierarchical paths to {output_path}")

    def export_thresholds(
        self,
        thresholds: dict,
        output_path: Path
    ):
        """
        Export optimized thresholds to CSV.

        Args:
            thresholds: Threshold dictionary from ThresholdOptimizer
            output_path: Path to save CSV

        Example:
            >>> exporter.export_thresholds(thresholds, Path('./thresholds.csv'))
        """
        logger.info(f"Exporting thresholds to {output_path}")

        threshold_data = []

        for taxon_id, info in thresholds.items():
            threshold_data.append({
                'taxon_id': taxon_id,
                'threshold': info['threshold'],
                'f1': info['metrics']['f1'],
                'accuracy': info['metrics']['accuracy'],
                'precision': info['metrics']['precision'],
                'recall': info['metrics']['recall'],
                'n_samples': info['n_samples']
            })

        thresholds_df = pd.DataFrame(threshold_data)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        thresholds_df.to_csv(output_path, index=False, encoding='utf-8')

        logger.info(f"Exported thresholds to {output_path}")

    def export_batch(
        self,
        results_df: pd.DataFrame,
        output_dir: Path,
        prefix: str = 'export'
    ):
        """
        Export multiple CSV files at once.

        Creates:
        - {prefix}_results.csv: Full results
        - {prefix}_statistics.csv: Summary statistics
        - {prefix}_confusion.csv: Confusion matrix data
        - {prefix}_paths.csv: Hierarchical paths

        Args:
            results_df: Classification results
            output_dir: Output directory
            prefix: Filename prefix

        Example:
            >>> exporter.export_batch(df, Path('./exports'), prefix='set1')
        """
        logger.info(f"Batch exporting to {output_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Results
        self.export_results(
            results_df,
            output_dir / f'{prefix}_results.csv'
        )

        # Statistics
        self.export_statistics(
            results_df,
            output_dir / f'{prefix}_statistics.csv'
        )

        # Confusion matrix data
        self.export_confusion_matrix_data(
            results_df,
            output_dir / f'{prefix}_confusion.csv'
        )

        # Hierarchical paths
        if self.data_manager:
            self.export_hierarchical_path(
                results_df,
                output_dir / f'{prefix}_paths.csv'
            )

        logger.info(f"Batch export complete: {output_dir}")
