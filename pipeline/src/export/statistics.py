"""
Statistical analysis and reporting for classification results.

This module calculates comprehensive statistics for publication,
including accuracy, precision, recall, F1 scores, and more.

Corresponds to Methods section "Statistical Analysis".
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import pandas as pd
import numpy as np

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class StatisticsCalculator:
    """
    Calculate comprehensive statistics for classification results.

    Provides methods for calculating:
    - Overall accuracy, precision, recall, F1
    - Per-taxon metrics
    - Confidence distributions
    - Error analysis
    - Publication-ready tables

    Example:
        >>> calculator = StatisticsCalculator()
        >>> stats = calculator.calculate_comprehensive_statistics(results_df)
        >>> calculator.export_for_publication(stats, Path('./stats.json'))
    """

    def __init__(self):
        """Initialize statistics calculator."""
        pass

    def calculate_binary_metrics(
        self,
        y_true: List,
        y_pred: List
    ) -> Dict[str, float]:
        """
        Calculate binary classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary with precision, recall, F1, accuracy
        """
        # Convert to arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # True positives, false positives, false negatives
        tp = np.sum((y_true == y_pred) & (y_pred == 1))
        fp = np.sum((y_true != y_pred) & (y_pred == 1))
        fn = np.sum((y_true != y_pred) & (y_pred == 0))
        tn = np.sum((y_true == y_pred) & (y_pred == 0))

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }

    def calculate_multiclass_metrics(
        self,
        results_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate multiclass classification metrics.

        Args:
            results_df: DataFrame with predicted and actual taxa

        Returns:
            Dictionary with overall metrics
        """
        # Overall accuracy
        correct = (
            results_df['predicted_taxon_id'] == results_df['correct_taxon_id']
        ).sum()
        total = len(results_df)
        accuracy = correct / total if total > 0 else 0.0

        # Average confidence
        avg_confidence = results_df['confidence'].mean()
        std_confidence = results_df['confidence'].std()

        # Confidence for correct vs incorrect
        correct_mask = (
            results_df['predicted_taxon_id'] == results_df['correct_taxon_id']
        )
        correct_confidences = results_df[correct_mask]['confidence']
        incorrect_confidences = results_df[~correct_mask]['confidence']

        avg_correct_conf = correct_confidences.mean() if len(correct_confidences) > 0 else 0.0
        avg_incorrect_conf = incorrect_confidences.mean() if len(incorrect_confidences) > 0 else 0.0

        return {
            'total_predictions': total,
            'correct_predictions': int(correct),
            'incorrect_predictions': total - int(correct),
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'std_confidence': std_confidence,
            'avg_correct_confidence': avg_correct_conf,
            'avg_incorrect_confidence': avg_incorrect_conf
        }

    def calculate_per_taxon_metrics(
        self,
        results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate metrics for each model/taxon.

        Args:
            results_df: Classification results

        Returns:
            DataFrame with per-taxon metrics
        """
        taxon_metrics = []

        for model_taxon in results_df['model_taxon_id'].unique():
            taxon_df = results_df[results_df['model_taxon_id'] == model_taxon]

            # Overall metrics
            total = len(taxon_df)
            correct = (
                taxon_df['predicted_taxon_id'] == taxon_df['correct_taxon_id']
            ).sum()
            accuracy = correct / total if total > 0 else 0.0

            # Confidence statistics
            avg_conf = taxon_df['confidence'].mean()
            std_conf = taxon_df['confidence'].std()

            # Test set metrics
            test_set = taxon_df[taxon_df.get('is_excluded', False) == True]
            test_accuracy = 0.0
            test_n = len(test_set)

            if test_n > 0:
                test_correct = (
                    test_set['predicted_taxon_id'] == test_set['correct_taxon_id']
                ).sum()
                test_accuracy = test_correct / test_n

            # Per-class breakdown
            n_classes = taxon_df['predicted_taxon_id'].nunique()

            taxon_metrics.append({
                'model_taxon': model_taxon,
                'n_predictions': total,
                'n_correct': int(correct),
                'accuracy': accuracy,
                'n_classes': n_classes,
                'avg_confidence': avg_conf,
                'std_confidence': std_conf,
                'test_set_size': test_n,
                'test_accuracy': test_accuracy
            })

        return pd.DataFrame(taxon_metrics)

    def calculate_confidence_distribution(
        self,
        results_df: pd.DataFrame,
        bins: int = 10
    ) -> Dict:
        """
        Calculate confidence score distribution.

        Args:
            results_df: Classification results
            bins: Number of bins for histogram

        Returns:
            Dictionary with bin edges and counts
        """
        confidences = results_df['confidence'].values

        counts, bin_edges = np.histogram(confidences, bins=bins, range=(0, 1))

        return {
            'bin_edges': bin_edges.tolist(),
            'counts': counts.tolist(),
            'mean': float(np.mean(confidences)),
            'median': float(np.median(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences))
        }

    def analyze_errors(
        self,
        results_df: pd.DataFrame,
        top_n: int = 10
    ) -> Dict:
        """
        Analyze classification errors.

        Args:
            results_df: Classification results
            top_n: Number of top error pairs to return

        Returns:
            Dictionary with error analysis
        """
        # Get incorrect predictions
        errors = results_df[
            results_df['predicted_taxon_id'] != results_df['correct_taxon_id']
        ].copy()

        if len(errors) == 0:
            return {
                'total_errors': 0,
                'error_rate': 0.0,
                'top_error_pairs': []
            }

        # Count error pairs (predicted, actual)
        error_pairs = errors.groupby(
            ['predicted_taxon_id', 'correct_taxon_id']
        ).size().reset_index(name='count')

        error_pairs = error_pairs.sort_values('count', ascending=False)

        # Get top N error pairs
        top_errors = error_pairs.head(top_n).to_dict('records')

        # Calculate error rate
        error_rate = len(errors) / len(results_df)

        # Average confidence for errors
        avg_error_conf = errors['confidence'].mean()

        return {
            'total_errors': len(errors),
            'error_rate': error_rate,
            'avg_error_confidence': avg_error_conf,
            'top_error_pairs': top_errors,
            'n_unique_error_pairs': len(error_pairs)
        }

    def calculate_hierarchical_accuracy(
        self,
        results_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate accuracy at different hierarchical levels.

        For hierarchical classification, we want to know:
        - How often we get the exact taxon right
        - How often we get the parent taxon right
        - How often we get the grandparent right, etc.

        Args:
            results_df: Classification results

        Returns:
            Dictionary with hierarchical accuracy metrics
        """
        # This would require taxonomy hierarchy information
        # Simplified version: just exact match vs parent match

        exact_matches = (
            results_df['predicted_taxon_id'] == results_df['correct_taxon_id']
        ).sum()

        total = len(results_df)

        exact_accuracy = exact_matches / total if total > 0 else 0.0

        return {
            'exact_match_accuracy': exact_accuracy,
            'total_predictions': total,
            'exact_matches': int(exact_matches)
        }

    def calculate_comprehensive_statistics(
        self,
        results_df: pd.DataFrame
    ) -> Dict:
        """
        Calculate all statistics (main entry point).

        Args:
            results_df: Classification results

        Returns:
            Comprehensive statistics dictionary

        Example:
            >>> stats = calculator.calculate_comprehensive_statistics(df)
            >>> print(json.dumps(stats, indent=2))
        """
        logger.info("Calculating comprehensive statistics")

        stats = {}

        # Overall metrics
        stats['overall'] = self.calculate_multiclass_metrics(results_df)

        # Per-taxon metrics
        per_taxon_df = self.calculate_per_taxon_metrics(results_df)
        stats['per_taxon'] = per_taxon_df.to_dict('records')

        # Confidence distribution
        stats['confidence_distribution'] = self.calculate_confidence_distribution(
            results_df
        )

        # Error analysis
        stats['errors'] = self.analyze_errors(results_df, top_n=10)

        # Hierarchical accuracy
        stats['hierarchical'] = self.calculate_hierarchical_accuracy(results_df)

        # Test set specific
        test_set = results_df[results_df.get('is_excluded', False) == True]
        if len(test_set) > 0:
            stats['test_set'] = self.calculate_multiclass_metrics(test_set)
        else:
            stats['test_set'] = None

        logger.info("Statistics calculation complete")
        return stats

    def export_for_publication(
        self,
        stats: Dict,
        output_path: Path
    ):
        """
        Export statistics in publication-ready format.

        Args:
            stats: Statistics dictionary
            output_path: Path to save JSON

        Example:
            >>> calculator.export_for_publication(stats, Path('./pub_stats.json'))
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Format for publication
        pub_stats = {
            'summary': {
                'total_specimens': stats['overall']['total_predictions'],
                'overall_accuracy': f"{stats['overall']['accuracy']:.3f}",
                'test_set_accuracy': (
                    f"{stats['test_set']['accuracy']:.3f}"
                    if stats['test_set'] else "N/A"
                ),
                'average_confidence': f"{stats['overall']['avg_confidence']:.3f}",
            },
            'detailed_metrics': stats
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pub_stats, f, indent=2)

        logger.info(f"Exported publication statistics to {output_path}")

    def create_performance_table(
        self,
        results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create publication-ready performance table.

        Args:
            results_df: Classification results

        Returns:
            Formatted DataFrame for publication

        Example:
            >>> table = calculator.create_performance_table(df)
            >>> table.to_latex('performance.tex')
        """
        per_taxon = self.calculate_per_taxon_metrics(results_df)

        # Format for publication
        table = per_taxon[['model_taxon', 'n_predictions', 'accuracy', 'test_accuracy']].copy()

        table.columns = ['Taxonomic Level', 'N', 'Accuracy (All)', 'Accuracy (Test)']

        # Format percentages
        table['Accuracy (All)'] = table['Accuracy (All)'].apply(lambda x: f'{x:.1%}')
        table['Accuracy (Test)'] = table['Accuracy (Test)'].apply(lambda x: f'{x:.1%}')

        return table
