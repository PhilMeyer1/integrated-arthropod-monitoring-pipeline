"""
Confidence threshold calibration for hierarchical classification.

This module implements the threshold calibration method described in the Methods section:
"Threshold Calculation". For each taxonomic class, thresholds are calibrated on the
test set to achieve ≥95% accuracy, starting at 0.95 and incrementing by 0.0001 until
the accuracy criterion is met or the threshold reaches 1.0.

Corresponds to Methods section "Threshold Calculation".
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ThresholdCalibrator:
    """
    Calibrate confidence thresholds for hierarchical classification.

    This class implements the threshold calibration method described in the Methods:
    - Start at confidence threshold of 0.95
    - If baseline accuracy ≥95%, use 0.95
    - Otherwise, increment by 0.0001 until accuracy ≥95% or threshold = 1.0
    - Classes that cannot achieve 95% accuracy are excluded

    Example:
        >>> calibrator = ThresholdCalibrator(target_accuracy=95.0)
        >>> calibrator.load_results('inference_results.json')
        >>> thresholds = calibrator.calibrate_thresholds()
        >>> calibrator.save_thresholds(thresholds, 'thresholds.json')
    """

    def __init__(
        self,
        target_accuracy: float = 95.0,
        start_threshold: float = 0.95,
        increment: float = 0.0001,
        data_manager=None
    ):
        """
        Initialize threshold calibrator.

        Args:
            target_accuracy: Target accuracy percentage (default: 95.0)
            start_threshold: Starting confidence threshold (default: 0.95)
            increment: Threshold increment step (default: 0.0001)
            data_manager: DataManager for database access (optional)
        """
        self.target_accuracy = target_accuracy
        self.start_threshold = start_threshold
        self.increment = increment
        self.data_manager = data_manager
        self.results_df = None

    def load_results_from_file(self, results_file: Path) -> pd.DataFrame:
        """
        Load inference results from JSON file.

        Args:
            results_file: Path to inference results JSON

        Returns:
            DataFrame with results

        Example:
            >>> df = calibrator.load_results_from_file('results.json')
            >>> print(f"Loaded {len(df)} results")
        """
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        df = pd.DataFrame(results)
        logger.info(f"Loaded {len(df)} results from {results_file}")

        return df

    def load_results_from_database(self, set_number: int) -> pd.DataFrame:
        """
        Load inference results from database.

        Args:
            set_number: Model set number

        Returns:
            DataFrame with results
        """
        if self.data_manager is None:
            logger.error("No data manager for database access")
            return pd.DataFrame()

        results = self.data_manager.get_inference_results(set_number)
        df = pd.DataFrame(results)

        logger.info(f"Loaded {len(df)} results from database (set {set_number})")
        return df

    def load_results(
        self,
        results_source: Optional[Path] = None,
        set_number: Optional[int] = None,
        only_test_set: bool = True
    ):
        """
        Load inference results from file or database.

        Args:
            results_source: Path to JSON file (if using files)
            set_number: Model set number (if using database)
            only_test_set: Only use test set (excluded) images (default: True)

        Example:
            >>> calibrator.load_results(results_source=Path('results.json'))
            >>> # or
            >>> calibrator.load_results(set_number=1)
        """
        if results_source:
            self.results_df = self.load_results_from_file(results_source)
        elif set_number is not None:
            self.results_df = self.load_results_from_database(set_number)
        else:
            logger.error("Must provide either results_source or set_number")
            return

        # Filter to test set only if requested
        if only_test_set and 'is_excluded' in self.results_df.columns:
            original_len = len(self.results_df)
            self.results_df = self.results_df[self.results_df['is_excluded'] == True].copy()
            logger.info(f"Filtered to test set: {len(self.results_df)}/{original_len} images")

        # Add correctness column
        if 'predicted_taxon_id' in self.results_df.columns and 'correct_taxon_id' in self.results_df.columns:
            self.results_df['is_correct'] = (
                self.results_df['predicted_taxon_id'] == self.results_df['correct_taxon_id']
            )

    def calculate_baseline_accuracy(self, df: pd.DataFrame) -> float:
        """
        Calculate baseline accuracy without threshold filtering.

        Args:
            df: DataFrame with inference results

        Returns:
            Baseline accuracy as percentage (0-100)
        """
        if len(df) == 0:
            return 0.0

        if 'is_correct' not in df.columns:
            return 0.0

        correct = df['is_correct'].sum()
        total = len(df)

        return (correct / total) * 100.0

    def calculate_accuracy_at_threshold(
        self,
        df: pd.DataFrame,
        threshold: float
    ) -> Tuple[float, int, int]:
        """
        Calculate accuracy at a specific confidence threshold.

        Args:
            df: DataFrame with inference results
            threshold: Confidence threshold to evaluate

        Returns:
            Tuple of (accuracy, correct_count, incorrect_count)
            - accuracy: Percentage (0-100)
            - correct_count: Number of correct predictions at/above threshold
            - incorrect_count: Number of incorrect predictions at/above threshold
        """
        # Filter by threshold
        filtered = df[df['confidence'] >= threshold].copy()

        if len(filtered) == 0:
            return 0.0, 0, 0

        correct = filtered['is_correct'].sum()
        incorrect = len(filtered) - correct
        accuracy = (correct / len(filtered)) * 100.0

        return accuracy, correct, incorrect

    def calibrate_threshold_for_taxon(
        self,
        taxon_id: str,
        taxon_df: pd.DataFrame
    ) -> Dict:
        """
        Calibrate threshold for a single taxon/class.

        Implements the algorithm from Methods section:
        1. Calculate baseline accuracy (no threshold)
        2. If baseline ≥ target, use start_threshold (0.95)
        3. Otherwise, increment from start_threshold by increment (0.0001)
           until accuracy ≥ target or threshold = 1.0

        Args:
            taxon_id: Taxon/class identifier
            taxon_df: DataFrame with results for this taxon

        Returns:
            Dictionary with threshold calibration results:
            - taxon_id: Taxon identifier
            - threshold: Calibrated threshold
            - baseline_accuracy: Accuracy without threshold
            - accuracy_with_threshold: Accuracy at calibrated threshold
            - correct_with_threshold: Correct predictions at threshold
            - incorrect_with_threshold: Incorrect predictions at threshold
            - undetermined: Predictions below threshold
            - test_count: Total test images
            - passes_criterion: Whether ≥95% accuracy achieved
        """
        test_count = len(taxon_df)

        # Calculate baseline accuracy
        baseline_accuracy = self.calculate_baseline_accuracy(taxon_df)

        # If baseline already meets criterion, use start threshold
        if baseline_accuracy >= self.target_accuracy:
            threshold = self.start_threshold
            accuracy, correct, incorrect = self.calculate_accuracy_at_threshold(
                taxon_df,
                threshold
            )

            logger.debug(
                f"{taxon_id}: Baseline {baseline_accuracy:.1f}% ≥ {self.target_accuracy}% "
                f"→ threshold = {threshold:.4f}"
            )
        else:
            # Increment threshold until criterion met or threshold = 1.0
            threshold = self.start_threshold
            accuracy = baseline_accuracy
            correct = 0
            incorrect = 0

            while threshold <= 1.0:
                accuracy, correct, incorrect = self.calculate_accuracy_at_threshold(
                    taxon_df,
                    threshold
                )

                if accuracy >= self.target_accuracy:
                    break

                threshold += self.increment

            # Ensure threshold doesn't exceed 1.0
            threshold = min(threshold, 1.0)

            # Recalculate at final threshold
            accuracy, correct, incorrect = self.calculate_accuracy_at_threshold(
                taxon_df,
                threshold
            )

            logger.debug(
                f"{taxon_id}: Baseline {baseline_accuracy:.1f}% < {self.target_accuracy}% "
                f"→ threshold = {threshold:.4f} (accuracy = {accuracy:.1f}%)"
            )

        # Calculate undetermined (below threshold)
        undetermined = test_count - correct - incorrect

        # Check if criterion met
        passes_criterion = accuracy >= self.target_accuracy

        result = {
            'taxon_id': taxon_id,
            'threshold': threshold,
            'baseline_accuracy': baseline_accuracy,
            'accuracy_with_threshold': accuracy,
            'correct_with_threshold': int(correct),
            'incorrect_with_threshold': int(incorrect),
            'undetermined': int(undetermined),
            'test_count': int(test_count),
            'passes_criterion': passes_criterion
        }

        return result

    def calibrate_thresholds(
        self,
        per_taxon: bool = True,
        group_by_column: str = 'predicted_taxon_id'
    ) -> Dict[str, Dict]:
        """
        Calibrate thresholds for all taxa/classes.

        Args:
            per_taxon: Calibrate separately per taxon (default: True)
            group_by_column: Column to group by ('predicted_taxon_id' or 'model_taxon_id')

        Returns:
            Dictionary mapping taxon_id to calibration results

        Example:
            >>> thresholds = calibrator.calibrate_thresholds()
            >>> for taxon, info in thresholds.items():
            ...     print(f"{taxon}: {info['threshold']:.4f} ({info['accuracy_with_threshold']:.1f}%)")
        """
        if self.results_df is None:
            logger.error("No results loaded")
            return {}

        logger.info("=" * 70)
        logger.info("CALIBRATING CONFIDENCE THRESHOLDS")
        logger.info("=" * 70)
        logger.info(f"Target accuracy: {self.target_accuracy}%")
        logger.info(f"Start threshold: {self.start_threshold}")
        logger.info(f"Increment: {self.increment}")
        logger.info(f"Test set size: {len(self.results_df)}")
        logger.info("")

        if not per_taxon:
            # Global threshold (all taxa together)
            logger.info("Calibrating global threshold...")
            result = self.calibrate_threshold_for_taxon('global', self.results_df)

            thresholds = {'global': result}
        else:
            # Per-taxon thresholds
            if group_by_column not in self.results_df.columns:
                logger.error(f"Column '{group_by_column}' not found in results")
                return {}

            taxa = self.results_df[group_by_column].unique()
            logger.info(f"Calibrating thresholds for {len(taxa)} taxa...")
            logger.info("")

            thresholds = {}

            for taxon_id in sorted(taxa):
                taxon_df = self.results_df[
                    self.results_df[group_by_column] == taxon_id
                ].copy()

                result = self.calibrate_threshold_for_taxon(taxon_id, taxon_df)
                thresholds[taxon_id] = result

                # Log result
                status = "✓" if result['passes_criterion'] else "✗"
                logger.info(
                    f"{status} {taxon_id}: "
                    f"threshold={result['threshold']:.4f}, "
                    f"accuracy={result['accuracy_with_threshold']:.1f}% "
                    f"(baseline={result['baseline_accuracy']:.1f}%), "
                    f"n={result['test_count']}"
                )

        logger.info("")
        logger.info("=" * 70)
        logger.info("CALIBRATION COMPLETE")
        logger.info("=" * 70)

        # Summary statistics
        passes = sum(1 for t in thresholds.values() if t['passes_criterion'])
        fails = len(thresholds) - passes

        logger.info(f"Total taxa: {len(thresholds)}")
        logger.info(f"Passes criterion (≥{self.target_accuracy}%): {passes}")
        logger.info(f"Fails criterion: {fails}")

        if fails > 0:
            logger.warning(f"{fails} taxa did not achieve {self.target_accuracy}% accuracy")
            logger.warning("These taxa will be excluded from downstream analysis")

        logger.info("=" * 70)
        logger.info("")

        return thresholds

    def get_valid_taxa(self, thresholds: Dict[str, Dict]) -> List[str]:
        """
        Get list of taxa that pass the accuracy criterion.

        Args:
            thresholds: Dictionary of threshold calibration results

        Returns:
            List of taxon IDs that achieve ≥95% accuracy
        """
        valid_taxa = [
            taxon_id
            for taxon_id, result in thresholds.items()
            if result['passes_criterion']
        ]

        return valid_taxa

    def get_invalid_taxa(self, thresholds: Dict[str, Dict]) -> List[str]:
        """
        Get list of taxa that fail the accuracy criterion.

        Args:
            thresholds: Dictionary of threshold calibration results

        Returns:
            List of taxon IDs that do not achieve ≥95% accuracy
        """
        invalid_taxa = [
            taxon_id
            for taxon_id, result in thresholds.items()
            if not result['passes_criterion']
        ]

        return invalid_taxa

    def save_thresholds(self, thresholds: Dict, output_path: Path):
        """
        Save calibrated thresholds to JSON file.

        Args:
            thresholds: Threshold dictionary
            output_path: Path to save JSON

        Example:
            >>> calibrator.save_thresholds(thresholds, Path('thresholds.json'))
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to native Python types for JSON serialization
        serializable_thresholds = {}
        for taxon_id, result in thresholds.items():
            serializable_thresholds[taxon_id] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in result.items()
            }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_thresholds, f, indent=2)

        logger.info(f"Saved thresholds to {output_path}")

    def load_thresholds(self, input_path: Path) -> Dict:
        """
        Load thresholds from JSON file.

        Args:
            input_path: Path to JSON file

        Returns:
            Dictionary with thresholds

        Example:
            >>> thresholds = calibrator.load_thresholds(Path('thresholds.json'))
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            thresholds = json.load(f)

        logger.info(f"Loaded thresholds from {input_path}")
        return thresholds

    def apply_thresholds(
        self,
        results_df: pd.DataFrame,
        thresholds: Dict,
        default_threshold: float = 0.95,
        taxon_column: str = 'predicted_taxon_id'
    ) -> pd.DataFrame:
        """
        Apply calibrated thresholds to filter results.

        Args:
            results_df: DataFrame with inference results
            thresholds: Threshold dictionary (per taxon or global)
            default_threshold: Default if taxon not in thresholds
            taxon_column: Column name for taxon ID

        Returns:
            Filtered DataFrame with only predictions meeting threshold

        Example:
            >>> filtered = calibrator.apply_thresholds(
            ...     results_df,
            ...     thresholds,
            ...     default_threshold=0.95
            ... )
            >>> print(f"Kept {len(filtered)}/{len(results_df)} predictions")
        """
        filtered_results = []

        for idx, row in results_df.iterrows():
            taxon = row[taxon_column]
            confidence = row['confidence']

            # Get threshold for this taxon
            if taxon in thresholds:
                threshold = thresholds[taxon]['threshold']
            elif 'global' in thresholds:
                threshold = thresholds['global']['threshold']
            else:
                threshold = default_threshold

            # Keep if above threshold
            if confidence >= threshold:
                filtered_results.append(row)

        filtered_df = pd.DataFrame(filtered_results)

        logger.info(
            f"Applied thresholds: {len(results_df)} → {len(filtered_df)} predictions "
            f"({len(filtered_df)/len(results_df)*100:.1f}% retained)"
        )

        return filtered_df

    def get_threshold_summary(self, thresholds: Dict) -> pd.DataFrame:
        """
        Get summary table of calibrated thresholds.

        Args:
            thresholds: Threshold dictionary

        Returns:
            DataFrame with summary

        Example:
            >>> summary = calibrator.get_threshold_summary(thresholds)
            >>> print(summary.to_string())
        """
        rows = []

        for taxon_id, info in thresholds.items():
            rows.append({
                'taxon_id': taxon_id,
                'threshold': info['threshold'],
                'baseline_accuracy': info['baseline_accuracy'],
                'accuracy_with_threshold': info['accuracy_with_threshold'],
                'correct': info['correct_with_threshold'],
                'incorrect': info['incorrect_with_threshold'],
                'undetermined': info['undetermined'],
                'test_count': info['test_count'],
                'passes_criterion': info['passes_criterion']
            })

        df = pd.DataFrame(rows)

        # Sort by taxon_id
        df = df.sort_values('taxon_id')

        return df


# Backward compatibility alias
ThresholdOptimizer = ThresholdCalibrator
