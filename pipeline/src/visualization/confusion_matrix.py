"""
Confusion matrix visualization for classification results.

This module creates confusion matrices for hierarchical classification,
showing prediction accuracy at each taxonomic level.

Corresponds to Methods section "Performance Evaluation".
"""

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ConfusionMatrixVisualizer:
    """
    Create confusion matrices for classification results.

    Generates publication-quality confusion matrix plots for each
    taxonomic level, showing which taxa are commonly confused.

    Example:
        >>> visualizer = ConfusionMatrixVisualizer()
        >>> visualizer.plot_confusion_matrix(
        ...     results_df,
        ...     model_taxon='Insecta',
        ...     output_path=Path('./confusion_insecta.png')
        ... )
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        """
        Initialize confusion matrix visualizer.

        Args:
            figsize: Default figure size (width, height)
        """
        self.figsize = figsize

    def create_confusion_matrix(
        self,
        results_df: pd.DataFrame,
        model_taxon: Optional[str] = None,
        normalize: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Create confusion matrix from results.

        Args:
            results_df: Classification results
            model_taxon: Specific taxon (or None for all)
            normalize: Normalize by row (show percentages)

        Returns:
            Tuple of (confusion_matrix, class_labels)
        """
        # Filter to specific taxon if requested
        if model_taxon:
            df = results_df[results_df['model_taxon_id'] == model_taxon].copy()
        else:
            df = results_df.copy()

        if len(df) == 0:
            logger.warning(f"No results for taxon {model_taxon}")
            return np.array([]), []

        # Get unique classes
        predicted = df['predicted_taxon_id']
        actual = df['correct_taxon_id']

        # Get all unique classes (union of predicted and actual)
        all_classes = sorted(set(predicted) | set(actual))

        # Create confusion matrix
        n_classes = len(all_classes)
        conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

        # Map class names to indices
        class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}

        # Fill confusion matrix
        for pred, act in zip(predicted, actual):
            pred_idx = class_to_idx[pred]
            act_idx = class_to_idx[act]
            conf_matrix[act_idx, pred_idx] += 1

        # Normalize if requested
        if normalize:
            # Normalize by row (actual class)
            row_sums = conf_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            conf_matrix = conf_matrix.astype(float) / row_sums

        return conf_matrix, all_classes

    def plot_confusion_matrix(
        self,
        results_df: pd.DataFrame,
        model_taxon: Optional[str] = None,
        output_path: Optional[Path] = None,
        normalize: bool = True,
        show_values: bool = True,
        cmap: str = 'Blues',
        title: Optional[str] = None
    ) -> Figure:
        """
        Plot confusion matrix.

        Args:
            results_df: Classification results
            model_taxon: Specific taxon to plot
            output_path: Path to save figure (or None to return)
            normalize: Show percentages instead of counts
            show_values: Show numbers in cells
            cmap: Matplotlib colormap
            title: Custom title (or None for default)

        Returns:
            Matplotlib Figure object

        Example:
            >>> fig = visualizer.plot_confusion_matrix(
            ...     results_df,
            ...     model_taxon='Insecta',
            ...     output_path=Path('./confusion.png')
            ... )
        """
        logger.info(f"Creating confusion matrix for {model_taxon or 'all taxa'}")

        # Create confusion matrix
        conf_matrix, class_labels = self.create_confusion_matrix(
            results_df,
            model_taxon=model_taxon,
            normalize=normalize
        )

        if len(class_labels) == 0:
            logger.warning("No data to plot")
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot matrix
        im = ax.imshow(conf_matrix, cmap=cmap, aspect='auto')

        # Set ticks and labels
        ax.set_xticks(np.arange(len(class_labels)))
        ax.set_yticks(np.arange(len(class_labels)))
        ax.set_xticklabels(class_labels, rotation=45, ha='right')
        ax.set_yticklabels(class_labels)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        if normalize:
            cbar.set_label('Proportion', rotation=270, labelpad=20)
        else:
            cbar.set_label('Count', rotation=270, labelpad=20)

        # Add values in cells
        if show_values:
            for i in range(len(class_labels)):
                for j in range(len(class_labels)):
                    value = conf_matrix[i, j]

                    if normalize:
                        text = f'{value:.2f}'
                    else:
                        text = f'{int(value)}'

                    # Choose text color based on background
                    color = 'white' if value > conf_matrix.max() / 2 else 'black'

                    ax.text(
                        j, i, text,
                        ha='center', va='center',
                        color=color,
                        fontsize=8
                    )

        # Labels
        ax.set_xlabel('Predicted Taxon', fontsize=12)
        ax.set_ylabel('Actual Taxon', fontsize=12)

        # Title
        if title is None:
            title = f'Confusion Matrix: {model_taxon or "All Taxa"}'
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Tight layout
        plt.tight_layout()

        # Save if requested
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {output_path}")

        return fig

    def plot_all_confusion_matrices(
        self,
        results_df: pd.DataFrame,
        output_dir: Path,
        normalize: bool = True
    ):
        """
        Plot confusion matrix for each model/taxon.

        Args:
            results_df: Classification results
            output_dir: Directory to save plots
            normalize: Show percentages

        Example:
            >>> visualizer.plot_all_confusion_matrices(
            ...     results_df,
            ...     Path('./confusion_matrices')
            ... )
        """
        logger.info("Creating confusion matrices for all taxa")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Get unique taxa
        taxa = results_df['model_taxon_id'].unique()

        for taxon in taxa:
            logger.info(f"Plotting confusion matrix for {taxon}")

            output_path = output_dir / f'confusion_{taxon}.png'

            self.plot_confusion_matrix(
                results_df,
                model_taxon=taxon,
                output_path=output_path,
                normalize=normalize
            )

            plt.close('all')  # Free memory

        logger.info(f"Saved {len(taxa)} confusion matrices to {output_dir}")

    def calculate_confusion_metrics(
        self,
        results_df: pd.DataFrame,
        model_taxon: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate per-class metrics from confusion matrix.

        Calculates precision, recall, F1 for each class.

        Args:
            results_df: Classification results
            model_taxon: Specific taxon (or None for all)

        Returns:
            DataFrame with per-class metrics

        Example:
            >>> metrics = visualizer.calculate_confusion_metrics(
            ...     results_df,
            ...     model_taxon='Insecta'
            ... )
            >>> print(metrics.to_string())
        """
        # Create confusion matrix (not normalized)
        conf_matrix, class_labels = self.create_confusion_matrix(
            results_df,
            model_taxon=model_taxon,
            normalize=False
        )

        if len(class_labels) == 0:
            return pd.DataFrame()

        metrics_data = []

        for idx, class_name in enumerate(class_labels):
            # True positives: diagonal element
            tp = conf_matrix[idx, idx]

            # False positives: column sum - tp
            fp = conf_matrix[:, idx].sum() - tp

            # False negatives: row sum - tp
            fn = conf_matrix[idx, :].sum() - tp

            # True negatives: everything else
            tn = conf_matrix.sum() - tp - fp - fn

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0
            )

            # Support (number of actual instances)
            support = conf_matrix[idx, :].sum()

            metrics_data.append({
                'class': class_name,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': int(support),
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn)
            })

        return pd.DataFrame(metrics_data)
