"""
Export module for classification results and statistics.

This module provides functionality to export classification results,
statistics, and metadata in various formats (Excel, CSV, JSON) for
analysis and publication.

Modules:
    excel_export: Excel export with formatting
    csv_export: Simple CSV export for analysis
    statistics: Statistical calculations and reporting

Example:
    >>> from src.export import ExcelExporter, CSVExporter, StatisticsCalculator
    >>>
    >>> # Export to Excel
    >>> excel_exporter = ExcelExporter()
    >>> excel_exporter.export_classification_results(
    ...     results_df,
    ...     Path('./results.xlsx')
    ... )
    >>>
    >>> # Export to CSV
    >>> csv_exporter = CSVExporter()
    >>> csv_exporter.export_batch(results_df, Path('./exports'))
    >>>
    >>> # Calculate statistics
    >>> calculator = StatisticsCalculator()
    >>> stats = calculator.calculate_comprehensive_statistics(results_df)
    >>> calculator.export_for_publication(stats, Path('./stats.json'))
"""

from src.export.excel_export import ExcelExporter
from src.export.csv_export import CSVExporter
from src.export.statistics import StatisticsCalculator

__all__ = [
    'ExcelExporter',
    'CSVExporter',
    'StatisticsCalculator',
]
