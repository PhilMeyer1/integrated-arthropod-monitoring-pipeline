#!/usr/bin/env python3
"""
Test Threshold Calibration Logic

This script tests the threshold calibration implementation using synthetic data
to validate the algorithm and compare Repository vs GUI implementations.

DO NOT modify GUI code - only document differences.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project paths
repo_path = Path(__file__).parent.parent

sys.path.insert(0, str(repo_path))

# Import repository implementation
from src.classification.thresholds import ThresholdCalibrator

# Note: GUI implementation cannot be imported due to module dependencies
# We will manually replicate its logic for testing


def create_synthetic_test_data():
    """
    Create synthetic test data with known properties.

    Returns three scenarios:
    - Taxon A: High baseline (98%), all confidences > 0.95
    - Taxon B: Medium baseline (90%), mixed confidences
    - Taxon C: Low baseline (60%), cannot reach 95%
    """
    print("="*70)
    print("CREATING SYNTHETIC TEST DATA")
    print("="*70)

    np.random.seed(42)

    data = []

    # ========================================
    # TAXON A: High baseline accuracy (98%)
    # BUT 3 correct images have confidence < 0.95
    # This is the CRITICAL TEST CASE for GUI bug!
    # ========================================
    print("\n📊 Taxon A: High baseline accuracy WITH LOW CONFIDENCE IMAGES")
    print("  - 100 images")
    print("  - 98% baseline accuracy (98 correct, 2 incorrect)")
    print("  - 3 CORRECT images have confidence < 0.95")
    print("  - Expected:")
    print("    * Repository: threshold=0.95, accuracy=95/97=97.9%")
    print("    * GUI BUG: threshold=0.95, accuracy=98% (WRONG - uses baseline)")

    for i in range(100):
        is_correct = i < 98  # 98 correct, 2 incorrect

        # CRITICAL: 3 correct images (i=0,1,2) have confidence < 0.95
        if is_correct and i < 3:
            confidence = np.random.uniform(0.85, 0.94)
        elif is_correct:
            confidence = np.random.uniform(0.96, 0.99)
        else:
            confidence = np.random.uniform(0.96, 0.99)

        data.append({
            'single_image_id': f'A_{i}',
            'model_taxon_id': 'model_A',
            'predicted_taxon_id': 'TaxonA',
            'correct_taxon_id': 'TaxonA' if is_correct else 'TaxonOther',
            'confidence': confidence,
            'is_excluded': True,  # Test set
            'Confidence': confidence,  # GUI format
            'Taxon_Model_ID': 'model_A',  # GUI format
            'Predicted_Taxon_ID': 'TaxonA',  # GUI format
            'Actual_Taxon_ID': 'TaxonA' if is_correct else 'TaxonOther',  # GUI format
            'excluded_image_id': True,  # GUI format
            'Parent_Taxon_ID': 'parent_A'  # GUI format
        })

    # ========================================
    # TAXON B: Medium baseline accuracy (90%)
    # ========================================
    print("\n📊 Taxon B: Medium baseline accuracy")
    print("  - 50 images")
    print("  - 90% baseline accuracy")
    print("  - Mixed confidences: 5 correct have conf < 0.95")
    print("  - Expected: threshold > 0.95 (incremented)")

    for i in range(50):
        is_correct = i < 45  # 45 correct, 5 incorrect (90% baseline)

        # 5 correct images have confidence < 0.95
        if is_correct and i < 5:
            confidence = np.random.uniform(0.85, 0.94)
        elif is_correct:
            confidence = np.random.uniform(0.96, 0.99)
        else:
            confidence = np.random.uniform(0.70, 0.90)

        data.append({
            'single_image_id': f'B_{i}',
            'model_taxon_id': 'model_B',
            'predicted_taxon_id': 'TaxonB',
            'correct_taxon_id': 'TaxonB' if is_correct else 'TaxonOther',
            'confidence': confidence,
            'is_excluded': True,
            'Confidence': confidence,
            'Taxon_Model_ID': 'model_B',
            'Predicted_Taxon_ID': 'TaxonB',
            'Actual_Taxon_ID': 'TaxonB' if is_correct else 'TaxonOther',
            'excluded_image_id': True,
            'Parent_Taxon_ID': 'parent_B'
        })

    # ========================================
    # TAXON C: Low baseline accuracy (60%)
    # ========================================
    print("\n📊 Taxon C: Low baseline accuracy")
    print("  - 20 images")
    print("  - 60% baseline accuracy")
    print("  - Cannot reach 95% even at threshold=1.0")
    print("  - Expected: threshold=1.0, excluded from analysis")

    for i in range(20):
        is_correct = i < 12  # 12 correct, 8 incorrect (60% baseline)
        confidence = np.random.uniform(0.80, 0.99) if is_correct else np.random.uniform(0.70, 0.95)

        data.append({
            'single_image_id': f'C_{i}',
            'model_taxon_id': 'model_C',
            'predicted_taxon_id': 'TaxonC',
            'correct_taxon_id': 'TaxonC' if is_correct else 'TaxonOther',
            'confidence': confidence,
            'is_excluded': True,
            'Confidence': confidence,
            'Taxon_Model_ID': 'model_C',
            'Predicted_Taxon_ID': 'TaxonC',
            'Actual_Taxon_ID': 'TaxonC' if is_correct else 'TaxonOther',
            'excluded_image_id': True,
            'Parent_Taxon_ID': 'parent_C'
        })

    df = pd.DataFrame(data)

    print(f"\n✓ Created {len(df)} synthetic test records")
    print(f"  - TaxonA: {len(df[df['predicted_taxon_id']=='TaxonA'])} images")
    print(f"  - TaxonB: {len(df[df['predicted_taxon_id']=='TaxonB'])} images")
    print(f"  - TaxonC: {len(df[df['predicted_taxon_id']=='TaxonC'])} images")
    print("="*70)
    print()

    return df


def test_repository_implementation(df):
    """Test the repository ThresholdCalibrator implementation."""
    print("="*70)
    print("TESTING REPOSITORY IMPLEMENTATION")
    print("="*70)
    print("File: src/classification/thresholds.py")
    print("Class: ThresholdCalibrator")
    print()

    # Initialize calibrator
    calibrator = ThresholdCalibrator(
        target_accuracy=95.0,
        start_threshold=0.95,
        increment=0.0001
    )

    # Set data directly
    calibrator.results_df = df.copy()

    # Add is_correct column
    calibrator.results_df['is_correct'] = (
        calibrator.results_df['predicted_taxon_id'] ==
        calibrator.results_df['correct_taxon_id']
    )

    print("Running calibration...")
    thresholds = calibrator.calibrate_thresholds(
        per_taxon=True,
        group_by_column='predicted_taxon_id'
    )

    print("\n" + "="*70)
    print("REPOSITORY RESULTS")
    print("="*70)

    results_repo = {}
    for taxon_id, result in thresholds.items():
        print(f"\n{taxon_id}:")
        print(f"  Threshold: {result['threshold']:.4f}")
        print(f"  Baseline accuracy: {result['baseline_accuracy']:.2f}%")
        print(f"  Accuracy with threshold: {result['accuracy_with_threshold']:.2f}%")
        print(f"  Correct: {result['correct_with_threshold']}")
        print(f"  Incorrect: {result['incorrect_with_threshold']}")
        print(f"  Undetermined: {result['undetermined']}")
        print(f"  Test count: {result['test_count']}")
        print(f"  Passes criterion: {result['passes_criterion']}")

        results_repo[taxon_id] = result

    print("\n" + "="*70)
    print()

    return results_repo


def test_gui_implementation(df):
    """Test the GUI ThresholdAccuracyAnalyzer implementation."""
    print("="*70)
    print("TESTING GUI IMPLEMENTATION")
    print("="*70)
    print("File: gui/.../threshold_analyzer.py")
    print("Class: ThresholdAccuracyAnalyzer")
    print()

    # Note: GUI implementation expects database session, so we'll test the logic manually
    print("⚠️  GUI implementation requires database session")
    print("    Testing threshold calculation logic manually...")
    print()

    # Manually replicate GUI logic
    results_gui = {}

    for taxon in ['TaxonA', 'TaxonB', 'TaxonC']:
        taxon_df = df[df['Predicted_Taxon_ID'] == taxon].copy()

        # Add correctness columns (GUI format)
        taxon_df['Correct_Classification'] = (
            taxon_df['Predicted_Taxon_ID'] == taxon_df['Actual_Taxon_ID']
        ).astype(int)
        taxon_df['Incorrect_Classification'] = (
            taxon_df['Predicted_Taxon_ID'] != taxon_df['Actual_Taxon_ID']
        ).astype(int)

        # Calculate baseline accuracy (GUI logic)
        test_count = len(taxon_df)
        correct_total = taxon_df['Correct_Classification'].sum()
        baseline_accuracy = (correct_total / test_count) * 100.0

        print(f"{taxon}:")
        print(f"  Test count: {test_count}")
        print(f"  Baseline accuracy: {baseline_accuracy:.2f}%")

        # GUI logic: Lines 111-117
        accuracy_threshold = 95.0  # Default in GUI

        if baseline_accuracy >= accuracy_threshold:
            # GUI uses baseline accuracy directly (LINE 113)
            threshold = 0.95
            accuracy_with_threshold = baseline_accuracy  # ❌ ISSUE: Uses baseline, not accuracy at threshold!
            correct_with_threshold = correct_total
            incorrect_with_threshold = test_count - correct_total
            print(f"  → Baseline ≥ {accuracy_threshold}%, using threshold = {threshold}")
            print(f"  ⚠️  GUI BUG: accuracy_with_threshold = baseline = {accuracy_with_threshold:.2f}%")
        else:
            # GUI increments threshold (Lines 126-137)
            threshold = 0.95
            accuracy_threshold_val = accuracy_threshold

            while threshold <= 1.0:
                thr_data = taxon_df[taxon_df['Confidence'] >= threshold]
                correct_thr = thr_data['Correct_Classification'].sum()
                incorrect_thr = thr_data['Incorrect_Classification'].sum()
                total_thr = correct_thr + incorrect_thr

                if total_thr > 0:
                    accuracy_with_threshold = (correct_thr / total_thr) * 100.0
                else:
                    accuracy_with_threshold = 0.0

                if accuracy_with_threshold >= accuracy_threshold_val:
                    break

                threshold += 0.0001

            threshold = min(threshold, 1.0)

            # Recalculate at final threshold
            thr_data = taxon_df[taxon_df['Confidence'] >= threshold]
            correct_with_threshold = thr_data['Correct_Classification'].sum()
            incorrect_with_threshold = thr_data['Incorrect_Classification'].sum()
            total_with_threshold = correct_with_threshold + incorrect_with_threshold

            if total_with_threshold > 0:
                accuracy_with_threshold = (correct_with_threshold / total_with_threshold) * 100.0
            else:
                accuracy_with_threshold = 0.0

            print(f"  → Baseline < {accuracy_threshold}%, incremented to threshold = {threshold:.4f}")
            print(f"  ✓ Correct: accuracy_with_threshold calculated at threshold = {accuracy_with_threshold:.2f}%")

        undetermined = test_count - correct_with_threshold - incorrect_with_threshold
        passes_criterion = accuracy_with_threshold >= accuracy_threshold

        print(f"  Threshold: {threshold:.4f}")
        print(f"  Accuracy with threshold: {accuracy_with_threshold:.2f}%")
        print(f"  Correct: {correct_with_threshold}")
        print(f"  Incorrect: {incorrect_with_threshold}")
        print(f"  Undetermined: {undetermined}")
        print(f"  Passes criterion: {passes_criterion}")
        print()

        results_gui[taxon] = {
            'taxon_id': taxon,
            'threshold': threshold,
            'baseline_accuracy': baseline_accuracy,
            'accuracy_with_threshold': accuracy_with_threshold,
            'correct_with_threshold': int(correct_with_threshold),
            'incorrect_with_threshold': int(incorrect_with_threshold),
            'undetermined': int(undetermined),
            'test_count': int(test_count),
            'passes_criterion': passes_criterion
        }

    print("="*70)
    print()

    return results_gui


def compare_implementations(results_repo, results_gui):
    """Compare results from both implementations."""
    print("="*70)
    print("COMPARISON: REPOSITORY vs GUI")
    print("="*70)
    print()

    comparison = []

    for taxon in ['TaxonA', 'TaxonB', 'TaxonC']:
        repo = results_repo[taxon]
        gui = results_gui[taxon]

        print(f"\n{taxon}:")
        print("-" * 50)

        # Threshold
        threshold_match = abs(repo['threshold'] - gui['threshold']) < 0.0001
        print(f"Threshold:")
        print(f"  Repository: {repo['threshold']:.4f}")
        print(f"  GUI:        {gui['threshold']:.4f}")
        print(f"  Match: {'✓' if threshold_match else '✗'}")

        # Baseline accuracy
        baseline_match = abs(repo['baseline_accuracy'] - gui['baseline_accuracy']) < 0.01
        print(f"\nBaseline Accuracy:")
        print(f"  Repository: {repo['baseline_accuracy']:.2f}%")
        print(f"  GUI:        {gui['baseline_accuracy']:.2f}%")
        print(f"  Match: {'✓' if baseline_match else '✗'}")

        # Accuracy with threshold (KEY DIFFERENCE)
        accuracy_match = abs(repo['accuracy_with_threshold'] - gui['accuracy_with_threshold']) < 0.01
        print(f"\nAccuracy with Threshold:")
        print(f"  Repository: {repo['accuracy_with_threshold']:.2f}%")
        print(f"  GUI:        {gui['accuracy_with_threshold']:.2f}%")
        print(f"  Match: {'✓' if accuracy_match else '⚠️  DIFFERENCE!'}")

        if not accuracy_match:
            diff = gui['accuracy_with_threshold'] - repo['accuracy_with_threshold']
            print(f"  Difference: {diff:+.2f}%")
            if repo['baseline_accuracy'] >= 95.0:
                print(f"  ⚠️  GUI BUG: Uses baseline accuracy instead of accuracy at threshold!")

        # Counts
        correct_match = repo['correct_with_threshold'] == gui['correct_with_threshold']
        print(f"\nCorrect Count:")
        print(f"  Repository: {repo['correct_with_threshold']}")
        print(f"  GUI:        {gui['correct_with_threshold']}")
        print(f"  Match: {'✓' if correct_match else '✗'}")

        incorrect_match = repo['incorrect_with_threshold'] == gui['incorrect_with_threshold']
        print(f"\nIncorrect Count:")
        print(f"  Repository: {repo['incorrect_with_threshold']}")
        print(f"  GUI:        {gui['incorrect_with_threshold']}")
        print(f"  Match: {'✓' if incorrect_match else '✗'}")

        undetermined_match = repo['undetermined'] == gui['undetermined']
        print(f"\nUndetermined Count:")
        print(f"  Repository: {repo['undetermined']}")
        print(f"  GUI:        {gui['undetermined']}")
        print(f"  Match: {'✓' if undetermined_match else '✗'}")

        # Passes criterion
        passes_match = repo['passes_criterion'] == gui['passes_criterion']
        print(f"\nPasses Criterion:")
        print(f"  Repository: {repo['passes_criterion']}")
        print(f"  GUI:        {gui['passes_criterion']}")
        print(f"  Match: {'✓' if passes_match else '✗'}")

        comparison.append({
            'taxon': taxon,
            'threshold_match': threshold_match,
            'baseline_match': baseline_match,
            'accuracy_match': accuracy_match,
            'correct_match': correct_match,
            'incorrect_match': incorrect_match,
            'undetermined_match': undetermined_match,
            'passes_match': passes_match,
            'all_match': all([threshold_match, baseline_match, accuracy_match,
                            correct_match, incorrect_match, undetermined_match, passes_match])
        })

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_perfect = all(c['all_match'] for c in comparison)

    if all_perfect:
        print("✓ ALL TESTS PASSED - Both implementations produce identical results!")
    else:
        print("⚠️  DIFFERENCES FOUND:")
        for c in comparison:
            if not c['all_match']:
                print(f"\n  {c['taxon']}:")
                if not c['accuracy_match']:
                    print(f"    ⚠️  Accuracy with threshold differs (GUI bug: uses baseline)")
                if not c['correct_match']:
                    print(f"    ✗ Correct count differs")
                if not c['incorrect_match']:
                    print(f"    ✗ Incorrect count differs")
                if not c['undetermined_match']:
                    print(f"    ✗ Undetermined count differs")

    print("\n" + "="*70)
    print()

    return comparison


def main():
    """Main test execution."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*12 + "THRESHOLD CALIBRATION TEST SUITE" + " "*24 + "║")
    print("╚" + "="*68 + "╝")
    print()

    # Create test data
    df = create_synthetic_test_data()

    # Test repository implementation
    results_repo = test_repository_implementation(df)

    # Test GUI implementation
    results_gui = test_gui_implementation(df)

    # Compare
    comparison = compare_implementations(results_repo, results_gui)

    print("\n✓ Test suite completed!")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
