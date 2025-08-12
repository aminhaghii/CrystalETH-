"""
Simple Validation Script for Your ETH/USD Predictions.

Replace the placeholder data below with your actual predictions and values,
then run this script to get comprehensive validation results.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from src.evaluation.validation_suite import ValidationSuite, format_validation_results

def main():
    """
    Main function to run validation on your data.
    
    INSTRUCTIONS:
    1. Replace the placeholder data below with your actual predictions and values
    2. Run this script: python run_validation.py
    3. Review the comprehensive validation results
    """
    
    # ============================================================================
    # REPLACE THIS SECTION WITH YOUR ACTUAL DATA
    # ============================================================================
    
    # Example placeholder data - REPLACE WITH YOUR DATA
    predictions = [
        0.0123, -0.0045, 0.0234, -0.0123, 0.0089, 0.0156, -0.0078, 0.0201,
        -0.0034, 0.0145, 0.0067, -0.0189, 0.0098, 0.0234, -0.0156, 0.0123,
        # Add your predictions here...
        # Continue with all your prediction values
    ]
    
    actual_values = [
        0.0134, -0.0056, 0.0245, -0.0134, 0.0078, 0.0167, -0.0089, 0.0212,
        -0.0045, 0.0156, 0.0078, -0.0178, 0.0089, 0.0223, -0.0145, 0.0134,
        # Add your actual values here...
        # Continue with all your actual values
    ]
    
    # Optional: Add timestamps if you have them
    timestamps = None  # Or provide your timestamp list
    
    # ============================================================================
    # VALIDATION CONFIGURATION
    # ============================================================================
    
    # Validation parameters (you can adjust these)
    train_ratio = 0.7           # 70% for training, 30% for testing
    min_train_samples = 50      # Minimum samples for training
    test_horizon = 20           # Test window size for walk-forward
    n_shuffles = 100            # Number of shuffled-label tests
    random_seed = 42            # For reproducibility
    
    # ============================================================================
    # RUN VALIDATION
    # ============================================================================
    
    print("=== ETH/USD PREDICTION VALIDATION ===\n")
    
    # Validate input data
    if len(predictions) != len(actual_values):
        raise ValueError(f"Length mismatch: predictions={len(predictions)}, actual={len(actual_values)}")
    
    if len(predictions) < min_train_samples * 2:
        raise ValueError(f"Insufficient data: need at least {min_train_samples * 2} samples")
    
    print(f"Data loaded: {len(predictions)} samples")
    print(f"Prediction range: [{min(predictions):.6f}, {max(predictions):.6f}]")
    print(f"Actual range: [{min(actual_values):.6f}, {max(actual_values):.6f}]")
    
    # Initialize validation suite
    validator = ValidationSuite(
        train_ratio=train_ratio,
        min_train_samples=min_train_samples,
        test_horizon=test_horizon,
        n_shuffles=n_shuffles,
        random_seed=random_seed
    )
    
    # Run comprehensive validation
    print("\nRunning comprehensive validation...\n")
    results = validator.comprehensive_validation(
        predictions=predictions,
        actual_values=actual_values,
        timestamps=timestamps
    )
    
    # Display results
    print(format_validation_results(results))
    
    # Extract key metrics for easy access
    test_metrics = results['test_set_metrics']
    wf_metrics = results['walk_forward_results']['aggregated_metrics']
    shuffle_stats = results['shuffled_label_results']['shuffled_statistics']
    
    print("\n=== KEY RESULTS SUMMARY ===")
    print(f"Test Set Log10 Loss: {test_metrics['Latest Log10 Loss']:.6f}")
    print(f"Test Set Score: {test_metrics['Latest Score']:.6f}")
    print(f"Walk-Forward Avg Log10 Loss: {wf_metrics['mean_log10_loss']:.6f}")
    print(f"Walk-Forward Avg Score: {wf_metrics['mean_score']:.6f}")
    print(f"Shuffled Avg Log10 Loss: {shuffle_stats['log10_loss']['mean']:.6f}")
    print(f"Shuffled Avg Score: {shuffle_stats['score']['mean']:.6f}")
    
    # Data leakage assessment
    leakage_risk = results['shuffled_label_results']['leakage_assessment']['risk_level']
    overall_grade = results['overall_assessment']['overall_grade']
    
    print(f"\nData Leakage Risk: {leakage_risk}")
    print(f"Overall Model Grade: {overall_grade}")
    
    return results

if __name__ == "__main__":
    # Check if this is being run with placeholder data
    print("WARNING: This script contains placeholder data!")
    print("Please replace the 'predictions' and 'actual_values' lists with your real data.\n")
    
    response = input("Continue with placeholder data for demonstration? (y/n): ")
    if response.lower() != 'y':
        print("Please edit the script with your data and run again.")
        exit()
    
    # Run validation
    try:
        results = main()
        print("\n✅ Validation completed successfully!")
    except Exception as e:
        print(f"\n❌ Validation failed: {str(e)}")
        print("Please check your data and try again.")