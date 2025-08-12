#!/usr/bin/env python3
"""
Data leakage detection script for ETH forecasting model.

This script implements automated leakage detection as required by Rule #1.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

def check_timestamp_alignment(predictions: pd.DataFrame) -> dict:
    """
    Check if prediction timestamps are properly aligned and don't use future information.
    
    Args:
        predictions: DataFrame with prediction data
        
    Returns:
        Dictionary with leakage detection results
    """
    results = {
        "timestamp_leakage": False,
        "issues": [],
        "details": {}
    }
    
    if len(predictions) == 0:
        results["issues"].append("No predictions to analyze")
        return results
    
    # Convert timestamps to datetime
    try:
        pred_times = pd.to_datetime(predictions['timestamp'])
        truth_times = pd.to_datetime(predictions['ground_truth_timestamp'])
        
        # Check if predictions are made before ground truth resolution
        future_predictions = pred_times >= truth_times
        if future_predictions.any():
            results["timestamp_leakage"] = True
            results["issues"].append(f"Found {future_predictions.sum()} predictions made after ground truth")
            results["details"]["future_prediction_count"] = int(future_predictions.sum())
        
        # Check for reasonable prediction horizons (should be ~24 hours)
        time_diffs = (truth_times - pred_times).dt.total_seconds() / 3600  # hours
        
        # Flag predictions with very short horizons (< 20 hours) as suspicious
        short_horizon = time_diffs < 20
        if short_horizon.any():
            results["issues"].append(f"Found {short_horizon.sum()} predictions with < 20 hour horizon")
            results["details"]["short_horizon_count"] = int(short_horizon.sum())
        
        # Flag predictions with very long horizons (> 30 hours) as suspicious  
        long_horizon = time_diffs > 30
        if long_horizon.any():
            results["issues"].append(f"Found {long_horizon.sum()} predictions with > 30 hour horizon")
            results["details"]["long_horizon_count"] = int(long_horizon.sum())
        
        results["details"]["mean_horizon_hours"] = float(time_diffs.mean())
        results["details"]["std_horizon_hours"] = float(time_diffs.std())
        
    except Exception as e:
        results["issues"].append(f"Error analyzing timestamps: {e}")
    
    return results

def check_feature_leakage(predictions: pd.DataFrame) -> dict:
    """
    Check for potential feature leakage by analyzing prediction patterns.
    
    Args:
        predictions: DataFrame with prediction data
        
    Returns:
        Dictionary with feature leakage detection results
    """
    results = {
        "feature_leakage": False,
        "issues": [],
        "details": {}
    }
    
    if len(predictions) == 0:
        results["issues"].append("No predictions to analyze")
        return results
    
    try:
        # Check correlation between predictions and ground truth
        pred_values = predictions['prediction_value'].values
        true_values = predictions['ground_truth_value'].values
        
        correlation = np.corrcoef(pred_values, true_values)[0, 1]
        results["details"]["prediction_correlation"] = float(correlation)
        
        # Flag suspiciously high correlation (> 0.8) as potential leakage
        if correlation > 0.8:
            results["feature_leakage"] = True
            results["issues"].append(f"Suspiciously high correlation: {correlation:.3f}")
        
        # Check directional accuracy
        pred_dir = predictions['prediction_direction'].values
        true_dir = predictions['ground_truth_direction'].values
        
        directional_acc = np.mean(pred_dir == true_dir)
        results["details"]["directional_accuracy"] = float(directional_acc)
        
        # Flag suspiciously high directional accuracy (> 0.75) as potential leakage
        if directional_acc > 0.75:
            results["feature_leakage"] = True
            results["issues"].append(f"Suspiciously high directional accuracy: {directional_acc:.3f}")
        
        # Check for perfect predictions (exact matches)
        exact_matches = np.sum(np.abs(pred_values - true_values) < 1e-6)
        if exact_matches > 0:
            results["feature_leakage"] = True
            results["issues"].append(f"Found {exact_matches} exact prediction matches")
            results["details"]["exact_matches"] = int(exact_matches)
        
        # Check prediction error distribution
        errors = np.abs(pred_values - true_values)
        results["details"]["mean_absolute_error"] = float(np.mean(errors))
        results["details"]["std_absolute_error"] = float(np.std(errors))
        
        # Flag if errors are suspiciously small
        if np.mean(errors) < 0.01:  # Very small errors for crypto predictions
            results["issues"].append(f"Suspiciously small prediction errors: {np.mean(errors):.6f}")
        
    except Exception as e:
        results["issues"].append(f"Error analyzing features: {e}")
    
    return results

def check_data_consistency(predictions: pd.DataFrame) -> dict:
    """
    Check for data consistency issues that might indicate leakage.
    
    Args:
        predictions: DataFrame with prediction data
        
    Returns:
        Dictionary with consistency check results
    """
    results = {
        "consistency_issues": False,
        "issues": [],
        "details": {}
    }
    
    if len(predictions) == 0:
        results["issues"].append("No predictions to analyze")
        return results
    
    try:
        # Check for missing values
        missing_pred = predictions['prediction_value'].isna().sum()
        missing_truth = predictions['ground_truth_value'].isna().sum()
        
        if missing_pred > 0:
            results["issues"].append(f"Found {missing_pred} missing prediction values")
        if missing_truth > 0:
            results["issues"].append(f"Found {missing_truth} missing ground truth values")
        
        # Check for duplicate predictions
        duplicates = predictions.duplicated(subset=['timestamp', 'topic_id']).sum()
        if duplicates > 0:
            results["consistency_issues"] = True
            results["issues"].append(f"Found {duplicates} duplicate predictions")
            results["details"]["duplicate_count"] = int(duplicates)
        
        # Check confidence scores
        if 'confidence_score' in predictions.columns:
            conf_scores = predictions['confidence_score'].values
            
            # Check for unrealistic confidence scores
            high_conf = np.sum(conf_scores > 0.95)
            if high_conf > len(predictions) * 0.5:  # More than 50% high confidence
                results["issues"].append(f"Suspiciously high confidence scores: {high_conf}/{len(predictions)}")
            
            results["details"]["mean_confidence"] = float(np.mean(conf_scores))
            results["details"]["std_confidence"] = float(np.std(conf_scores))
        
    except Exception as e:
        results["issues"].append(f"Error checking consistency: {e}")
    
    return results

def main():
    """Main leakage detection function."""
    
    print("🔍 ETH Forecasting Model - Data Leakage Detection")
    print("=" * 60)
    
    # Load predictions
    predictions_file = 'logs/predictions/predictions_log.csv'
    if not os.path.exists(predictions_file):
        print(f"❌ Predictions file not found: {predictions_file}")
        return
    
    try:
        predictions = pd.read_csv(predictions_file)
        resolved_predictions = predictions[predictions['prediction_status'] == 'resolved'].copy()
        
        print(f"📊 Loaded {len(predictions)} total predictions")
        print(f"📊 Analyzing {len(resolved_predictions)} resolved predictions")
        print()
        
        # Run leakage detection tests
        results = {
            "timestamp": check_timestamp_alignment(resolved_predictions),
            "features": check_feature_leakage(resolved_predictions),
            "consistency": check_data_consistency(resolved_predictions)
        }
        
        # Summary
        total_issues = 0
        leakage_detected = False
        
        for test_name, test_results in results.items():
            print(f"🧪 {test_name.upper()} ANALYSIS:")
            
            if test_results.get("timestamp_leakage") or test_results.get("feature_leakage") or test_results.get("consistency_issues"):
                leakage_detected = True
                print("  ❌ POTENTIAL LEAKAGE DETECTED")
            else:
                print("  ✅ No leakage detected")
            
            for issue in test_results.get("issues", []):
                print(f"    • {issue}")
                total_issues += 1
            
            if test_results.get("details"):
                print("  📈 Details:")
                for key, value in test_results["details"].items():
                    if isinstance(value, float):
                        print(f"    - {key}: {value:.4f}")
                    else:
                        print(f"    - {key}: {value}")
            print()
        
        # Final assessment
        print("🎯 FINAL ASSESSMENT:")
        if leakage_detected:
            print("  ❌ DATA LEAKAGE DETECTED - Model results may be invalid")
            print(f"  📊 Total issues found: {total_issues}")
            print("  🔧 Recommendation: Investigate and fix data pipeline")
        else:
            print("  ✅ NO DATA LEAKAGE DETECTED")
            print("  📊 Model results appear valid")
            print("  🎉 Safe to proceed with model evaluation")
        
        # Save results
        output_file = 'reports/data_leakage_report.json'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        final_results = {
            "timestamp": datetime.now().isoformat(),
            "total_predictions": len(predictions),
            "resolved_predictions": len(resolved_predictions),
            "leakage_detected": leakage_detected,
            "total_issues": total_issues,
            "test_results": results
        }
        
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error during leakage detection: {e}")

if __name__ == "__main__":
    main()