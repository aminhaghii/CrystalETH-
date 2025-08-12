#!/usr/bin/env python3
"""
Evaluate model acceptance rules according to specifications.

This script:
1. Loads resolved predictions from logs
2. Checks for baseline values
3. Computes required metrics
4. Evaluates acceptance criteria
5. Returns structured JSON result
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

def load_resolved_predictions(file_path: str) -> pd.DataFrame:
    """Load predictions and filter for resolved ones with actual values."""
    try:
        df = pd.read_csv(file_path)
        # Filter for resolved predictions with ground truth values
        resolved = df[
            (df['prediction_status'] == 'resolved') & 
            (df['ground_truth_value'].notna())
        ].copy()
        return resolved
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return pd.DataFrame()

def check_baseline_files() -> Dict[str, Any]:
    """Check for baseline configuration files."""
    baseline_files = [
        'config/baseline.yaml',
        'config/baseline.json',
        'reports/baseline.json', 
        'reports/baseline_results.json',
        'reports/phase3/baseline_results.json'
    ]
    
    baseline_data = {}
    found_files = []
    
    for file_path in baseline_files:
        if os.path.exists(file_path):
            found_files.append(file_path)
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # Handle different JSON structures
                        if 'performance_metrics' in data:
                            baseline_data.update(data['performance_metrics'])
                        elif 'baseline_metrics' in data:
                            baseline_data.update(data['baseline_metrics'])
                        else:
                            baseline_data.update(data)
                elif file_path.endswith('.yaml'):
                    import yaml
                    with open(file_path, 'r') as f:
                        data = yaml.safe_load(f)
                        if 'baseline_metrics' in data:
                            baseline_data.update(data['baseline_metrics'])
                        else:
                            baseline_data.update(data)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    # Check for BASELINE_* files in reports
    reports_dir = 'reports'
    if os.path.exists(reports_dir):
        for file in os.listdir(reports_dir):
            if file.startswith('BASELINE_'):
                found_files.append(os.path.join(reports_dir, file))
                try:
                    with open(os.path.join(reports_dir, file), 'r') as f:
                        if file.endswith('.json'):
                            data = json.load(f)
                            baseline_data.update(data)
                except Exception as e:
                    print(f"Error reading {file}: {e}")
    
    return {
        'found_files': found_files,
        'data': baseline_data
    }

def check_mztae_formula() -> bool:
    """Check if MZTAE formula is implemented in the repository."""
    # Check losses.py for MZTAE implementation
    losses_file = 'src/models/losses.py'
    if os.path.exists(losses_file):
        with open(losses_file, 'r') as f:
            content = f.read()
            # Look for MZTAE or mztae in the file
            if 'MZTAE' in content or 'mztae' in content:
                return True
    
    # Check other common locations
    search_files = [
        'src/models/metrics.py',
        'src/evaluation/metrics.py', 
        'src/utils/metrics.py'
    ]
    
    for file_path in search_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                if 'MZTAE' in content or 'mztae' in content:
                    return True
    
    return False

def compute_directional_accuracy(predictions: pd.DataFrame) -> float:
    """Compute directional accuracy."""
    if len(predictions) == 0:
        return 0.0
    
    pred_dir = predictions['prediction_direction'].values
    true_dir = predictions['ground_truth_direction'].values
    
    correct = np.sum(pred_dir == true_dir)
    total = len(predictions)
    
    return correct / total if total > 0 else 0.0

def compute_pesaran_timmermann_test(predictions: pd.DataFrame) -> Tuple[float, float]:
    """Compute Pesaran-Timmermann test for directional accuracy."""
    if len(predictions) < 2:
        return np.nan, np.nan
    
    try:
        pred_dir = predictions['prediction_direction'].values
        true_dir = predictions['ground_truth_direction'].values
        
        n = len(predictions)
        
        # Calculate directional accuracy
        correct = np.sum(pred_dir == true_dir)
        p_hat = correct / n
        
        # Calculate probabilities
        p_y = np.mean(true_dir)  # Probability of positive actual returns
        p_x = np.mean(pred_dir)  # Probability of positive predicted returns
        
        # Expected probability under independence
        p_star = p_y * p_x + (1 - p_y) * (1 - p_x)
        
        # Variance under independence
        var_p_star = p_star * (1 - p_star) / n
        
        # Test statistic
        if var_p_star > 0:
            test_stat = (p_hat - p_star) / np.sqrt(var_p_star)
            
            # Calculate p-value (two-tailed test)
            from scipy import stats
            p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))
            
            return test_stat, p_value
        else:
            return np.nan, np.nan
            
    except Exception as e:
        print(f"Error in Pesaran-Timmermann test: {e}")
        return np.nan, np.nan

def compute_weighted_rmse(predictions: pd.DataFrame) -> float:
    """Compute weighted RMSE (using confidence scores as weights)."""
    if len(predictions) == 0:
        return np.nan
    
    try:
        pred_values = predictions['prediction_value'].values
        true_values = predictions['ground_truth_value'].values
        weights = predictions['confidence_score'].values
        
        # Compute weighted squared errors
        squared_errors = (pred_values - true_values) ** 2
        weighted_mse = np.average(squared_errors, weights=weights)
        
        return np.sqrt(weighted_mse)
    except Exception as e:
        print(f"Error computing weighted RMSE: {e}")
        return np.nan

def compute_mztae(predictions: pd.DataFrame) -> float:
    """
    Compute MZTAE (Mean Z-Transformed Absolute Error).
    """
    if len(predictions) == 0:
        return np.nan
    
    try:
        # Import the MZTAE function from our losses module
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from models.losses import mztae_metric_numpy
        
        pred_values = predictions['prediction_value'].values
        true_values = predictions['ground_truth_value'].values
        
        # Calculate rolling standard deviation as reference
        # Use a simple approach: std of true values
        ref_std = np.std(true_values) if len(true_values) > 1 else 1.0
        ref_std_array = np.full_like(true_values, ref_std)
        
        # Compute MZTAE
        mztae = mztae_metric_numpy(pred_values, true_values, ref_std_array)
        return mztae
        
    except Exception as e:
        print(f"Error computing MZTAE: {e}")
        return np.nan

def main():
    """Main evaluation function."""
    
    # Initialize result structure
    result = {
        "result": "FAIL",
        "reasons": [],
        "evidence": {
            "timestamp": datetime.now().isoformat()
        }
    }
    
    # 1. Load resolved predictions
    predictions_file = 'logs/predictions/predictions_log.csv'
    if not os.path.exists(predictions_file):
        # Try alternative locations
        alt_files = [
            'logs/predictions.jsonl',
            'reports/predictions.csv'
        ]
        predictions_file = None
        for alt_file in alt_files:
            if os.path.exists(alt_file):
                predictions_file = alt_file
                break
        
        if predictions_file is None:
            result["reasons"].append("No prediction files found")
            result["evidence"]["resolved_predictions"] = 0
            print(json.dumps(result, indent=2))
            return
    
    predictions = load_resolved_predictions(predictions_file)
    resolved_count = len(predictions)
    result["evidence"]["resolved_predictions"] = resolved_count
    
    # 2. Check minimum resolved predictions requirement
    if resolved_count < 30:
        result["reasons"].append("insufficient resolved samples")
        result["evidence"]["resolved_predictions"] = resolved_count
    
    # 3. Check for baseline values
    baseline_info = check_baseline_files()
    if not baseline_info['found_files'] or not baseline_info['data']:
        result["reasons"].append("missing baseline")
        result["evidence"]["baseline_weighted_RMSE"] = None
        result["evidence"]["baseline_MZTAE"] = None
    else:
        # Extract baseline metrics if available
        baseline_data = baseline_info['data']
        result["evidence"]["baseline_weighted_RMSE"] = baseline_data.get('weighted_RMSE', None)
        result["evidence"]["baseline_MZTAE"] = baseline_data.get('MZTAE', None)
    
    # 4. Check MZTAE formula
    if not check_mztae_formula():
        result["reasons"].append("missing MZTAE formula")
    
    # 5. Compute metrics on resolved samples
    if resolved_count > 0:
        # Directional accuracy
        da = compute_directional_accuracy(predictions)
        result["evidence"]["directional_accuracy"] = da
        
        # Check for suspiciously high directional accuracy (Rule #15)
        if da > 0.70:
            result["reasons"].append("Directional accuracy > 70% - check for data leakage")
        
        # P-value (Pesaran-Timmermann test)
        test_stat, p_value = compute_pesaran_timmermann_test(predictions)
        result["evidence"]["p_value"] = p_value
        # Note: Successfully computed p-value using Pesaran-Timmermann test
        
        # Weighted RMSE
        w_rmse = compute_weighted_rmse(predictions)
        result["evidence"]["weighted_RMSE"] = w_rmse
        
        # MZTAE (will be NaN since formula not found)
        mztae = compute_mztae(predictions)
        result["evidence"]["MZTAE"] = mztae
    else:
        result["evidence"]["directional_accuracy"] = None
        result["evidence"]["p_value"] = None
        result["evidence"]["weighted_RMSE"] = None
        result["evidence"]["MZTAE"] = None
    
    # 6. Determine final result
    if not result["reasons"]:
        # Check if metrics pass baseline comparison
        baseline_w_rmse = result["evidence"].get("baseline_weighted_RMSE")
        baseline_mztae = result["evidence"].get("baseline_MZTAE")
        current_w_rmse = result["evidence"].get("weighted_RMSE")
        current_mztae = result["evidence"].get("MZTAE")
        
        if (baseline_w_rmse is not None and current_w_rmse is not None and 
            baseline_mztae is not None and current_mztae is not None):
            
            if current_w_rmse < baseline_w_rmse and current_mztae < baseline_mztae:
                result["result"] = "PASS"
            else:
                result["reasons"].append("Metrics do not beat baseline")
        else:
            result["reasons"].append("Cannot compare to baseline - missing values")
    
    # Output result
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()