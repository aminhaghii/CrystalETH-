"""
Data Leakage Detection Module
Implements comprehensive checks for temporal data leakage in time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path
import warnings

class DataLeakageDetector:
    """
    Comprehensive data leakage detection for time series forecasting.
    
    Implements Rule #1: Data Integrity is Sacred
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize leakage detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.leakage_report = {}
        
    def detect_temporal_leakage(self, features_data: pd.DataFrame, 
                              target_col: str) -> Dict[str, Any]:
        """
        Detect temporal data leakage in features.
        
        Args:
            features_data: DataFrame with features and target
            target_col: Name of target column
            
        Returns:
            Dictionary with leakage detection results
        """
        self.logger.info("Starting temporal leakage detection...")
        
        report = {
            'timestamp_alignment': self._check_timestamp_alignment(features_data),
            'future_information': self._check_future_information(features_data, target_col),
            'feature_target_correlation': self._check_feature_target_correlation(features_data, target_col),
            'rolling_window_integrity': self._check_rolling_window_integrity(features_data),
            'lag_consistency': self._check_lag_consistency(features_data),
            'suspicious_correlations': self._find_suspicious_correlations(features_data, target_col),
            'temporal_ordering': self._verify_temporal_ordering(features_data),
            'data_availability_check': self._check_data_availability(features_data)
        }
        
        # Overall assessment
        report['overall_assessment'] = self._assess_overall_leakage(report)
        
        self.leakage_report = report
        return report
    
    def _check_timestamp_alignment(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check if timestamps are properly aligned and UTC."""
        result = {
            'status': 'PASS',
            'issues': [],
            'details': {}
        }
        
        # Check if index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            result['status'] = 'FAIL'
            result['issues'].append("Index is not DatetimeIndex")
            return result
        
        # Check timezone
        if data.index.tz is None:
            result['issues'].append("No timezone information - should be UTC")
            result['status'] = 'WARNING'
        elif str(data.index.tz) != 'UTC':
            result['issues'].append(f"Timezone is {data.index.tz}, should be UTC")
            result['status'] = 'WARNING'
        
        # Check for gaps
        expected_freq = pd.infer_freq(data.index)
        if expected_freq is None:
            result['issues'].append("Cannot infer frequency - irregular timestamps")
            result['status'] = 'WARNING'
        
        result['details']['inferred_frequency'] = expected_freq
        result['details']['date_range'] = f"{data.index.min()} to {data.index.max()}"
        result['details']['total_periods'] = len(data)
        
        return result
    
    def _check_future_information(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Check for features that might contain future information."""
        result = {
            'status': 'PASS',
            'issues': [],
            'suspicious_features': []
        }
        
        # Look for features with suspicious names
        suspicious_patterns = [
            'future', 'next', 'tomorrow', 'ahead', 'forward',
            'lead', 'shift_-', '_t+1', '_t1', 'target_lag_-'
        ]
        
        feature_cols = [col for col in data.columns if col != target_col]
        
        for col in feature_cols:
            col_lower = col.lower()
            for pattern in suspicious_patterns:
                if pattern in col_lower:
                    result['suspicious_features'].append(col)
                    result['issues'].append(f"Feature '{col}' has suspicious name pattern: '{pattern}'")
                    result['status'] = 'WARNING'
        
        return result
    
    def _check_feature_target_correlation(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Check for suspiciously high correlations between features and target."""
        result = {
            'status': 'PASS',
            'issues': [],
            'high_correlations': {}
        }
        
        if target_col not in data.columns:
            result['status'] = 'FAIL'
            result['issues'].append(f"Target column '{target_col}' not found")
            return result
        
        feature_cols = [col for col in data.columns if col != target_col and not col.endswith('_direction')]
        
        # Calculate correlations
        correlations = {}
        for col in feature_cols:
            try:
                corr = data[col].corr(data[target_col])
                if not np.isnan(corr):
                    correlations[col] = abs(corr)
            except:
                continue
        
        # Flag suspiciously high correlations (>0.95)
        high_corr_threshold = 0.95
        for col, corr in correlations.items():
            if corr > high_corr_threshold:
                result['high_correlations'][col] = corr
                result['issues'].append(f"Suspiciously high correlation: {col} ({corr:.4f})")
                result['status'] = 'WARNING'
        
        # Store top correlations for analysis
        sorted_corrs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        result['top_correlations'] = dict(sorted_corrs[:10])
        
        return result
    
    def _check_rolling_window_integrity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check if rolling window features are properly calculated."""
        result = {
            'status': 'PASS',
            'issues': [],
            'rolling_features': []
        }
        
        # Find rolling window features
        rolling_patterns = ['_ma_', '_std_', '_rolling_', '_ewm_', '_sma_', '_ema_']
        
        # Patterns for binary event features that are derived from rolling stats but are not rolling features themselves
        binary_event_patterns = ['_extreme_event', '_large_positive', '_large_negative', 
                                '_new_high', '_new_low', '_break', 'vol_breakout', 
                                'vol_compression', '_consecutive_']
        
        for col in data.columns:
            col_lower = col.lower()
            
            # Skip binary event features - they're derived from rolling stats but are binary flags
            is_binary_event = any(pattern in col_lower for pattern in binary_event_patterns)
            if is_binary_event:
                continue
                
            for pattern in rolling_patterns:
                if pattern in col_lower:
                    result['rolling_features'].append(col)
                    
                    # Check for NaN values at the beginning (expected for rolling windows)
                    first_valid_idx = data[col].first_valid_index()
                    if first_valid_idx is None:
                        result['issues'].append(f"Rolling feature '{col}' has no valid values")
                        result['status'] = 'WARNING'
                    elif first_valid_idx == data.index[0]:
                        result['issues'].append(f"Rolling feature '{col}' has no initial NaN values - potential leakage")
                        result['status'] = 'WARNING'
                    break
        
        return result
    
    def _check_lag_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check if lag features are consistently applied."""
        result = {
            'status': 'PASS',
            'issues': [],
            'lag_features': {}
        }
        
        # Find lag features
        lag_patterns = ['_lag_', '_shift_', '_t-']
        
        for col in data.columns:
            col_lower = col.lower()
            for pattern in lag_patterns:
                if pattern in col_lower:
                    # Extract lag number if possible
                    try:
                        if '_lag_' in col_lower:
                            lag_num = int(col_lower.split('_lag_')[1].split('_')[0])
                        elif '_shift_' in col_lower:
                            lag_num = int(col_lower.split('_shift_')[1].split('_')[0])
                        elif '_t-' in col_lower:
                            lag_num = int(col_lower.split('_t-')[1].split('_')[0])
                        else:
                            continue
                            
                        result['lag_features'][col] = lag_num
                        
                        # Check if lag is positive (should be for historical data)
                        if lag_num <= 0:
                            result['issues'].append(f"Lag feature '{col}' has non-positive lag: {lag_num}")
                            result['status'] = 'WARNING'
                            
                    except (ValueError, IndexError):
                        result['issues'].append(f"Cannot parse lag number from feature: {col}")
                        result['status'] = 'WARNING'
                    break
        
        return result
    
    def _find_suspicious_correlations(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Find features with suspiciously perfect correlations."""
        result = {
            'status': 'PASS',
            'issues': [],
            'perfect_correlations': [],
            'near_perfect_correlations': []
        }
        
        if target_col not in data.columns:
            return result
        
        feature_cols = [col for col in data.columns if col != target_col and not col.endswith('_direction')]
        
        for col in feature_cols:
            try:
                corr = abs(data[col].corr(data[target_col]))
                if corr > 0.999:
                    result['perfect_correlations'].append((col, corr))
                    result['issues'].append(f"Perfect correlation detected: {col} ({corr:.6f})")
                    result['status'] = 'FAIL'
                elif corr > 0.98:
                    result['near_perfect_correlations'].append((col, corr))
                    result['issues'].append(f"Near-perfect correlation: {col} ({corr:.4f})")
                    result['status'] = 'WARNING'
            except:
                continue
        
        return result
    
    def _verify_temporal_ordering(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Verify that data is properly ordered temporally."""
        result = {
            'status': 'PASS',
            'issues': []
        }
        
        # Check if index is sorted
        if not data.index.is_monotonic_increasing:
            result['status'] = 'FAIL'
            result['issues'].append("Data is not sorted in chronological order")
        
        # Check for duplicate timestamps
        if data.index.duplicated().any():
            result['status'] = 'FAIL'
            result['issues'].append("Duplicate timestamps found")
            result['duplicate_count'] = data.index.duplicated().sum()
        
        return result
    
    def _check_data_availability(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check if features would have been available at prediction time."""
        result = {
            'status': 'PASS',
            'issues': [],
            'availability_analysis': {}
        }
        
        # This is a simplified check - in practice, you'd need domain knowledge
        # about when each type of data becomes available
        
        # Check for features that might not be available in real-time
        realtime_suspicious = ['volume_24h', 'trades_24h', 'market_cap']
        
        for col in data.columns:
            col_lower = col.lower()
            for suspicious in realtime_suspicious:
                if suspicious in col_lower and 'lag' not in col_lower:
                    result['issues'].append(f"Feature '{col}' might not be available in real-time")
                    result['status'] = 'WARNING'
        
        return result
    
    def _assess_overall_leakage(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall leakage risk based on all checks."""
        assessment = {
            'risk_level': 'LOW',
            'critical_issues': 0,
            'warning_issues': 0,
            'recommendations': []
        }
        
        # Count issues
        for check_name, check_result in report.items():
            if isinstance(check_result, dict) and 'status' in check_result:
                if check_result['status'] == 'FAIL':
                    assessment['critical_issues'] += len(check_result.get('issues', []))
                elif check_result['status'] == 'WARNING':
                    assessment['warning_issues'] += len(check_result.get('issues', []))
        
        # Determine risk level
        if assessment['critical_issues'] > 0:
            assessment['risk_level'] = 'HIGH'
            assessment['recommendations'].append("HALT TRAINING - Critical data leakage detected")
        elif assessment['warning_issues'] > 5:
            assessment['risk_level'] = 'MEDIUM'
            assessment['recommendations'].append("Investigate warnings before proceeding")
        else:
            assessment['risk_level'] = 'LOW'
            assessment['recommendations'].append("Proceed with caution")
        
        return assessment
    
    def generate_report(self, output_path: str = None) -> str:
        """Generate a comprehensive leakage detection report."""
        if not self.leakage_report:
            return "No leakage detection has been performed yet."
        
        report_lines = [
            "=" * 80,
            "DATA LEAKAGE DETECTION REPORT",
            "=" * 80,
            "",
            f"Overall Risk Level: {self.leakage_report['overall_assessment']['risk_level']}",
            f"Critical Issues: {self.leakage_report['overall_assessment']['critical_issues']}",
            f"Warning Issues: {self.leakage_report['overall_assessment']['warning_issues']}",
            "",
            "RECOMMENDATIONS:",
        ]
        
        for rec in self.leakage_report['overall_assessment']['recommendations']:
            report_lines.append(f"- {rec}")
        
        report_lines.extend([
            "",
            "DETAILED FINDINGS:",
            "-" * 40
        ])
        
        # Add detailed findings
        for check_name, result in self.leakage_report.items():
            if check_name == 'overall_assessment':
                continue
                
            report_lines.extend([
                f"\n{check_name.upper().replace('_', ' ')}:",
                f"Status: {result.get('status', 'N/A')}"
            ])
            
            if result.get('issues'):
                report_lines.append("Issues:")
                for issue in result['issues']:
                    report_lines.append(f"  - {issue}")
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Leakage detection report saved to {output_path}")
        
        return report_text


def run_leakage_detection(features_data: pd.DataFrame, target_col: str, 
                         config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run comprehensive data leakage detection.
    
    Args:
        features_data: DataFrame with features and target
        target_col: Name of target column
        config: Configuration dictionary
        
    Returns:
        Leakage detection report
    """
    logging.info("Starting comprehensive data leakage detection")
    
    detector = DataLeakageDetector(config)
    report = detector.detect_temporal_leakage(features_data, target_col)
    
    # Generate and save report
    report_path = Path(config['paths']['reports']) / 'leakage_detection_report.txt'
    detector.generate_report(str(report_path))
    
    return report