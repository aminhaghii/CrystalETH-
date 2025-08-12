"""
Comprehensive Validation Suite for ETH Forecasting Project.

This module implements proper time-based validation following Rule #13 and Rule #1:
- Time-based train/test splits
- Walk-forward cross-validation
- Shuffled-label tests for data leakage detection
- Statistical significance testing

Following Rule #13: Walk-Forward CV Protocol and Rule #1: Data Integrity.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings
from datetime import datetime, timedelta
from scipy import stats
import json

warnings.filterwarnings('ignore', category=RuntimeWarning)


class ValidationSuite:
    """
    Comprehensive validation suite for time series forecasting models.
    
    Implements:
    1. Time-based train/test splits
    2. Walk-forward cross-validation
    3. Shuffled-label tests for leakage detection
    4. Statistical significance testing
    """
    
    def __init__(self, 
                 train_ratio: float = 0.7,
                 min_train_samples: int = 100,
                 test_horizon: int = 30,
                 n_shuffles: int = 100,
                 random_seed: int = 42):
        """
        Initialize validation suite.
        
        Args:
            train_ratio: Fraction of data for training (0.7 = 70%)
            min_train_samples: Minimum samples required for training
            test_horizon: Number of samples in each test window
            n_shuffles: Number of shuffled-label tests
            random_seed: Random seed for reproducibility
        """
        self.train_ratio = train_ratio
        self.min_train_samples = min_train_samples
        self.test_horizon = test_horizon
        self.n_shuffles = n_shuffles
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility (Rule #6)
        np.random.seed(random_seed)
        
        logging.info(f"ValidationSuite initialized: train_ratio={train_ratio}, "
                    f"min_train={min_train_samples}, test_horizon={test_horizon}")
    
    def time_based_split(self, 
                        predictions: Union[List[float], np.ndarray],
                        actual_values: Union[List[float], np.ndarray],
                        timestamps: Optional[Union[List, np.ndarray]] = None) -> Dict:
        """
        Perform time-based train/test split.
        
        Args:
            predictions: Model predictions
            actual_values: Actual values
            timestamps: Optional timestamps for data
            
        Returns:
            Dictionary with train/test splits and metadata
        """
        pred = np.array(predictions, dtype=float)
        actual = np.array(actual_values, dtype=float)
        
        # Validate inputs
        self._validate_inputs(pred, actual)
        
        n_samples = len(pred)
        split_idx = int(n_samples * self.train_ratio)
        
        # Ensure minimum training samples
        if split_idx < self.min_train_samples:
            split_idx = self.min_train_samples
            logging.warning(f"Adjusted split to ensure min_train_samples: {self.min_train_samples}")
        
        # Create splits (chronological order preserved)
        train_pred = pred[:split_idx]
        train_actual = actual[:split_idx]
        test_pred = pred[split_idx:]
        test_actual = actual[split_idx:]
        
        # Handle timestamps if provided
        if timestamps is not None:
            timestamps = np.array(timestamps)
            train_timestamps = timestamps[:split_idx]
            test_timestamps = timestamps[split_idx:]
        else:
            train_timestamps = np.arange(split_idx)
            test_timestamps = np.arange(split_idx, n_samples)
        
        split_info = {
            'train': {
                'predictions': train_pred,
                'actual': train_actual,
                'timestamps': train_timestamps,
                'size': len(train_pred)
            },
            'test': {
                'predictions': test_pred,
                'actual': test_actual,
                'timestamps': test_timestamps,
                'size': len(test_pred)
            },
            'split_ratio': split_idx / n_samples,
            'split_index': split_idx,
            'total_samples': n_samples
        }
        
        logging.info(f"Time-based split: train={len(train_pred)}, test={len(test_pred)}, "
                    f"actual_ratio={split_idx/n_samples:.3f}")
        
        return split_info
    
    def walk_forward_validation(self, 
                               predictions: Union[List[float], np.ndarray],
                               actual_values: Union[List[float], np.ndarray],
                               n_folds: Optional[int] = None) -> Dict:
        """
        Perform walk-forward cross-validation.
        
        Following Rule #13: expanding window training, fixed validation horizon.
        
        Args:
            predictions: Model predictions
            actual_values: Actual values
            n_folds: Number of folds (auto-calculated if None)
            
        Returns:
            Dictionary with fold-wise results and aggregated metrics
        """
        pred = np.array(predictions, dtype=float)
        actual = np.array(actual_values, dtype=float)
        
        self._validate_inputs(pred, actual)
        
        n_samples = len(pred)
        
        # Auto-calculate number of folds if not specified
        if n_folds is None:
            # Ensure 8-12 folds minimum (Rule #13)
            max_folds = (n_samples - self.min_train_samples) // self.test_horizon
            n_folds = max(8, min(12, max_folds))
        
        if n_folds < 8:
            logging.warning(f"Only {n_folds} folds possible, less than recommended minimum of 8")
        
        fold_results = []
        fold_metrics = []
        
        for fold in range(n_folds):
            # Expanding window: train on all data up to current point
            train_end = self.min_train_samples + fold * self.test_horizon
            test_start = train_end
            test_end = min(test_start + self.test_horizon, n_samples)
            
            if test_end <= test_start:
                break
            
            # Extract fold data
            fold_train_pred = pred[:train_end]
            fold_train_actual = actual[:train_end]
            fold_test_pred = pred[test_start:test_end]
            fold_test_actual = actual[test_start:test_end]
            
            # Calculate metrics for this fold
            fold_metric = self._calculate_competition_metrics(fold_test_pred, fold_test_actual)
            fold_metric['fold'] = fold
            fold_metric['train_size'] = len(fold_train_pred)
            fold_metric['test_size'] = len(fold_test_pred)
            fold_metric['train_end'] = train_end
            fold_metric['test_start'] = test_start
            fold_metric['test_end'] = test_end
            
            fold_results.append({
                'fold': fold,
                'train_predictions': fold_train_pred,
                'train_actual': fold_train_actual,
                'test_predictions': fold_test_pred,
                'test_actual': fold_test_actual,
                'metrics': fold_metric
            })
            
            fold_metrics.append(fold_metric)
        
        # Aggregate metrics across folds
        aggregated_metrics = self._aggregate_fold_metrics(fold_metrics)
        
        wf_results = {
            'fold_results': fold_results,
            'fold_metrics': fold_metrics,
            'aggregated_metrics': aggregated_metrics,
            'n_folds': len(fold_results),
            'validation_type': 'walk_forward_expanding_window'
        }
        
        logging.info(f"Walk-forward validation completed: {len(fold_results)} folds, "
                    f"avg_log10_loss={aggregated_metrics['mean_log10_loss']:.6f}")
        
        return wf_results
    
    def shuffled_label_test(self, 
                           predictions: Union[List[float], np.ndarray],
                           actual_values: Union[List[float], np.ndarray]) -> Dict:
        """
        Perform shuffled-label test to detect data leakage.
        
        If model has data leakage, it should perform well even with shuffled labels.
        Valid models should show significant performance drop with shuffled labels.
        
        Args:
            predictions: Model predictions
            actual_values: Actual values
            
        Returns:
            Dictionary with shuffled test results and statistical analysis
        """
        pred = np.array(predictions, dtype=float)
        actual = np.array(actual_values, dtype=float)
        
        self._validate_inputs(pred, actual)
        
        # Calculate baseline metrics (original labels)
        baseline_metrics = self._calculate_competition_metrics(pred, actual)
        
        # Perform multiple shuffled tests
        shuffled_results = []
        shuffled_log10_losses = []
        shuffled_scores = []
        
        for i in range(self.n_shuffles):
            # Shuffle the actual values (breaking time relationship)
            np.random.seed(self.random_seed + i)  # Different seed for each shuffle
            shuffled_actual = np.random.permutation(actual)
            
            # Calculate metrics with shuffled labels
            shuffled_metrics = self._calculate_competition_metrics(pred, shuffled_actual)
            shuffled_results.append(shuffled_metrics)
            shuffled_log10_losses.append(shuffled_metrics['Latest Log10 Loss'])
            shuffled_scores.append(shuffled_metrics['Latest Score'])
        
        # Statistical analysis
        shuffled_log10_losses = np.array(shuffled_log10_losses)
        shuffled_scores = np.array(shuffled_scores)
        
        # Performance degradation analysis
        log10_degradation = baseline_metrics['Latest Log10 Loss'] - np.mean(shuffled_log10_losses)
        score_degradation = baseline_metrics['Latest Score'] - np.mean(shuffled_scores)
        
        # Statistical significance tests
        log10_ttest = stats.ttest_1samp(shuffled_log10_losses, baseline_metrics['Latest Log10 Loss'])
        score_ttest = stats.ttest_1samp(shuffled_scores, baseline_metrics['Latest Score'])
        
        # Percentile analysis
        log10_percentile = stats.percentileofscore(shuffled_log10_losses, baseline_metrics['Latest Log10 Loss'])
        score_percentile = stats.percentileofscore(shuffled_scores, baseline_metrics['Latest Score'])
        
        shuffle_results = {
            'baseline_metrics': baseline_metrics,
            'shuffled_results': shuffled_results,
            'shuffled_statistics': {
                'log10_loss': {
                    'mean': np.mean(shuffled_log10_losses),
                    'std': np.std(shuffled_log10_losses),
                    'min': np.min(shuffled_log10_losses),
                    'max': np.max(shuffled_log10_losses),
                    'median': np.median(shuffled_log10_losses),
                    'degradation': log10_degradation,
                    'percentile': log10_percentile,
                    'ttest_statistic': log10_ttest.statistic,
                    'ttest_pvalue': log10_ttest.pvalue
                },
                'score': {
                    'mean': np.mean(shuffled_scores),
                    'std': np.std(shuffled_scores),
                    'min': np.min(shuffled_scores),
                    'max': np.max(shuffled_scores),
                    'median': np.median(shuffled_scores),
                    'degradation': score_degradation,
                    'percentile': score_percentile,
                    'ttest_statistic': score_ttest.statistic,
                    'ttest_pvalue': score_ttest.pvalue
                }
            },
            'n_shuffles': self.n_shuffles,
            'leakage_assessment': self._assess_data_leakage(
                log10_degradation, score_degradation, 
                log10_ttest.pvalue, score_ttest.pvalue
            )
        }
        
        logging.info(f"Shuffled-label test completed: {self.n_shuffles} shuffles, "
                    f"score_degradation={score_degradation:.6f}, "
                    f"leakage_risk={shuffle_results['leakage_assessment']['risk_level']}")
        
        return shuffle_results
    
    def comprehensive_validation(self, 
                                predictions: Union[List[float], np.ndarray],
                                actual_values: Union[List[float], np.ndarray],
                                timestamps: Optional[Union[List, np.ndarray]] = None) -> Dict:
        """
        Run complete validation suite.
        
        Args:
            predictions: Model predictions
            actual_values: Actual values
            timestamps: Optional timestamps
            
        Returns:
            Comprehensive validation results
        """
        logging.info("Starting comprehensive validation suite...")
        
        # 1. Time-based train/test split
        split_results = self.time_based_split(predictions, actual_values, timestamps)
        
        # 2. Test set metrics
        test_metrics = self._calculate_competition_metrics(
            split_results['test']['predictions'],
            split_results['test']['actual']
        )
        
        # 3. Walk-forward validation
        wf_results = self.walk_forward_validation(predictions, actual_values)
        
        # 4. Shuffled-label test
        shuffle_results = self.shuffled_label_test(predictions, actual_values)
        
        # 5. Overall assessment
        overall_assessment = self._generate_overall_assessment(
            test_metrics, wf_results, shuffle_results
        )
        
        comprehensive_results = {
            'test_set_metrics': test_metrics,
            'walk_forward_results': wf_results,
            'shuffled_label_results': shuffle_results,
            'time_split_info': split_results,
            'overall_assessment': overall_assessment,
            'validation_timestamp': datetime.now().isoformat(),
            'configuration': {
                'train_ratio': self.train_ratio,
                'min_train_samples': self.min_train_samples,
                'test_horizon': self.test_horizon,
                'n_shuffles': self.n_shuffles,
                'random_seed': self.random_seed
            }
        }
        
        logging.info("Comprehensive validation completed successfully")
        
        return comprehensive_results
    
    def _calculate_competition_metrics(self, predictions: np.ndarray, actual_values: np.ndarray) -> Dict[str, float]:
        """Calculate competition metrics (Log10 Loss and Score)."""
        # Log10 Loss calculation
        errors = np.abs(predictions - actual_values)
        mean_error = max(np.mean(errors), 1e-10)
        log10_loss = np.log10(mean_error)
        
        # Score calculation (correlation)
        if len(predictions) < 2 or np.std(predictions) == 0 or np.std(actual_values) == 0:
            score = 0.0
        else:
            correlation = np.corrcoef(predictions, actual_values)[0, 1]
            score = correlation if not np.isnan(correlation) else 0.0
        
        return {
            'Latest Log10 Loss': float(log10_loss),
            'Latest Score': float(score)
        }
    
    def _aggregate_fold_metrics(self, fold_metrics: List[Dict]) -> Dict:
        """Aggregate metrics across folds."""
        log10_losses = [m['Latest Log10 Loss'] for m in fold_metrics]
        scores = [m['Latest Score'] for m in fold_metrics]
        
        return {
            'mean_log10_loss': np.mean(log10_losses),
            'std_log10_loss': np.std(log10_losses),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_log10_loss': np.min(log10_losses),
            'max_log10_loss': np.max(log10_losses),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'stability_log10_loss': np.std(log10_losses) / abs(np.mean(log10_losses)) if np.mean(log10_losses) != 0 else float('inf'),
            'stability_score': np.std(scores) / abs(np.mean(scores)) if np.mean(scores) != 0 else float('inf')
        }
    
    def _assess_data_leakage(self, log10_degradation: float, score_degradation: float, 
                           log10_pvalue: float, score_pvalue: float) -> Dict:
        """Assess data leakage risk based on shuffled-label test results."""
        
        # Thresholds for leakage detection
        SCORE_DEGRADATION_THRESHOLD = 0.1  # Should drop by at least 0.1 in correlation
        PVALUE_THRESHOLD = 0.05  # Statistical significance
        
        risk_factors = []
        risk_level = "LOW"
        
        # Check score degradation
        if abs(score_degradation) < SCORE_DEGRADATION_THRESHOLD:
            risk_factors.append("Insufficient score degradation with shuffled labels")
            risk_level = "MEDIUM"
        
        # Check statistical significance
        if score_pvalue > PVALUE_THRESHOLD:
            risk_factors.append("No significant difference from shuffled labels")
            risk_level = "HIGH"
        
        # Check for suspicious improvements
        if score_degradation < -0.05:  # Score improved with shuffled labels
            risk_factors.append("Score improved with shuffled labels (highly suspicious)")
            risk_level = "CRITICAL"
        
        return {
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'score_degradation': score_degradation,
            'log10_degradation': log10_degradation,
            'statistical_significance': score_pvalue < PVALUE_THRESHOLD,
            'recommendation': self._get_leakage_recommendation(risk_level)
        }
    
    def _get_leakage_recommendation(self, risk_level: str) -> str:
        """Get recommendation based on leakage risk level."""
        recommendations = {
            'LOW': "Model appears valid. Continue with deployment.",
            'MEDIUM': "Some concerns detected. Review feature engineering and data preprocessing.",
            'HIGH': "Significant leakage risk. Audit data pipeline and feature creation process.",
            'CRITICAL': "Critical leakage detected. HALT deployment and investigate immediately."
        }
        return recommendations.get(risk_level, "Unknown risk level")
    
    def _generate_overall_assessment(self, test_metrics: Dict, wf_results: Dict, shuffle_results: Dict) -> Dict:
        """Generate overall model assessment."""
        
        # Performance assessment
        test_log10_loss = test_metrics['Latest Log10 Loss']
        test_score = test_metrics['Latest Score']
        wf_mean_score = wf_results['aggregated_metrics']['mean_score']
        wf_std_score = wf_results['aggregated_metrics']['std_score']
        
        # Stability assessment
        stability_score = wf_std_score / abs(wf_mean_score) if wf_mean_score != 0 else float('inf')
        
        # Overall grade
        grade = "FAIL"
        if shuffle_results['leakage_assessment']['risk_level'] in ['LOW', 'MEDIUM']:
            if test_score > 0.3 and stability_score < 0.5:
                grade = "PASS"
            elif test_score > 0.5 and stability_score < 0.3:
                grade = "GOOD"
            elif test_score > 0.7 and stability_score < 0.2:
                grade = "EXCELLENT"
        
        return {
            'overall_grade': grade,
            'test_performance': {
                'log10_loss': test_log10_loss,
                'score': test_score,
                'interpretation': self._interpret_performance(test_log10_loss, test_score)
            },
            'stability_analysis': {
                'cv_mean_score': wf_mean_score,
                'cv_std_score': wf_std_score,
                'stability_coefficient': stability_score,
                'stability_rating': 'HIGH' if stability_score < 0.2 else 'MEDIUM' if stability_score < 0.5 else 'LOW'
            },
            'leakage_assessment': shuffle_results['leakage_assessment'],
            'recommendations': self._generate_recommendations(grade, shuffle_results['leakage_assessment']['risk_level'])
        }
    
    def _interpret_performance(self, log10_loss: float, score: float) -> str:
        """Interpret model performance."""
        if score > 0.7:
            return "Excellent predictive performance"
        elif score > 0.5:
            return "Good predictive performance"
        elif score > 0.3:
            return "Moderate predictive performance"
        elif score > 0.1:
            return "Weak predictive performance"
        else:
            return "No significant predictive performance"
    
    def _generate_recommendations(self, grade: str, leakage_risk: str) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if leakage_risk in ['HIGH', 'CRITICAL']:
            recommendations.append("URGENT: Investigate data leakage in feature engineering")
            recommendations.append("Review all features for future information")
            recommendations.append("Audit data preprocessing pipeline")
        
        if grade == 'FAIL':
            recommendations.append("Model not ready for production")
            recommendations.append("Consider alternative modeling approaches")
            recommendations.append("Increase training data or improve features")
        elif grade == 'PASS':
            recommendations.append("Model shows promise but needs improvement")
            recommendations.append("Consider ensemble methods or hyperparameter tuning")
        elif grade in ['GOOD', 'EXCELLENT']:
            recommendations.append("Model ready for production deployment")
            recommendations.append("Monitor performance in live environment")
        
        return recommendations
    
    def _validate_inputs(self, predictions: np.ndarray, actual_values: np.ndarray):
        """Validate input arrays."""
        if len(predictions) == 0 or len(actual_values) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        if len(predictions) != len(actual_values):
            raise ValueError(f"Length mismatch: predictions={len(predictions)}, actual={len(actual_values)}")
        
        if not np.all(np.isfinite(predictions)):
            raise ValueError("Predictions contain non-finite values (NaN or inf)")
        
        if not np.all(np.isfinite(actual_values)):
            raise ValueError("Actual values contain non-finite values (NaN or inf)")
        
        if len(predictions) < self.min_train_samples:
            raise ValueError(f"Insufficient data: {len(predictions)} < {self.min_train_samples}")


def format_validation_results(validation_results: Dict) -> str:
    """
    Format validation results for display.
    
    Args:
        validation_results: Results from comprehensive_validation
        
    Returns:
        Formatted string output
    """
    test_metrics = validation_results['test_set_metrics']
    wf_metrics = validation_results['walk_forward_results']['aggregated_metrics']
    shuffle_stats = validation_results['shuffled_label_results']['shuffled_statistics']
    assessment = validation_results['overall_assessment']
    
    output = f"""=== COMPREHENSIVE VALIDATION RESULTS ===

TEST SET METRICS:
Latest Log10 Loss: {test_metrics['Latest Log10 Loss']:.6f}
Latest Score: {test_metrics['Latest Score']:.6f}

WALK-FORWARD AVERAGE METRICS:
Latest Log10 Loss: {wf_metrics['mean_log10_loss']:.6f} ± {wf_metrics['std_log10_loss']:.6f}
Latest Score: {wf_metrics['mean_score']:.6f} ± {wf_metrics['std_score']:.6f}

SHUFFLED-LABEL METRICS:
Latest Log10 Loss: {shuffle_stats['log10_loss']['mean']:.6f} ± {shuffle_stats['log10_loss']['std']:.6f}
Latest Score: {shuffle_stats['score']['mean']:.6f} ± {shuffle_stats['score']['std']:.6f}

PERFORMANCE DEGRADATION (Original vs Shuffled):
Log10 Loss Degradation: {shuffle_stats['log10_loss']['degradation']:.6f}
Score Degradation: {shuffle_stats['score']['degradation']:.6f}

OVERALL ASSESSMENT:
Grade: {assessment['overall_grade']}
Leakage Risk: {assessment['leakage_assessment']['risk_level']}
Stability: {assessment['stability_analysis']['stability_rating']}

RECOMMENDATIONS:
{chr(10).join(f"- {rec}" for rec in assessment['recommendations'])}
"""
    
    return output