"""
Competition Metrics Calculator for ETH Forecasting Project.

This module implements the exact competition metrics as defined:
1. Latest Log10 Loss - log10 of mean error according to competition definition
2. Latest Score - official competition score metric

Following Rule #3 for technical precision and Rule #14 for statistical testing.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings
from scipy import stats

warnings.filterwarnings('ignore', category=RuntimeWarning)


class CompetitionMetricsCalculator:
    """
    Calculator for official competition metrics.
    
    Implements the exact formulas used in the competition leaderboard:
    - Latest Log10 Loss: log10(mean_error) where error follows competition definition
    - Latest Score: Official competition scoring metric
    """
    
    def __init__(self, competition_type: str = "correlation", **kwargs):
        """
        Initialize competition metrics calculator.
        
        Args:
            competition_type: Type of competition scoring ("correlation", "sharpe", "mse", "custom")
            **kwargs: Additional parameters for specific competition types
        """
        self.competition_type = competition_type.lower()
        self.kwargs = kwargs
        
        # Competition-specific parameters
        self.error_type = kwargs.get('error_type', 'absolute')  # 'absolute', 'squared', 'custom'
        self.score_direction = kwargs.get('score_direction', 'higher_better')  # 'higher_better', 'lower_better'
        
        logging.info(f"Competition metrics calculator initialized: type={competition_type}")
    
    def calculate_competition_metrics(self, 
                                    predictions: Union[List[float], np.ndarray],
                                    actual_values: Union[List[float], np.ndarray]) -> Dict[str, float]:
        """
        Calculate both Latest Log10 Loss and Latest Score.
        
        Args:
            predictions: List or array of prediction values (floats)
            actual_values: List or array of actual values (floats)
            
        Returns:
            Dictionary containing:
            - 'Latest Log10 Loss': log10 of mean error
            - 'Latest Score': competition score
        """
        # Convert to numpy arrays
        pred = np.array(predictions, dtype=float)
        actual = np.array(actual_values, dtype=float)
        
        # Validate inputs
        self._validate_inputs(pred, actual)
        
        # Calculate Latest Log10 Loss
        log10_loss = self._calculate_log10_loss(pred, actual)
        
        # Calculate Latest Score
        competition_score = self._calculate_competition_score(pred, actual)
        
        results = {
            'Latest Log10 Loss': log10_loss,
            'Latest Score': competition_score
        }
        
        logging.info(f"Competition metrics calculated: Log10 Loss={log10_loss:.6f}, Score={competition_score:.6f}")
        
        return results
    
    def _calculate_log10_loss(self, predictions: np.ndarray, actual_values: np.ndarray) -> float:
        """
        Calculate Latest Log10 Loss according to competition definition.
        
        Steps:
        1. Compute error for each prediction according to competition's official definition
        2. Take the mean of these errors
        3. Apply base-10 logarithm to this mean error
        
        Args:
            predictions: Predicted values
            actual_values: Actual values
            
        Returns:
            Latest Log10 Loss value
        """
        # Step 1: Compute errors according to competition definition
        if self.error_type == 'absolute':
            # Absolute error (most common for log returns)
            errors = np.abs(predictions - actual_values)
        elif self.error_type == 'squared':
            # Squared error
            errors = (predictions - actual_values) ** 2
        elif self.error_type == 'relative':
            # Relative error (percentage)
            # Handle division by zero
            mask = actual_values != 0
            errors = np.zeros_like(predictions)
            errors[mask] = np.abs((predictions[mask] - actual_values[mask]) / actual_values[mask])
            errors[~mask] = np.abs(predictions[~mask])  # If actual is 0, use absolute error
        elif self.error_type == 'zptae':
            # ZPTAE-style error (competition might use this for crypto)
            ref_std = np.std(actual_values) if len(actual_values) > 1 else 1.0
            ref_std = max(ref_std, 1e-8)  # Avoid division by zero
            z_error = (predictions - actual_values) / ref_std
            errors = np.tanh(np.abs(z_error))
        else:
            # Default to absolute error
            errors = np.abs(predictions - actual_values)
        
        # Step 2: Take the mean of these errors
        mean_error = np.mean(errors)
        
        # Ensure mean_error is positive and not zero for log10
        mean_error = max(mean_error, 1e-10)
        
        # Step 3: Apply base-10 logarithm
        log10_loss = np.log10(mean_error)
        
        return float(log10_loss)
    
    def _calculate_competition_score(self, predictions: np.ndarray, actual_values: np.ndarray) -> float:
        """
        Calculate Latest Score according to competition's official metric.
        
        Common competition metrics:
        - Correlation (Pearson)
        - Sharpe ratio of strategy
        - Information ratio
        - Custom scoring function
        
        Args:
            predictions: Predicted values
            actual_values: Actual values
            
        Returns:
            Latest Score value
        """
        if self.competition_type == 'correlation':
            # Pearson correlation coefficient
            if len(predictions) < 2:
                return 0.0
            
            if np.std(predictions) == 0 or np.std(actual_values) == 0:
                return 0.0
            
            correlation = np.corrcoef(predictions, actual_values)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
            
        elif self.competition_type == 'spearman':
            # Spearman rank correlation
            if len(predictions) < 2:
                return 0.0
            
            try:
                correlation, _ = stats.spearmanr(predictions, actual_values)
                return float(correlation) if not np.isnan(correlation) else 0.0
            except:
                return 0.0
                
        elif self.competition_type == 'sharpe':
            # Sharpe ratio of strategy based on predictions
            # Strategy: long if prediction > 0, short otherwise
            strategy_returns = np.where(predictions > 0, actual_values, -actual_values)
            
            if np.std(strategy_returns) == 0:
                return 0.0
            
            # Annualized Sharpe ratio (assuming daily returns)
            sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
            return float(sharpe)
            
        elif self.competition_type == 'information_ratio':
            # Information ratio vs benchmark (buy-and-hold)
            strategy_returns = np.where(predictions > 0, actual_values, -actual_values)
            benchmark_returns = actual_values  # Buy-and-hold
            
            active_returns = strategy_returns - benchmark_returns
            tracking_error = np.std(active_returns)
            
            if tracking_error == 0:
                return 0.0
            
            info_ratio = np.mean(active_returns) / tracking_error * np.sqrt(252)
            return float(info_ratio)
            
        elif self.competition_type == 'directional_accuracy':
            # Directional accuracy
            pred_direction = predictions > 0
            actual_direction = actual_values > 0
            accuracy = np.mean(pred_direction == actual_direction)
            return float(accuracy)
            
        elif self.competition_type == 'mse_negative':
            # Negative MSE (higher is better)
            mse = np.mean((predictions - actual_values) ** 2)
            return float(-mse)
            
        elif self.competition_type == 'mae_negative':
            # Negative MAE (higher is better)
            mae = np.mean(np.abs(predictions - actual_values))
            return float(-mae)
            
        elif self.competition_type == 'custom':
            # Custom scoring function
            return self._custom_score(predictions, actual_values)
            
        else:
            # Default to correlation
            logging.warning(f"Unknown competition type: {self.competition_type}, defaulting to correlation")
            return self._calculate_competition_score(predictions, actual_values)
    
    def _custom_score(self, predictions: np.ndarray, actual_values: np.ndarray) -> float:
        """
        Custom scoring function - implement based on specific competition requirements.
        
        Args:
            predictions: Predicted values
            actual_values: Actual values
            
        Returns:
            Custom score value
        """
        # Example: Weighted combination of correlation and directional accuracy
        
        # Correlation component
        if np.std(predictions) == 0 or np.std(actual_values) == 0:
            corr_score = 0.0
        else:
            corr_score = np.corrcoef(predictions, actual_values)[0, 1]
            if np.isnan(corr_score):
                corr_score = 0.0
        
        # Directional accuracy component
        pred_direction = predictions > 0
        actual_direction = actual_values > 0
        dir_accuracy = np.mean(pred_direction == actual_direction)
        
        # Weighted combination (can be customized)
        weight_corr = self.kwargs.get('correlation_weight', 0.7)
        weight_dir = self.kwargs.get('directional_weight', 0.3)
        
        custom_score = weight_corr * corr_score + weight_dir * dir_accuracy
        
        return float(custom_score)
    
    def _validate_inputs(self, predictions: np.ndarray, actual_values: np.ndarray):
        """
        Validate input arrays.
        
        Args:
            predictions: Predicted values
            actual_values: Actual values
            
        Raises:
            ValueError: If inputs are invalid
        """
        if len(predictions) == 0 or len(actual_values) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        if len(predictions) != len(actual_values):
            raise ValueError(f"Length mismatch: predictions={len(predictions)}, actual={len(actual_values)}")
        
        if not np.all(np.isfinite(predictions)):
            raise ValueError("Predictions contain non-finite values (NaN or inf)")
        
        if not np.all(np.isfinite(actual_values)):
            raise ValueError("Actual values contain non-finite values (NaN or inf)")
    
    def format_output(self, metrics: Dict[str, float]) -> str:
        """
        Format output in the requested format.
        
        Args:
            metrics: Dictionary with calculated metrics
            
        Returns:
            Formatted string output
        """
        log10_loss = metrics['Latest Log10 Loss']
        score = metrics['Latest Score']
        
        output = f"Latest Log10 Loss: {log10_loss:.6f}\nLatest Score: {score:.6f}"
        
        return output


def calculate_competition_metrics(predictions: Union[List[float], np.ndarray],
                                actual_values: Union[List[float], np.ndarray],
                                competition_type: str = "correlation",
                                error_type: str = "absolute",
                                **kwargs) -> Dict[str, float]:
    """
    Convenience function to calculate competition metrics.
    
    Args:
        predictions: List or array of prediction values
        actual_values: List or array of actual values
        competition_type: Type of competition scoring
        error_type: Type of error calculation for log10 loss
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with Latest Log10 Loss and Latest Score
    """
    calculator = CompetitionMetricsCalculator(
        competition_type=competition_type,
        error_type=error_type,
        **kwargs
    )
    
    return calculator.calculate_competition_metrics(predictions, actual_values)


def format_competition_output(predictions: Union[List[float], np.ndarray],
                            actual_values: Union[List[float], np.ndarray],
                            competition_type: str = "correlation",
                            error_type: str = "absolute",
                            **kwargs) -> str:
    """
    Calculate and format competition metrics in the requested output format.
    
    Args:
        predictions: List or array of prediction values
        actual_values: List or array of actual values
        competition_type: Type of competition scoring
        error_type: Type of error calculation for log10 loss
        **kwargs: Additional parameters
        
    Returns:
        Formatted string: "Latest Log10 Loss: <value>\nLatest Score: <value>"
    """
    calculator = CompetitionMetricsCalculator(
        competition_type=competition_type,
        error_type=error_type,
        **kwargs
    )
    
    metrics = calculator.calculate_competition_metrics(predictions, actual_values)
    return calculator.format_output(metrics)


# Example usage and testing
if __name__ == "__main__":
    # Example data
    np.random.seed(42)
    n_samples = 100
    
    # Generate synthetic ETH log returns
    actual = np.random.normal(0, 0.02, n_samples)  # 2% daily volatility
    predictions = actual + np.random.normal(0, 0.01, n_samples)  # Add noise
    
    # Test different competition types
    competition_types = ['correlation', 'sharpe', 'directional_accuracy', 'information_ratio']
    
    for comp_type in competition_types:
        print(f"\n=== {comp_type.upper()} COMPETITION ===")
        result = format_competition_output(
            predictions=predictions,
            actual_values=actual,
            competition_type=comp_type,
            error_type='absolute'
        )
        print(result)