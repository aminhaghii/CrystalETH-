"""
Comprehensive metrics module for ETH forecasting project.

This module implements various evaluation metrics including:
- Regression metrics (MSE, MAE, RMSE, ZPTAE)
- Classification metrics (Directional Accuracy, Precision, Recall)
- Financial metrics (Sharpe ratio, Maximum Drawdown, Calmar ratio)
- Risk metrics (VaR, CVaR, Volatility)
- Economic metrics with transaction costs

Following Rule #22 for economic backtesting standards.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import warnings
from scipy import stats

warnings.filterwarnings('ignore', category=RuntimeWarning)


class MetricsCalculator:
    """
    Comprehensive metrics calculator for forecasting evaluation.
    
    Implements various metrics for regression, classification, and
    financial performance evaluation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MetricsCalculator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.metrics_config = config.get('evaluation', {}).get('metrics', {})
        
        # Economic parameters
        self.transaction_cost = self.metrics_config.get('transaction_cost', 0.0005)  # 0.05%
        self.slippage = self.metrics_config.get('slippage', 0.0005)  # 0.05%
        self.risk_free_rate = self.metrics_config.get('risk_free_rate', 0.02)  # 2% annual
        
        # Risk parameters
        self.var_confidence = self.metrics_config.get('var_confidence', 0.05)  # 5% VaR
        self.cvar_confidence = self.metrics_config.get('cvar_confidence', 0.05)  # 5% CVaR
        
        logging.info("MetricsCalculator initialized")
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                            include_economic: bool = True) -> Dict[str, float]:
        """
        Calculate all available metrics.
        
        Args:
            y_true: True values (log returns)
            y_pred: Predicted values (log returns)
            include_economic: Whether to include economic metrics
            
        Returns:
            Dictionary with all calculated metrics
        """
        logging.info("Calculating comprehensive metrics")
        
        metrics = {}
        
        # Regression metrics
        metrics.update(self.calculate_regression_metrics(y_true, y_pred))
        
        # Classification metrics
        metrics.update(self.calculate_classification_metrics(y_true, y_pred))
        
        # Statistical metrics
        metrics.update(self.calculate_statistical_metrics(y_true, y_pred))
        
        # Risk metrics
        metrics.update(self.calculate_risk_metrics(y_true, y_pred))
        
        # Economic metrics
        if include_economic:
            metrics.update(self.calculate_economic_metrics(y_true, y_pred))
        
        logging.info(f"Calculated {len(metrics)} metrics")
        
        return metrics
    
    def calculate_regression_metrics(self, y_true: np.ndarray, 
                                   y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with regression metrics
        """
        metrics = {}
        
        # Basic regression metrics
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        
        # Mean Absolute Percentage Error (handle division by zero)
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            metrics['mape'] = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            metrics['mape'] = np.nan
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r_squared'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Explained variance
        metrics['explained_variance'] = 1 - np.var(y_true - y_pred) / np.var(y_true)
        
        # ZPTAE metric
        try:
            from ..models.losses import zptae_metric_numpy
            zptae_params = self.config.get('loss_functions', {}).get('zptae', {})
            a = zptae_params.get('a', 1.0)
            p = zptae_params.get('p', 1.5)
            metrics['zptae'] = zptae_metric_numpy(y_true, y_pred, a, p)
        except ImportError:
            logging.warning("ZPTAE metric not available")
            metrics['zptae'] = np.nan
        
        # Median Absolute Error
        metrics['median_ae'] = np.median(np.abs(y_true - y_pred))
        
        # Max Error
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))
        
        return metrics
    
    def calculate_classification_metrics(self, y_true: np.ndarray, 
                                       y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate classification metrics for directional prediction.
        
        Args:
            y_true: True values (log returns)
            y_pred: Predicted values (log returns)
            
        Returns:
            Dictionary with classification metrics
        """
        metrics = {}
        
        # Convert to directional indicators
        y_true_dir = (y_true > 0).astype(int)
        y_pred_dir = (y_pred > 0).astype(int)
        
        # Directional accuracy
        metrics['directional_accuracy'] = np.mean(y_true_dir == y_pred_dir)
        
        # Confusion matrix components
        tp = np.sum((y_true_dir == 1) & (y_pred_dir == 1))  # True Positive
        tn = np.sum((y_true_dir == 0) & (y_pred_dir == 0))  # True Negative
        fp = np.sum((y_true_dir == 0) & (y_pred_dir == 1))  # False Positive
        fn = np.sum((y_true_dir == 1) & (y_pred_dir == 0))  # False Negative
        
        # Precision, Recall, F1-score
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
        
        # Specificity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Balanced accuracy
        sensitivity = metrics['recall']
        metrics['balanced_accuracy'] = (sensitivity + metrics['specificity']) / 2
        
        # Matthews Correlation Coefficient
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        metrics['mcc'] = (tp * tn - fp * fn) / denominator if denominator > 0 else 0
        
        # Hit rate for different thresholds
        for threshold in [0.001, 0.005, 0.01]:
            hit_rate = np.mean(np.abs(y_true - y_pred) < threshold)
            metrics[f'hit_rate_{int(threshold*1000)}bp'] = hit_rate
        
        return metrics
    
    def calculate_statistical_metrics(self, y_true: np.ndarray, 
                                    y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistical metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with statistical metrics
        """
        metrics = {}
        
        # Correlation
        if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
            metrics['correlation'] = np.corrcoef(y_true, y_pred)[0, 1]
        else:
            metrics['correlation'] = 0
        
        # Rank correlation (Spearman)
        try:
            metrics['spearman_correlation'] = stats.spearmanr(y_true, y_pred)[0]
        except:
            metrics['spearman_correlation'] = 0
        
        # Prediction bias
        metrics['bias'] = np.mean(y_pred - y_true)
        
        # Prediction variance ratio
        metrics['variance_ratio'] = np.var(y_pred) / np.var(y_true) if np.var(y_true) > 0 else 0
        
        # Theil's U statistic
        mse_pred = np.mean((y_true - y_pred) ** 2)
        mse_naive = np.mean((y_true[1:] - y_true[:-1]) ** 2)
        metrics['theil_u'] = np.sqrt(mse_pred) / np.sqrt(mse_naive) if mse_naive > 0 else np.inf
        
        # Forecast error statistics
        errors = y_pred - y_true
        metrics['error_mean'] = np.mean(errors)
        metrics['error_std'] = np.std(errors)
        metrics['error_skewness'] = stats.skew(errors)
        metrics['error_kurtosis'] = stats.kurtosis(errors)
        
        # Ljung-Box test for error autocorrelation (simplified)
        if len(errors) > 10:
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lb_stat = acorr_ljungbox(errors, lags=5, return_df=True)
                metrics['ljung_box_pvalue'] = lb_stat['lb_pvalue'].iloc[-1]
            except:
                metrics['ljung_box_pvalue'] = np.nan
        else:
            metrics['ljung_box_pvalue'] = np.nan
        
        return metrics
    
    def calculate_risk_metrics(self, y_true: np.ndarray, 
                             y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate risk metrics.
        
        Args:
            y_true: True values (log returns)
            y_pred: Predicted values (log returns)
            
        Returns:
            Dictionary with risk metrics
        """
        metrics = {}
        
        # Strategy returns (simple long/short based on predictions)
        strategy_returns = np.where(y_pred > 0, y_true, -y_true)
        
        # Volatility (annualized)
        metrics['volatility_annual'] = np.std(strategy_returns) * np.sqrt(252)
        metrics['true_volatility_annual'] = np.std(y_true) * np.sqrt(252)
        
        # Value at Risk (VaR)
        metrics[f'var_{int(self.var_confidence*100)}'] = np.percentile(strategy_returns, self.var_confidence * 100)
        
        # Conditional Value at Risk (CVaR)
        var_threshold = metrics[f'var_{int(self.var_confidence*100)}']
        cvar_returns = strategy_returns[strategy_returns <= var_threshold]
        metrics[f'cvar_{int(self.cvar_confidence*100)}'] = np.mean(cvar_returns) if len(cvar_returns) > 0 else 0
        
        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        metrics['max_drawdown'] = np.min(drawdown)
        
        # Downside deviation
        negative_returns = strategy_returns[strategy_returns < 0]
        metrics['downside_deviation'] = np.std(negative_returns) if len(negative_returns) > 0 else 0
        
        # Upside/Downside capture
        positive_true = y_true[y_true > 0]
        positive_strategy = strategy_returns[y_true > 0]
        negative_true = y_true[y_true < 0]
        negative_strategy = strategy_returns[y_true < 0]
        
        metrics['upside_capture'] = np.mean(positive_strategy) / np.mean(positive_true) if len(positive_true) > 0 and np.mean(positive_true) != 0 else 0
        metrics['downside_capture'] = np.mean(negative_strategy) / np.mean(negative_true) if len(negative_true) > 0 and np.mean(negative_true) != 0 else 0
        
        return metrics
    
    def calculate_economic_metrics(self, y_true: np.ndarray, 
                                 y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate economic and financial performance metrics.
        
        Args:
            y_true: True values (log returns)
            y_pred: Predicted values (log returns)
            
        Returns:
            Dictionary with economic metrics
        """
        metrics = {}
        
        # Simple strategy: long if prediction > 0, short otherwise
        positions = np.where(y_pred > 0, 1, -1)
        
        # Calculate strategy returns with transaction costs
        strategy_returns = self._calculate_strategy_returns(y_true, positions)
        
        # Performance metrics
        metrics['total_return'] = np.prod(1 + strategy_returns) - 1
        metrics['annualized_return'] = (1 + metrics['total_return']) ** (252 / len(strategy_returns)) - 1
        
        # Sharpe ratio
        excess_returns = strategy_returns - self.risk_free_rate / 252
        metrics['sharpe_ratio'] = np.mean(excess_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0
        
        # Sortino ratio
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(strategy_returns)
        metrics['sortino_ratio'] = np.mean(excess_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # Calmar ratio
        max_dd = self._calculate_max_drawdown(strategy_returns)
        metrics['calmar_ratio'] = metrics['annualized_return'] / abs(max_dd) if max_dd != 0 else 0
        
        # Information ratio (vs buy-and-hold)
        benchmark_returns = y_true  # Buy-and-hold
        active_returns = strategy_returns - benchmark_returns
        tracking_error = np.std(active_returns)
        metrics['information_ratio'] = np.mean(active_returns) / tracking_error * np.sqrt(252) if tracking_error > 0 else 0
        
        # Win rate
        metrics['win_rate'] = np.mean(strategy_returns > 0)
        
        # Profit factor
        winning_returns = strategy_returns[strategy_returns > 0]
        losing_returns = strategy_returns[strategy_returns < 0]
        gross_profit = np.sum(winning_returns) if len(winning_returns) > 0 else 0
        gross_loss = abs(np.sum(losing_returns)) if len(losing_returns) > 0 else 1e-10
        metrics['profit_factor'] = gross_profit / gross_loss
        
        # Average win/loss
        metrics['avg_win'] = np.mean(winning_returns) if len(winning_returns) > 0 else 0
        metrics['avg_loss'] = np.mean(losing_returns) if len(losing_returns) > 0 else 0
        
        # Expectancy
        win_rate = metrics['win_rate']
        avg_win = metrics['avg_win']
        avg_loss = abs(metrics['avg_loss'])
        metrics['expectancy'] = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Transaction costs impact
        total_trades = np.sum(np.abs(np.diff(positions, prepend=0)))
        total_transaction_costs = total_trades * (self.transaction_cost + self.slippage)
        metrics['total_transaction_costs'] = total_transaction_costs
        metrics['transaction_cost_impact'] = total_transaction_costs / len(strategy_returns)
        
        # Compare with buy-and-hold
        bh_total_return = np.prod(1 + y_true) - 1
        bh_annualized_return = (1 + bh_total_return) ** (252 / len(y_true)) - 1
        bh_sharpe = np.mean(y_true - self.risk_free_rate / 252) / np.std(y_true) * np.sqrt(252) if np.std(y_true) > 0 else 0
        
        metrics['excess_return_vs_bh'] = metrics['annualized_return'] - bh_annualized_return
        metrics['excess_sharpe_vs_bh'] = metrics['sharpe_ratio'] - bh_sharpe
        
        return metrics
    
    def _calculate_strategy_returns(self, returns: np.ndarray, 
                                  positions: np.ndarray) -> np.ndarray:
        """
        Calculate strategy returns including transaction costs.
        
        Args:
            returns: Market returns
            positions: Position signals (1 for long, -1 for short)
            
        Returns:
            Strategy returns after costs
        """
        # Calculate position changes
        position_changes = np.abs(np.diff(positions, prepend=0))
        
        # Calculate gross returns
        gross_returns = positions * returns
        
        # Calculate transaction costs
        transaction_costs = position_changes * (self.transaction_cost + self.slippage)
        
        # Net returns
        net_returns = gross_returns - transaction_costs
        
        return net_returns
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            returns: Return series
            
        Returns:
            Maximum drawdown
        """
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def create_metrics_summary(self, metrics: Dict[str, float]) -> pd.DataFrame:
        """
        Create a formatted summary of metrics.
        
        Args:
            metrics: Dictionary of calculated metrics
            
        Returns:
            DataFrame with formatted metrics
        """
        # Group metrics by category
        categories = {
            'Regression': ['mse', 'rmse', 'mae', 'mape', 'r_squared', 'zptae'],
            'Classification': ['directional_accuracy', 'precision', 'recall', 'f1_score', 'mcc'],
            'Statistical': ['correlation', 'spearman_correlation', 'bias', 'theil_u'],
            'Risk': ['volatility_annual', 'max_drawdown', 'var_5', 'cvar_5'],
            'Economic': ['annualized_return', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'win_rate']
        }
        
        summary_data = []
        
        for category, metric_names in categories.items():
            for metric_name in metric_names:
                if metric_name in metrics:
                    value = metrics[metric_name]
                    
                    # Format value based on metric type
                    if metric_name in ['mse', 'rmse', 'mae', 'zptae']:
                        formatted_value = f"{value:.6f}"
                    elif metric_name in ['directional_accuracy', 'precision', 'recall', 'win_rate']:
                        formatted_value = f"{value:.4f} ({value*100:.2f}%)"
                    elif metric_name in ['annualized_return', 'sharpe_ratio', 'volatility_annual']:
                        formatted_value = f"{value:.4f}"
                    elif metric_name in ['max_drawdown', 'var_5', 'cvar_5']:
                        formatted_value = f"{value:.4f} ({value*100:.2f}%)"
                    else:
                        formatted_value = f"{value:.4f}"
                    
                    summary_data.append({
                        'Category': category,
                        'Metric': metric_name,
                        'Value': value,
                        'Formatted': formatted_value
                    })
        
        return pd.DataFrame(summary_data)


def main():
    """
    Main function for testing metrics calculation.
    """
    # Load configuration
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.utils.helpers import load_config, setup_logging
    
    # Setup
    config = load_config()
    setup_logging()
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # True returns with some pattern
    true_returns = np.random.normal(0.001, 0.02, n_samples)
    
    # Predictions with some skill
    predictions = 0.3 * true_returns + np.random.normal(0, 0.015, n_samples)
    
    # Initialize calculator
    calculator = MetricsCalculator(config)
    
    # Calculate all metrics
    metrics = calculator.calculate_all_metrics(true_returns, predictions)
    
    # Create summary
    summary = calculator.create_metrics_summary(metrics)
    
    print("Metrics Summary:")
    print("=" * 60)
    
    for category in summary['Category'].unique():
        print(f"\n{category} Metrics:")
        category_metrics = summary[summary['Category'] == category]
        for _, row in category_metrics.iterrows():
            print(f"  {row['Metric']}: {row['Formatted']}")
    
    # Print key metrics
    print(f"\nKey Performance Indicators:")
    print(f"  Directional Accuracy: {metrics.get('directional_accuracy', 0):.4f}")
    print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
    print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.4f}")
    print(f"  Annualized Return: {metrics.get('annualized_return', 0):.4f}")


if __name__ == "__main__":
    main()