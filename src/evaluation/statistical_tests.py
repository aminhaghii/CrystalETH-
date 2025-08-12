"""
Statistical testing module for ETH forecasting project.

This module implements statistical tests following Rule #14:
- Pesaran-Timmermann test for directional accuracy
- Diebold-Mariano test for model comparison
- REPORT p-values and confidence intervals
- INTERPRET statistical significance correctly

Additional tests for robustness and validation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from scipy import stats
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class StatisticalTester:
    """
    Statistical testing class for forecasting evaluation.
    
    Implements comprehensive statistical tests for model validation
    and comparison following academic standards.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize StatisticalTester.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.test_config = config.get('statistical_tests', {})
        
        # Test parameters
        self.confidence_level = self.test_config.get('confidence_level', 0.95)
        self.alpha = 1 - self.confidence_level
        
        logging.info("StatisticalTester initialized")
    
    def pesaran_timmermann_test(self, y_true: np.ndarray, y_pred: np.ndarray,
                               return_details: bool = True) -> Dict[str, Any]:
        """
        Pesaran-Timmermann test for directional accuracy.
        
        Tests the null hypothesis that directional predictions are independent
        of actual directions (no forecasting ability).
        
        Args:
            y_true: True values (log returns)
            y_pred: Predicted values (log returns)
            return_details: Whether to return detailed results
            
        Returns:
            Dictionary with test results
        """
        logging.info("Performing Pesaran-Timmermann test for directional accuracy")
        
        # Convert to directional indicators
        R_t = (y_true > 0).astype(int)  # Actual directions
        P_t = (y_pred > 0).astype(int)  # Predicted directions
        
        n = len(R_t)
        
        # Calculate components
        P_hat = np.mean(R_t)  # Proportion of positive actual returns
        P_y = np.mean(P_t)    # Proportion of positive predicted returns
        
        # Success rate (directional accuracy)
        S_n = np.mean(R_t == P_t)
        
        # Expected success rate under independence
        P_star = P_hat * P_y + (1 - P_hat) * (1 - P_y)
        
        # Variance under independence
        V_star = P_star * (1 - P_star) / n
        
        # Test statistic
        if V_star > 0:
            S_n_star = (S_n - P_star) / np.sqrt(V_star)
        else:
            S_n_star = 0
            logging.warning("Zero variance in Pesaran-Timmermann test")
        
        # P-value (two-tailed test)
        p_value = 2 * (1 - norm.cdf(abs(S_n_star)))
        
        # Critical value
        critical_value = norm.ppf(1 - self.alpha / 2)
        
        # Test result
        is_significant = abs(S_n_star) > critical_value
        
        results = {
            'test_name': 'Pesaran-Timmermann',
            'test_statistic': S_n_star,
            'p_value': p_value,
            'critical_value': critical_value,
            'is_significant': is_significant,
            'directional_accuracy': S_n,
            'expected_accuracy_under_independence': P_star,
            'confidence_level': self.confidence_level
        }
        
        if return_details:
            results.update({
                'n_observations': n,
                'proportion_positive_actual': P_hat,
                'proportion_positive_predicted': P_y,
                'variance_under_independence': V_star,
                'interpretation': self._interpret_pt_test(results)
            })
        
        logging.info(f"Pesaran-Timmermann test completed: DA={S_n:.4f}, p-value={p_value:.4f}")
        
        return results
    
    def diebold_mariano_test(self, y_true: np.ndarray, pred1: np.ndarray, 
                           pred2: np.ndarray, loss_function: str = 'mse',
                           h: int = 1, return_details: bool = True) -> Dict[str, Any]:
        """
        Diebold-Mariano test for comparing forecast accuracy.
        
        Tests the null hypothesis that two forecasts have equal accuracy.
        
        Args:
            y_true: True values
            pred1: Predictions from model 1
            pred2: Predictions from model 2
            loss_function: Loss function ('mse', 'mae', 'zptae')
            h: Forecast horizon (for HAC correction)
            return_details: Whether to return detailed results
            
        Returns:
            Dictionary with test results
        """
        logging.info(f"Performing Diebold-Mariano test with {loss_function} loss")
        
        # Calculate loss differentials
        if loss_function == 'mse':
            loss1 = (y_true - pred1) ** 2
            loss2 = (y_true - pred2) ** 2
        elif loss_function == 'mae':
            loss1 = np.abs(y_true - pred1)
            loss2 = np.abs(y_true - pred2)
        elif loss_function == 'zptae':
            # Use ZPTAE loss (requires additional parameters)
            from ..models.losses import zptae_metric_numpy
            zptae_params = self.config.get('loss_functions', {}).get('zptae', {})
            a = zptae_params.get('a', 1.0)
            p = zptae_params.get('p', 1.5)
            
            # Calculate ZPTAE for each prediction
            loss1 = np.array([zptae_metric_numpy(np.array([yt]), np.array([p1]), a, p) 
                             for yt, p1 in zip(y_true, pred1)])
            loss2 = np.array([zptae_metric_numpy(np.array([yt]), np.array([p2]), a, p) 
                             for yt, p2 in zip(y_true, pred2)])
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
        
        # Loss differential
        d = loss1 - loss2
        
        # Mean loss differential
        d_bar = np.mean(d)
        
        # Calculate variance with HAC correction for autocorrelation
        n = len(d)
        
        if h == 1:
            # Simple variance for 1-step ahead forecasts
            var_d = np.var(d, ddof=1) / n
        else:
            # HAC variance correction for multi-step forecasts
            var_d = self._calculate_hac_variance(d, h)
        
        # Test statistic
        if var_d > 0:
            dm_stat = d_bar / np.sqrt(var_d)
        else:
            dm_stat = 0
            logging.warning("Zero variance in Diebold-Mariano test")
        
        # P-value (two-tailed test)
        p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
        
        # Critical value
        critical_value = norm.ppf(1 - self.alpha / 2)
        
        # Test result
        is_significant = abs(dm_stat) > critical_value
        
        results = {
            'test_name': 'Diebold-Mariano',
            'test_statistic': dm_stat,
            'p_value': p_value,
            'critical_value': critical_value,
            'is_significant': is_significant,
            'mean_loss_differential': d_bar,
            'loss_function': loss_function,
            'confidence_level': self.confidence_level
        }
        
        if return_details:
            results.update({
                'n_observations': n,
                'forecast_horizon': h,
                'variance_loss_differential': var_d,
                'model1_mean_loss': np.mean(loss1),
                'model2_mean_loss': np.mean(loss2),
                'interpretation': self._interpret_dm_test(results)
            })
        
        logging.info(f"Diebold-Mariano test completed: stat={dm_stat:.4f}, p-value={p_value:.4f}")
        
        return results
    
    def _calculate_hac_variance(self, d: np.ndarray, h: int) -> float:
        """
        Calculate HAC (Heteroskedasticity and Autocorrelation Consistent) variance.
        
        Args:
            d: Loss differential series
            h: Forecast horizon
            
        Returns:
            HAC variance estimate
        """
        n = len(d)
        d_bar = np.mean(d)
        
        # Variance
        gamma_0 = np.mean((d - d_bar) ** 2)
        
        # Autocovariances
        gamma_sum = 0
        for j in range(1, h):
            if j < n:
                gamma_j = np.mean((d[:-j] - d_bar) * (d[j:] - d_bar))
                gamma_sum += 2 * gamma_j
        
        var_d = (gamma_0 + gamma_sum) / n
        
        return max(var_d, 1e-10)  # Ensure positive variance
    
    def superior_predictive_ability_test(self, y_true: np.ndarray, 
                                       predictions: Dict[str, np.ndarray],
                                       benchmark_model: str,
                                       loss_function: str = 'mse',
                                       bootstrap_samples: int = 1000) -> Dict[str, Any]:
        """
        Superior Predictive Ability (SPA) test by Hansen (2005).
        
        Tests whether any model is significantly better than a benchmark.
        
        Args:
            y_true: True values
            predictions: Dictionary of model predictions
            benchmark_model: Name of benchmark model
            loss_function: Loss function to use
            bootstrap_samples: Number of bootstrap samples
            
        Returns:
            Dictionary with test results
        """
        logging.info(f"Performing SPA test with benchmark: {benchmark_model}")
        
        if benchmark_model not in predictions:
            raise ValueError(f"Benchmark model {benchmark_model} not found in predictions")
        
        benchmark_pred = predictions[benchmark_model]
        n = len(y_true)
        
        # Calculate loss differentials for all models vs benchmark
        loss_diffs = {}
        
        for model_name, pred in predictions.items():
            if model_name == benchmark_model:
                continue
                
            if loss_function == 'mse':
                loss_benchmark = (y_true - benchmark_pred) ** 2
                loss_model = (y_true - pred) ** 2
            elif loss_function == 'mae':
                loss_benchmark = np.abs(y_true - benchmark_pred)
                loss_model = np.abs(y_true - pred)
            else:
                raise ValueError(f"Loss function {loss_function} not implemented for SPA test")
            
            loss_diffs[model_name] = loss_benchmark - loss_model
        
        # Calculate test statistic
        max_t_stat = -np.inf
        
        for model_name, diff in loss_diffs.items():
            d_bar = np.mean(diff)
            var_d = np.var(diff, ddof=1) / n
            
            if var_d > 0:
                t_stat = d_bar / np.sqrt(var_d)
                max_t_stat = max(max_t_stat, t_stat)
        
        # Bootstrap procedure
        bootstrap_stats = []
        
        for _ in range(bootstrap_samples):
            # Resample with replacement
            indices = np.random.choice(n, size=n, replace=True)
            
            max_bootstrap_stat = -np.inf
            
            for model_name, diff in loss_diffs.items():
                diff_bootstrap = diff[indices]
                d_bar_bootstrap = np.mean(diff_bootstrap)
                var_d_bootstrap = np.var(diff_bootstrap, ddof=1) / n
                
                if var_d_bootstrap > 0:
                    t_stat_bootstrap = d_bar_bootstrap / np.sqrt(var_d_bootstrap)
                    max_bootstrap_stat = max(max_bootstrap_stat, t_stat_bootstrap)
            
            bootstrap_stats.append(max_bootstrap_stat)
        
        # Calculate p-value
        bootstrap_stats = np.array(bootstrap_stats)
        p_value = np.mean(bootstrap_stats >= max_t_stat)
        
        results = {
            'test_name': 'Superior Predictive Ability',
            'test_statistic': max_t_stat,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'benchmark_model': benchmark_model,
            'n_competing_models': len(loss_diffs),
            'bootstrap_samples': bootstrap_samples,
            'confidence_level': self.confidence_level
        }
        
        logging.info(f"SPA test completed: max_t={max_t_stat:.4f}, p-value={p_value:.4f}")
        
        return results
    
    def reality_check_test(self, y_true: np.ndarray, 
                          predictions: Dict[str, np.ndarray],
                          benchmark_model: str,
                          loss_function: str = 'mse',
                          bootstrap_samples: int = 1000) -> Dict[str, Any]:
        """
        Reality Check test by White (2000).
        
        Tests whether any model significantly outperforms a benchmark.
        
        Args:
            y_true: True values
            predictions: Dictionary of model predictions
            benchmark_model: Name of benchmark model
            loss_function: Loss function to use
            bootstrap_samples: Number of bootstrap samples
            
        Returns:
            Dictionary with test results
        """
        logging.info(f"Performing Reality Check test with benchmark: {benchmark_model}")
        
        # Similar to SPA test but with different null hypothesis
        spa_results = self.superior_predictive_ability_test(
            y_true, predictions, benchmark_model, loss_function, bootstrap_samples
        )
        
        # Modify interpretation for Reality Check
        spa_results['test_name'] = 'Reality Check'
        spa_results['interpretation'] = self._interpret_rc_test(spa_results)
        
        return spa_results
    
    def encompassing_test(self, y_true: np.ndarray, pred1: np.ndarray, 
                         pred2: np.ndarray, return_details: bool = True) -> Dict[str, Any]:
        """
        Forecast encompassing test.
        
        Tests whether forecast 1 encompasses forecast 2 (i.e., forecast 2
        provides no additional information beyond forecast 1).
        
        Args:
            y_true: True values
            pred1: Predictions from model 1
            pred2: Predictions from model 2
            return_details: Whether to return detailed results
            
        Returns:
            Dictionary with test results
        """
        logging.info("Performing forecast encompassing test")
        
        # Regression: y_true = alpha + beta1 * pred1 + beta2 * pred2 + error
        # H0: beta2 = 0 (model 2 is encompassed by model 1)
        
        n = len(y_true)
        X = np.column_stack([np.ones(n), pred1, pred2])
        
        try:
            # OLS estimation
            beta = np.linalg.solve(X.T @ X, X.T @ y_true)
            
            # Residuals
            residuals = y_true - X @ beta
            
            # Standard errors
            mse = np.sum(residuals ** 2) / (n - 3)
            var_beta = mse * np.linalg.inv(X.T @ X)
            se_beta2 = np.sqrt(var_beta[2, 2])
            
            # Test statistic for beta2 = 0
            t_stat = beta[2] / se_beta2
            
            # P-value (two-tailed)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-3))
            
            # Critical value
            critical_value = stats.t.ppf(1 - self.alpha / 2, df=n-3)
            
            is_significant = abs(t_stat) > critical_value
            
            results = {
                'test_name': 'Forecast Encompassing',
                'test_statistic': t_stat,
                'p_value': p_value,
                'critical_value': critical_value,
                'is_significant': is_significant,
                'beta2_coefficient': beta[2],
                'beta2_std_error': se_beta2,
                'confidence_level': self.confidence_level
            }
            
            if return_details:
                results.update({
                    'n_observations': n,
                    'beta_coefficients': beta,
                    'mse': mse,
                    'r_squared': 1 - np.sum(residuals ** 2) / np.sum((y_true - np.mean(y_true)) ** 2),
                    'interpretation': self._interpret_encompassing_test(results)
                })
            
        except np.linalg.LinAlgError:
            logging.error("Singular matrix in encompassing test")
            results = {
                'test_name': 'Forecast Encompassing',
                'error': 'Singular matrix - cannot perform test',
                'is_significant': False
            }
        
        logging.info(f"Encompassing test completed: t-stat={results.get('test_statistic', 'N/A')}")
        
        return results
    
    def _interpret_pt_test(self, results: Dict[str, Any]) -> str:
        """Interpret Pesaran-Timmermann test results."""
        da = results['directional_accuracy']
        expected_da = results['expected_accuracy_under_independence']
        is_sig = results['is_significant']
        p_val = results['p_value']
        
        interpretation = f"Directional accuracy: {da:.4f} vs expected {expected_da:.4f} under independence. "
        
        if is_sig:
            if da > expected_da:
                interpretation += f"Significantly better than random (p={p_val:.4f}). "
                interpretation += "Model has genuine forecasting ability."
            else:
                interpretation += f"Significantly worse than random (p={p_val:.4f}). "
                interpretation += "Model performs worse than chance."
        else:
            interpretation += f"Not significantly different from random (p={p_val:.4f}). "
            interpretation += "No evidence of forecasting ability."
        
        return interpretation
    
    def _interpret_dm_test(self, results: Dict[str, Any]) -> str:
        """Interpret Diebold-Mariano test results."""
        dm_stat = results['test_statistic']
        is_sig = results['is_significant']
        p_val = results['p_value']
        
        interpretation = f"DM statistic: {dm_stat:.4f}. "
        
        if is_sig:
            if dm_stat > 0:
                interpretation += f"Model 1 significantly worse than Model 2 (p={p_val:.4f})."
            else:
                interpretation += f"Model 1 significantly better than Model 2 (p={p_val:.4f})."
        else:
            interpretation += f"No significant difference between models (p={p_val:.4f})."
        
        return interpretation
    
    def _interpret_rc_test(self, results: Dict[str, Any]) -> str:
        """Interpret Reality Check test results."""
        p_val = results['p_value']
        is_sig = results['is_significant']
        
        if is_sig:
            interpretation = f"At least one model significantly outperforms benchmark (p={p_val:.4f})."
        else:
            interpretation = f"No model significantly outperforms benchmark (p={p_val:.4f})."
        
        return interpretation
    
    def _interpret_encompassing_test(self, results: Dict[str, Any]) -> str:
        """Interpret encompassing test results."""
        is_sig = results['is_significant']
        p_val = results['p_value']
        beta2 = results['beta2_coefficient']
        
        if is_sig:
            interpretation = f"Model 2 provides significant additional information (β₂={beta2:.4f}, p={p_val:.4f}). "
            interpretation += "Model 1 does not encompass Model 2."
        else:
            interpretation = f"Model 2 provides no additional information (β₂={beta2:.4f}, p={p_val:.4f}). "
            interpretation += "Model 1 encompasses Model 2."
        
        return interpretation
    
    def run_comprehensive_tests(self, y_true: np.ndarray, 
                              predictions: Dict[str, np.ndarray],
                              benchmark_model: str = None) -> Dict[str, Any]:
        """
        Run comprehensive statistical tests.
        
        Args:
            y_true: True values
            predictions: Dictionary of model predictions
            benchmark_model: Benchmark model name (if None, use first model)
            
        Returns:
            Dictionary with all test results
        """
        logging.info("Running comprehensive statistical tests")
        
        if benchmark_model is None:
            benchmark_model = list(predictions.keys())[0]
        
        results = {
            'benchmark_model': benchmark_model,
            'n_models': len(predictions),
            'n_observations': len(y_true)
        }
        
        # Pesaran-Timmermann tests for each model
        results['pesaran_timmermann'] = {}
        for model_name, pred in predictions.items():
            pt_result = self.pesaran_timmermann_test(y_true, pred)
            results['pesaran_timmermann'][model_name] = pt_result
        
        # Pairwise Diebold-Mariano tests
        results['diebold_mariano'] = {}
        model_names = list(predictions.keys())
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                pair_name = f"{model1}_vs_{model2}"
                dm_result = self.diebold_mariano_test(
                    y_true, predictions[model1], predictions[model2]
                )
                results['diebold_mariano'][pair_name] = dm_result
        
        # SPA test (if more than one model)
        if len(predictions) > 1:
            spa_result = self.superior_predictive_ability_test(
                y_true, predictions, benchmark_model
            )
            results['spa_test'] = spa_result
        
        # Encompassing tests (pairwise)
        results['encompassing'] = {}
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                pair_name = f"{model1}_encompasses_{model2}"
                enc_result = self.encompassing_test(
                    y_true, predictions[model1], predictions[model2]
                )
                results['encompassing'][pair_name] = enc_result
        
        logging.info("Comprehensive statistical tests completed")
        
        return results


def main():
    """
    Main function for testing statistical tests.
    """
    # Load configuration
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.utils.helpers import load_config, setup_logging
    
    # Setup
    config = load_config()
    setup_logging()
    
    # Create synthetic data for testing
    np.random.seed(42)
    n = 500
    
    # True returns with some predictable pattern
    true_returns = np.random.normal(0, 0.02, n)
    true_returns[100:200] += 0.01  # Add some trend
    
    # Model predictions
    # Model 1: Good model with some skill
    pred1 = true_returns + np.random.normal(0, 0.01, n)
    
    # Model 2: Random model
    pred2 = np.random.normal(0, 0.02, n)
    
    # Model 3: Slightly better than random
    pred3 = 0.3 * true_returns + np.random.normal(0, 0.015, n)
    
    predictions = {
        'model1': pred1,
        'model2': pred2,
        'model3': pred3
    }
    
    # Initialize tester
    tester = StatisticalTester(config)
    
    # Run comprehensive tests
    results = tester.run_comprehensive_tests(true_returns, predictions, 'model2')
    
    # Print results
    print("Statistical Test Results:")
    print("=" * 50)
    
    # Pesaran-Timmermann results
    print("\nPesaran-Timmermann Tests:")
    for model, pt_result in results['pesaran_timmermann'].items():
        da = pt_result['directional_accuracy']
        p_val = pt_result['p_value']
        sig = "***" if pt_result['is_significant'] else ""
        print(f"  {model}: DA={da:.4f}, p-value={p_val:.4f} {sig}")
    
    # Diebold-Mariano results
    print("\nDiebold-Mariano Tests:")
    for pair, dm_result in results['diebold_mariano'].items():
        stat = dm_result['test_statistic']
        p_val = dm_result['p_value']
        sig = "***" if dm_result['is_significant'] else ""
        print(f"  {pair}: DM={stat:.4f}, p-value={p_val:.4f} {sig}")
    
    # SPA test
    if 'spa_test' in results:
        spa = results['spa_test']
        print(f"\nSPA Test: stat={spa['test_statistic']:.4f}, p-value={spa['p_value']:.4f}")


if __name__ == "__main__":
    main()