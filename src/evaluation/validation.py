"""
Validation module for ETH forecasting project.

This module implements walk-forward cross-validation following Rule #13:
- USE expanding window training (no sliding window)
- MAINTAIN fixed validation horizon
- ENSURE no data contamination between folds
- GENERATE 8-12 folds minimum
- CALCULATE fold-wise metrics and stability measures

Includes comprehensive validation strategies and metrics calculation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Callable
from datetime import datetime, timedelta
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=RuntimeWarning)


class WalkForwardValidator:
    """
    Walk-forward cross-validation implementation.
    
    Implements expanding window training with strict temporal ordering
    to prevent data leakage and ensure realistic validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize WalkForwardValidator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.validation_config = config.get('validation', {})
        
        # Validation parameters
        self.min_train_size = self.validation_config.get('min_train_size', 252)  # 1 year
        self.validation_horizon = self.validation_config.get('validation_horizon', 30)  # 30 days
        self.step_size = self.validation_config.get('step_size', 30)  # 30 days
        self.min_folds = self.validation_config.get('min_folds', 8)
        self.max_folds = self.validation_config.get('max_folds', 12)
        
        # Quality checks
        self.enable_leakage_detection = True
        self.enable_overlap_detection = True
        
        logging.info("WalkForwardValidator initialized")
    
    def create_folds(self, data: pd.DataFrame, 
                    date_column: str = 'date') -> List[Dict[str, Any]]:
        """
        Create walk-forward cross-validation folds.
        
        Args:
            data: Input data with datetime index or column
            date_column: Name of date column (if not index)
            
        Returns:
            List of fold dictionaries with train/validation indices
        """
        logging.info("Creating walk-forward cross-validation folds")
        
        # Ensure data is sorted by date
        if date_column in data.columns:
            data = data.sort_values(date_column).reset_index(drop=True)
            dates = data[date_column]
        else:
            data = data.sort_index()
            dates = data.index
        
        n_samples = len(data)
        
        # Calculate fold parameters
        max_possible_folds = (n_samples - self.min_train_size) // self.step_size
        n_folds = min(max_possible_folds, self.max_folds)
        n_folds = max(n_folds, self.min_folds)
        
        if n_folds < self.min_folds:
            raise ValueError(f"Insufficient data for {self.min_folds} folds. "
                           f"Need at least {self.min_train_size + self.min_folds * self.step_size} samples")
        
        folds = []
        
        for fold_idx in range(n_folds):
            # Calculate indices
            train_end_idx = self.min_train_size + fold_idx * self.step_size
            val_start_idx = train_end_idx
            val_end_idx = min(val_start_idx + self.validation_horizon, n_samples)
            
            # Skip if validation set is too small
            if val_end_idx - val_start_idx < 5:
                continue
            
            # Create fold
            # Handle both Series (from column) and DatetimeIndex (from index) cases
            if hasattr(dates, 'iloc'):
                # dates is a Series
                train_start_date = dates.iloc[0]
                train_end_date = dates.iloc[train_end_idx - 1]
                val_start_date = dates.iloc[val_start_idx]
                val_end_date = dates.iloc[val_end_idx - 1]
            else:
                # dates is a DatetimeIndex
                train_start_date = dates[0]
                train_end_date = dates[train_end_idx - 1]
                val_start_date = dates[val_start_idx]
                val_end_date = dates[val_end_idx - 1]
            
            fold = {
                'fold_id': fold_idx,
                'train_start_idx': 0,
                'train_end_idx': train_end_idx,
                'val_start_idx': val_start_idx,
                'val_end_idx': val_end_idx,
                'train_indices': list(range(0, train_end_idx)),
                'val_indices': list(range(val_start_idx, val_end_idx)),
                'train_start_date': train_start_date,
                'train_end_date': train_end_date,
                'val_start_date': val_start_date,
                'val_end_date': val_end_date,
                'train_size': train_end_idx,
                'val_size': val_end_idx - val_start_idx
            }
            
            folds.append(fold)
        
        # Quality checks
        self._validate_folds(folds, data)
        
        logging.info(f"Created {len(folds)} walk-forward CV folds")
        
        return folds
    
    def _validate_folds(self, folds: List[Dict[str, Any]], 
                       data: pd.DataFrame) -> None:
        """
        Validate fold integrity and detect potential issues.
        
        Args:
            folds: List of fold dictionaries
            data: Original data
        """
        logging.info("Validating fold integrity")
        
        # Check for overlaps between train and validation sets
        if self.enable_overlap_detection:
            for fold in folds:
                train_indices = set(fold['train_indices'])
                val_indices = set(fold['val_indices'])
                
                overlap = train_indices.intersection(val_indices)
                if overlap:
                    raise ValueError(f"Fold {fold['fold_id']} has overlapping train/val indices: {overlap}")
        
        # Check temporal ordering
        for fold in folds:
            if fold['train_end_date'] > fold['val_start_date']:
                raise ValueError(f"Fold {fold['fold_id']} violates temporal ordering")
        
        # Check minimum sizes
        for fold in folds:
            if fold['train_size'] < self.min_train_size:
                raise ValueError(f"Fold {fold['fold_id']} train size too small: {fold['train_size']}")
            
            if fold['val_size'] < 5:
                logging.warning(f"Fold {fold['fold_id']} validation size very small: {fold['val_size']}")
        
        # Check for data leakage (future information in features)
        if self.enable_leakage_detection:
            self._detect_potential_leakage(folds, data)
        
        logging.info("Fold validation completed successfully")
    
    def _detect_potential_leakage(self, folds: List[Dict[str, Any]], 
                                 data: pd.DataFrame) -> None:
        """
        Detect potential data leakage in features.
        
        Args:
            folds: List of fold dictionaries
            data: Original data
        """
        logging.info("Detecting potential data leakage")
        
        # Check for features that might contain future information
        suspicious_patterns = [
            'future_', 'next_', 'tomorrow_', 'ahead_',
            '_t+1', '_t+2', '_lead', '_forward'
        ]
        
        feature_columns = [col for col in data.columns 
                          if col not in ['date', 'target', 'direction']]
        
        for col in feature_columns:
            col_lower = col.lower()
            for pattern in suspicious_patterns:
                if pattern in col_lower:
                    logging.warning(f"Suspicious feature name detected: {col}")
        
        # Statistical test for leakage
        for fold in folds[:3]:  # Check first 3 folds
            train_data = data.iloc[fold['train_indices']]
            val_data = data.iloc[fold['val_indices']]
            
            if 'target' in data.columns:
                # Check correlation between features and future targets
                for col in feature_columns:
                    # Only consider numeric columns for correlation
                    if col in train_data.columns:
                        series_train = train_data[col]
                        if not pd.api.types.is_numeric_dtype(series_train):
                            continue
                        if series_train.isna().all():
                            continue
                        # Correlation with next period target
                        if len(val_data) > 0 and 'target' in val_data.columns:
                            series_future_target = val_data['target']
                            if not pd.api.types.is_numeric_dtype(series_future_target):
                                continue
                            try:
                                corr = np.corrcoef(
                                    series_train.iloc[-len(val_data):].fillna(0).astype(float),
                                    series_future_target.fillna(0).astype(float)
                                )[0, 1]
                                if np.isfinite(corr) and abs(corr) > 0.8:  # High correlation threshold
                                    logging.warning(f"High correlation between {col} and future target: {corr:.4f}")
                            except Exception as e:
                                logging.debug(f"Correlation check skipped for column {col} due to error: {e}")
        
        logging.info("Data leakage detection completed")
    
    def validate_model(self, model: Any, X: pd.DataFrame, y: pd.Series,
                      folds: List[Dict[str, Any]], 
                      metrics: List[str] = None,
                      fit_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Validate model using walk-forward cross-validation.
        
        Args:
            model: Model to validate (must have fit/predict methods)
            X: Feature matrix
            y: Target variable
            folds: Cross-validation folds
            metrics: List of metrics to calculate
            fit_params: Additional parameters for model fitting
            
        Returns:
            Dictionary with validation results
        """
        logging.info("Starting walk-forward cross-validation")
        
        if metrics is None:
            metrics = ['mse', 'mae', 'directional_accuracy', 'zptae']
        
        if fit_params is None:
            fit_params = {}
        
        # Rule #16: Quality Gate Checks (NON-NEGOTIABLE)
        self._run_quality_gate_checks(X, y)
        
        # Initialize results storage
        fold_results = []
        all_predictions = []
        all_actuals = []
        
        # Feature scaling
        scaler = StandardScaler()
        
        for fold_idx, fold in enumerate(folds):
            logging.info(f"Processing fold {fold_idx + 1}/{len(folds)}")
            
            try:
                # Get fold data
                X_train = X.iloc[fold['train_indices']].copy()
                y_train = y.iloc[fold['train_indices']].copy()
                X_val = X.iloc[fold['val_indices']].copy()
                y_val = y.iloc[fold['val_indices']].copy()
                
                # Handle missing values
                X_train = X_train.fillna(X_train.mean())
                X_val = X_val.fillna(X_train.mean())  # Use training mean
                
                # Scale features (fit on train, transform both)
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Convert back to DataFrame to maintain column names
                X_train_scaled = pd.DataFrame(X_train_scaled, 
                                            columns=X_train.columns,
                                            index=X_train.index)
                X_val_scaled = pd.DataFrame(X_val_scaled,
                                          columns=X_val.columns,
                                          index=X_val.index)
                
                # Fit model
                if hasattr(model, 'fit'):
                    model.fit(X_train_scaled, y_train, **fit_params)
                else:
                    raise ValueError("Model must have a 'fit' method")
                
                # Make predictions
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_val_scaled)
                    
                    # Handle tuple predictions (LightGBM returns tuple)
                    if isinstance(y_pred, tuple):
                        y_pred_reg, y_pred_dir = y_pred
                        # Use regression predictions for metrics
                        y_pred = y_pred_reg
                    else:
                        y_pred_dir = None
                else:
                    raise ValueError("Model must have a 'predict' method")
                
                # Ensure y_pred is 1D numpy array
                y_pred = np.asarray(y_pred).reshape(-1)
                y_val_array = np.asarray(y_val.values).reshape(-1)
                
                # Align lengths - use minimum length
                min_len = min(len(y_pred), len(y_val_array))
                if min_len == 0:
                    logging.warning(f"Fold {fold_idx}: Empty predictions or targets")
                    continue
                    
                y_pred = y_pred[:min_len]
                y_val_array = y_val_array[:min_len]
                
                # Rule #16: Post-prediction quality checks
                self._validate_predictions(y_pred, y_val_array)
                    
                # Calculate metrics for this fold
                fold_metrics = self._calculate_metrics(y_val_array, y_pred, metrics)
                
                # Store fold results
                fold_result = {
                    'fold_id': fold['fold_id'],
                    'train_size': fold['train_size'],
                    'val_size': fold['val_size'],
                    'train_period': f"{fold['train_start_date']} to {fold['train_end_date']}",
                    'val_period': f"{fold['val_start_date']} to {fold['val_end_date']}",
                    **fold_metrics
                }
                
                fold_results.append(fold_result)
                
                # Store predictions for overall metrics
                y_pred_array = np.array(y_pred).reshape(-1)
                y_true_array = np.array(y_val.values).reshape(-1)
                all_predictions.extend(y_pred_array.tolist())
                all_actuals.extend(y_true_array.tolist())
                
            except Exception as e:
                logging.error(f"Error in fold {fold_idx}: {str(e)}")
                # Store error information
                fold_result = {
                    'fold_id': fold['fold_id'],
                    'error': str(e),
                    'train_size': fold['train_size'],
                    'val_size': fold['val_size']
                }
                fold_results.append(fold_result)
                continue
        
        # Calculate overall metrics
        if all_predictions and all_actuals:
            y_true_all = np.array(all_actuals).reshape(-1)
            y_pred_all = np.array(all_predictions).reshape(-1)
            # Align lengths if needed
            min_len = min(len(y_true_all), len(y_pred_all))
            y_true_all = y_true_all[:min_len]
            y_pred_all = y_pred_all[:min_len]
            
            # Final quality gate check on overall predictions
            self._validate_predictions(y_pred_all, y_true_all)
            
            overall_metrics = self._calculate_metrics(
                y_true_all,
                y_pred_all,
                metrics
            )
        else:
            overall_metrics = {}
        
        # Calculate stability metrics
        stability_metrics = self._calculate_stability_metrics(fold_results, metrics)
        
        # Rule #15: Performance expectations check
        self._check_performance_expectations(overall_metrics, stability_metrics)
        
        # Rule #14: Statistical testing (if sufficient data)
        statistical_tests = {}
        if len(all_predictions) >= 30:  # Minimum for meaningful tests
            statistical_tests = self._run_statistical_tests(all_actuals, all_predictions)
        
        # Compile results
        results = {
            'n_folds': len(folds),
            'successful_folds': len([f for f in fold_results if 'error' not in f]),
            'fold_results': fold_results,
            'overall_metrics': overall_metrics,
            'stability_metrics': stability_metrics,
            'statistical_tests': statistical_tests,  # New
            'validation_config': {
                'min_train_size': self.min_train_size,
                'validation_horizon': self.validation_horizon,
                'step_size': self.step_size
            }
        }
        
        logging.info("Walk-forward cross-validation completed")
        
        return results
    
    def _run_statistical_tests(self, y_true: List[float], y_pred: List[float]) -> Dict[str, Any]:
        """
        Run statistical tests per Rule #14.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with statistical test results
        """
        logging.info("Running statistical tests (Rule #14)")
        
        try:
            from .statistical_tests import StatisticalTester
            
            # Convert to numpy arrays
            y_true_arr = np.array(y_true)
            y_pred_arr = np.array(y_pred)
            
            # Initialize statistical tester
            tester = StatisticalTester(self.config)
            
            # Pesaran-Timmermann test for directional accuracy
            pt_result = tester.pesaran_timmermann_test(y_true_arr, y_pred_arr)
            
            # Create benchmark predictions (random walk/persistence)
            benchmark_pred = np.roll(y_true_arr, 1)  # Lagged values as benchmark
            benchmark_pred[0] = 0  # Handle first value
            
            # Diebold-Mariano test vs benchmark
            dm_result = tester.diebold_mariano_test(
                y_true_arr, y_pred_arr, benchmark_pred, loss_function='mse'
            )
            
            results = {
                'pesaran_timmermann': pt_result,
                'diebold_mariano_vs_benchmark': dm_result,
                'interpretation': self._interpret_statistical_tests(pt_result, dm_result)
            }
            
            logging.info(f"Statistical tests completed")
            logging.info(f"  PT test p-value: {pt_result['p_value']:.4f}")
            logging.info(f"  DM test p-value: {dm_result['p_value']:.4f}")
            
            return results
            
        except Exception as e:
            logging.error(f"Error running statistical tests: {str(e)}")
            return {'error': str(e)}
    
    def _interpret_statistical_tests(self, pt_result: Dict[str, Any], 
                                   dm_result: Dict[str, Any]) -> str:
        """
        Interpret statistical test results.
        
        Args:
            pt_result: Pesaran-Timmermann test result
            dm_result: Diebold-Mariano test result
            
        Returns:
            Human-readable interpretation
        """
        interpretation = []
        
        # Pesaran-Timmermann interpretation
        pt_pval = pt_result.get('p_value', 1.0)
        pt_da = pt_result.get('directional_accuracy', 0.5)
        pt_sig = pt_result.get('is_significant', False)
        
        if pt_sig and pt_pval < 0.05:
            interpretation.append(f"✅ Model shows significant directional forecasting ability (DA={pt_da:.3f}, p={pt_pval:.4f})")
        elif pt_da > 0.52:
            interpretation.append(f"⚠️  Model shows modest directional accuracy (DA={pt_da:.3f}) but not statistically significant (p={pt_pval:.4f})")
        else:
            interpretation.append(f"❌ Model lacks directional forecasting ability (DA={pt_da:.3f}, p={pt_pval:.4f})")
        
        # Diebold-Mariano interpretation
        dm_pval = dm_result.get('p_value', 1.0)
        dm_sig = dm_result.get('is_significant', False)
        dm_diff = dm_result.get('mean_loss_differential', 0)
        
        if dm_sig and dm_pval < 0.05:
            if dm_diff < 0:
                interpretation.append(f"✅ Model significantly outperforms benchmark (p={dm_pval:.4f})")
            else:
                interpretation.append(f"❌ Model significantly underperforms benchmark (p={dm_pval:.4f})")
        else:
            interpretation.append(f"➡️  No significant difference vs benchmark (p={dm_pval:.4f})")
        
        # Combined assessment
        if pt_sig and dm_sig and dm_diff < 0:
            interpretation.append("🎯 STRONG: Model shows both directional skill and superior accuracy")
        elif pt_sig or (dm_sig and dm_diff < 0):
            interpretation.append("👍 MODERATE: Model shows some forecasting skill")
        else:
            interpretation.append("❓ WEAK: Limited evidence of forecasting skill")
        
        return " | ".join(interpretation)

    def _run_quality_gate_checks(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Run comprehensive quality gate checks before validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Raises:
            ValueError: If critical quality checks fail
        """
        logging.info("Running quality gate checks (Rule #16)")
        
        try:
            # Check 1: Basic data integrity
            if X.empty or y.empty:
                raise ValueError("Empty features or target data")
            
            if len(X) != len(y):
                raise ValueError(f"Feature ({len(X)}) and target ({len(y)}) length mismatch")
                
            # Check 2: No data leakage detection
            if not self._no_data_leakage_detected(X, y):
                raise ValueError("CRITICAL: Data leakage detected! Training halted.")
            
            # Check 3: Features properly scaled
            if not self._all_features_properly_scaled(X):
                logging.warning("Some features may not be properly scaled")
            
            # Check 4: Rolling windows have sufficient samples
            if not self._rolling_std_has_sufficient_samples(X):
                logging.warning("Some rolling features may have insufficient samples")
            
            # Check 5: No NaN in critical columns
            critical_nans = X.isna().sum().sum() + y.isna().sum()
            if critical_nans > 0:
                logging.warning(f"Found {critical_nans} NaN values in features/targets")
                
            logging.info("✅ Quality gate checks passed")
            
        except Exception as e:
            logging.error(f"❌ Quality gate check failed: {str(e)}")
            raise ValueError(f"Quality gate failure: {str(e)}")
    
    def _no_data_leakage_detected(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Check for data leakage using basic heuristics."""
        try:
            # Check 1: No future information in feature names
            suspicious_patterns = ['future_', 'next_', 'tomorrow_', 'ahead_', '_t+1', '_t+2']
            for col in X.columns:
                if any(pattern in col.lower() for pattern in suspicious_patterns):
                    logging.error(f"Suspicious feature name: {col}")
                    return False
            
            # Check 2: No perfect correlation (>0.99) with target
            for col in X.select_dtypes(include=[np.number]).columns:
                if not X[col].isna().all():
                    corr = np.corrcoef(X[col].fillna(0), y.fillna(0))[0, 1]
                    if abs(corr) > 0.99:
                        logging.error(f"Suspiciously high correlation: {col} -> {corr:.4f}")
                        return False
            
            return True
        except Exception as e:
            logging.error(f"Error in leakage detection: {str(e)}")
            return False
    
    def _all_features_properly_scaled(self, X: pd.DataFrame) -> bool:
        """Check if features are properly scaled."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        scaling_issues = 0
        
        for col in numeric_cols:
            if not X[col].isna().all():
                col_std = X[col].std()
                col_mean = X[col].mean()
                
                # Check for unscaled features (high variance)
                if col_std > 100:
                    logging.warning(f"Feature {col} may be unscaled (std={col_std:.2f})")
                    scaling_issues += 1
                    
                # Check for constant features
                if col_std == 0:
                    logging.warning(f"Feature {col} is constant")
                    scaling_issues += 1
        
        return scaling_issues == 0
    
    def _rolling_std_has_sufficient_samples(self, X: pd.DataFrame) -> bool:
        """Check if rolling window features have sufficient samples."""
        rolling_patterns = ['_ma_', '_std_', '_rolling_', '_ewm_']
        issues = 0
        
        for col in X.columns:
            if any(pattern in col.lower() for pattern in rolling_patterns):
                first_valid = X[col].first_valid_index()
                if first_valid is not None:
                    n_initial_nans = X.index.get_loc(first_valid)
                    if n_initial_nans < 10:  # Expect at least 10 initial NaNs for rolling features
                        logging.warning(f"Rolling feature {col} has only {n_initial_nans} initial NaNs")
                        issues += 1
        
        return issues == 0
    
    def _validate_predictions(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """
        Validate prediction quality.
        
        Args:
            y_pred: Predictions
            y_true: True values
            
        Raises:
            ValueError: If predictions fail quality checks
        """
        # Check for NaN predictions
        if np.isnan(y_pred).any():
            raise ValueError("NaN values found in predictions")
        
        # Check for infinite predictions
        if np.isinf(y_pred).any():
            raise ValueError("Infinite values found in predictions")
        
        # Check prediction range (for log returns, should be reasonable)
        pred_std = np.std(y_pred)
        if pred_std > 1.0:
            logging.warning(f"High prediction variance: std={pred_std:.4f}")
        
        # Check for constant predictions
        if pred_std == 0:
            logging.warning("All predictions are identical")
    
    def _check_performance_expectations(self, overall_metrics: Dict[str, float], 
                                      stability_metrics: Dict[str, float]) -> None:
        """
        Check performance against Rule #15 expectations.
        
        Args:
            overall_metrics: Overall validation metrics
            stability_metrics: Cross-fold stability metrics
        """
        logging.info("Checking performance expectations (Rule #15)")
        
        # Check directional accuracy
        da = overall_metrics.get('directional_accuracy', 0)
        if da > 0.70:
            logging.warning(f"⚠️  SUSPICIOUS: High DA ({da:.3f}) - check for data leakage!")
        elif da < 0.45:
            logging.warning(f"⚠️  LOW: Directional accuracy ({da:.3f}) below random")
        elif 0.52 <= da <= 0.58:
            logging.info(f"✅ GOOD: Directional accuracy ({da:.3f}) in expected range")
        
        # Check stability
        da_std = stability_metrics.get('directional_accuracy_std', 0)
        if da_std > 0.10:
            logging.warning(f"⚠️  UNSTABLE: High DA std ({da_std:.3f}) across folds")
        
        # Check ZPTAE convergence
        zptae = overall_metrics.get('zptae', np.nan)
        if not np.isnan(zptae):
            if zptae < 0:
                logging.info(f"✅ ZPTAE converged: {zptae:.6f}")
            else:
                logging.warning(f"⚠️  ZPTAE did not converge properly: {zptae:.6f}")

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          metrics: List[str]) -> Dict[str, float]:
        """
        Calculate specified metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metrics: List of metric names
            
        Returns:
            Dictionary of calculated metrics
        """
        results = {}
        
        for metric in metrics:
            try:
                if metric == 'mse':
                    results[metric] = mean_squared_error(y_true, y_pred)
                elif metric == 'mae':
                    results[metric] = mean_absolute_error(y_true, y_pred)
                elif metric == 'rmse':
                    results[metric] = np.sqrt(mean_squared_error(y_true, y_pred))
                elif metric == 'directional_accuracy':
                    results[metric] = self._directional_accuracy(y_true, y_pred)
                elif metric == 'zptae':
                    results[metric] = self._zptae_metric(y_true, y_pred)
                elif metric == 'hit_rate':
                    results[metric] = self._hit_rate(y_true, y_pred)
                elif metric == 'sharpe_ratio':
                    results[metric] = self._sharpe_ratio(y_true, y_pred)
                else:
                    logging.warning(f"Unknown metric: {metric}")
            except Exception as e:
                logging.error(f"Error calculating {metric}: {str(e)}")
                results[metric] = np.nan
        
        return results
    
    def _directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy."""
        y_true_arr = np.asarray(y_true).reshape(-1)
        y_pred_arr = np.asarray(y_pred).reshape(-1)
        min_len = min(len(y_true_arr), len(y_pred_arr))
        if min_len == 0:
            return np.nan
        y_true_arr = y_true_arr[:min_len]
        y_pred_arr = y_pred_arr[:min_len]
        return float(np.mean((y_true_arr > 0) == (y_pred_arr > 0)))
    
    def _zptae_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate ZPTAE metric."""
        try:
            from ..models.losses import zptae_metric_numpy, calculate_rolling_std
            zptae_params = self.config.get('loss_functions', {}).get('zptae', {})
            a = zptae_params.get('a', 1.0)
            p = zptae_params.get('p', 1.5)
            y_true_arr = np.asarray(y_true).reshape(-1)
            y_pred_arr = np.asarray(y_pred).reshape(-1)
            min_len = min(len(y_true_arr), len(y_pred_arr))
            if min_len == 0:
                return np.nan
            y_true_arr = y_true_arr[:min_len]
            y_pred_arr = y_pred_arr[:min_len]
            ref_std = calculate_rolling_std(y_true_arr, window=100, min_periods=30)
            return float(zptae_metric_numpy(y_pred_arr, y_true_arr, ref_std, a=a, p=p))
        except ImportError:
            logging.warning("ZPTAE metric not available")
            return np.nan
    
    def _hit_rate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 threshold: float = 0.001) -> float:
        """Calculate hit rate (predictions within threshold)."""
        return np.mean(np.abs(y_true - y_pred) < threshold)
    
    def _sharpe_ratio(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Sharpe ratio of prediction-based strategy."""
        # Simple strategy: go long if prediction > 0
        strategy_returns = np.where(y_pred > 0, y_true, -y_true)
        
        if np.std(strategy_returns) == 0:
            return 0.0
        
        return np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
    
    def _calculate_stability_metrics(self, fold_results: List[Dict[str, Any]],
                                   metrics: List[str]) -> Dict[str, float]:
        """
        Calculate stability metrics across folds.
        
        Args:
            fold_results: Results from each fold
            metrics: List of metric names
            
        Returns:
            Dictionary of stability metrics
        """
        stability = {}
        
        # Filter successful folds
        successful_folds = [f for f in fold_results if 'error' not in f]
        
        if not successful_folds:
            return stability
        
        for metric in metrics:
            values = [f.get(metric, np.nan) for f in successful_folds]
            values = [v for v in values if not np.isnan(v)]
            
            if values:
                stability[f'{metric}_mean'] = np.mean(values)
                stability[f'{metric}_std'] = np.std(values)
                stability[f'{metric}_min'] = np.min(values)
                stability[f'{metric}_max'] = np.max(values)
                stability[f'{metric}_cv'] = np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf
        
        return stability
    
    def create_holdout_split(self, data: pd.DataFrame, 
                           holdout_size: float = 0.2,
                           date_column: str = 'date') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create holdout test set.
        
        Args:
            data: Input data
            holdout_size: Proportion for holdout (0.2 = 20%)
            date_column: Name of date column
            
        Returns:
            Tuple of (train_data, holdout_data)
        """
        logging.info(f"Creating holdout split with {holdout_size:.1%} holdout")
        
        # Ensure data is sorted by date
        if date_column in data.columns:
            data = data.sort_values(date_column).reset_index(drop=True)
        else:
            data = data.sort_index()
        
        n_samples = len(data)
        holdout_start_idx = int(n_samples * (1 - holdout_size))
        
        train_data = data.iloc[:holdout_start_idx].copy()
        holdout_data = data.iloc[holdout_start_idx:].copy()
        
        logging.info(f"Train set: {len(train_data)} samples, "
                    f"Holdout set: {len(holdout_data)} samples")
        
        return train_data, holdout_data


def main():
    """
    Main function for testing validation module.
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
    
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Create features
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples)
    })
    
    # Create target with some predictable pattern
    y = 0.1 * X['feature1'] + 0.05 * X['feature2'] + np.random.randn(n_samples) * 0.02
    
    data = pd.DataFrame({
        'date': dates,
        'target': y,
        **X
    })
    
    # Initialize validator
    validator = WalkForwardValidator(config)
    
    # Create folds
    folds = validator.create_folds(data)
    
    print(f"Created {len(folds)} folds:")
    for fold in folds[:3]:  # Show first 3 folds
        print(f"Fold {fold['fold_id']}: Train {fold['train_size']}, Val {fold['val_size']}")
        print(f"  Train: {fold['train_start_date']} to {fold['train_end_date']}")
        print(f"  Val: {fold['val_start_date']} to {fold['val_end_date']}")
    
    # Test with simple linear model
    from sklearn.linear_model import LinearRegression
    
    model = LinearRegression()
    
    # Run validation
    results = validator.validate_model(
        model, X, y, folds, 
        metrics=['mse', 'mae', 'directional_accuracy']
    )
    
    print("\nValidation Results:")
    print(f"Successful folds: {results['successful_folds']}/{results['n_folds']}")
    print("\nOverall metrics:")
    for metric, value in results['overall_metrics'].items():
        print(f"  {metric}: {value:.6f}")
    
    print("\nStability metrics:")
    for metric, value in results['stability_metrics'].items():
        print(f"  {metric}: {value:.6f}")


if __name__ == "__main__":
    main()