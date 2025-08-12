"""
LightGBM model implementation for ETH forecasting project.

This module implements LightGBM with ZPTAE loss integration following Rule #10.
Includes hyperparameter optimization with Optuna and proper validation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import joblib
import os
from datetime import datetime
from pathlib import Path

from .losses import ZPTAELoss, LightGBMZPTAEObjective, zptae_metric_numpy


class LightGBMForecaster:
    """
    LightGBM forecaster with ZPTAE loss integration.
    
    Implements baseline model with comprehensive hyperparameter optimization
    and proper validation following project rules.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LightGBM forecaster.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config.get('models', {}).get('lightgbm', {})
        
        # Model parameters
        self.use_zptae = self.model_config.get('use_zptae_loss', True)
        self.zptae_params = config.get('loss_functions', {}).get('zptae', {})
        self.optuna_trials = self.model_config.get('optuna_trials', 200)
        
        # Model storage
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.training_history = {}
        
        # ZPTAE objective (will be initialized during training)
        self.zptae_objective = None
        
        logging.info("LightGBM forecaster initialized with ZPTAE loss integration")
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame, y_val: pd.Series,
                                direction_train: pd.Series = None,
                                direction_val: pd.Series = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training targets (log returns)
            X_val: Validation features
            y_val: Validation targets
            direction_train: Training direction labels (optional)
            direction_val: Validation direction labels (optional)
            
        Returns:
            Best hyperparameters
        """
        logging.info(f"Starting hyperparameter optimization with {self.optuna_trials} trials")
        
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'objective': 'regression',
                'metric': 'None' if self.use_zptae else 'rmse',  # Use rmse when ZPTAE disabled
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'verbosity': -1,
                'random_state': 42
            }
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model
            if self.use_zptae:
                # Use ZPTAE objective
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=1000,
                    fobj=self.zptae_objective,
                    feval=self._zptae_eval_metric,
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
            else:
                # Use standard regression
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=1000,
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
            
            # Make predictions
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            
            # Calculate ZPTAE metric with reference std
            from .losses import calculate_rolling_std
            ref_std = calculate_rolling_std(y_val.values, window=100, min_periods=30)
            
            zptae_score = zptae_metric_numpy(
                y_pred,
                y_val.values, 
                ref_std,
                a=self.zptae_params.get('a', 1.0),
                p=self.zptae_params.get('p', 1.5)
            )
            
            return zptae_score
        
        # Create study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.optuna_trials)
        
        self.best_params = study.best_params
        self.best_params.update({
            'objective': 'regression',
            'metric': 'None' if self.use_zptae else 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42
        })
        
        logging.info(f"Hyperparameter optimization completed. Best ZPTAE: {study.best_value:.6f}")
        logging.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              direction_train: pd.Series = None,
              direction_val: pd.Series = None) -> None:
        """
        Train LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training targets (log returns)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            direction_train: Training direction labels (optional)
            direction_val: Validation direction labels (optional)
        """
        logging.info("Training LightGBM model")
        
        # Use best parameters if available, otherwise use defaults
        if self.best_params is None:
            params = {
                'objective': 'regression',
                'metric': 'rmse',  # Always use rmse for early stopping
                'boosting_type': 'gbdt',
                'num_leaves': 100,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'max_depth': 8,
                'verbosity': -1,
                'random_state': 42
            }
        else:
            params = self.best_params.copy()
        
        # If no validation data provided, create a validation split
        if X_val is None or y_val is None:
            # Use last 20% of data for validation
            split_idx = int(len(X_train) * 0.8)
            X_val = X_train.iloc[split_idx:]
            y_val = y_train.iloc[split_idx:]
            X_train = X_train.iloc[:split_idx]
            y_train = y_train.iloc[:split_idx]
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        valid_sets = [train_data, val_data]
        
        # Initialize ZPTAE objective if needed
        if self.use_zptae and self.zptae_objective is None:
            # Calculate reference standard deviation from training data
            from .losses import calculate_rolling_std
            ref_std = calculate_rolling_std(y_train.values, window=100, min_periods=30)
            
            self.zptae_objective = LightGBMZPTAEObjective(
                ref_std=ref_std,
                a=self.zptae_params.get('a', 1.0),
                p=self.zptae_params.get('p', 1.5)
            )
        
        # Train model
        callbacks = [lgb.log_evaluation(100)]
        # Always use early stopping since we always have validation data now
        callbacks.append(lgb.early_stopping(100))
        
        if self.use_zptae:
            # Train with ZPTAE objective
            self.model = lgb.train(
                params,
                train_data,
                valid_sets=valid_sets,
                num_boost_round=2000,
                fobj=self.zptae_objective,
                feval=self._zptae_eval_metric,
                callbacks=callbacks
            )
        else:
            # Train with standard regression
            self.model = lgb.train(
                params,
                train_data,
                valid_sets=valid_sets,
                num_boost_round=2000,
                callbacks=callbacks
            )
        
        # Store feature importance
        self.feature_importance = dict(zip(X_train.columns, self.model.feature_importance()))
        
        # Store training history
        self.training_history = {
            'best_iteration': self.model.best_iteration,
            'best_score': self.model.best_score,
            'feature_importance': self.feature_importance
        }
        
        logging.info(f"Training completed. Best iteration: {self.model.best_iteration}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, 
            y_val: pd.Series = None, direction_train: pd.Series = None, 
            direction_val: pd.Series = None) -> 'LightGBMForecaster':
        """
        Fit method for compatibility with validation module.
        This is an alias for the train method.
        
        Args:
            X: Training features
            y: Training targets (log returns)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            direction_train: Training direction labels (optional)
            direction_val: Validation direction labels (optional)
            
        Returns:
            Self for method chaining
        """
        self.train(X, y, X_val, y_val, direction_train, direction_val)
        return self
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Tuple of (log_return_predictions, direction_predictions)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Predict log returns
        log_return_pred = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        # Convert to direction predictions
        direction_pred = (log_return_pred > 0).astype(int)
        
        return log_return_pred, direction_pred
    
    def evaluate(self, X: pd.DataFrame, y_true: pd.Series, 
                direction_true: pd.Series = None) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y_true: True log returns
            direction_true: True directions (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred, direction_pred = self.predict(X)
        
        # Calculate regression metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Calculate ZPTAE with reference std
        from .losses import calculate_rolling_std
        ref_std = calculate_rolling_std(y_true.values, window=100, min_periods=30)
        
        zptae = zptae_metric_numpy(
            y_pred,
            y_true.values, 
            ref_std,
            a=self.zptae_params.get('a', 1.0),
            p=self.zptae_params.get('p', 1.5)
        )
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'zptae': zptae
        }
        
        # Calculate directional accuracy if direction labels provided
        if direction_true is not None:
            direction_accuracy = accuracy_score(direction_true, direction_pred)
            metrics['directional_accuracy'] = direction_accuracy
        else:
            # Calculate directional accuracy from log returns
            direction_true_from_returns = (y_true > 0).astype(int)
            direction_accuracy = accuracy_score(direction_true_from_returns, direction_pred)
            metrics['directional_accuracy'] = direction_accuracy
        
        return metrics
    
    def _zptae_eval_metric(self, y_pred: np.ndarray, y_true: lgb.Dataset) -> Tuple[str, float, bool]:
        """
        ZPTAE evaluation metric for LightGBM.
        
        Args:
            y_pred: Predictions
            y_true: True values (LightGBM Dataset)
            
        Returns:
            Tuple of (metric_name, metric_value, is_higher_better)
        """
        y_true_values = y_true.get_label()
        
        # Calculate reference std for evaluation
        from .losses import calculate_rolling_std
        ref_std = calculate_rolling_std(y_true_values, window=100, min_periods=30)
        
        zptae = zptae_metric_numpy(
            y_pred,
            y_true_values, 
            ref_std,
            a=self.zptae_params.get('a', 1.0),
            p=self.zptae_params.get('p', 1.5)
        )
        
        return 'zptae', zptae, False  # Lower is better
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """
        Get feature importance.
        
        Args:
            importance_type: Type of importance ('gain', 'split')
            
        Returns:
            Dictionary of feature importances
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if importance_type == 'gain':
            importance = self.model.feature_importance(importance_type='gain')
        elif importance_type == 'split':
            importance = self.model.feature_importance(importance_type='split')
        else:
            raise ValueError("importance_type must be 'gain' or 'split'")
        
        feature_names = self.model.feature_name()
        return dict(zip(feature_names, importance))
    
    def save_model(self, model_name: str) -> str:
        """
        Save trained model.
        
        Args:
            model_name: Name of the model (will be saved in models directory)
            
        Returns:
            Full path where model was saved
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Construct full filepath
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        filepath = models_dir / f"{model_name}.txt"
        filepath = str(filepath)
        
        # Save model
        self.model.save_model(filepath)
        
        # Save additional information
        model_info = {
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'use_zptae': self.use_zptae,
            'zptae_params': self.zptae_params
        }
        
        info_filepath = filepath.replace('.txt', '_info.joblib')
        joblib.dump(model_info, info_filepath)
        
        logging.info(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model.
        
        Args:
            filepath: Path to load model from
        """
        # Load model
        self.model = lgb.Booster(model_file=filepath)
        
        # Load additional information
        info_filepath = filepath.replace('.txt', '_info.joblib')
        if os.path.exists(info_filepath):
            model_info = joblib.load(info_filepath)
            self.best_params = model_info.get('best_params')
            self.feature_importance = model_info.get('feature_importance')
            self.training_history = model_info.get('training_history')
            self.use_zptae = model_info.get('use_zptae', True)
            self.zptae_params = model_info.get('zptae_params', {})
        
        logging.info(f"Model loaded from {filepath}")
    
    def plot_feature_importance(self, top_n: int = 20, output_dir: str = "reports/figures") -> None:
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to plot
            output_dir: Output directory for plots
        """
        if self.feature_importance is None:
            raise ValueError("No feature importance available. Train model first.")
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Sort features by importance
        sorted_features = sorted(self.feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)[:top_n]
        
        features, importances = zip(*sorted_features)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x=list(importances), y=list(features))
        plt.title(f'LightGBM Feature Importance (Top {top_n})')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        
        # Save plot
        filepath = os.path.join(output_dir, 'lightgbm_feature_importance.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Feature importance plot saved to {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model summary.
        
        Returns:
            Dictionary with model summary
        """
        if self.model is None:
            return {'status': 'not_trained'}
        
        summary = {
            'status': 'trained',
            'model_type': 'LightGBM',
            'use_zptae': self.use_zptae,
            'best_iteration': self.training_history.get('best_iteration'),
            'best_score': self.training_history.get('best_score'),
            'num_features': len(self.feature_importance) if self.feature_importance else 0,
            'hyperparameters': self.best_params
        }
        
        return summary


def main():
    """
    Main function for testing LightGBM model.
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
    n_samples = 1000
    n_features = 50
    
    # Generate features
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    
    # Generate target (log returns with some signal)
    signal = X.iloc[:, :5].sum(axis=1) * 0.01  # Use first 5 features as signal
    noise = np.random.normal(0, 0.02, n_samples)
    y = signal + noise
    
    # Generate direction labels
    direction = (y > 0).astype(int)
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    direction_train, direction_val = direction[:split_idx], direction[split_idx:]
    
    # Initialize model
    forecaster = LightGBMForecaster(config)
    
    # Optimize hyperparameters (small number of trials for testing)
    forecaster.optuna_trials = 10
    best_params = forecaster.optimize_hyperparameters(
        X_train, y_train, X_val, y_val, direction_train, direction_val
    )
    
    # Train model
    forecaster.train(X_train, y_train, X_val, y_val, direction_train, direction_val)
    
    # Evaluate model
    metrics = forecaster.evaluate(X_val, y_val, direction_val)
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    # Get model summary
    summary = forecaster.get_model_summary()
    print("\nModel Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test save/load
    model_path = "test_lightgbm_model.txt"
    forecaster.save_model(model_path)
    
    # Create new forecaster and load model
    new_forecaster = LightGBMForecaster(config)
    new_forecaster.load_model(model_path)
    
    # Test predictions
    y_pred, direction_pred = new_forecaster.predict(X_val)
    print(f"\nPrediction test: {len(y_pred)} predictions generated")
    
    # Clean up
    if os.path.exists(model_path):
        os.remove(model_path)
    info_path = model_path.replace('.txt', '_info.joblib')
    if os.path.exists(info_path):
        os.remove(info_path)


if __name__ == "__main__":
    main()