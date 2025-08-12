"""
Ensemble Model Implementation

This module implements ensemble methods for combining multiple models,
following Rule #10 (Multi-Model Mandate) and Rule #12 (Multi-Task Loss Implementation).

The ensemble uses stacking with cross-validation to combine predictions from
LightGBM, TFT, and N-BEATS models.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

from .losses import ZPTAELoss, zptae_metric_numpy

logger = logging.getLogger(__name__)


class StackingEnsemble:
    """
    Stacking ensemble that combines multiple base models using a meta-learner.
    
    This implementation supports both regression and classification tasks,
    with custom ZPTAE loss integration for the meta-learner.
    """
    
    def __init__(self, 
                 base_models: Dict[str, Any],
                 meta_model_type: str = 'lightgbm',
                 cv_folds: int = 5,
                 config: Optional[Dict[str, Any]] = None,
                 device: Optional[str] = None):
        """
        Initialize stacking ensemble.
        
        Args:
            base_models: Dictionary of base models {name: model}
            meta_model_type: Type of meta-learner ('lightgbm', 'ridge', 'rf')
            cv_folds: Number of CV folds for stacking
            config: Configuration dictionary
            device: Device for GPU acceleration ('cuda' or 'cpu')
        """
        self.base_models = base_models
        self.meta_model_type = meta_model_type
        self.cv_folds = cv_folds
        self.config = config or {}
        
        # Device management for GPU acceleration
        import torch
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Meta-learners
        self.meta_regressor = None
        self.meta_classifier = None
        
        # Base model predictions storage
        self.base_predictions_train = None
        self.base_predictions_val = None
        
        # Fitted status
        self.is_fitted = False
        
        # Initialize ZPTAE loss
        loss_config = self.config.get('loss', {})
        self.zptae_loss = ZPTAELoss(
            a=loss_config.get('zptae', {}).get('a', 1.0),
            p=loss_config.get('zptae', {}).get('p', 1.5),
            epsilon=loss_config.get('zptae', {}).get('epsilon', 1e-8)
        )
        
        logger.info(f"Stacking ensemble initialized with {len(base_models)} base models on {self.device}")
    
    def _create_meta_learner(self, task_type: str = 'regression') -> Any:
        """
        Create meta-learner based on configuration.
        
        Args:
            task_type: 'regression' or 'classification'
            
        Returns:
            Initialized meta-learner
        """
        if task_type == 'regression':
            if self.meta_model_type == 'lightgbm':
                return lgb.LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42,
                    verbose=-1
                )
            elif self.meta_model_type == 'ridge':
                return Ridge(alpha=1.0, random_state=42)
            elif self.meta_model_type == 'rf':
                return RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                raise ValueError(f"Unknown meta-model type: {self.meta_model_type}")
        
        else:  # classification
            if self.meta_model_type == 'lightgbm':
                return lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42,
                    verbose=-1
                )
            elif self.meta_model_type == 'ridge':
                return LogisticRegression(
                    C=1.0,
                    random_state=42,
                    max_iter=1000
                )
            elif self.meta_model_type == 'rf':
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                raise ValueError(f"Unknown meta-model type: {self.meta_model_type}")
    
    def _get_base_predictions(self, 
                             X: pd.DataFrame, 
                             y: pd.Series,
                             mode: str = 'train') -> np.ndarray:
        """
        Get base model predictions using cross-validation.
        
        Args:
            X: Input features
            y: Target values
            mode: 'train' for training mode, 'predict' for prediction mode
            
        Returns:
            Array of base model predictions
        """
        n_samples = len(X)
        n_models = len(self.base_models)
        
        # Initialize prediction arrays
        reg_predictions = np.zeros((n_samples, n_models))
        cls_predictions = np.zeros((n_samples, n_models))
        
        if mode == 'train':
            # Use cross-validation for training
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                logger.info(f"Processing fold {fold + 1}/{self.cv_folds}")
                
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                
                for i, (model_name, model) in enumerate(self.base_models.items()):
                    try:
                        # Fit model on fold training data
                        model_copy = self._copy_model(model)
                        
                        # Handle GPU acceleration for deep learning models
                        if hasattr(model_copy, 'device') and hasattr(model_copy, 'to'):
                            model_copy.to(self.device)
                        
                        model_copy.fit(X_train_fold, y_train_fold)
                        
                        # Predict on fold validation data
                        reg_pred = model_copy.predict(X_val_fold)
                        
                        # Ensure predictions are numpy arrays
                        if hasattr(reg_pred, 'cpu'):
                            reg_pred = reg_pred.cpu().numpy()
                        elif hasattr(reg_pred, 'detach'):
                            reg_pred = reg_pred.detach().numpy()
                        
                        reg_predictions[val_idx, i] = reg_pred.flatten()
                        
                        # Get classification predictions if available
                        if hasattr(model_copy, 'predict_proba'):
                            cls_pred = model_copy.predict_proba(X_val_fold)
                            if hasattr(cls_pred, 'cpu'):
                                cls_pred = cls_pred.cpu().numpy()
                            elif hasattr(cls_pred, 'detach'):
                                cls_pred = cls_pred.detach().numpy()
                            
                            if cls_pred.ndim > 1 and cls_pred.shape[1] > 1:
                                cls_pred = cls_pred[:, 1]
                            else:
                                cls_pred = cls_pred.flatten()
                        else:
                            cls_pred = (reg_pred > 0).astype(float)
                        
                        cls_predictions[val_idx, i] = cls_pred.flatten()
                        
                        # Clean up GPU memory
                        if hasattr(model_copy, 'device') and str(self.device) == 'cuda':
                            import torch
                            torch.cuda.empty_cache()
                        
                    except Exception as e:
                        logger.warning(f"Error with model {model_name} in fold {fold}: {str(e)}")
                        # Fill with zeros if model fails
                        reg_predictions[val_idx, i] = 0
                        cls_predictions[val_idx, i] = 0.5
        
        else:  # prediction mode
            for i, (model_name, model) in enumerate(self.base_models.items()):
                try:
                    # Use fitted models for prediction
                    reg_pred = model.predict(X)
                    
                    # Ensure predictions are numpy arrays
                    if hasattr(reg_pred, 'cpu'):
                        reg_pred = reg_pred.cpu().numpy()
                    elif hasattr(reg_pred, 'detach'):
                        reg_pred = reg_pred.detach().numpy()
                    
                    reg_predictions[:, i] = reg_pred.flatten()
                    
                    # Get classification predictions if available
                    if hasattr(model, 'predict_proba'):
                        cls_pred = model.predict_proba(X)
                        if hasattr(cls_pred, 'cpu'):
                            cls_pred = cls_pred.cpu().numpy()
                        elif hasattr(cls_pred, 'detach'):
                            cls_pred = cls_pred.detach().numpy()
                        
                        if cls_pred.ndim > 1 and cls_pred.shape[1] > 1:
                            cls_pred = cls_pred[:, 1]
                        else:
                            cls_pred = cls_pred.flatten()
                    else:
                        cls_pred = (reg_pred > 0).astype(float)
                    
                    cls_predictions[:, i] = cls_pred.flatten()
                    
                except Exception as e:
                    logger.warning(f"Error with model {model_name} in prediction: {str(e)}")
                    # Fill with zeros if model fails
                    reg_predictions[:, i] = 0
                    cls_predictions[:, i] = 0.5
        
        # Combine regression and classification predictions
        combined_predictions = np.hstack([reg_predictions, cls_predictions])
        
        return combined_predictions
    
    def _copy_model(self, model: Any) -> Any:
        """
        Create a copy of the model for cross-validation.
        
        Args:
            model: Model to copy
            
        Returns:
            Copy of the model
        """
        try:
            # Try to use sklearn's clone if available
            from sklearn.base import clone
            return clone(model)
        except:
            # Fallback: create new instance with same parameters
            model_class = type(model)
            if hasattr(model, 'get_params'):
                params = model.get_params()
                return model_class(**params)
            else:
                # Last resort: return the original model
                logger.warning("Could not copy model, using original")
                return model
    
    def fit(self, 
            X_train: pd.DataFrame, 
            y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'StackingEnsemble':
        """
        Fit the stacking ensemble.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Self for method chaining
        """
        try:
            logger.info("Starting ensemble training...")
            
            # Step 1: Fit base models and get cross-validated predictions
            logger.info("Fitting base models and generating meta-features...")
            
            # Fit base models on full training data
            for model_name, model in self.base_models.items():
                logger.info(f"Fitting base model: {model_name}")
                model.fit(X_train, y_train)
            
            # Get base model predictions for meta-learner training
            self.base_predictions_train = self._get_base_predictions(
                X_train, y_train, mode='train'
            )
            
            # Step 2: Train meta-learners
            logger.info("Training meta-learners...")
            
            # Regression meta-learner
            self.meta_regressor = self._create_meta_learner('regression')
            self.meta_regressor.fit(self.base_predictions_train, y_train)
            
            # Classification meta-learner
            y_train_binary = (y_train > 0).astype(int)
            self.meta_classifier = self._create_meta_learner('classification')
            self.meta_classifier.fit(self.base_predictions_train, y_train_binary)
            
            # Step 3: Validate on validation set if provided
            if X_val is not None and y_val is not None:
                logger.info("Validating ensemble performance...")
                
                # Get base predictions for validation set
                val_base_predictions = self._get_base_predictions(
                    X_val, y_val, mode='predict'
                )
                
                # Meta-learner predictions
                reg_pred = self.meta_regressor.predict(val_base_predictions)
                cls_pred = self.meta_classifier.predict_proba(val_base_predictions)[:, 1]
                
                # Calculate validation metrics
                reg_mse = mean_squared_error(y_val, reg_pred)
                cls_acc = accuracy_score((y_val > 0).astype(int), (cls_pred > 0.5).astype(int))
                zptae_score = zptae_metric_numpy(y_val.values, reg_pred)
                
                logger.info(f"Validation - MSE: {reg_mse:.6f}, Accuracy: {cls_acc:.4f}, ZPTAE: {zptae_score:.6f}")
            
            self.is_fitted = True
            logger.info("Ensemble training completed successfully")
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting ensemble: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make regression predictions using the ensemble.
        
        Args:
            X: Input features
            
        Returns:
            Regression predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        try:
            # Get base model predictions
            base_predictions = self._get_base_predictions(
                X, pd.Series(np.zeros(len(X))), mode='predict'
            )
            
            # Meta-learner prediction
            predictions = self.meta_regressor.predict(base_predictions)
            
            logger.info(f"Generated {len(predictions)} ensemble predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Error making ensemble predictions: {str(e)}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make classification predictions using the ensemble.
        
        Args:
            X: Input features
            
        Returns:
            Classification probabilities
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        try:
            # Get base model predictions
            base_predictions = self._get_base_predictions(
                X, pd.Series(np.zeros(len(X))), mode='predict'
            )
            
            # Meta-learner prediction
            probabilities = self.meta_classifier.predict_proba(base_predictions)
            
            logger.info(f"Generated {len(probabilities)} ensemble probability predictions")
            return probabilities
            
        except Exception as e:
            logger.error(f"Error making ensemble probability predictions: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from meta-learners.
        
        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before extracting feature importance")
        
        try:
            importance_dict = {}
            
            # Get importance from regression meta-learner
            if hasattr(self.meta_regressor, 'feature_importances_'):
                reg_importance = self.meta_regressor.feature_importances_
            elif hasattr(self.meta_regressor, 'coef_'):
                reg_importance = np.abs(self.meta_regressor.coef_)
            else:
                reg_importance = np.ones(len(self.base_models) * 2) / (len(self.base_models) * 2)
            
            # Map to model names
            model_names = list(self.base_models.keys())
            for i, model_name in enumerate(model_names):
                importance_dict[f"{model_name}_reg"] = reg_importance[i]
                importance_dict[f"{model_name}_cls"] = reg_importance[i + len(model_names)]
            
            logger.info("Extracted ensemble feature importance")
            return importance_dict
            
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
            return {}
    
    def get_base_model_weights(self) -> Dict[str, float]:
        """
        Get the effective weights of base models in the ensemble.
        
        Returns:
            Dictionary of model weights
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before extracting weights")
        
        try:
            # Get meta-learner coefficients
            if hasattr(self.meta_regressor, 'coef_'):
                weights = np.abs(self.meta_regressor.coef_)
            elif hasattr(self.meta_regressor, 'feature_importances_'):
                weights = self.meta_regressor.feature_importances_
            else:
                weights = np.ones(len(self.base_models)) / len(self.base_models)
            
            # Normalize weights
            weights = weights[:len(self.base_models)]  # Take only regression weights
            weights = weights / np.sum(weights)
            
            # Map to model names
            weight_dict = {
                model_name: weight 
                for model_name, weight in zip(self.base_models.keys(), weights)
            }
            
            logger.info("Extracted base model weights")
            return weight_dict
            
        except Exception as e:
            logger.warning(f"Could not extract model weights: {str(e)}")
            return {name: 1.0/len(self.base_models) for name in self.base_models.keys()}
    
    def evaluate_base_models(self, 
                           X_test: pd.DataFrame, 
                           y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evaluate individual base models and ensemble.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of model performance metrics
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before evaluation")
        
        results = {}
        
        try:
            # Evaluate base models
            for model_name, model in self.base_models.items():
                pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, pred)
                zptae_score = zptae_metric_numpy(y_test.values, pred)
                
                # Classification metrics
                y_test_binary = (y_test > 0).astype(int)
                pred_binary = (pred > 0).astype(int)
                accuracy = accuracy_score(y_test_binary, pred_binary)
                
                results[model_name] = {
                    'mse': mse,
                    'zptae': zptae_score,
                    'accuracy': accuracy
                }
            
            # Evaluate ensemble
            ensemble_pred = self.predict(X_test)
            ensemble_mse = mean_squared_error(y_test, ensemble_pred)
            ensemble_zptae = zptae_metric_numpy(y_test.values, ensemble_pred)
            ensemble_accuracy = accuracy_score(
                (y_test > 0).astype(int), 
                (ensemble_pred > 0).astype(int)
            )
            
            results['ensemble'] = {
                'mse': ensemble_mse,
                'zptae': ensemble_zptae,
                'accuracy': ensemble_accuracy
            }
            
            logger.info("Base model evaluation completed")
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating base models: {str(e)}")
            raise
    
    def save_ensemble(self, filepath: str) -> None:
        """
        Save the ensemble model.
        
        Args:
            filepath: Path to save the ensemble
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before saving")
        
        try:
            ensemble_data = {
                'base_models': self.base_models,
                'meta_regressor': self.meta_regressor,
                'meta_classifier': self.meta_classifier,
                'meta_model_type': self.meta_model_type,
                'cv_folds': self.cv_folds,
                'config': self.config,
                'is_fitted': self.is_fitted
            }
            
            joblib.dump(ensemble_data, filepath)
            logger.info(f"Ensemble saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving ensemble: {str(e)}")
            raise
    
    def load_ensemble(self, filepath: str) -> 'StackingEnsemble':
        """
        Load an ensemble model.
        
        Args:
            filepath: Path to load the ensemble from
            
        Returns:
            Self for method chaining
        """
        try:
            ensemble_data = joblib.load(filepath)
            
            self.base_models = ensemble_data['base_models']
            self.meta_regressor = ensemble_data['meta_regressor']
            self.meta_classifier = ensemble_data['meta_classifier']
            self.meta_model_type = ensemble_data['meta_model_type']
            self.cv_folds = ensemble_data['cv_folds']
            self.config = ensemble_data['config']
            self.is_fitted = ensemble_data['is_fitted']
            
            logger.info(f"Ensemble loaded from {filepath}")
            return self
            
        except Exception as e:
            logger.error(f"Error loading ensemble: {str(e)}")
            raise


def create_ensemble(base_models: Dict[str, Any], 
                   config: Dict[str, Any],
                   device: Optional[str] = None) -> StackingEnsemble:
    """
    Factory function to create ensemble model.
    
    Args:
        base_models: Dictionary of base models
        config: Configuration dictionary
        device: Device for GPU acceleration ('cuda' or 'cpu')
        
    Returns:
        Initialized ensemble model
    """
    ensemble_config = config.get('ensemble', {})
    
    return StackingEnsemble(
        base_models=base_models,
        meta_model_type=ensemble_config.get('meta_model', 'lightgbm'),
        cv_folds=ensemble_config.get('cv_folds', 5),
        config=config,
        device=device
    )