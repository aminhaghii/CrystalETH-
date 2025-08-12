"""
N-BEATS Model Implementation

This module implements the N-BEATS (Neural Basis Expansion Analysis for Time Series)
model for ETH price forecasting, following Rule #4 (Production-Ready Code) and 
Rule #10 (Multi-Model Mandate).

N-BEATS is a deep neural architecture based on backward and forward residual links
and a very deep stack of fully-connected layers.
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.preprocessing import StandardScaler
import optuna
import warnings
warnings.filterwarnings('ignore')

from .losses import ZPTAELoss, MultiTaskLoss

logger = logging.getLogger(__name__)


class NBeatsBlock(nn.Module):
    """
    Basic building block of N-BEATS architecture.
    """
    
    def __init__(self, 
                 input_size: int,
                 theta_size: int,
                 basis_function: nn.Module,
                 layers: int = 4,
                 layer_size: int = 512):
        """
        Initialize N-BEATS block.
        
        Args:
            input_size: Size of input sequence
            theta_size: Size of theta parameter vector
            basis_function: Basis function for generating forecasts/backcasts
            layers: Number of fully connected layers
            layer_size: Size of each layer
        """
        super().__init__()
        
        self.input_size = input_size
        self.theta_size = theta_size
        self.basis_function = basis_function
        
        # Fully connected layers
        layers_list = []
        for i in range(layers):
            layers_list.append(nn.Linear(
                input_size if i == 0 else layer_size, 
                layer_size
            ))
            layers_list.append(nn.ReLU())
        
        self.layers = nn.Sequential(*layers_list)
        
        # Theta layer
        self.theta_layer = nn.Linear(layer_size, theta_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through N-BEATS block.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (backcast, forecast)
        """
        # Pass through fully connected layers
        h = self.layers(x)
        
        # Generate theta parameters
        theta = self.theta_layer(h)
        
        # Generate backcast and forecast using basis function
        backcast, forecast = self.basis_function(theta)
        
        return backcast, forecast


class GenericBasis(nn.Module):
    """
    Generic basis function for N-BEATS.
    """
    
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
    
    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate backcast and forecast from theta parameters.
        
        Args:
            theta: Parameter tensor
            
        Returns:
            Tuple of (backcast, forecast)
        """
        # Split theta into backcast and forecast parts
        backcast = theta[:, :self.backcast_size]
        forecast = theta[:, self.backcast_size:]
        
        return backcast, forecast


class TrendBasis(nn.Module):
    """
    Trend basis function for N-BEATS.
    """
    
    def __init__(self, backcast_size: int, forecast_size: int, degree: int = 3):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.degree = degree
        
        # Create polynomial basis
        self.register_buffer('backcast_time', torch.arange(backcast_size, dtype=torch.float32) / backcast_size)
        self.register_buffer('forecast_time', torch.arange(forecast_size, dtype=torch.float32) / forecast_size)
    
    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate trend-based backcast and forecast.
        
        Args:
            theta: Parameter tensor
            
        Returns:
            Tuple of (backcast, forecast)
        """
        batch_size = theta.size(0)
        
        # Create polynomial features
        backcast_basis = torch.stack([
            self.backcast_time ** i for i in range(self.degree + 1)
        ], dim=1)  # [backcast_size, degree+1]
        
        forecast_basis = torch.stack([
            self.forecast_time ** i for i in range(self.degree + 1)
        ], dim=1)  # [forecast_size, degree+1]
        
        # Expand for batch
        backcast_basis = backcast_basis.unsqueeze(0).expand(batch_size, -1, -1)
        forecast_basis = forecast_basis.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Generate backcast and forecast
        theta_expanded = theta.unsqueeze(1)  # [batch_size, 1, degree+1]
        
        backcast = torch.bmm(backcast_basis, theta_expanded.transpose(1, 2)).squeeze(-1)
        forecast = torch.bmm(forecast_basis, theta_expanded.transpose(1, 2)).squeeze(-1)
        
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    """
    Seasonality basis function for N-BEATS.
    """
    
    def __init__(self, backcast_size: int, forecast_size: int, harmonics: int = 10):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.harmonics = harmonics
        
        # Create time indices
        self.register_buffer('backcast_time', 
                           2 * np.pi * torch.arange(backcast_size, dtype=torch.float32) / backcast_size)
        self.register_buffer('forecast_time', 
                           2 * np.pi * torch.arange(forecast_size, dtype=torch.float32) / forecast_size)
    
    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate seasonality-based backcast and forecast.
        
        Args:
            theta: Parameter tensor
            
        Returns:
            Tuple of (backcast, forecast)
        """
        batch_size = theta.size(0)
        
        # Create harmonic features
        backcast_basis = []
        forecast_basis = []
        
        for i in range(1, self.harmonics + 1):
            backcast_basis.extend([
                torch.cos(i * self.backcast_time),
                torch.sin(i * self.backcast_time)
            ])
            forecast_basis.extend([
                torch.cos(i * self.forecast_time),
                torch.sin(i * self.forecast_time)
            ])
        
        backcast_basis = torch.stack(backcast_basis, dim=1)  # [backcast_size, 2*harmonics]
        forecast_basis = torch.stack(forecast_basis, dim=1)  # [forecast_size, 2*harmonics]
        
        # Expand for batch
        backcast_basis = backcast_basis.unsqueeze(0).expand(batch_size, -1, -1)
        forecast_basis = forecast_basis.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Generate backcast and forecast
        theta_expanded = theta.unsqueeze(1)  # [batch_size, 1, 2*harmonics]
        
        backcast = torch.bmm(backcast_basis, theta_expanded.transpose(1, 2)).squeeze(-1)
        forecast = torch.bmm(forecast_basis, theta_expanded.transpose(1, 2)).squeeze(-1)
        
        return backcast, forecast


class NBeatsModel(nn.Module):
    """
    N-BEATS model implementation with custom ZPTAE loss integration.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 input_size: int = 30,
                 forecast_size: int = 1,
                 stack_types: List[str] = ['trend', 'seasonality', 'generic'],
                 nb_blocks_per_stack: int = 3,
                 hidden_layer_units: int = 512):
        """
        Initialize N-BEATS model.
        
        Args:
            config: Configuration dictionary
            input_size: Length of input sequence
            forecast_size: Length of forecast sequence
            stack_types: Types of stacks to use
            nb_blocks_per_stack: Number of blocks per stack
            hidden_layer_units: Number of units in hidden layers
        """
        super().__init__()
        
        self.config = config
        self.input_size = input_size
        self.forecast_size = forecast_size
        self.stack_types = stack_types
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.hidden_layer_units = hidden_layer_units
        
        # Initialize loss functions
        loss_config = config.get('loss', {})
        self.zptae_loss = ZPTAELoss(
            a=loss_config.get('zptae', {}).get('a', 1.0),
            p=loss_config.get('zptae', {}).get('p', 1.5),
            epsilon=loss_config.get('zptae', {}).get('epsilon', 1e-8)
        )
        
        self.multi_task_loss = MultiTaskLoss(
            zptae_weight=loss_config.get('multi_task', {}).get('zptae_weight', 0.7),
            bce_weight=loss_config.get('multi_task', {}).get('bce_weight', 0.3)
        )
        
        # Build stacks
        self.stacks = nn.ModuleList()
        
        for stack_type in stack_types:
            stack = nn.ModuleList()
            
            for _ in range(nb_blocks_per_stack):
                if stack_type == 'trend':
                    basis_function = TrendBasis(input_size, forecast_size, degree=3)
                    theta_size = 4  # degree + 1
                elif stack_type == 'seasonality':
                    basis_function = SeasonalityBasis(input_size, forecast_size, harmonics=10)
                    theta_size = 20  # 2 * harmonics
                else:  # generic
                    basis_function = GenericBasis(input_size, forecast_size)
                    theta_size = input_size + forecast_size
                
                block = NBeatsBlock(
                    input_size=input_size,
                    theta_size=theta_size,
                    basis_function=basis_function,
                    layers=4,
                    layer_size=hidden_layer_units
                )
                stack.append(block)
            
            self.stacks.append(stack)
        
        # Classification head for directional prediction (output logits, not probabilities)
        self.classifier = nn.Sequential(
            nn.Linear(forecast_size, hidden_layer_units // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_layer_units // 4, 1)
        )
        
        logger.info(f"N-BEATS model initialized with {len(stack_types)} stacks")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through N-BEATS model.
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            Tuple of (regression_output, classification_output)
        """
        residual = x
        forecast = torch.zeros(x.size(0), self.forecast_size, device=x.device)
        
        # Pass through each stack
        for stack in self.stacks:
            stack_forecast = torch.zeros(x.size(0), self.forecast_size, device=x.device)
            
            # Pass through each block in the stack
            for block in stack:
                backcast, block_forecast = block(residual)
                residual = residual - backcast
                stack_forecast = stack_forecast + block_forecast
            
            forecast = forecast + stack_forecast
        
        # Generate classification output
        classification_output = self.classifier(forecast)
        
        return forecast, classification_output


class NBeatsWrapper:
    """
    Wrapper class for N-BEATS model with scikit-learn compatible interface.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize N-BEATS wrapper.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                 config.get('performance', {}).get('gpu_enabled', True) else 'cpu')
        
        # Set random seeds for reproducibility (Rule #6)
        torch.manual_seed(config.get('random_seed', 42))
        np.random.seed(config.get('random_seed', 42))
        
        logger.info(f"N-BEATS wrapper initialized on device: {self.device}")
    
    def _prepare_sequences(self, X: pd.DataFrame, y: pd.Series = None, 
                          sequence_length: int = 30) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare sequences for N-BEATS training.
        
        Args:
            X: Input features
            y: Target values (optional)
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (input_sequences, target_sequences)
        """
        # Select only numeric columns, excluding timestamp
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found for N-BEATS sequence preparation")
        
        # Use the first numeric column (typically log_return or similar)
        X_values = X[numeric_cols[0]].values
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(X_values)):
            # Input sequence
            seq = X_values[i-sequence_length:i]
            sequences.append(seq)
            
            # Target (next value)
            if y is not None:
                targets.append(y.iloc[i])
        
        sequences = torch.tensor(sequences, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32) if targets else None
        
        return sequences, targets
    
    def optimize_hyperparameters(self, 
                                X_train: pd.DataFrame, 
                                y_train: pd.Series,
                                X_val: pd.DataFrame,
                                y_val: pd.Series,
                                n_trials: int = 80) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of optimization trials
            
        Returns:
            Dict containing best hyperparameters
        """
        def objective(trial):
            try:
                # Suggest hyperparameters
                params = {
                    'hidden_layer_units': trial.suggest_categorical('hidden_layer_units', [256, 512, 1024]),
                    'nb_blocks_per_stack': trial.suggest_int('nb_blocks_per_stack', 2, 4),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                    'sequence_length': trial.suggest_int('sequence_length', 20, 40),
                }
                
                # Prepare data
                X_train_seq, y_train_seq = self._prepare_sequences(
                    X_train, y_train, params['sequence_length']
                )
                X_val_seq, y_val_seq = self._prepare_sequences(
                    X_val, y_val, params['sequence_length']
                )
                
                # Create model
                model = NBeatsModel(
                    config=self.config,
                    input_size=params['sequence_length'],
                    hidden_layer_units=params['hidden_layer_units'],
                    nb_blocks_per_stack=params['nb_blocks_per_stack']
                ).to(self.device)
                
                # Train model
                optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
                
                model.train()
                for epoch in range(20):  # Reduced epochs for optimization
                    # Mini-batch training
                    batch_size = params['batch_size']
                    total_loss = 0
                    
                    for i in range(0, len(X_train_seq), batch_size):
                        batch_X = X_train_seq[i:i+batch_size].to(self.device)
                        batch_y = y_train_seq[i:i+batch_size].to(self.device)
                        
                        optimizer.zero_grad()
                        
                        # Forward pass
                        reg_output, cls_output = model(batch_X)
                        
                        # Calculate loss
                        # Calculate reference standard deviation for ZPTAE
                        ref_std = torch.std(batch_y) + 1e-8  # Add epsilon for numerical stability
                        reg_loss = model.zptae_loss(reg_output.squeeze(), batch_y, ref_std)
                        cls_targets = (batch_y > 0).float()
                        # Ensure shapes are [N,1] for BCEWithLogitsLoss
                        cls_logits = cls_output.view(-1, 1)
                        cls_targets = cls_targets.view(-1, 1)
                        cls_loss = nn.BCEWithLogitsLoss()(cls_logits, cls_targets)
                        
                        loss = 0.7 * reg_loss + 0.3 * cls_loss
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()
                
                # Validation
                if X_val_seq is not None and y_val_seq is not None:
                    model.eval()
                    with torch.no_grad():
                        val_X = X_val_seq.to(self.device)
                        val_y = y_val_seq.to(self.device)
                        
                        reg_output, _ = model(val_X)
                        # Calculate reference standard deviation for validation
                        val_ref_std = torch.std(val_y) + 1e-8
                        val_loss = model.zptae_loss(reg_output.squeeze(), val_y, val_ref_std).item()
                    
                    return val_loss
                else:
                    # If validation sequences couldn't be created, return high loss
                    return float('inf')
                
            except Exception as e:
                logger.warning(f"Trial failed: {str(e)}")
                return float('inf')
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        logger.info(f"Best N-BEATS hyperparameters: {best_params}")
        
        return best_params
    
    def fit(self, 
            X_train: pd.DataFrame, 
            y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            optimize_hyperparams: bool = True) -> 'NBeatsWrapper':
        """
        Fit the N-BEATS model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            optimize_hyperparams: Whether to optimize hyperparameters
            
        Returns:
            Self for method chaining
        """
        try:
            # Optimize hyperparameters if requested
            if optimize_hyperparams and X_val is not None and y_val is not None:
                n_trials = self.config.get('nbeats', {}).get('n_trials', 80)
                best_params = self.optimize_hyperparameters(
                    X_train, y_train, X_val, y_val, n_trials
                )
            else:
                # Use default parameters
                best_params = {
                    'hidden_layer_units': 512,
                    'nb_blocks_per_stack': 3,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'sequence_length': 30,
                }
            
            # Prepare training data
            sequence_length = best_params['sequence_length']
            X_train_seq, y_train_seq = self._prepare_sequences(X_train, y_train, sequence_length)
            
            # Create model
            self.model = NBeatsModel(
                config=self.config,
                input_size=sequence_length,
                hidden_layer_units=best_params['hidden_layer_units'],
                nb_blocks_per_stack=best_params['nb_blocks_per_stack']
            ).to(self.device)
            
            # Setup optimizer
            optimizer = optim.Adam(self.model.parameters(), lr=best_params['learning_rate'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
            
            # Training loop
            max_epochs = self.config.get('nbeats', {}).get('max_epochs', 100)
            batch_size = best_params['batch_size']
            best_val_loss = float('inf')
            patience_counter = 0
            patience = self.config.get('nbeats', {}).get('early_stopping_patience', 10)
            
            for epoch in range(max_epochs):
                # Training
                self.model.train()
                total_loss = 0
                
                # Shuffle training data
                indices = torch.randperm(len(X_train_seq))
                X_train_shuffled = X_train_seq[indices]
                y_train_shuffled = y_train_seq[indices]
                
                for i in range(0, len(X_train_shuffled), batch_size):
                    batch_X = X_train_shuffled[i:i+batch_size].to(self.device)
                    batch_y = y_train_shuffled[i:i+batch_size].to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    reg_output, cls_output = self.model(batch_X)
                    
                    # Calculate loss
                    # Calculate reference standard deviation for training batch
                    ref_std = torch.std(batch_y) + 1e-8
                    reg_loss = self.model.zptae_loss(reg_output.squeeze(), batch_y, ref_std)
                    cls_targets = (batch_y > 0).float()
                    # Ensure shapes are [N,1] for BCEWithLogitsLoss
                    cls_logits = cls_output.view(-1, 1)
                    cls_targets = cls_targets.view(-1, 1)
                    cls_loss = nn.BCEWithLogitsLoss()(cls_logits, cls_targets)
                    
                    loss = 0.7 * reg_loss + 0.3 * cls_loss
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    total_loss += loss.item()
                
                # Validation
                if X_val is not None and y_val is not None:
                    X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val, sequence_length)
                    
                    # Check if sequences were created successfully
                    if X_val_seq is not None and y_val_seq is not None:
                        self.model.eval()
                        with torch.no_grad():
                            val_X = X_val_seq.to(self.device)
                            val_y = y_val_seq.to(self.device)
                            
                            reg_output, _ = self.model(val_X)
                            # Calculate reference standard deviation for validation
                            val_ref_std = torch.std(val_y) + 1e-8
                            val_loss = self.model.zptae_loss(reg_output.squeeze(), val_y, val_ref_std).item()
                    else:
                        logger.warning("Validation sequences could not be created, skipping validation for this epoch")
                        continue
                    
                    scheduler.step(val_loss)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {total_loss/len(X_train_seq)*batch_size:.6f}")
            
            self.is_fitted = True
            self.sequence_length = sequence_length
            logger.info("N-BEATS model training completed successfully")
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting N-BEATS model: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the fitted N-BEATS model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Prepare sequences
            X_seq, _ = self._prepare_sequences(X, sequence_length=self.sequence_length)
            
            # Check if sequences were created
            if X_seq is None or len(X_seq) == 0:
                logger.warning("No sequences could be created for prediction")
                return np.array([])
            
            self.model.eval()
            predictions = []
            
            with torch.no_grad():
                for i in range(0, len(X_seq), 64):  # Batch prediction
                    batch_X = X_seq[i:i+64].to(self.device)
                    reg_output, _ = self.model(batch_X)
                    predictions.append(reg_output.cpu().numpy())
            
            if len(predictions) == 0:
                logger.warning("No predictions generated")
                return np.array([])
            
            predictions = np.concatenate(predictions, axis=0).squeeze()
            
            logger.info(f"Generated {len(predictions)} N-BEATS predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Error making N-BEATS predictions: {str(e)}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for directional forecasting.
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Prepare sequences
            X_seq, _ = self._prepare_sequences(X, sequence_length=self.sequence_length)
            
            # Check if sequences were created
            if X_seq is None or len(X_seq) == 0:
                logger.warning("No sequences could be created for probability prediction")
                return np.array([]).reshape(0, 2)
            
            self.model.eval()
            probabilities = []
            
            with torch.no_grad():
                for i in range(0, len(X_seq), 64):  # Batch prediction
                    batch_X = X_seq[i:i+64].to(self.device)
                    _, cls_output = self.model(batch_X)
                    probabilities.append(cls_output.cpu().numpy())
            
            if len(probabilities) == 0:
                logger.warning("No probabilities generated")
                return np.array([]).reshape(0, 2)
            
            probabilities = np.concatenate(probabilities, axis=0).squeeze()
            
            # Apply sigmoid to logits to get probabilities
            prob_up = 1 / (1 + np.exp(-probabilities))
            prob_down = 1 - prob_up
            
            return np.column_stack([prob_down, prob_up])
            
        except Exception as e:
            logger.error(f"Error making N-BEATS probability predictions: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Extract feature importance (simplified for N-BEATS).
        
        Returns:
            Dictionary of feature names and their importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before extracting feature importance")
        
        # N-BEATS doesn't have explicit feature importance
        # Return uniform importance for sequence positions
        importance_dict = {
            f"lag_{i}": 1.0 / self.sequence_length 
            for i in range(self.sequence_length)
        }
        
        logger.info("Generated uniform feature importance for N-BEATS")
        return importance_dict
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'sequence_length': self.sequence_length,
            }, filepath)
            logger.info(f"N-BEATS model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving N-BEATS model: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> 'NBeatsWrapper':
        """
        Load a trained model.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Self for method chaining
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.config = checkpoint['config']
            self.sequence_length = checkpoint['sequence_length']
            
            # Recreate model
            self.model = NBeatsModel(
                config=self.config,
                input_size=self.sequence_length
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.is_fitted = True
            
            logger.info(f"N-BEATS model loaded from {filepath}")
            return self
            
        except Exception as e:
            logger.error(f"Error loading N-BEATS model: {str(e)}")
            raise


def create_nbeats_model(config: Dict[str, Any]) -> NBeatsWrapper:
    """
    Factory function to create N-BEATS model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized N-BEATS model wrapper
    """
    return NBeatsWrapper(config)