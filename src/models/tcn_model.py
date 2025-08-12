"""
Temporal Convolutional Network (TCN) Model Implementation

This module implements TCN and CNN-LSTM models for time series forecasting,
following Rule #10 (Multi-Model Mandate) and Rule #3 (ZPTAE Loss Implementation).

TCN uses dilated causal convolutions for long-range dependencies,
while CNN-LSTM combines convolutional feature extraction with LSTM memory.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import optuna
import joblib
import warnings
warnings.filterwarnings('ignore')

from .losses import ZPTAELoss, MultiTaskLoss, zptae_metric_numpy

logger = logging.getLogger(__name__)


class TemporalBlock(nn.Module):
    """
    Temporal block for TCN with dilated causal convolutions.
    """
    
    def __init__(self, 
                 n_inputs: int, 
                 n_outputs: int, 
                 kernel_size: int, 
                 stride: int, 
                 dilation: int, 
                 padding: int, 
                 dropout: float = 0.2):
        """
        Initialize temporal block.
        
        Args:
            n_inputs: Number of input channels
            n_outputs: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            dilation: Dilation factor
            padding: Padding size
            dropout: Dropout rate
        """
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        """Forward pass."""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """
    Chomp layer to ensure causal convolutions.
    """
    
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network.
    """
    
    def __init__(self, 
                 num_inputs: int, 
                 num_channels: List[int], 
                 kernel_size: int = 2, 
                 dropout: float = 0.2):
        """
        Initialize TCN.
        
        Args:
            num_inputs: Number of input features
            num_channels: List of channel sizes for each layer
            kernel_size: Convolution kernel size
            dropout: Dropout rate
        """
        super(TemporalConvNet, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                   stride=1, dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size, 
                                   dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        return self.network(x)


class TCNModel(nn.Module):
    """
    Complete TCN model for time series forecasting.
    """
    
    def __init__(self, 
                 input_size: int,
                 sequence_length: int,
                 num_channels: List[int] = [25, 25, 25],
                 kernel_size: int = 2,
                 dropout: float = 0.2,
                 output_size: int = 1):
        """
        Initialize TCN model.
        
        Args:
            input_size: Number of input features
            sequence_length: Length of input sequences
            num_channels: List of channel sizes
            kernel_size: Convolution kernel size
            dropout: Dropout rate
            output_size: Number of output features
        """
        super(TCNModel, self).__init__()
        
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.dropout = nn.Dropout(dropout)
        
        # For classification
        self.classifier = nn.Linear(num_channels[-1], 1)
        
    def forward(self, x):
        """Forward pass."""
        # x shape: (batch_size, sequence_length, input_size)
        # TCN expects: (batch_size, input_size, sequence_length)
        x = x.transpose(1, 2)
        
        # TCN forward
        tcn_out = self.tcn(x)  # (batch_size, num_channels[-1], sequence_length)
        
        # Take the last time step
        tcn_out = tcn_out[:, :, -1]  # (batch_size, num_channels[-1])
        
        # Apply dropout
        tcn_out = self.dropout(tcn_out)
        
        # Regression output
        regression_out = self.linear(tcn_out)
        
        # Classification output
        classification_out = self.classifier(tcn_out)
        
        return regression_out, classification_out


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM model combining convolutional feature extraction with LSTM memory.
    """
    
    def __init__(self,
                 input_size: int,
                 sequence_length: int,
                 cnn_filters: int = 64,
                 cnn_kernel_size: int = 3,
                 lstm_hidden_size: int = 50,
                 lstm_layers: int = 2,
                 dropout: float = 0.2,
                 output_size: int = 1):
        """
        Initialize CNN-LSTM model.
        
        Args:
            input_size: Number of input features
            sequence_length: Length of input sequences
            cnn_filters: Number of CNN filters
            cnn_kernel_size: CNN kernel size
            lstm_hidden_size: LSTM hidden size
            lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size: Number of output features
        """
        super(CNNLSTMModel, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(input_size, cnn_filters, cnn_kernel_size, padding=1)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters, cnn_kernel_size, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout_cnn = nn.Dropout(dropout)
        
        # Calculate LSTM input size after CNN
        cnn_output_length = sequence_length // 2  # After pooling
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout_lstm = nn.Dropout(dropout)
        
        # Output layers
        self.linear = nn.Linear(lstm_hidden_size, output_size)
        self.classifier = nn.Linear(lstm_hidden_size, 1)
        
    def forward(self, x):
        """Forward pass."""
        # x shape: (batch_size, sequence_length, input_size)
        # CNN expects: (batch_size, input_size, sequence_length)
        x = x.transpose(1, 2)
        
        # CNN forward
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout_cnn(x)
        
        # Prepare for LSTM: (batch_size, sequence_length, features)
        x = x.transpose(1, 2)
        
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the last time step
        lstm_out = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_size)
        lstm_out = self.dropout_lstm(lstm_out)
        
        # Regression output
        regression_out = self.linear(lstm_out)
        
        # Classification output
        classification_out = self.classifier(lstm_out)
        
        return regression_out, classification_out


class TCNWrapper:
    """
    Scikit-learn compatible wrapper for TCN models.
    """
    
    def __init__(self, 
                 model_type: str = 'tcn',
                 sequence_length: int = 30,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize TCN wrapper.
        
        Args:
            model_type: 'tcn' or 'cnn_lstm'
            sequence_length: Length of input sequences
            config: Configuration dictionary
        """
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.config = config or {}
        
        # Model parameters
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training parameters
        self.is_fitted = False
        self.input_size = None
        
        # Loss functions
        loss_config = self.config.get('loss', {})
        self.zptae_loss = ZPTAELoss(
            a=loss_config.get('zptae', {}).get('a', 1.0),
            p=loss_config.get('zptae', {}).get('p', 1.5),
            epsilon=loss_config.get('zptae', {}).get('epsilon', 1e-8)
        )
        
        self.multitask_loss = MultiTaskLoss(
            zptae_weight=loss_config.get('multitask', {}).get('zptae_weight', 0.7),
            bce_weight=loss_config.get('multitask', {}).get('bce_weight', 0.3)
        )
        
        logger.info(f"TCN wrapper initialized with model type: {model_type}")
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series modeling.
        
        Args:
            X: Input features
            y: Target values (optional)
            
        Returns:
            Tuple of (sequences, targets)
        """
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(X)):
            sequences.append(X[i-self.sequence_length:i])
            if y is not None:
                targets.append(y[i])
        
        sequences = np.array(sequences)
        targets = np.array(targets) if y is not None else None
        
        return sequences, targets
    
    def _create_model(self, input_size: int) -> nn.Module:
        """
        Create the neural network model.
        
        Args:
            input_size: Number of input features
            
        Returns:
            Initialized model
        """
        if self.model_type == 'tcn':
            model_config = self.config.get('tcn', {})
            return TCNModel(
                input_size=input_size,
                sequence_length=self.sequence_length,
                num_channels=model_config.get('num_channels', [25, 25, 25]),
                kernel_size=model_config.get('kernel_size', 2),
                dropout=model_config.get('dropout', 0.2),
                output_size=1
            )
        
        elif self.model_type == 'cnn_lstm':
            model_config = self.config.get('cnn_lstm', {})
            return CNNLSTMModel(
                input_size=input_size,
                sequence_length=self.sequence_length,
                cnn_filters=model_config.get('cnn_filters', 64),
                cnn_kernel_size=model_config.get('cnn_kernel_size', 3),
                lstm_hidden_size=model_config.get('lstm_hidden_size', 50),
                lstm_layers=model_config.get('lstm_layers', 2),
                dropout=model_config.get('dropout', 0.2),
                output_size=1
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'TCNWrapper':
        """
        Fit the TCN model.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Self for method chaining
        """
        try:
            logger.info(f"Starting {self.model_type.upper()} training...")
            
            # Prepare data
            X_scaled = self.scaler.fit_transform(X)
            y_values = y.values
            
            # Create sequences
            X_seq, y_seq = self._create_sequences(X_scaled, y_values)
            
            if len(X_seq) == 0:
                raise ValueError("Not enough data to create sequences")
            
            self.input_size = X_seq.shape[2]
            
            # Create model
            self.model = self._create_model(self.input_size)
            self.model.to(self.device)
            
            # Training parameters
            training_config = self.config.get('training', {})
            batch_size = training_config.get('batch_size', 32)
            epochs = training_config.get('epochs', 100)
            learning_rate = training_config.get('learning_rate', 0.001)
            patience = training_config.get('patience', 10)
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_seq),
                torch.FloatTensor(y_seq)
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Validation data
            val_loader = None
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val.values)
                
                if len(X_val_seq) > 0:
                    val_dataset = TensorDataset(
                        torch.FloatTensor(X_val_seq),
                        torch.FloatTensor(y_val_seq)
                    )
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Optimizer
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training
                self.model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    reg_pred, cls_pred = self.model(batch_X)
                    reg_pred = reg_pred.squeeze()
                    
                    # Create binary targets
                    cls_target = (batch_y > 0).long()
                    
                    # Calculate reference standard deviation for ZPTAE
                    ref_std = torch.std(batch_y) + 1e-8  # Add epsilon for numerical stability
                    
                    # Calculate loss
                    loss, _, _ = self.multitask_loss(
                        reg_pred, cls_pred, batch_y, cls_target.float(), ref_std
                    )
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                # Validation
                val_loss = 0.0
                if val_loader is not None:
                    self.model.eval()
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            batch_X = batch_X.to(self.device)
                            batch_y = batch_y.to(self.device)
                            
                            reg_pred, cls_pred = self.model(batch_X)
                            reg_pred = reg_pred.squeeze()
                            cls_target = (batch_y > 0).long()
                            
                            # Calculate reference standard deviation for validation
                            val_ref_std = torch.std(batch_y) + 1e-8
                            
                            loss, _, _ = self.multitask_loss(
                                reg_pred, cls_pred, batch_y, cls_target.float(), val_ref_std
                            )
                            val_loss += loss.item()
                    
                    val_loss /= len(val_loader)
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
                
                # Logging
                if epoch % 10 == 0:
                    if val_loader is not None:
                        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                    else:
                        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}")
            
            self.is_fitted = True
            logger.info(f"{self.model_type.upper()} training completed")
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting {self.model_type}: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Prepare data
            X_scaled = self.scaler.transform(X)
            X_seq, _ = self._create_sequences(X_scaled)
            
            if len(X_seq) == 0:
                # Return zeros if not enough data
                return np.zeros(len(X))
            
            # Make predictions
            self.model.eval()
            predictions = []
            
            with torch.no_grad():
                for i in range(0, len(X_seq), 32):  # Batch processing
                    batch = X_seq[i:i+32]
                    batch_tensor = torch.FloatTensor(batch).to(self.device)
                    
                    reg_pred, _ = self.model(batch_tensor)
                    predictions.extend(reg_pred.squeeze().cpu().numpy())
            
            # Pad predictions to match input length
            full_predictions = np.zeros(len(X))
            full_predictions[self.sequence_length:] = predictions
            
            logger.info(f"Generated {len(predictions)} {self.model_type} predictions")
            return full_predictions
            
        except Exception as e:
            logger.error(f"Error making {self.model_type} predictions: {str(e)}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions using the fitted model.
        
        Args:
            X: Input features
            
        Returns:
            Probability predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Prepare data
            X_scaled = self.scaler.transform(X)
            X_seq, _ = self._create_sequences(X_scaled)
            
            if len(X_seq) == 0:
                # Return neutral probabilities if not enough data
                return np.full((len(X), 2), 0.5)
            
            # Make predictions
            self.model.eval()
            probabilities = []
            
            with torch.no_grad():
                for i in range(0, len(X_seq), 64):
                    batch_X = torch.FloatTensor(X_seq[i:i+64]).to(self.device)
                    _, cls_pred = self.model(batch_X)
                    probabilities.append(cls_pred.cpu().numpy())
            
            probabilities = np.concatenate(probabilities, axis=0).squeeze()
            # Convert logits to probabilities using sigmoid
            prob_up = 1 / (1 + np.exp(-probabilities))
            prob_down = 1 - prob_up
            
            # Align probabilities with input length
            padding = np.full((len(X) - len(prob_up),), 0.5)
            prob_up_full = np.concatenate([padding, prob_up])
            prob_down_full = np.concatenate([padding, prob_down])
            
            return np.column_stack([prob_down_full, prob_up_full])
            
        except Exception as e:
            logger.error(f"Error making {self.model_type} probability predictions: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance (placeholder for neural networks).
        
        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before extracting feature importance")
        
        # For neural networks, feature importance is not straightforward
        # Return uniform importance
        if self.input_size is not None:
            importance = 1.0 / self.input_size
            return {f"feature_{i}": importance for i in range(self.input_size)}
        else:
            return {}
    
    def optimize_hyperparameters(self, 
                                X_train: pd.DataFrame, 
                                y_train: pd.Series,
                                X_val: pd.DataFrame,
                                y_val: pd.Series,
                                n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of optimization trials
            
        Returns:
            Best hyperparameters
        """
        def objective(trial):
            try:
                # Suggest hyperparameters
                if self.model_type == 'tcn':
                    params = {
                        'num_channels': [
                            trial.suggest_int('ch1', 16, 64),
                            trial.suggest_int('ch2', 16, 64),
                            trial.suggest_int('ch3', 16, 64)
                        ],
                        'kernel_size': trial.suggest_int('kernel_size', 2, 5),
                        'dropout': trial.suggest_float('dropout', 0.1, 0.5)
                    }
                else:  # cnn_lstm
                    params = {
                        'cnn_filters': trial.suggest_int('cnn_filters', 32, 128),
                        'cnn_kernel_size': trial.suggest_int('cnn_kernel_size', 3, 7),
                        'lstm_hidden_size': trial.suggest_int('lstm_hidden_size', 32, 128),
                        'lstm_layers': trial.suggest_int('lstm_layers', 1, 3),
                        'dropout': trial.suggest_float('dropout', 0.1, 0.5)
                    }
                
                # Update config
                temp_config = self.config.copy()
                temp_config[self.model_type] = params
                temp_config['training'] = {
                    'epochs': 50,  # Reduced for optimization
                    'batch_size': trial.suggest_int('batch_size', 16, 64),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                    'patience': 10
                }
                
                # Create temporary model
                temp_model = TCNWrapper(
                    model_type=self.model_type,
                    sequence_length=self.sequence_length,
                    config=temp_config
                )
                
                # Fit and evaluate
                temp_model.fit(X_train, y_train, X_val, y_val)
                predictions = temp_model.predict(X_val)
                
                # Calculate ZPTAE metric
                zptae_score = zptae_metric_numpy(y_val.values, predictions[self.sequence_length:])
                
                return zptae_score
                
            except Exception as e:
                logger.warning(f"Trial failed: {str(e)}")
                return float('inf')
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Hyperparameter optimization completed. Best ZPTAE: {study.best_value:.6f}")
        
        return study.best_params
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        try:
            model_data = {
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'model_type': self.model_type,
                'sequence_length': self.sequence_length,
                'input_size': self.input_size,
                'config': self.config,
                'is_fitted': self.is_fitted
            }
            
            torch.save(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> 'TCNWrapper':
        """
        Load a model.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Self for method chaining
        """
        try:
            model_data = torch.load(filepath, map_location=self.device)
            
            self.scaler = model_data['scaler']
            self.model_type = model_data['model_type']
            self.sequence_length = model_data['sequence_length']
            self.input_size = model_data['input_size']
            self.config = model_data['config']
            self.is_fitted = model_data['is_fitted']
            
            # Recreate model
            self.model = self._create_model(self.input_size)
            self.model.load_state_dict(model_data['model_state_dict'])
            self.model.to(self.device)
            
            logger.info(f"Model loaded from {filepath}")
            return self
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


def create_tcn_model(config: Dict[str, Any]) -> TCNWrapper:
    """
    Factory function to create TCN model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized TCN model
    """
    model_config = config.get('tcn', {})
    
    return TCNWrapper(
        model_type='tcn',
        sequence_length=model_config.get('sequence_length', 30),
        config=config
    )


def create_cnn_lstm_model(config: Dict[str, Any]) -> TCNWrapper:
    """
    Factory function to create CNN-LSTM model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized CNN-LSTM model
    """
    model_config = config.get('cnn_lstm', {})
    
    return TCNWrapper(
        model_type='cnn_lstm',
        sequence_length=model_config.get('sequence_length', 30),
        config=config
    )