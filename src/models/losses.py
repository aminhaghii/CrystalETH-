"""
Custom loss functions for ETH forecasting project.

This module implements the ZPTAE (Z-Transformed Power-Tanh Absolute Error) loss
function with proper gradient support and numerical stability.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Optional, Tuple
import warnings


class ZPTAELoss(nn.Module):
    """
    Z-Transformed Power-Tanh Absolute Error Loss Function.
    
    This custom loss function implements the ZPTAE metric as a differentiable
    loss for neural network training. It includes numerical stability measures
    and proper gradient computation.
    
    Formula:
    ZPTAE = tanh(a * |z_error|^p)
    where z_error = (pred - true) / ref_std
    """
    
    def __init__(self, a: float = 1.0, p: float = 1.5, epsilon: float = 1e-8):
        """
        Initialize ZPTAE loss function.
        
        Args:
            a: Scale parameter (default 1.0)
            p: Power parameter (default 1.5)
            epsilon: Small constant for numerical stability
        """
        super(ZPTAELoss, self).__init__()
        # Ensure numeric types even if config provided strings
        try:
            self.a = float(a)
            self.p = float(p)
            self.epsilon = float(epsilon)
        except Exception as e:
            logging.error(f"Invalid ZPTAE parameters a={a}, p={p}, epsilon={epsilon}: {e}")
            raise
        
        logging.info(f"ZPTAE Loss initialized with a={self.a}, p={self.p}, epsilon={self.epsilon}")
    
    def forward(self, pred: torch.Tensor, true: torch.Tensor, 
                ref_std: torch.Tensor) -> torch.Tensor:
        """
        Compute ZPTAE loss.
        
        Args:
            pred: Predicted values [batch_size, ...]
            true: True values [batch_size, ...]
            ref_std: Reference standard deviation [batch_size, ...] or scalar
            
        Returns:
            ZPTAE loss value
        """
        # Ensure ref_std is not zero or too small
        ref_std = torch.clamp(ref_std, min=self.epsilon)
        
        # Calculate z-score error
        z_error = (pred - true) / ref_std
        
        # Calculate absolute z-error
        abs_z_error = torch.abs(z_error)
        
        # Apply power transformation with numerical stability
        if self.p != 1.0:
            # Add small epsilon to avoid issues with p < 1 and zero values
            powered_error = torch.pow(abs_z_error + self.epsilon, self.p)
        else:
            powered_error = abs_z_error
        
        # Apply tanh transformation
        zptae = torch.tanh(self.a * powered_error)
        
        # Return mean loss
        return torch.mean(zptae)
    
    def __repr__(self) -> str:
        return f"ZPTAELoss(a={self.a}, p={self.p}, epsilon={self.epsilon})"


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining ZPTAE and Binary Cross-Entropy.
    
    This loss function combines regression (ZPTAE) and classification (BCE)
    objectives for multi-task learning.
    """
    
    def __init__(self, zptae_weight: float = 0.7, bce_weight: float = 0.3,
                 a: float = 1.0, p: float = 1.5, epsilon: float = 1e-8):
        """
        Initialize multi-task loss.
        
        Args:
            zptae_weight: Weight for ZPTAE loss
            bce_weight: Weight for BCE loss
            a: ZPTAE scale parameter
            p: ZPTAE power parameter
            epsilon: Numerical stability constant
        """
        super(MultiTaskLoss, self).__init__()
        
        if abs(zptae_weight + bce_weight - 1.0) > 1e-6:
            logging.warning(f"Loss weights don't sum to 1.0: {zptae_weight + bce_weight}")
        
        self.zptae_weight = zptae_weight
        self.bce_weight = bce_weight
        
        self.zptae_loss = ZPTAELoss(a=a, p=p, epsilon=epsilon)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        logging.info(f"MultiTask Loss initialized: ZPTAE={zptae_weight}, BCE={bce_weight}")
    
    def forward(self, pred_regression: torch.Tensor, pred_classification: torch.Tensor,
                true_regression: torch.Tensor, true_classification: torch.Tensor,
                ref_std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            pred_regression: Regression predictions
            pred_classification: Classification logits
            true_regression: True regression values
            true_classification: True classification labels
            ref_std: Reference standard deviation
            
        Returns:
            Tuple of (total_loss, zptae_loss, bce_loss)
        """
        # Compute individual losses
        zptae = self.zptae_loss(pred_regression, true_regression, ref_std)
        
        # Align shapes for BCE
        logits = pred_classification
        targets = true_classification.float()
        try:
            if logits.shape != targets.shape:
                # Common case: logits [N,1], targets [N]
                if logits.dim() == targets.dim() + 1 and logits.size(-1) == 1 and list(logits.shape[:-1]) == list(targets.shape):
                    targets = targets.unsqueeze(-1)
                # Or logits [N], targets [N,1]
                elif targets.dim() == logits.dim() + 1 and targets.size(-1) == 1 and list(targets.shape[:-1]) == list(logits.shape):
                    logits = logits.unsqueeze(-1)
                # If total elements match, just view
                elif targets.numel() == logits.numel():
                    targets = targets.view_as(logits)
                else:
                    raise ValueError(f"Shape mismatch for BCE: logits {logits.shape}, targets {targets.shape}")
        except Exception as e:
            logging.error(f"Error aligning shapes for BCE: {e}")
            raise
        
        bce = self.bce_loss(logits, targets)
        
        # Combine losses
        total_loss = self.zptae_weight * zptae + self.bce_weight * bce
        
        return total_loss, zptae, bce


def mztae_metric_numpy(pred: np.ndarray, true: np.ndarray, 
                      ref_std: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Compute MZTAE (Mean Z-Transformed Absolute Error) metric using NumPy.
    
    MZTAE Formula:
    MZTAE = mean(|z_error|)
    where z_error = (pred - true) / ref_std
    
    Args:
        pred: Predicted values
        true: True values
        ref_std: Reference standard deviation
        epsilon: Numerical stability constant
        
    Returns:
        MZTAE metric value
    """
    # Ensure ref_std is not zero
    ref_std = np.maximum(ref_std, epsilon)
    
    # Calculate z-score error
    z_error = (pred - true) / ref_std
    
    # Calculate absolute z-error and return mean
    abs_z_error = np.abs(z_error)
    
    return np.mean(abs_z_error)


def zptae_metric_numpy(pred: np.ndarray, true: np.ndarray, 
                      ref_std: np.ndarray, a: float = 1.0, 
                      p: float = 1.5, epsilon: float = 1e-8) -> float:
    """
    Compute ZPTAE metric using NumPy (for evaluation).
    
    Args:
        pred: Predicted values
        true: True values
        ref_std: Reference standard deviation
        a: Scale parameter
        p: Power parameter
        epsilon: Numerical stability constant
        
    Returns:
        ZPTAE metric value
    """
    # Ensure ref_std is not zero
    ref_std = np.maximum(ref_std, epsilon)
    
    # Calculate z-score error
    z_error = (pred - true) / ref_std
    
    # Calculate absolute z-error
    abs_z_error = np.abs(z_error)
    
    # Apply power transformation
    if p != 1.0:
        powered_error = np.power(abs_z_error + epsilon, p)
    else:
        powered_error = abs_z_error
    
    # Apply tanh transformation
    zptae = np.tanh(a * powered_error)
    
    return np.mean(zptae)


def calculate_rolling_std(values: np.ndarray, window: int = 100, 
                         min_periods: int = 30) -> np.ndarray:
    """
    Calculate rolling standard deviation for ZPTAE reference.
    
    Args:
        values: Input values
        window: Rolling window size
        min_periods: Minimum periods required
        
    Returns:
        Rolling standard deviation array
    """
    import pandas as pd
    
    series = pd.Series(values)
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()
    
    # Forward fill initial NaN values
    rolling_std = rolling_std.bfill()
    
    # Ensure no zero values
    rolling_std = np.maximum(rolling_std.values, 1e-8)
    
    return rolling_std


class LightGBMZPTAEObjective:
    """
    ZPTAE objective function for LightGBM.
    
    Since LightGBM doesn't support the exact ZPTAE loss, this provides
    an approximation using a smooth loss function.
    """
    
    def __init__(self, ref_std: np.ndarray, a: float = 1.0, p: float = 1.5):
        """
        Initialize LightGBM ZPTAE objective.
        
        Args:
            ref_std: Reference standard deviation array
            a: Scale parameter
            p: Power parameter
        """
        self.ref_std = ref_std
        self.a = a
        self.p = p
        
    def __call__(self, y_pred: np.ndarray, y_true) -> Tuple[np.ndarray, np.ndarray]:
        """
        LightGBM objective function.
        
        Args:
            y_pred: Predicted values
            y_true: LightGBM dataset with true values
            
        Returns:
            Tuple of (gradient, hessian)
        """
        y_true_values = y_true.get_label()
        
        # Ensure we have the right reference std for this batch
        if len(self.ref_std) != len(y_pred):
            # Use the last available std values
            ref_std_batch = np.full(len(y_pred), self.ref_std[-1])
        else:
            ref_std_batch = self.ref_std
        
        # Calculate z-error
        z_error = (y_pred - y_true_values) / np.maximum(ref_std_batch, 1e-8)
        
        # For gradient computation, we use a smooth approximation
        # grad = sign(z_error) * a * p * |z_error|^(p-1) * sech^2(a * |z_error|^p) / ref_std
        
        abs_z_error = np.abs(z_error)
        sign_z_error = np.sign(z_error)
        
        # Power term
        if self.p != 1.0:
            power_term = np.power(abs_z_error + 1e-8, self.p - 1)
        else:
            power_term = np.ones_like(abs_z_error)
        
        # Tanh term for gradient
        tanh_arg = self.a * np.power(abs_z_error + 1e-8, self.p)
        sech_squared = 1 - np.tanh(tanh_arg) ** 2
        
        # Gradient
        grad = (sign_z_error * self.a * self.p * power_term * sech_squared / 
                np.maximum(ref_std_batch, 1e-8))
        
        # Hessian (approximation)
        hess = np.ones_like(grad) * 0.1  # Simple approximation
        
        return grad, hess


def test_zptae_loss():
    """
    Test ZPTAE loss function with synthetic data.
    """
    logging.info("Testing ZPTAE loss function...")
    
    # Create synthetic data
    torch.manual_seed(42)
    batch_size = 100
    
    pred = torch.randn(batch_size, requires_grad=True)
    true = torch.randn(batch_size)
    ref_std = torch.ones(batch_size) * 0.1
    
    # Test ZPTAE loss
    zptae_loss = ZPTAELoss(a=1.0, p=1.5)
    loss = zptae_loss(pred, true, ref_std)
    
    # Test gradient computation
    loss.backward()
    
    assert pred.grad is not None, "Gradient not computed"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is infinite"
    
    logging.info(f"ZPTAE loss test passed. Loss value: {loss.item():.6f}")
    
    # Test multi-task loss
    pred_class = torch.randn(batch_size, requires_grad=True)
    true_class = torch.randint(0, 2, (batch_size,))
    
    multi_loss = MultiTaskLoss()
    total_loss, zptae, bce = multi_loss(pred, pred_class, true, true_class, ref_std)
    
    total_loss.backward()
    
    assert not torch.isnan(total_loss), "Multi-task loss is NaN"
    logging.info(f"Multi-task loss test passed. Total: {total_loss.item():.6f}")
    
    # Test NumPy version
    pred_np = pred.detach().numpy()
    true_np = true.detach().numpy()
    ref_std_np = ref_std.detach().numpy()
    
    zptae_np = zptae_metric_numpy(pred_np, true_np, ref_std_np)
    
    # Should be close to PyTorch version
    torch_zptae = zptae_loss(pred.detach(), true, ref_std).item()
    assert abs(zptae_np - torch_zptae) < 1e-5, "NumPy and PyTorch versions don't match"
    
    logging.info("All ZPTAE tests passed!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    test_zptae_loss()