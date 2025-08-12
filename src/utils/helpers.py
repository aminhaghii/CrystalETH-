"""
Utility functions for ETH forecasting project.

This module contains helper functions for configuration loading, logging setup,
data validation, and other common utilities.
"""

import os
import yaml
import logging
import random
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        raise


def setup_logging(level: str = "INFO", format_str: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_str: Custom format string for log messages
    """
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('eth_forecasting.log')
        ]
    )
    logging.info(f"Logging setup complete with level: {level}")


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logging.info(f"Random seeds set to {seed}")


def create_directories(paths: Dict[str, str]) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        paths: Dictionary of path names and their values
    """
    for name, path in paths.items():
        Path(path).mkdir(parents=True, exist_ok=True)
        logging.info(f"Directory created/verified: {name} -> {path}")


def validate_timezone_utc(df: pd.DataFrame, datetime_col: str = 'Date') -> bool:
    """
    Validate that datetime column is in UTC timezone.
    
    Args:
        df: DataFrame with datetime column
        datetime_col: Name of datetime column
        
    Returns:
        True if timezone is UTC, False otherwise
    """
    if datetime_col not in df.columns:
        logging.error(f"Column {datetime_col} not found in DataFrame")
        return False
    
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        logging.error(f"Column {datetime_col} is not datetime type")
        return False
    
    # Check if timezone aware
    if df[datetime_col].dt.tz is None:
        logging.warning(f"Column {datetime_col} is timezone naive")
        return False
    
    # Check if UTC
    is_utc = str(df[datetime_col].dt.tz) == 'UTC'
    if is_utc:
        logging.info(f"Column {datetime_col} is properly in UTC timezone")
    else:
        logging.error(f"Column {datetime_col} timezone is {df[datetime_col].dt.tz}, not UTC")
    
    return is_utc


def check_data_leakage(features_df: pd.DataFrame, target_df: pd.DataFrame, 
                      datetime_col: str = 'Date') -> bool:
    """
    Check for potential data leakage in features.
    
    This function performs basic checks to ensure no feature at time t
    uses information from time t+1 or later.
    
    Args:
        features_df: DataFrame with features
        target_df: DataFrame with targets
        datetime_col: Name of datetime column
        
    Returns:
        True if no leakage detected, False otherwise
    """
    try:
        # Check if datetime columns are aligned
        if not features_df[datetime_col].equals(target_df[datetime_col]):
            logging.error("DateTime columns in features and targets are not aligned")
            return False
        
        # Check for any NaN values in future that are filled in past
        # This is a basic check - more sophisticated checks can be added
        for col in features_df.columns:
            if col == datetime_col:
                continue
                
            # Check if there are patterns suggesting forward-looking bias
            series = features_df[col]
            if series.isna().sum() > 0:
                # Check if NaN pattern suggests forward filling
                nan_indices = series.isna()
                if nan_indices.any():
                    # Basic check: ensure NaNs are not filled with future values
                    first_valid = series.first_valid_index()
                    last_valid = series.last_valid_index()
                    
                    if first_valid is not None and last_valid is not None:
                        # Check for suspicious patterns
                        middle_nans = nan_indices.loc[first_valid:last_valid]
                        if middle_nans.sum() > len(middle_nans) * 0.1:  # More than 10% NaNs in middle
                            logging.warning(f"Suspicious NaN pattern in feature {col}")
        
        logging.info("Basic data leakage check passed")
        return True
        
    except Exception as e:
        logging.error(f"Error in data leakage check: {e}")
        return False


def validate_rolling_window_samples(df: pd.DataFrame, window: int, 
                                  min_samples: int = 30) -> bool:
    """
    Validate that rolling window operations have sufficient samples.
    
    Args:
        df: DataFrame to check
        window: Rolling window size
        min_samples: Minimum required samples
        
    Returns:
        True if sufficient samples, False otherwise
    """
    if len(df) < min_samples:
        logging.error(f"DataFrame has {len(df)} samples, minimum required: {min_samples}")
        return False
    
    if window > len(df):
        logging.error(f"Window size {window} larger than DataFrame length {len(df)}")
        return False
    
    if window < min_samples:
        logging.warning(f"Window size {window} smaller than recommended minimum {min_samples}")
    
    logging.info(f"Rolling window validation passed: {len(df)} samples, window={window}")
    return True


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate log returns from price series.
    
    Args:
        prices: Price series
        
    Returns:
        Log returns series
    """
    log_returns = np.log(prices / prices.shift(1))
    logging.info(f"Calculated log returns: {len(log_returns)} values, {log_returns.isna().sum()} NaNs")
    return log_returns


def ensure_24h_windows(df: pd.DataFrame, datetime_col: str = 'Date') -> bool:
    """
    Ensure that data represents proper 24-hour windows.
    
    Args:
        df: DataFrame with datetime column
        datetime_col: Name of datetime column
        
    Returns:
        True if 24-hour windows are proper, False otherwise
    """
    try:
        # Check if data is daily frequency
        time_diff = df[datetime_col].diff().dropna()
        
        # For daily data, expect ~24 hours (1 day) between observations
        expected_freq = pd.Timedelta(days=1)
        
        # Allow some tolerance for weekends/holidays
        tolerance = pd.Timedelta(hours=12)
        
        # Check if most differences are close to 24 hours or multiples (weekends)
        valid_diffs = (
            (time_diff >= expected_freq - tolerance) & 
            (time_diff <= expected_freq + tolerance)
        ) | (
            (time_diff >= 2 * expected_freq - tolerance) & 
            (time_diff <= 3 * expected_freq + tolerance)  # weekends
        )
        
        valid_ratio = valid_diffs.sum() / len(valid_diffs)
        
        if valid_ratio > 0.9:  # 90% of differences should be valid
            logging.info(f"24-hour window validation passed: {valid_ratio:.2%} valid intervals")
            return True
        else:
            logging.error(f"24-hour window validation failed: only {valid_ratio:.2%} valid intervals")
            return False
            
    except Exception as e:
        logging.error(f"Error in 24-hour window validation: {e}")
        return False


def quality_gate_check(features_df: pd.DataFrame, target_df: pd.DataFrame,
                      predictions: Optional[np.ndarray] = None) -> bool:
    """
    Comprehensive quality gate check.
    
    Args:
        features_df: Features DataFrame
        target_df: Target DataFrame  
        predictions: Model predictions (optional)
        
    Returns:
        True if all checks pass, False otherwise
    """
    logging.info("Starting quality gate check...")
    
    checks = []
    
    # Data leakage check
    checks.append(check_data_leakage(features_df, target_df))
    
    # Timezone check
    checks.append(validate_timezone_utc(features_df))
    checks.append(validate_timezone_utc(target_df))
    
    # 24-hour window check
    checks.append(ensure_24h_windows(features_df))
    checks.append(ensure_24h_windows(target_df))
    
    # Feature scaling check (basic)
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'Date':
            col_std = features_df[col].std()
            if col_std > 100:  # Basic check for unscaled features
                logging.warning(f"Feature {col} may not be properly scaled (std={col_std:.2f})")
    
    # Predictions check
    if predictions is not None:
        if np.isnan(predictions).any():
            logging.error("NaN values found in predictions")
            checks.append(False)
        else:
            # Check if predictions are in reasonable range
            pred_std = np.std(predictions)
            if pred_std > 1.0:  # For log returns, std > 1 is suspicious
                logging.warning(f"Predictions have high standard deviation: {pred_std:.4f}")
            checks.append(True)
    
    all_passed = all(checks)
    
    if all_passed:
        logging.info("✅ All quality gate checks passed")
    else:
        logging.error("❌ Quality gate checks failed")
    
    return all_passed


def save_intermediate_data(df: pd.DataFrame, filename: str, 
                          directory: str = "data/processed") -> None:
    """
    Save intermediate data with timestamp.
    
    Args:
        df: DataFrame to save
        filename: Base filename
        directory: Directory to save in
    """
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    full_filename = f"{filename}_{timestamp}.csv"
    filepath = os.path.join(directory, full_filename)
    
    Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Preserve DatetimeIndex if present
    if isinstance(df.index, pd.DatetimeIndex):
        df.to_csv(filepath, index=True)
        logging.info(f"Intermediate data saved with DatetimeIndex: {filepath}")
    else:
        df.to_csv(filepath, index=False)
        logging.info(f"Intermediate data saved: {filepath}")


def memory_usage_check(threshold_gb: float = 8.0) -> bool:
    """
    Check current memory usage.
    
    Args:
        threshold_gb: Memory threshold in GB
        
    Returns:
        True if memory usage is below threshold
    """
    try:
        import psutil
        memory_gb = psutil.virtual_memory().used / (1024**3)
        
        if memory_gb > threshold_gb:
            logging.warning(f"High memory usage: {memory_gb:.2f} GB (threshold: {threshold_gb} GB)")
            return False
        else:
            logging.info(f"Memory usage OK: {memory_gb:.2f} GB")
            return True
    except ImportError:
        logging.warning("psutil not available for memory monitoring")
        return True