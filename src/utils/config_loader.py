"""
Configuration loader utility for ETH forecasting project.

This module provides utilities to load and manage configuration files.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (defaults to config/config.yaml)
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default to config/config.yaml in project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from: {config_path}")
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for production demo.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'paths': {
            'data': 'data',
            'models': 'models',
            'reports': 'reports',
            'logs': 'logs'
        },
        'data': {
            'symbols': ['ETH-USD', 'BTC-USD'],
            'start_date': '2020-01-01',
            'end_date': None,
            'interval': '1d'
        },
        'features': {
            'lookback_days': 30,
            'target_column': 'ETH_log_return',
            'feature_groups': {
                'basic': True,
                'returns_lags': True,
                'rolling_stats': True,
                'volatility': True,
                'momentum': True,
                'cross_asset': True,
                'calendar': True,
                'event_flags': True
            }
        },
        'models': {
            'lightgbm': {
                'use_zptae_loss': True,
                'optuna_trials': 50
            },
            'tft': {
                'hidden_size': 64,
                'num_attention_heads': 4,
                'dropout': 0.1
            },
            'nbeats': {
                'stack_types': ['trend', 'seasonality'],
                'num_blocks': [3, 3],
                'num_layers': 4,
                'layer_widths': 256
            },
            'ensemble': {
                'use_stacking': True,
                'cv_folds': 5
            }
        },
        'validation': {
            'test_size': 0.3,
            'cv_folds': 8,
            'shuffle_tests': 100
        },
        'production': {
            'max_loaded_models': 3,
            'memory_threshold_mb': 4096,
            'model_timeout_hours': 2,
            'ensemble_weights': {
                'lightgbm': 0.3,
                'tft': 0.3,
                'nbeats': 0.3,
                'ensemble': 0.1
            },
            'min_data_quality_score': 0.8,
            'max_prediction_age_minutes': 5
        },
        'live_pipeline': {
            'fetch_interval_seconds': 30,
            'processing_interval_seconds': 15,
            'quality_threshold': 0.7,
            'buffer_size': 100,
            'data_sources': {
                'yahoo_finance': {
                    'enabled': True,
                    'symbols': ['ETH-USD', 'BTC-USD'],
                    'interval': '1m',
                    'period': '1d'
                },
                'coingecko': {
                    'enabled': False,
                    'coin_ids': ['ethereum', 'bitcoin'],
                    'vs_currency': 'usd'
                }
            }
        },
        'api': {
            'cache_ttl_seconds': 300,
            'rate_limit_per_minute': 60,
            'max_concurrent_requests': 10
        },
        'loss_functions': {
            'zptae': {
                'a': 1.0,
                'p': 1.5
            }
        },
        'preprocessing': {
            'nan_threshold': 0.20,
            'hampel_window': 5,
            'hampel_n_sigma': 3,
            'winsorize_lower': 0.005,
            'winsorize_upper': 0.995,
            'wavelet_type': 'db4',
            'rolling_median_window': 3,
            'extreme_event_multiplier': 8,
            'rolling_std_window': 30
        }
    }