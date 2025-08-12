"""
Production Ensemble Manager for ETH Forecasting

This module implements a production-ready ensemble system with:
- On-demand model loading and unloading
- Live data source connections
- Real-time feature engineering pipeline
- Memory-efficient model management
- Automated quality gates and monitoring

Following Rules #10 (Multi-Model Mandate), #16 (Automated Quality Gates),
and #18 (Memory and Performance Management).
"""

import logging
import numpy as np
import pandas as pd
import torch
import joblib
import gc
import psutil
import threading
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from ..models.ensemble import StackingEnsemble
from ..models.lightgbm_model import LightGBMForecaster
from ..models.tft_model import TFTModel
from ..models.nbeats_model import NBeatsWrapper
from ..models.tcn_model import TCNWrapper
from ..data.acquisition import DataAcquisition
from ..features.engineering import FeatureEngineer
from ..evaluation.validation_suite import ValidationSuite

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    model: Any
    load_time: datetime
    last_used: datetime
    memory_usage_mb: float
    performance_metrics: Dict[str, float]
    is_loaded: bool = True


@dataclass
class PredictionResult:
    """Result of ensemble prediction."""
    prediction: float
    direction: int  # 1 for up, 0 for down
    confidence: float
    model_contributions: Dict[str, float]
    feature_importance: Dict[str, float]
    timestamp: datetime
    data_quality_score: float


class EnsembleManager:
    """
    Production-ready ensemble manager with on-demand loading and live data.
    
    Features:
    - Lazy loading of models to conserve memory
    - Automatic model unloading based on usage patterns
    - Live data acquisition and feature engineering
    - Real-time quality monitoring
    - Thread-safe operations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize EnsembleManager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.production_config = config.get('production', {})
        
        # Model management
        self.models: Dict[str, ModelInfo] = {}
        self.model_paths = self._get_model_paths()
        self.max_loaded_models = self.production_config.get('max_loaded_models', 3)
        self.memory_threshold_mb = self.production_config.get('memory_threshold_mb', 4096)
        self.model_timeout_hours = self.production_config.get('model_timeout_hours', 2)
        
        # Live data components
        self.data_acquisition = DataAcquisition(config)
        self.feature_engineer = FeatureEngineer(config)
        self.validation_suite = ValidationSuite(config)
        
        # Ensemble configuration
        self.ensemble_weights = self.production_config.get('ensemble_weights', {
            'lightgbm': 0.25,
            'tft': 0.25,
            'nbeats': 0.25,
            'tcn': 0.15,
            'cnn_lstm': 0.10
        })
        
        # Quality gates
        self.min_data_quality_score = self.production_config.get('min_data_quality_score', 0.8)
        self.max_prediction_age_minutes = self.production_config.get('max_prediction_age_minutes', 5)
        
        # Threading
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="EnsembleManager")
        
        # Cache for recent predictions and data
        self._prediction_cache = {}
        self._data_cache = {}
        self._cache_timeout_minutes = 5
        
        logger.info("EnsembleManager initialized for production use")
    
    def _get_model_paths(self) -> Dict[str, Path]:
        """Get paths to saved model files."""
        models_dir = Path(self.config['paths']['models'])
        
        model_paths = {
            'lightgbm': models_dir / 'lightgbm_baseline_info.joblib',
            'tft': models_dir / 'tft_model.pkl',
            'nbeats': models_dir / 'nbeats_model.pkl',
            'tcn': models_dir / 'tcn_model.pkl',
            'cnn_lstm': models_dir / 'cnn_lstm_model.pkl'
        }
        
        # Verify model files exist and log status
        available_models = []
        for name, path in model_paths.items():
            if path.exists():
                available_models.append(name)
                logger.info(f"✅ Model available: {name} at {path}")
            else:
                logger.warning(f"❌ Model file not found: {path}")
        
        logger.info(f"Available models: {available_models}")
        return model_paths
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _estimate_model_memory(self, model: Any) -> float:
        """Estimate memory usage of a model in MB."""
        try:
            if hasattr(model, 'get_memory_usage'):
                return model.get_memory_usage()
            
            # Rough estimation based on model type
            if isinstance(model, (LightGBMForecaster,)):
                return 50  # LightGBM is typically lightweight
            elif isinstance(model, (TFTModel, NBeatsWrapper)):
                return 200  # Deep learning models are heavier
            elif isinstance(model, StackingEnsemble):
                return 100  # Ensemble combines multiple models
            else:
                return 100  # Default estimate
        except:
            return 100  # Fallback estimate
    
    def _should_unload_model(self, model_info: ModelInfo) -> bool:
        """Determine if a model should be unloaded."""
        # Check if model hasn't been used recently
        time_since_use = datetime.now() - model_info.last_used
        if time_since_use > timedelta(hours=self.model_timeout_hours):
            return True
        
        # Check memory pressure
        current_memory = self._get_memory_usage()
        if current_memory > self.memory_threshold_mb:
            return True
        
        return False
    
    def _unload_least_used_model(self) -> None:
        """Unload the least recently used model."""
        if not self.models:
            return
        
        # Find least recently used model
        lru_model = min(
            self.models.values(),
            key=lambda m: m.last_used if m.is_loaded else datetime.min
        )
        
        if lru_model.is_loaded:
            self._unload_model(lru_model.name)
    
    def _unload_model(self, model_name: str) -> None:
        """Unload a specific model from memory."""
        with self._lock:
            if model_name in self.models and self.models[model_name].is_loaded:
                logger.info(f"Unloading model: {model_name}")
                
                # Clear model from memory
                del self.models[model_name].model
                self.models[model_name].model = None
                self.models[model_name].is_loaded = False
                
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info(f"Model {model_name} unloaded successfully")
    
    def _load_model(self, model_name: str) -> Any:
        """Load a specific model on-demand."""
        with self._lock:
            # Check if model is already loaded
            if model_name in self.models and self.models[model_name].is_loaded:
                self.models[model_name].last_used = datetime.now()
                return self.models[model_name].model
            
            # Check memory constraints
            current_memory = self._get_memory_usage()
            loaded_count = sum(1 for m in self.models.values() if m.is_loaded)
            
            if (loaded_count >= self.max_loaded_models or 
                current_memory > self.memory_threshold_mb):
                self._unload_least_used_model()
            
            # Load the model
            model_path = self.model_paths.get(model_name)
            if not model_path or not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            logger.info(f"Loading model: {model_name}")
            load_start = time.time()
            
            try:
                if model_name == 'lightgbm':
                    model = joblib.load(model_path)
                elif model_name in ['tft', 'nbeats', 'tcn', 'cnn_lstm']:
                    # Load pickled models (these are already trained and saved)
                    import pickle
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                else:
                    raise ValueError(f"Unknown model type: {model_name}")
                
                load_time = time.time() - load_start
                memory_usage = self._estimate_model_memory(model)
                
                # Store model info
                self.models[model_name] = ModelInfo(
                    name=model_name,
                    model=model,
                    load_time=datetime.now(),
                    last_used=datetime.now(),
                    memory_usage_mb=memory_usage,
                    performance_metrics={},
                    is_loaded=True
                )
                
                logger.info(f"Model {model_name} loaded in {load_time:.2f}s, "
                           f"using {memory_usage:.1f}MB")
                
                return model
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {str(e)}")
                raise
    
    def _get_live_data(self, lookback_days: int = 90) -> pd.DataFrame:
        """Get live data for prediction."""
        cache_key = f"live_data_{lookback_days}"
        
        # Check cache first
        if cache_key in self._data_cache:
            cached_data, cache_time = self._data_cache[cache_key]
            if datetime.now() - cache_time < timedelta(minutes=self._cache_timeout_minutes):
                logger.info("Using cached live data")
                return cached_data
        
        logger.info(f"Fetching live data (last {lookback_days} days)")
        
        try:
            # Calculate date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=lookback_days)
            
            # Update data acquisition config for live data
            self.data_acquisition.start_date = start_date.strftime('%Y-%m-%d')
            self.data_acquisition.end_date = end_date.strftime('%Y-%m-%d')
            
            # Get combined data
            raw_data = self.data_acquisition.get_combined_data()
            
            # Cache the data
            self._data_cache[cache_key] = (raw_data, datetime.now())
            
            logger.info(f"Fetched {len(raw_data)} records of live data")
            return raw_data
            
        except Exception as e:
            logger.error(f"Failed to fetch live data: {str(e)}")
            raise
    
    def _engineer_live_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for live data."""
        try:
            logger.info("Engineering features for live data")
            
            # Use feature engineer (not training mode)
            features = self.feature_engineer.engineer_features(
                raw_data, is_training=False
            )
            
            logger.info(f"Engineered {len(features.columns)} features")
            return features
            
        except Exception as e:
            logger.error(f"Failed to engineer features: {str(e)}")
            raise
    
    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """Assess quality of input data."""
        try:
            quality_score = 1.0
            
            # Check for missing values
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            quality_score -= missing_ratio * 0.3
            
            # Check for infinite values
            inf_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
            if inf_count > 0:
                quality_score -= 0.2
            
            # Check data recency
            if hasattr(data.index, 'max'):
                latest_data = data.index.max()
                if isinstance(latest_data, pd.Timestamp):
                    hours_old = (datetime.now(timezone.utc) - latest_data.tz_localize('UTC')).total_seconds() / 3600
                    if hours_old > 24:
                        quality_score -= 0.1
            
            # Check for reasonable value ranges
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 0:
                # Check for extreme outliers
                for col in numeric_data.columns:
                    if col.endswith('_return') or 'return' in col.lower():
                        # Returns should be reasonable
                        extreme_returns = (np.abs(numeric_data[col]) > 0.5).sum()
                        if extreme_returns > len(data) * 0.01:  # More than 1% extreme
                            quality_score -= 0.1
                            break
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.warning(f"Error assessing data quality: {str(e)}")
            return 0.5  # Neutral score if assessment fails
    
    def _make_model_prediction(self, model_name: str, features: pd.DataFrame) -> Tuple[float, float]:
        """Make prediction with a specific model."""
        try:
            model = self._load_model(model_name)
            
            # Get latest features for prediction
            latest_features = features.iloc[-1:].copy()
            
            # Make prediction
            if hasattr(model, 'predict'):
                prediction = model.predict(latest_features)
                if hasattr(prediction, 'item'):
                    prediction = prediction.item()
                elif isinstance(prediction, np.ndarray):
                    prediction = prediction[0] if len(prediction) > 0 else 0.0
            else:
                raise ValueError(f"Model {model_name} does not have predict method")
            
            # Get confidence if available
            confidence = 0.5  # Default confidence
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(latest_features)
                    if hasattr(proba, 'max'):
                        confidence = float(proba.max())
                    elif isinstance(proba, np.ndarray) and len(proba) > 0:
                        confidence = float(np.max(proba))
                except:
                    pass
            
            return float(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Error making prediction with {model_name}: {str(e)}")
            return 0.0, 0.0
    
    def predict(self, 
                custom_data: Optional[pd.DataFrame] = None,
                use_cache: bool = True) -> PredictionResult:
        """
        Make ensemble prediction with live data.
        
        Args:
            custom_data: Optional custom data (if None, fetches live data)
            use_cache: Whether to use cached predictions
            
        Returns:
            PredictionResult with ensemble prediction and metadata
        """
        try:
            prediction_start = time.time()
            
            # Check cache if enabled
            cache_key = "latest_prediction"
            if use_cache and cache_key in self._prediction_cache:
                cached_result, cache_time = self._prediction_cache[cache_key]
                if datetime.now() - cache_time < timedelta(minutes=self.max_prediction_age_minutes):
                    logger.info("Using cached prediction")
                    return cached_result
            
            # Get data
            if custom_data is not None:
                raw_data = custom_data.copy()
                logger.info("Using provided custom data")
            else:
                raw_data = self._get_live_data()
            
            # Engineer features
            features = self._engineer_live_features(raw_data)
            
            # Assess data quality
            data_quality_score = self._assess_data_quality(features)
            
            if data_quality_score < self.min_data_quality_score:
                logger.warning(f"Low data quality score: {data_quality_score:.3f}")
            
            # Make predictions with available models
            model_predictions = {}
            model_confidences = {}
            
            # Use ThreadPoolExecutor for parallel predictions
            with ThreadPoolExecutor(max_workers=len(self.model_paths)) as executor:
                future_to_model = {
                    executor.submit(self._make_model_prediction, model_name, features): model_name
                    for model_name in self.model_paths.keys()
                    if self.model_paths[model_name].exists()
                }
                
                for future in as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        prediction, confidence = future.result(timeout=30)
                        model_predictions[model_name] = prediction
                        model_confidences[model_name] = confidence
                    except Exception as e:
                        logger.error(f"Model {model_name} prediction failed: {str(e)}")
                        model_predictions[model_name] = 0.0
                        model_confidences[model_name] = 0.0
            
            # Combine predictions using ensemble weights
            ensemble_prediction = 0.0
            total_weight = 0.0
            
            for model_name, prediction in model_predictions.items():
                weight = self.ensemble_weights.get(model_name, 0.25)
                ensemble_prediction += prediction * weight
                total_weight += weight
            
            if total_weight > 0:
                ensemble_prediction /= total_weight
            
            # Calculate overall confidence
            confidences = list(model_confidences.values())
            overall_confidence = np.mean(confidences) if confidences else 0.5
            
            # Determine direction
            direction = 1 if ensemble_prediction > 0 else 0
            
            # Get feature importance (simplified)
            feature_importance = {}
            if len(features.columns) > 0:
                # Use variance as a proxy for importance
                feature_vars = features.var()
                top_features = feature_vars.nlargest(10)
                total_var = top_features.sum()
                if total_var > 0:
                    feature_importance = {
                        col: float(var / total_var) 
                        for col, var in top_features.items()
                    }
            
            # Create result
            result = PredictionResult(
                prediction=float(ensemble_prediction),
                direction=direction,
                confidence=float(overall_confidence),
                model_contributions=model_predictions,
                feature_importance=feature_importance,
                timestamp=datetime.now(timezone.utc),
                data_quality_score=float(data_quality_score)
            )
            
            # Cache result
            if use_cache:
                self._prediction_cache[cache_key] = (result, datetime.now())
            
            prediction_time = time.time() - prediction_start
            logger.info(f"Ensemble prediction completed in {prediction_time:.2f}s: "
                       f"{ensemble_prediction:.6f} (direction: {direction}, "
                       f"confidence: {overall_confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to make ensemble prediction: {str(e)}")
            raise
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models."""
        with self._lock:
            status = {
                'loaded_models': len([m for m in self.models.values() if m.is_loaded]),
                'total_models': len(self.model_paths),
                'memory_usage_mb': self._get_memory_usage(),
                'memory_threshold_mb': self.memory_threshold_mb,
                'models': {}
            }
            
            for name, path in self.model_paths.items():
                model_status = {
                    'path_exists': path.exists(),
                    'is_loaded': name in self.models and self.models[name].is_loaded,
                    'last_used': None,
                    'memory_usage_mb': 0
                }
                
                if name in self.models:
                    model_info = self.models[name]
                    model_status.update({
                        'last_used': model_info.last_used.isoformat(),
                        'memory_usage_mb': model_info.memory_usage_mb,
                        'load_time': model_info.load_time.isoformat()
                    })
                
                status['models'][name] = model_status
            
            return status
    
    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up EnsembleManager resources")
        
        # Unload all models
        with self._lock:
            for model_name in list(self.models.keys()):
                if self.models[model_name].is_loaded:
                    self._unload_model(model_name)
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        # Clear caches
        self._prediction_cache.clear()
        self._data_cache.clear()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("EnsembleManager cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


def main():
    """Test the EnsembleManager."""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.utils.config_loader import load_config
    
    # Load configuration
    config = load_config()
    
    # Test ensemble manager
    with EnsembleManager(config) as manager:
        print("🚀 Testing Production Ensemble Manager")
        print("=" * 50)
        
        # Get model status
        status = manager.get_model_status()
        print(f"Model Status:")
        print(f"  Loaded: {status['loaded_models']}/{status['total_models']}")
        print(f"  Memory: {status['memory_usage_mb']:.1f}MB")
        
        # Make prediction
        try:
            result = manager.predict()
            print(f"\nPrediction Result:")
            print(f"  Prediction: {result.prediction:.6f}")
            print(f"  Direction: {'UP' if result.direction == 1 else 'DOWN'}")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Data Quality: {result.data_quality_score:.3f}")
            print(f"  Timestamp: {result.timestamp}")
            
            print(f"\nModel Contributions:")
            for model, contrib in result.model_contributions.items():
                print(f"  {model}: {contrib:.6f}")
            
        except Exception as e:
            print(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()