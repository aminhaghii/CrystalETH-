"""
Production Prediction API for ETH Forecasting

This module implements a production-ready API that:
- Integrates EnsembleManager with LiveDataPipeline
- Provides real-time predictions via REST API
- Implements caching and rate limiting
- Monitors model performance and data quality
- Handles graceful degradation and failover

Following Rules #16 (Automated Quality Gates), #17 (Error Handling),
and #18 (Memory and Performance Management).
"""

import logging
import asyncio
import json
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available. Install with: pip install fastapi uvicorn")

from .ensemble_manager import EnsembleManager
from .live_data_pipeline import LiveDataPipeline, DataQualityMetrics
from ..evaluation.validation_suite import ValidationSuite

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Prediction result with metadata."""
    timestamp: datetime
    prediction: float
    confidence: float
    model_contributions: Dict[str, float]
    data_quality: float
    features_used: int
    processing_time_ms: float
    model_version: str
    warnings: List[str] = None


@dataclass
class SystemStatus:
    """System health and status information."""
    status: str  # 'healthy', 'degraded', 'critical'
    uptime_seconds: float
    models_loaded: List[str]
    data_pipeline_status: str
    last_prediction: Optional[datetime]
    predictions_count: int
    error_rate: float
    memory_usage_mb: float
    quality_score: float


# Pydantic models for API
if FASTAPI_AVAILABLE:
    class PredictionRequest(BaseModel):
        """Request model for predictions."""
        use_live_data: bool = Field(True, description="Use live data or provide custom data")
        custom_data: Optional[Dict[str, Any]] = Field(None, description="Custom data for prediction")
        model_names: Optional[List[str]] = Field(None, description="Specific models to use")
        include_explanations: bool = Field(False, description="Include model explanations")
    
    class PredictionResponse(BaseModel):
        """Response model for predictions."""
        success: bool
        prediction: Optional[float]
        confidence: Optional[float]
        timestamp: str
        model_contributions: Dict[str, float]
        data_quality: float
        processing_time_ms: float
        warnings: List[str]
        metadata: Dict[str, Any]
    
    class StatusResponse(BaseModel):
        """Response model for system status."""
        status: str
        uptime_seconds: float
        models_loaded: List[str]
        data_pipeline_status: str
        last_prediction: Optional[str]
        predictions_count: int
        error_rate: float
        memory_usage_mb: float
        quality_score: float
        timestamp: str


class PredictionCache:
    """Simple in-memory cache for predictions."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 60):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[PredictionResult, datetime]] = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[PredictionResult]:
        """Get cached prediction if valid."""
        with self._lock:
            if key in self.cache:
                result, timestamp = self.cache[key]
                if (datetime.now() - timestamp).total_seconds() < self.ttl_seconds:
                    return result
                else:
                    del self.cache[key]
            return None
    
    def put(self, key: str, result: PredictionResult) -> None:
        """Cache prediction result."""
        with self._lock:
            # Clean old entries if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(), 
                               key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            
            self.cache[key] = (result, datetime.now())
    
    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self.cache.clear()


class RateLimiter:
    """Simple rate limiter."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: List[datetime] = []
        self._lock = threading.Lock()
    
    def is_allowed(self) -> bool:
        """Check if request is allowed."""
        now = datetime.now()
        
        with self._lock:
            # Remove old requests
            cutoff = now - timedelta(seconds=self.window_seconds)
            self.requests = [req_time for req_time in self.requests if req_time > cutoff]
            
            # Check limit
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False


class PredictionAPI:
    """
    Production Prediction API for ETH Forecasting.
    
    Integrates ensemble models with live data pipeline to provide
    real-time predictions via REST API.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PredictionAPI.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.api_config = config.get('prediction_api', {})
        
        # Core components
        self.ensemble_manager = EnsembleManager(config)
        self.live_pipeline = LiveDataPipeline(config)
        self.validation_suite = ValidationSuite(config)
        
        # API components
        self.cache = PredictionCache(
            max_size=self.api_config.get('cache_size', 1000),
            ttl_seconds=self.api_config.get('cache_ttl_seconds', 60)
        )
        
        self.rate_limiter = RateLimiter(
            max_requests=self.api_config.get('rate_limit_requests', 100),
            window_seconds=self.api_config.get('rate_limit_window', 60)
        )
        
        # Status tracking
        self.start_time = datetime.now()
        self.predictions_count = 0
        self.error_count = 0
        self.last_prediction_time = None
        
        # Threading
        self.executor = ThreadPoolExecutor(
            max_workers=self.api_config.get('max_workers', 4)
        )
        
        # FastAPI app
        if FASTAPI_AVAILABLE:
            self.app = self._create_fastapi_app()
        else:
            self.app = None
            logger.warning("FastAPI not available. API endpoints disabled.")
        
        logger.info("PredictionAPI initialized")
    
    def _create_fastapi_app(self) -> FastAPI:
        """Create FastAPI application."""
        app = FastAPI(
            title="ETH Forecasting Prediction API",
            description="Production API for ETH price predictions using ensemble models",
            version="1.0.0"
        )
        
        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Routes
        @app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint."""
            return {
                "message": "ETH Forecasting Prediction API",
                "version": "1.0.0",
                "status": "operational"
            }
        
        @app.get("/health", response_model=StatusResponse)
        async def health():
            """Health check endpoint."""
            status = self.get_system_status()
            return StatusResponse(**asdict(status), timestamp=datetime.now().isoformat())
        
        @app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
            """Main prediction endpoint."""
            # Rate limiting
            if not self.rate_limiter.is_allowed():
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            try:
                # Make prediction
                result = await self._make_prediction_async(
                    use_live_data=request.use_live_data,
                    custom_data=request.custom_data,
                    model_names=request.model_names,
                    include_explanations=request.include_explanations
                )
                
                if result:
                    return PredictionResponse(
                        success=True,
                        prediction=result.prediction,
                        confidence=result.confidence,
                        timestamp=result.timestamp.isoformat(),
                        model_contributions=result.model_contributions,
                        data_quality=result.data_quality,
                        processing_time_ms=result.processing_time_ms,
                        warnings=result.warnings or [],
                        metadata={
                            'features_used': result.features_used,
                            'model_version': result.model_version
                        }
                    )
                else:
                    raise HTTPException(status_code=500, detail="Prediction failed")
            
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/models", response_model=Dict[str, Any])
        async def get_models():
            """Get available models and their status."""
            return {
                "available_models": self.ensemble_manager.get_available_models(),
                "loaded_models": self.ensemble_manager.get_loaded_models(),
                "model_status": self.ensemble_manager.get_model_status()
            }
        
        @app.post("/models/{model_name}/load")
        async def load_model(model_name: str):
            """Load a specific model."""
            try:
                success = self.ensemble_manager.load_model(model_name)
                if success:
                    return {"message": f"Model {model_name} loaded successfully"}
                else:
                    raise HTTPException(status_code=400, detail=f"Failed to load model {model_name}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/models/{model_name}/unload")
        async def unload_model(model_name: str):
            """Unload a specific model."""
            try:
                self.ensemble_manager.unload_model(model_name)
                return {"message": f"Model {model_name} unloaded successfully"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/data/quality", response_model=Dict[str, Any])
        async def get_data_quality():
            """Get current data quality metrics."""
            return self.live_pipeline.get_quality_summary()
        
        @app.post("/cache/clear")
        async def clear_cache():
            """Clear prediction cache."""
            self.cache.clear()
            return {"message": "Cache cleared successfully"}
        
        return app
    
    async def _make_prediction_async(
        self,
        use_live_data: bool = True,
        custom_data: Optional[Dict[str, Any]] = None,
        model_names: Optional[List[str]] = None,
        include_explanations: bool = False
    ) -> Optional[PredictionResult]:
        """Make prediction asynchronously."""
        start_time = time.time()
        warnings = []
        
        try:
            # Generate cache key
            cache_key = f"{use_live_data}_{hash(str(custom_data))}_{str(model_names)}"
            
            # Check cache
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info("Returning cached prediction")
                return cached_result
            
            # Get data
            if use_live_data:
                data = self.live_pipeline.get_latest_data(max_age_minutes=5)
                if data is None:
                    warnings.append("No recent live data available")
                    return None
            else:
                if custom_data is None:
                    warnings.append("No custom data provided")
                    return None
                
                # Convert custom data to DataFrame
                data = pd.DataFrame(custom_data)
                if 'Date' in data.columns:
                    data['Date'] = pd.to_datetime(data['Date'])
                    data.set_index('Date', inplace=True)
            
            # Assess data quality
            data_quality = 1.0
            if use_live_data:
                quality_summary = self.live_pipeline.get_quality_summary()
                data_quality = quality_summary.get('current_score', 0.5)
            
            # Quality gate
            min_quality = self.api_config.get('min_data_quality', 0.6)
            if data_quality < min_quality:
                warnings.append(f"Data quality below threshold: {data_quality:.3f}")
            
            # Make ensemble prediction
            prediction_result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.ensemble_manager.predict,
                data,
                model_names
            )
            
            if prediction_result is None:
                warnings.append("Ensemble prediction failed")
                return None
            
            prediction, confidence, contributions = prediction_result
            
            # Create result
            processing_time = (time.time() - start_time) * 1000
            
            result = PredictionResult(
                timestamp=datetime.now(timezone.utc),
                prediction=float(prediction),
                confidence=float(confidence),
                model_contributions=contributions,
                data_quality=data_quality,
                features_used=len(data.columns) if data is not None else 0,
                processing_time_ms=processing_time,
                model_version="ensemble_v1.0",
                warnings=warnings if warnings else None
            )
            
            # Cache result
            self.cache.put(cache_key, result)
            
            # Update counters
            self.predictions_count += 1
            self.last_prediction_time = datetime.now()
            
            logger.info(f"Prediction completed: {prediction:.6f} "
                       f"(confidence: {confidence:.3f}, time: {processing_time:.1f}ms)")
            
            return result
        
        except Exception as e:
            self.error_count += 1
            logger.error(f"Prediction error: {e}")
            return None
    
    def get_system_status(self) -> SystemStatus:
        """Get current system status."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        # Calculate error rate
        total_requests = self.predictions_count + self.error_count
        error_rate = self.error_count / max(total_requests, 1)
        
        # Get memory usage (simplified)
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Determine overall status
        data_quality = self.live_pipeline.get_quality_summary().get('current_score', 0)
        
        if error_rate > 0.1 or data_quality < 0.5:
            status = 'critical'
        elif error_rate > 0.05 or data_quality < 0.7:
            status = 'degraded'
        else:
            status = 'healthy'
        
        return SystemStatus(
            status=status,
            uptime_seconds=uptime,
            models_loaded=self.ensemble_manager.get_loaded_models(),
            data_pipeline_status=self.live_pipeline.get_quality_summary().get('status', 'unknown'),
            last_prediction=self.last_prediction_time,
            predictions_count=self.predictions_count,
            error_rate=error_rate,
            memory_usage_mb=memory_mb,
            quality_score=data_quality
        )
    
    def start(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the prediction API server."""
        if not FASTAPI_AVAILABLE:
            logger.error("FastAPI not available. Cannot start server.")
            return
        
        logger.info("Starting Prediction API server")
        
        # Start live data pipeline
        self.live_pipeline.start()
        
        # Load default models
        default_models = self.api_config.get('default_models', ['lightgbm', 'tft'])
        for model_name in default_models:
            try:
                self.ensemble_manager.load_model(model_name)
                logger.info(f"Loaded default model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load default model {model_name}: {e}")
        
        # Start server
        try:
            uvicorn.run(
                self.app,
                host=host,
                port=port,
                log_level="info",
                access_log=True
            )
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop the prediction API."""
        logger.info("Stopping Prediction API")
        
        # Stop live data pipeline
        self.live_pipeline.stop()
        
        # Unload models
        self.ensemble_manager.cleanup()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Prediction API stopped")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def create_api_config() -> Dict[str, Any]:
    """Create default API configuration."""
    return {
        'prediction_api': {
            'cache_size': 1000,
            'cache_ttl_seconds': 60,
            'rate_limit_requests': 100,
            'rate_limit_window': 60,
            'max_workers': 4,
            'min_data_quality': 0.6,
            'default_models': ['lightgbm', 'tft']
        },
        'live_pipeline': {
            'fetch_interval_seconds': 60,
            'processing_interval_seconds': 30,
            'quality_threshold': 0.7,
            'buffer_size': 100,
            'data_sources': {
                'yahoo_finance': {
                    'enabled': True,
                    'symbols': ['ETH-USD', 'BTC-USD'],
                    'interval': '1m',
                    'period': '1d'
                }
            }
        },
        'ensemble': {
            'memory_limit_gb': 4.0,
            'model_timeout_seconds': 30,
            'default_weights': {
                'lightgbm': 0.4,
                'tft': 0.3,
                'nbeats': 0.3
            }
        }
    }


def main():
    """Test the PredictionAPI."""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Create test configuration
    config = create_api_config()
    
    # Test API
    print("🚀 Testing Prediction API")
    print("=" * 50)
    
    try:
        with PredictionAPI(config) as api:
            print("✅ API initialized successfully")
            
            # Test system status
            status = api.get_system_status()
            print(f"📊 System Status: {status.status}")
            print(f"   Memory: {status.memory_usage_mb:.1f} MB")
            print(f"   Models: {status.models_loaded}")
            
            if FASTAPI_AVAILABLE:
                print("\n🌐 Starting API server on http://localhost:8000")
                print("   Available endpoints:")
                print("   - GET  /health")
                print("   - POST /predict")
                print("   - GET  /models")
                print("   - GET  /data/quality")
                print("\n   Press Ctrl+C to stop")
                
                api.start(host="localhost", port=8000)
            else:
                print("\n❌ FastAPI not available. Install with:")
                print("   pip install fastapi uvicorn")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()