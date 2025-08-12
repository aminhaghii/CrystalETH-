"""
Live Data Pipeline for ETH Forecasting

This module implements a real-time data pipeline that:
- Connects to live data sources (Yahoo Finance, APIs)
- Performs real-time feature engineering
- Maintains data quality monitoring
- Handles data streaming and buffering
- Implements automated data validation

Following Rules #1 (Data Integrity), #7 (Timezone Discipline),
and #16 (Automated Quality Gates).
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
import asyncio
import aiohttp
import threading
import time
import queue
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import warnings

from ..data.acquisition import DataAcquisition
from ..features.engineering import FeatureEngineer
from ..data.denoising import DataDenoiser

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class DataPoint:
    """Single data point with metadata."""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    source: str
    quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics."""
    completeness: float  # Ratio of non-null values
    timeliness: float    # How recent the data is
    accuracy: float      # Data validation score
    consistency: float   # Cross-source consistency
    overall_score: float # Combined quality score


class LiveDataSource:
    """Base class for live data sources."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.is_active = False
        self.last_update = None
        self.error_count = 0
        self.max_errors = config.get('max_errors', 5)
    
    async def fetch_data(self) -> List[DataPoint]:
        """Fetch data from source. To be implemented by subclasses."""
        raise NotImplementedError
    
    def is_healthy(self) -> bool:
        """Check if data source is healthy."""
        return self.error_count < self.max_errors


class YahooFinanceSource(LiveDataSource):
    """Yahoo Finance live data source."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("yahoo_finance", config)
        self.symbols = config.get('symbols', ['ETH-USD', 'BTC-USD'])
        self.interval = config.get('interval', '1m')
        self.period = config.get('period', '1d')
    
    async def fetch_data(self) -> List[DataPoint]:
        """Fetch latest data from Yahoo Finance."""
        try:
            data_points = []
            
            for symbol in self.symbols:
                ticker = yf.Ticker(symbol)
                
                # Get latest data
                hist = ticker.history(period=self.period, interval=self.interval)
                
                if not hist.empty:
                    latest = hist.iloc[-1]
                    timestamp = hist.index[-1]
                    
                    # Ensure timezone awareness
                    if timestamp.tz is None:
                        timestamp = timestamp.tz_localize('UTC')
                    else:
                        timestamp = timestamp.tz_convert('UTC')
                    
                    data_point = DataPoint(
                        timestamp=timestamp.to_pydatetime(),
                        symbol=symbol,
                        price=float(latest['Close']),
                        volume=float(latest['Volume']),
                        source=self.name,
                        metadata={
                            'open': float(latest['Open']),
                            'high': float(latest['High']),
                            'low': float(latest['Low']),
                            'interval': self.interval
                        }
                    )
                    
                    data_points.append(data_point)
            
            self.last_update = datetime.now(timezone.utc)
            self.error_count = 0  # Reset error count on success
            
            return data_points
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Yahoo Finance fetch error: {str(e)}")
            return []


class CoinGeckoSource(LiveDataSource):
    """CoinGecko API data source (alternative/backup)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("coingecko", config)
        self.base_url = "https://api.coingecko.com/api/v3"
        self.coin_ids = config.get('coin_ids', ['ethereum', 'bitcoin'])
        self.vs_currency = config.get('vs_currency', 'usd')
    
    async def fetch_data(self) -> List[DataPoint]:
        """Fetch data from CoinGecko API."""
        try:
            data_points = []
            
            async with aiohttp.ClientSession() as session:
                for coin_id in self.coin_ids:
                    url = f"{self.base_url}/simple/price"
                    params = {
                        'ids': coin_id,
                        'vs_currencies': self.vs_currency,
                        'include_24hr_vol': 'true',
                        'include_last_updated_at': 'true'
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if coin_id in data:
                                coin_data = data[coin_id]
                                
                                # Convert to standard format
                                symbol = f"{coin_id.upper()}-USD"
                                if coin_id == 'ethereum':
                                    symbol = 'ETH-USD'
                                elif coin_id == 'bitcoin':
                                    symbol = 'BTC-USD'
                                
                                timestamp = datetime.fromtimestamp(
                                    coin_data.get('last_updated_at', time.time()),
                                    tz=timezone.utc
                                )
                                
                                data_point = DataPoint(
                                    timestamp=timestamp,
                                    symbol=symbol,
                                    price=float(coin_data[self.vs_currency]),
                                    volume=float(coin_data.get(f'{self.vs_currency}_24h_vol', 0)),
                                    source=self.name,
                                    metadata={
                                        'coin_id': coin_id,
                                        'vs_currency': self.vs_currency
                                    }
                                )
                                
                                data_points.append(data_point)
            
            self.last_update = datetime.now(timezone.utc)
            self.error_count = 0
            
            return data_points
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"CoinGecko fetch error: {str(e)}")
            return []


class LiveDataPipeline:
    """
    Live data pipeline for real-time ETH forecasting.
    
    Features:
    - Multiple data source support with failover
    - Real-time data quality monitoring
    - Automated feature engineering
    - Data buffering and streaming
    - Quality gates and validation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LiveDataPipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.pipeline_config = config.get('live_pipeline', {})
        
        # Data sources
        self.data_sources: List[LiveDataSource] = []
        self._initialize_data_sources()
        
        # Data processing components
        self.feature_engineer = FeatureEngineer(config)
        self.data_denoiser = DataDenoiser(config)
        
        # Data buffer and streaming
        self.data_buffer = queue.Queue(maxsize=self.pipeline_config.get('buffer_size', 1000))
        self.processed_data_buffer = queue.Queue(maxsize=100)
        
        # Quality monitoring
        self.quality_threshold = self.pipeline_config.get('quality_threshold', 0.8)
        self.quality_history = []
        self.max_quality_history = 100
        
        # Threading and async
        self.is_running = False
        self.fetch_interval = self.pipeline_config.get('fetch_interval_seconds', 60)
        self.processing_interval = self.pipeline_config.get('processing_interval_seconds', 30)
        
        self._fetch_thread = None
        self._processing_thread = None
        self._loop = None
        
        # Callbacks
        self.data_callbacks: List[Callable] = []
        self.quality_callbacks: List[Callable] = []
        
        logger.info("LiveDataPipeline initialized")
    
    def _initialize_data_sources(self) -> None:
        """Initialize configured data sources."""
        sources_config = self.pipeline_config.get('data_sources', {})
        
        # Yahoo Finance (primary)
        if sources_config.get('yahoo_finance', {}).get('enabled', True):
            yahoo_config = sources_config.get('yahoo_finance', {})
            self.data_sources.append(YahooFinanceSource(yahoo_config))
        
        # CoinGecko (backup)
        if sources_config.get('coingecko', {}).get('enabled', False):
            coingecko_config = sources_config.get('coingecko', {})
            self.data_sources.append(CoinGeckoSource(coingecko_config))
        
        logger.info(f"Initialized {len(self.data_sources)} data sources")
    
    def add_data_callback(self, callback: Callable[[pd.DataFrame], None]) -> None:
        """Add callback for new processed data."""
        self.data_callbacks.append(callback)
    
    def add_quality_callback(self, callback: Callable[[DataQualityMetrics], None]) -> None:
        """Add callback for quality metrics updates."""
        self.quality_callbacks.append(callback)
    
    def _assess_data_quality(self, data_points: List[DataPoint]) -> DataQualityMetrics:
        """Assess quality of fetched data points."""
        if not data_points:
            return DataQualityMetrics(0, 0, 0, 0, 0)
        
        # Completeness: ratio of valid data points
        valid_points = [dp for dp in data_points if dp.price > 0 and dp.volume >= 0]
        completeness = len(valid_points) / len(data_points)
        
        # Timeliness: how recent the data is
        now = datetime.now(timezone.utc)
        avg_age = np.mean([(now - dp.timestamp).total_seconds() for dp in data_points])
        timeliness = max(0, 1 - avg_age / 3600)  # 1 hour = 0 score
        
        # Accuracy: basic validation checks
        accuracy = 1.0
        for dp in data_points:
            # Check for reasonable price ranges
            if dp.symbol == 'ETH-USD' and (dp.price < 100 or dp.price > 10000):
                accuracy -= 0.1
            elif dp.symbol == 'BTC-USD' and (dp.price < 10000 or dp.price > 100000):
                accuracy -= 0.1
        accuracy = max(0, accuracy)
        
        # Consistency: check across sources (if multiple)
        consistency = 1.0
        if len(set(dp.source for dp in data_points)) > 1:
            # Group by symbol and check price consistency
            symbol_groups = {}
            for dp in data_points:
                if dp.symbol not in symbol_groups:
                    symbol_groups[dp.symbol] = []
                symbol_groups[dp.symbol].append(dp)
            
            for symbol, points in symbol_groups.items():
                if len(points) > 1:
                    prices = [dp.price for dp in points]
                    price_std = np.std(prices) / np.mean(prices)  # Coefficient of variation
                    if price_std > 0.01:  # More than 1% variation
                        consistency -= 0.2
        
        consistency = max(0, consistency)
        
        # Overall score
        overall_score = (completeness * 0.3 + timeliness * 0.3 + 
                        accuracy * 0.2 + consistency * 0.2)
        
        return DataQualityMetrics(
            completeness=completeness,
            timeliness=timeliness,
            accuracy=accuracy,
            consistency=consistency,
            overall_score=overall_score
        )
    
    async def _fetch_data_async(self) -> List[DataPoint]:
        """Fetch data from all sources asynchronously."""
        all_data_points = []
        
        # Fetch from all healthy sources
        tasks = []
        for source in self.data_sources:
            if source.is_healthy():
                tasks.append(source.fetch_data())
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    all_data_points.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Data fetch error: {result}")
        
        return all_data_points
    
    def _fetch_data_loop(self) -> None:
        """Main data fetching loop."""
        logger.info("Starting data fetch loop")
        
        # Create event loop for this thread
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            while self.is_running:
                try:
                    # Fetch data
                    data_points = self._loop.run_until_complete(self._fetch_data_async())
                    
                    if data_points:
                        # Assess quality
                        quality_metrics = self._assess_data_quality(data_points)
                        
                        # Store quality history
                        self.quality_history.append(quality_metrics)
                        if len(self.quality_history) > self.max_quality_history:
                            self.quality_history.pop(0)
                        
                        # Check quality gate
                        if quality_metrics.overall_score >= self.quality_threshold:
                            # Add to buffer
                            for dp in data_points:
                                try:
                                    self.data_buffer.put_nowait(dp)
                                except queue.Full:
                                    # Remove oldest item and add new one
                                    try:
                                        self.data_buffer.get_nowait()
                                        self.data_buffer.put_nowait(dp)
                                    except queue.Empty:
                                        pass
                            
                            logger.info(f"Fetched {len(data_points)} data points "
                                       f"(quality: {quality_metrics.overall_score:.3f})")
                        else:
                            logger.warning(f"Data quality below threshold: "
                                         f"{quality_metrics.overall_score:.3f}")
                        
                        # Notify quality callbacks
                        for callback in self.quality_callbacks:
                            try:
                                callback(quality_metrics)
                            except Exception as e:
                                logger.error(f"Quality callback error: {e}")
                    
                    # Wait for next fetch
                    time.sleep(self.fetch_interval)
                    
                except Exception as e:
                    logger.error(f"Error in fetch loop: {e}")
                    time.sleep(self.fetch_interval)
        
        finally:
            self._loop.close()
    
    def _process_data_loop(self) -> None:
        """Main data processing loop."""
        logger.info("Starting data processing loop")
        
        while self.is_running:
            try:
                # Collect data points from buffer
                data_points = []
                
                # Get all available data points (non-blocking)
                while True:
                    try:
                        dp = self.data_buffer.get_nowait()
                        data_points.append(dp)
                    except queue.Empty:
                        break
                
                if data_points:
                    # Convert to DataFrame
                    df_data = []
                    for dp in data_points:
                        row = {
                            'Date': dp.timestamp,
                            f'{dp.symbol.split("-")[0]}_Close': dp.price,
                            f'{dp.symbol.split("-")[0]}_Volume': dp.volume,
                            'source': dp.source
                        }
                        # Add metadata
                        for key, value in dp.metadata.items():
                            row[f'{dp.symbol.split("-")[0]}_{key}'] = value
                        
                        df_data.append(row)
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                        
                        # Sort by timestamp
                        df.sort_index(inplace=True)
                        
                        # Remove duplicates (keep latest)
                        df = df[~df.index.duplicated(keep='last')]
                        
                        # Engineer features (not training mode)
                        try:
                            processed_df = self.feature_engineer.engineer_features(
                                df, is_training=False
                            )
                            
                            # Add to processed buffer
                            try:
                                self.processed_data_buffer.put_nowait(processed_df)
                            except queue.Full:
                                # Remove oldest and add new
                                try:
                                    self.processed_data_buffer.get_nowait()
                                    self.processed_data_buffer.put_nowait(processed_df)
                                except queue.Empty:
                                    pass
                            
                            # Notify data callbacks
                            for callback in self.data_callbacks:
                                try:
                                    callback(processed_df)
                                except Exception as e:
                                    logger.error(f"Data callback error: {e}")
                            
                            logger.info(f"Processed {len(df)} data points into "
                                       f"{len(processed_df.columns)} features")
                        
                        except Exception as e:
                            logger.error(f"Feature engineering error: {e}")
                
                # Wait for next processing cycle
                time.sleep(self.processing_interval)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(self.processing_interval)
    
    def start(self) -> None:
        """Start the live data pipeline."""
        if self.is_running:
            logger.warning("Pipeline is already running")
            return
        
        logger.info("Starting live data pipeline")
        self.is_running = True
        
        # Start fetch thread
        self._fetch_thread = threading.Thread(
            target=self._fetch_data_loop,
            name="DataFetchThread",
            daemon=True
        )
        self._fetch_thread.start()
        
        # Start processing thread
        self._processing_thread = threading.Thread(
            target=self._process_data_loop,
            name="DataProcessingThread",
            daemon=True
        )
        self._processing_thread.start()
        
        logger.info("Live data pipeline started successfully")
    
    def stop(self) -> None:
        """Stop the live data pipeline."""
        if not self.is_running:
            logger.warning("Pipeline is not running")
            return
        
        logger.info("Stopping live data pipeline")
        self.is_running = False
        
        # Wait for threads to finish
        if self._fetch_thread and self._fetch_thread.is_alive():
            self._fetch_thread.join(timeout=10)
        
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=10)
        
        logger.info("Live data pipeline stopped")
    
    def get_latest_data(self, max_age_minutes: int = 5) -> Optional[pd.DataFrame]:
        """Get latest processed data."""
        try:
            processed_df = self.processed_data_buffer.get_nowait()
            
            # Check data age
            if not processed_df.empty:
                latest_timestamp = processed_df.index.max()
                age = (datetime.now(timezone.utc) - latest_timestamp.tz_localize('UTC')).total_seconds() / 60
                
                if age <= max_age_minutes:
                    return processed_df
                else:
                    logger.warning(f"Latest data is {age:.1f} minutes old")
            
            return None
            
        except queue.Empty:
            return None
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get data quality summary."""
        if not self.quality_history:
            return {'status': 'no_data'}
        
        recent_quality = self.quality_history[-10:]  # Last 10 measurements
        
        return {
            'status': 'healthy' if self.quality_history[-1].overall_score >= self.quality_threshold else 'degraded',
            'current_score': self.quality_history[-1].overall_score,
            'average_score': np.mean([q.overall_score for q in recent_quality]),
            'completeness': self.quality_history[-1].completeness,
            'timeliness': self.quality_history[-1].timeliness,
            'accuracy': self.quality_history[-1].accuracy,
            'consistency': self.quality_history[-1].consistency,
            'source_health': {source.name: source.is_healthy() for source in self.data_sources},
            'last_update': self.data_sources[0].last_update.isoformat() if self.data_sources[0].last_update else None
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def main():
    """Test the LiveDataPipeline."""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.utils.config_loader import load_config
    
    # Load configuration
    config = load_config()
    
    # Add live pipeline configuration
    config['live_pipeline'] = {
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
    }
    
    # Test callbacks
    def data_callback(df):
        print(f"📊 New data: {len(df)} rows, {len(df.columns)} features")
        print(f"   Latest timestamp: {df.index.max()}")
    
    def quality_callback(metrics):
        print(f"📈 Quality: {metrics.overall_score:.3f} "
              f"(C:{metrics.completeness:.2f}, T:{metrics.timeliness:.2f}, "
              f"A:{metrics.accuracy:.2f}, S:{metrics.consistency:.2f})")
    
    # Test pipeline
    with LiveDataPipeline(config) as pipeline:
        print("🚀 Testing Live Data Pipeline")
        print("=" * 50)
        
        # Add callbacks
        pipeline.add_data_callback(data_callback)
        pipeline.add_quality_callback(quality_callback)
        
        # Run for a short time
        print("Running pipeline for 2 minutes...")
        time.sleep(120)
        
        # Get latest data
        latest_data = pipeline.get_latest_data()
        if latest_data is not None:
            print(f"\n📋 Latest Data Shape: {latest_data.shape}")
            print(f"   Columns: {list(latest_data.columns[:10])}...")
        else:
            print("\n❌ No recent data available")
        
        # Get quality summary
        quality = pipeline.get_quality_summary()
        print(f"\n📊 Quality Summary:")
        print(f"   Status: {quality['status']}")
        print(f"   Current Score: {quality.get('current_score', 'N/A')}")
        print(f"   Source Health: {quality.get('source_health', {})}")


if __name__ == "__main__":
    main()