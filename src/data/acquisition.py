"""
Data acquisition module for ETH forecasting project.

This module handles downloading and initial processing of ETH and BTC data
from Yahoo Finance with proper timezone handling and data validation.
"""

import logging
import pandas as pd
import yfinance as yf
import numpy as np
import time
from typing import Optional, Tuple, Dict, Any
from datetime import datetime, timezone, timedelta
import warnings

# Suppress yfinance warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class DataAcquisition:
    """
    Data acquisition class for downloading and processing cryptocurrency data.
    
    This class ensures proper timezone handling, data validation, and
    adherence to the 24-hour window requirements.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataAcquisition with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config.get('data', {})
        self.symbol = self.data_config.get('symbol', 'ETH-USD')
        self.btc_symbol = self.data_config.get('btc_symbol', 'BTC-USD')
        self.start_date = self.data_config.get('start_date', '2020-01-01')
        self.end_date = self.data_config.get('end_date', None)
        self.interval = self.data_config.get('interval', '1d')
        self.timezone = self.data_config.get('timezone', 'UTC')
        
        logging.info(f"DataAcquisition initialized for {self.symbol} and {self.btc_symbol}")
    
    def download_data(self, symbol: str, start_date: str, 
                     end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Download data from Yahoo Finance with proper error handling.
        
        Args:
            symbol: Trading symbol (e.g., 'ETH-USD')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (None for current date)
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            ValueError: If data download fails or returns empty
        """
        max_retries = 5
        retry_delay = 5  # Start with 5 second delay
        
        for attempt in range(max_retries):
            try:
                logging.info(f"Downloading data for {symbol} from {start_date} to {end_date or 'current'} (attempt {attempt + 1})")
                
                # Add initial delay to avoid immediate rate limiting
                if attempt > 0:
                    time.sleep(retry_delay)
                
                # Create ticker object
                ticker = yf.Ticker(symbol)
                
                # Download data
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=self.interval,
                    auto_adjust=True,
                    prepost=True
                )
                
                if data.empty:
                    raise ValueError(f"No data downloaded for {symbol}")
                
                # Reset index to make Date a column
                data = data.reset_index()
                
                # Ensure timezone is UTC
                if data['Date'].dt.tz is None:
                    # If timezone naive, assume UTC
                    data['Date'] = data['Date'].dt.tz_localize('UTC')
                    logging.warning(f"Data for {symbol} was timezone naive, assumed UTC")
                else:
                    # Convert to UTC if not already
                    data['Date'] = data['Date'].dt.tz_convert('UTC')
                
                # Validate data
                self._validate_downloaded_data(data, symbol)
                
                logging.info(f"Successfully downloaded {len(data)} records for {symbol}")
                return data
                
            except Exception as e:
                if "Rate limited" in str(e) or "Too Many Requests" in str(e) or "YFRateLimitError" in str(type(e)):
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        logging.warning(f"Rate limited for {symbol}, retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logging.error(f"Rate limit exceeded for {symbol} after {max_retries} attempts")
                        raise
                else:
                    logging.error(f"Failed to download data for {symbol}: {e}")
                    raise
    
    def _validate_downloaded_data(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Validate downloaded data for completeness and quality.
        
        Args:
            data: Downloaded data DataFrame
            symbol: Trading symbol for logging
            
        Raises:
            ValueError: If data validation fails
        """
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns for {symbol}: {missing_columns}")
        
        # Check for reasonable data ranges
        if data['Close'].min() <= 0:
            raise ValueError(f"Invalid price data for {symbol}: negative or zero prices")
        
        if data['Volume'].min() < 0:
            raise ValueError(f"Invalid volume data for {symbol}: negative volume")
        
        # Check for excessive NaN values
        nan_percentage = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if nan_percentage > 0.1:  # More than 10% NaN
            logging.warning(f"High NaN percentage in {symbol} data: {nan_percentage:.2%}")
        
        # Check date continuity (allowing for weekends in crypto)
        date_diff = data['Date'].diff().dropna()
        expected_diff = pd.Timedelta(days=1)
        
        # Allow for some gaps (weekends, holidays)
        large_gaps = date_diff > pd.Timedelta(days=3)
        if large_gaps.sum() > len(date_diff) * 0.05:  # More than 5% large gaps
            logging.warning(f"Many large gaps in {symbol} date series")
        
        logging.info(f"Data validation passed for {symbol}")
    
    def get_eth_data(self) -> pd.DataFrame:
        """
        Download ETH data with proper preprocessing.
        
        Returns:
            DataFrame with ETH OHLCV data
        """
        eth_data = self.download_data(self.symbol, self.start_date, self.end_date)
        
        # Add prefix to distinguish ETH columns
        eth_columns = {col: f'ETH_{col}' if col != 'Date' else col 
                      for col in eth_data.columns}
        eth_data = eth_data.rename(columns=eth_columns)
        
        return eth_data
    
    def get_btc_data(self) -> pd.DataFrame:
        """
        Download BTC data for cross-asset features.
        
        Returns:
            DataFrame with BTC OHLCV data
        """
        btc_data = self.download_data(self.btc_symbol, self.start_date, self.end_date)
        
        # Add prefix to distinguish BTC columns
        btc_columns = {col: f'BTC_{col}' if col != 'Date' else col 
                      for col in btc_data.columns}
        btc_data = btc_data.rename(columns=btc_columns)
        
        return btc_data
    
    def get_combined_data(self) -> pd.DataFrame:
        """
        Get combined ETH and BTC data with proper alignment.
        Uses cached data as fallback if Yahoo Finance rate limiting occurs.
        
        Returns:
            DataFrame with both ETH and BTC data aligned by date
        """
        try:
            logging.info("Downloading and combining ETH and BTC data")
            
            # Download both datasets with delay to avoid rate limiting
            eth_data = self.get_eth_data()
            
            # Wait between downloads to avoid rate limiting
            logging.info("Waiting 10 seconds before downloading BTC data to avoid rate limiting...")
            time.sleep(10)
            
            btc_data = self.get_btc_data()
            
            # Merge on Date with inner join to ensure alignment
            combined_data = pd.merge(eth_data, btc_data, on='Date', how='inner')
            
            # Sort by date
            combined_data = combined_data.sort_values('Date').reset_index(drop=True)
            
            # Validate combined data
            self._validate_combined_data(combined_data)
            
            logging.info(f"Combined data created with {len(combined_data)} records")
            return combined_data
            
        except Exception as e:
            if "Rate limited" in str(e) or "Too Many Requests" in str(e):
                logging.warning(f"Yahoo Finance rate limiting detected: {e}")
                return self._load_cached_data()
            else:
                raise e
    
    def _load_cached_data(self) -> pd.DataFrame:
        """
        Load cached data as fallback when Yahoo Finance is rate limited.
        
        Returns:
            DataFrame with cached combined data
        """
        import os
        
        cached_file_path = "data/raw/eth_btc_combined.csv"
        
        if not os.path.exists(cached_file_path):
            raise FileNotFoundError(f"No cached data found at {cached_file_path}. Cannot proceed without data.")
        
        logging.info(f"Loading cached data from {cached_file_path}")
        
        # Load cached data
        cached_data = pd.read_csv(cached_file_path)
        
        # Convert Date column to datetime
        cached_data['Date'] = pd.to_datetime(cached_data['Date'])
        
        # Validate cached data
        self._validate_combined_data(cached_data)
        
        logging.info(f"Cached data loaded successfully with {len(cached_data)} records")
        logging.info(f"Date range: {cached_data['Date'].min()} to {cached_data['Date'].max()}")
        
        return cached_data
    
    def _validate_combined_data(self, data: pd.DataFrame) -> None:
        """
        Validate combined ETH and BTC data.
        
        Args:
            data: Combined data DataFrame
            
        Raises:
            ValueError: If validation fails
        """
        # Check minimum data requirements
        min_required_days = 365  # At least 1 year of data
        if len(data) < min_required_days:
            raise ValueError(f"Insufficient data: {len(data)} days, minimum required: {min_required_days}")
        
        # Check for ETH and BTC columns
        eth_columns = [col for col in data.columns if col.startswith('ETH_')]
        btc_columns = [col for col in data.columns if col.startswith('BTC_')]
        
        if len(eth_columns) == 0:
            raise ValueError("No ETH columns found in combined data")
        
        if len(btc_columns) == 0:
            raise ValueError("No BTC columns found in combined data")
        
        # Check date alignment
        if not data['Date'].is_monotonic_increasing:
            raise ValueError("Date column is not properly sorted")
        
        # Check for duplicate dates
        if data['Date'].duplicated().any():
            raise ValueError("Duplicate dates found in combined data")
        
        logging.info("Combined data validation passed")
    
    def calculate_target_variable(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate 24-hour log returns as target variable.
        
        Args:
            data: DataFrame with ETH price data
            
        Returns:
            DataFrame with target variable added
        """
        # Calculate 24-hour log returns
        data['ETH_log_return_24h'] = np.log(data['ETH_Close'] / data['ETH_Close'].shift(1))
        
        # Create binary direction target
        data['ETH_direction'] = (data['ETH_log_return_24h'] > 0).astype(int)
        
        # Remove first row with NaN log return
        data = data.dropna(subset=['ETH_log_return_24h']).reset_index(drop=True)
        
        logging.info(f"Target variables calculated: {len(data)} valid observations")
        
        # Log target statistics
        log_returns = data['ETH_log_return_24h']
        logging.info(f"Log returns statistics:")
        logging.info(f"  Mean: {log_returns.mean():.6f}")
        logging.info(f"  Std: {log_returns.std():.6f}")
        logging.info(f"  Min: {log_returns.min():.6f}")
        logging.info(f"  Max: {log_returns.max():.6f}")
        logging.info(f"  Positive direction ratio: {data['ETH_direction'].mean():.3f}")
        
        return data
    
    def save_raw_data(self, data: pd.DataFrame, filename: str = "raw_data.csv") -> str:
        """
        Save raw data to file.
        
        Args:
            data: DataFrame to save
            filename: Output filename
            
        Returns:
            Full path to saved file
        """
        output_path = f"data/raw/{filename}"
        
        # Ensure directory exists
        import os
        os.makedirs("data/raw", exist_ok=True)
        
        # Save data
        data.to_csv(output_path, index=False)
        
        logging.info(f"Raw data saved to {output_path}")
        return output_path
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the data.
        
        Args:
            data: DataFrame to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_records': len(data),
            'date_range': {
                'start': data['Date'].min().strftime('%Y-%m-%d'),
                'end': data['Date'].max().strftime('%Y-%m-%d')
            },
            'missing_values': data.isnull().sum().to_dict(),
            'eth_price_stats': {
                'min': float(data['ETH_Close'].min()),
                'max': float(data['ETH_Close'].max()),
                'mean': float(data['ETH_Close'].mean()),
                'std': float(data['ETH_Close'].std())
            }
        }
        
        if 'ETH_log_return_24h' in data.columns:
            summary['target_stats'] = {
                'mean_log_return': float(data['ETH_log_return_24h'].mean()),
                'std_log_return': float(data['ETH_log_return_24h'].std()),
                'positive_direction_ratio': float(data['ETH_direction'].mean())
            }
        
        return summary


def main():
    """
    Main function for testing data acquisition.
    """
    # Load configuration
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.utils.helpers import load_config, setup_logging, set_random_seeds
    
    # Setup
    config = load_config()
    setup_logging()
    set_random_seeds(config['models']['random_seed'])
    
    # Initialize data acquisition
    data_acq = DataAcquisition(config)
    
    # Download and process data
    combined_data = data_acq.get_combined_data()
    data_with_target = data_acq.calculate_target_variable(combined_data)
    
    # Save raw data
    output_path = data_acq.save_raw_data(data_with_target)
    
    # Print summary
    summary = data_acq.get_data_summary(data_with_target)
    print("Data Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()