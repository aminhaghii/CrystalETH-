"""
Feature engineering module for ETH forecasting project.

This module implements comprehensive feature engineering following Rule #9:
- IMPLEMENT all required feature groups (no exceptions)
- USE only historical data available at time t
- APPLY scalers only on training data, transform validation/test
- SAVE feature importance rankings for analysis

Feature Groups:
1. Basic Features (OHLCV, log returns)
2. Returns & Lags (multiple horizons)
3. Rolling Statistics (mean, std, min, max)
4. Volatility Features (realized vol, GARCH-style)
5. Momentum Features (RSI, MACD, Bollinger Bands)
6. Cross-Asset Features (BTC correlation, ratios)
7. Calendar Features (day of week, month, holidays)
8. Event Flags (extreme events, regime changes)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import ta
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class FeatureEngineer:
    """
    Comprehensive feature engineering class for ETH forecasting.
    
    Implements all required feature groups with strict temporal constraints
    to prevent data leakage.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FeatureEngineer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_config = config.get('feature_engineering', {})
        
        # Feature engineering parameters
        self.lag_periods = self.feature_config.get('lag_periods', [1, 2, 3, 5, 7, 14, 21])
        self.rolling_windows = self.feature_config.get('rolling_windows', [7, 14, 30, 60])
        self.volatility_windows = self.feature_config.get('volatility_windows', [7, 14, 30])
        self.momentum_windows = self.feature_config.get('momentum_windows', [14, 21, 30])
        self.correlation_window = self.feature_config.get('correlation_window', 30)
        
        # Scaling configuration
        self.scaler_type = self.feature_config.get('scaler_type', 'robust')
        self.scalers = {}
        
        # Feature tracking
        self.feature_groups = {}
        self.feature_importance = {}
        
        logging.info("FeatureEngineer initialized with comprehensive feature groups")
    
    def engineer_features(self, data: pd.DataFrame, 
                         is_training: bool = True) -> pd.DataFrame:
        """
        Engineer all feature groups for the dataset.
        
        Args:
            data: Input DataFrame with OHLCV data
            is_training: Whether this is training data (for scaler fitting)
            
        Returns:
            DataFrame with engineered features
        """
        logging.info(f"Engineering features (training={is_training})")
        
        # Initialize feature groups tracking
        self.feature_groups = {
            'basic': [],
            'returns': [],
            'rolling': [],
            'volatility': [],
            'momentum': [],
            'cross_asset': [],
            'calendar': [],
            'events': []
        }
        
        # Start with copy of original data
        features_df = data.copy()
        
        # Ensure datetime index
        if 'Date' in features_df.columns:
            features_df['Date'] = pd.to_datetime(features_df['Date'])
            features_df.set_index('Date', inplace=True)
        elif not isinstance(features_df.index, pd.DatetimeIndex):
            # If index is not DatetimeIndex but should be, try to convert it
            try:
                features_df.index = pd.to_datetime(features_df.index)
                logging.info("Converted existing index to DatetimeIndex")
            except Exception as e:
                logging.warning(f"Could not convert index to DatetimeIndex: {e}")
        
        # Apply feature engineering groups in order
        features_df = self._group1_basic_features(features_df)
        features_df = self._group2_returns_and_lags(features_df)
        features_df = self._group3_rolling_statistics(features_df)
        features_df = self._group4_volatility_features(features_df)
        features_df = self._group5_momentum_features(features_df)
        features_df = self._group6_cross_asset_features(features_df)
        features_df = self._group7_calendar_features(features_df)
        features_df = self._group8_event_flags(features_df)
        
        # Apply scaling (only fit on training data)
        features_df = self._apply_scaling(features_df, is_training)
        
        # Handle NaN values while preserving initial NaN values for rolling features
        initial_rows = len(features_df)
        
        # Calculate the maximum lag/window to determine minimum valid start
        max_lag = max(self.lag_periods) if self.lag_periods else 0
        max_window = max(self.rolling_windows) if self.rolling_windows else 0
        min_valid_start = max(max_lag, max_window)
        
        # Identify rolling features that should have initial NaN values
        rolling_features = []
        for col in features_df.columns:
            if any(pattern in col.lower() for pattern in ['rolling_', '_ma_', '_sma_', '_ema_']):
                rolling_features.append(col)
        
        # For non-rolling features, handle NaN values normally
        non_rolling_features = [col for col in features_df.columns if col not in rolling_features]
        
        # Only drop rows that have excessive NaN values in non-rolling features
        if len(non_rolling_features) > 0:
            nan_threshold = 0.5
            non_rolling_df = features_df[non_rolling_features]
            nan_counts = non_rolling_df.isna().sum(axis=1)
            total_non_rolling = len(non_rolling_features)
            
            # Keep rows that have less than 50% NaN values in non-rolling features
            valid_rows = (nan_counts / total_non_rolling < nan_threshold)
            features_df = features_df[valid_rows]
        
        # For non-rolling features, use forward fill then backward fill
        for col in non_rolling_features:
            if features_df[col].isna().any():
                features_df[col] = features_df[col].ffill().bfill()
        
        # For non-rolling features, fill remaining NaN values with appropriate defaults
        for col in non_rolling_features:
            if features_df[col].isna().any():
                if any(pattern in col.lower() for pattern in ['return', 'change', 'ratio']):
                    features_df[col] = features_df[col].fillna(0)
                elif any(pattern in col.lower() for pattern in ['volume', 'count']):
                    features_df[col] = features_df[col].fillna(0)
                else:
                    # For other features, use median
                    features_df[col] = features_df[col].fillna(features_df[col].median())
        
        # For rolling features, preserve initial NaN values but handle other NaN values
        for col in rolling_features:
            if features_df[col].isna().any():
                # Only fill NaN values that are not in the initial window period
                # Extract window size from column name if possible
                window_size = None
                for window in self.rolling_windows:
                    if f'_{window}' in col or f'{window}d' in col:
                        window_size = window
                        break
                
                if window_size is not None:
                    # Keep initial NaN values (first window_size-1 values should be NaN)
                    # Only fill NaN values after the initial window period
                    non_initial_mask = features_df.index >= features_df.index[window_size-1] if len(features_df) >= window_size else features_df.index >= features_df.index[0]
                    nan_mask = features_df[col].isna()
                    fill_mask = nan_mask & non_initial_mask
                    
                    if fill_mask.any():
                        # Use forward fill for non-initial NaN values
                        features_df.loc[fill_mask, col] = features_df[col].ffill().loc[fill_mask]
                else:
                    # If we can't determine window size, use conservative approach
                    # Only fill NaN values after the maximum window size
                    if len(features_df) > max_window:
                        non_initial_mask = features_df.index >= features_df.index[max_window]
                        nan_mask = features_df[col].isna()
                        fill_mask = nan_mask & non_initial_mask
                        
                        if fill_mask.any():
                            features_df.loc[fill_mask, col] = features_df[col].ffill().loc[fill_mask]
        
        # Handle infinite values
        for col in features_df.columns:
            if np.isinf(features_df[col]).any():
                logging.warning(f"Replacing infinite values in feature {col}")
                # Replace positive infinity with 99th percentile
                pos_inf_mask = np.isposinf(features_df[col])
                if pos_inf_mask.any():
                    percentile_99 = features_df[col][~np.isinf(features_df[col])].quantile(0.99)
                    features_df.loc[pos_inf_mask, col] = percentile_99
                
                # Replace negative infinity with 1st percentile
                neg_inf_mask = np.isneginf(features_df[col])
                if neg_inf_mask.any():
                    percentile_01 = features_df[col][~np.isinf(features_df[col])].quantile(0.01)
                    features_df.loc[neg_inf_mask, col] = percentile_01
        
        final_rows = len(features_df)
        
        if initial_rows != final_rows:
            logging.warning(f"Dropped {initial_rows - final_rows} rows due to excessive NaN values")
        
        # Final NaN cleanup - ensure no unexpected NaN values remain (preserve rolling feature initial NaNs)
        remaining_nans = features_df.isna().sum().sum()
        if remaining_nans > 0:
            logging.info(f"Final NaN check: {remaining_nans} NaN values found (may include expected rolling feature initial NaNs)")
            
            # Count NaN values in non-rolling features only
            non_rolling_nans = 0
            for col in non_rolling_features:
                non_rolling_nans += features_df[col].isna().sum()
            
            if non_rolling_nans > 0:
                logging.warning(f"Unexpected NaN values in non-rolling features: {non_rolling_nans}")
                
                # Apply final cleanup only to non-rolling features
                for col in non_rolling_features:
                    if features_df[col].isna().any():
                        if features_df[col].dtype in ['object', 'category']:
                            # For categorical columns, use mode or 'unknown'
                            mode_val = features_df[col].mode()
                            fill_val = mode_val[0] if len(mode_val) > 0 else 'unknown'
                            features_df[col] = features_df[col].fillna(fill_val)
                        else:
                            # For numeric columns, use median or 0
                            if features_df[col].notna().any():
                                median_val = features_df[col].median()
                                features_df[col] = features_df[col].fillna(median_val)
                            else:
                                features_df[col] = features_df[col].fillna(0)
                
                # Verify no NaN values remain in non-rolling features
                final_non_rolling_nans = sum(features_df[col].isna().sum() for col in non_rolling_features)
                if final_non_rolling_nans > 0:
                    logging.error(f"Failed to remove all NaN values from non-rolling features: {final_non_rolling_nans} remain")
                else:
                    logging.info("All unexpected NaN values successfully removed")
            else:
                logging.info("All NaN values are in rolling features (expected initial NaNs)")
        
        logging.info(f"Feature engineering completed: {features_df.shape[1]} features, {features_df.shape[0]} samples")
        
        return features_df
    
    def _group1_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Group 1: Basic Features (OHLCV, log returns).
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with basic features added
        """
        logging.info("Engineering Group 1: Basic Features")
        
        features = []
        
        # Price columns
        price_columns = [col for col in data.columns 
                        if any(price_type in col.lower() 
                              for price_type in ['open', 'high', 'low', 'close'])]
        
        for col in price_columns:
            if col not in data.columns:
                continue
            
            # Skip binary/flag columns that shouldn't have log applied
            if any(pattern in col.lower() for pattern in ['event', 'flag', 'binary']):
                continue
                
            # Log prices (handle zero/negative values)
            log_col = f'{col}_log'
            data[log_col] = np.log(np.maximum(data[col], 1e-8))  # Prevent log(0) or log(negative)
            features.append(log_col)
            
            # Log returns (already calculated in acquisition, but ensure consistency)
            if 'close' in col.lower():
                log_return_col = col.replace('Close', 'LogReturn')
                if log_return_col not in data.columns:
                    data[log_return_col] = np.log(data[col] / data[col].shift(1))
                features.append(log_return_col)
        
        # Volume features (if available)
        volume_columns = [col for col in data.columns if 'volume' in col.lower()]
        for col in volume_columns:
            # Log volume
            data[f'{col}_log'] = np.log(data[col] + 1)  # +1 to handle zero volumes
            features.append(f'{col}_log')
            
            # Volume change
            data[f'{col}_change'] = data[col].pct_change()
            features.append(f'{col}_change')
        
        # Price spreads and ratios
        if 'ETH_High' in data.columns and 'ETH_Low' in data.columns:
            data['ETH_HL_spread'] = (data['ETH_High'] - data['ETH_Low']) / data['ETH_Close']
            features.append('ETH_HL_spread')
        
        if 'ETH_Open' in data.columns and 'ETH_Close' in data.columns:
            data['ETH_OC_ratio'] = data['ETH_Close'] / data['ETH_Open']
            features.append('ETH_OC_ratio')
        
        self.feature_groups['basic'] = features
        logging.info(f"  Added {len(features)} basic features")
        
        return data
    
    def _group2_returns_and_lags(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Group 2: Returns & Lags (multiple horizons).
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with returns and lag features added
        """
        logging.info("Engineering Group 2: Returns & Lags")
        
        features = []
        
        # Return columns
        return_columns = [col for col in data.columns if 'logreturn' in col.lower()]
        
        for col in return_columns:
            # Lag features
            for lag in self.lag_periods:
                lag_col = f'{col}_lag_{lag}'
                data[lag_col] = data[col].shift(lag)
                features.append(lag_col)
            
            # Multi-period returns (cumulative)
            for period in [2, 3, 5, 7, 14, 21]:
                if period <= max(self.lag_periods):
                    cumret_col = f'{col}_cumret_{period}'
                    data[cumret_col] = data[col].rolling(window=period).sum()
                    features.append(cumret_col)
        
        # Price lag features
        price_columns = [col for col in data.columns 
                        if 'close' in col.lower() and 'log' not in col.lower()]
        
        for col in price_columns:
            # Key price lags
            for lag in [1, 2, 3, 7]:
                if lag in self.lag_periods:
                    lag_col = f'{col}_lag_{lag}'
                    data[lag_col] = data[col].shift(lag)
                    features.append(lag_col)
        
        self.feature_groups['returns_lags'] = features
        logging.info(f"  Added {len(features)} returns & lag features")
        
        return data
    
    def _group3_rolling_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Group 3: Rolling Statistics (mean, std, min, max).
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with rolling statistics added
        """
        logging.info("Engineering Group 3: Rolling Statistics")
        
        features = []
        
        # Target columns for rolling stats
        target_columns = [col for col in data.columns 
                         if any(target in col.lower() 
                               for target in ['close', 'logreturn', 'volume'])]
        
        for col in target_columns:
            if 'lag' in col.lower():  # Skip lag features
                continue
                
            for window in self.rolling_windows:
                # Rolling mean - FIXED: Use min_periods=window to prevent data leakage
                mean_col = f'{col}_rolling_mean_{window}'
                data[mean_col] = data[col].rolling(window=window, min_periods=window).mean()
                features.append(mean_col)
                
                # Rolling std - FIXED: Use min_periods=window to prevent data leakage
                std_col = f'{col}_rolling_std_{window}'
                data[std_col] = data[col].rolling(window=window, min_periods=window).std()
                features.append(std_col)
                
                # Rolling min/max (for price columns) - FIXED: Use min_periods=window
                if 'close' in col.lower() or 'price' in col.lower():
                    min_col = f'{col}_rolling_min_{window}'
                    max_col = f'{col}_rolling_max_{window}'
                    data[min_col] = data[col].rolling(window=window, min_periods=window).min()
                    data[max_col] = data[col].rolling(window=window, min_periods=window).max()
                    features.extend([min_col, max_col])
                    
                    # Price position within range
                    range_col = f'{col}_range_position_{window}'
                    data[range_col] = (data[col] - data[min_col]) / (data[max_col] - data[min_col] + 1e-8)
                    features.append(range_col)
                
                # Rolling skewness and kurtosis (for return columns) - FIXED: Use min_periods=window
                if 'return' in col.lower() and window >= 14:
                    skew_col = f'{col}_rolling_skew_{window}'
                    kurt_col = f'{col}_rolling_kurt_{window}'
                    data[skew_col] = data[col].rolling(window=window, min_periods=window).skew()
                    data[kurt_col] = data[col].rolling(window=window, min_periods=window).kurt()
                    features.extend([skew_col, kurt_col])
        
        self.feature_groups['rolling_stats'] = features
        logging.info(f"  Added {len(features)} rolling statistics features")
        
        return data
    
    def _group4_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Group 4: Volatility Features (realized vol, GARCH-style).
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with volatility features added
        """
        logging.info("Engineering Group 4: Volatility Features")
        
        features = []
        
        # Return columns for volatility calculation
        return_columns = [col for col in data.columns if 'logreturn' in col.lower()]
        
        for col in return_columns:
            for window in self.volatility_windows:
                # Realized volatility (annualized) - FIXED: Use min_periods=window
                vol_col = f'{col}_realized_vol_{window}'
                data[vol_col] = data[col].rolling(window=window, min_periods=window).std() * np.sqrt(365)
                features.append(vol_col)
                
                # GARCH-style volatility (EWMA) - EWMA is OK as it doesn't use future data
                ewma_col = f'{col}_ewma_vol_{window}'
                alpha = 2 / (window + 1)
                data[ewma_col] = data[col].ewm(alpha=alpha).std() * np.sqrt(365)
                features.append(ewma_col)
                
                # Volatility of volatility - FIXED: Use proper min_periods
                if window >= 14:
                    vol_vol_col = f'{col}_vol_of_vol_{window}'
                    vol_window = window//2
                    data[vol_vol_col] = data[vol_col].rolling(window=vol_window, min_periods=vol_window).std()
                    features.append(vol_vol_col)
        
        # High-Low volatility (Garman-Klass estimator)
        if 'ETH_High' in data.columns and 'ETH_Low' in data.columns and 'ETH_Close' in data.columns:
            for window in self.volatility_windows:
                gk_col = f'ETH_garman_klass_vol_{window}'
                
                # Garman-Klass volatility estimator
                log_hl = np.log(data['ETH_High'] / data['ETH_Low'])
                log_co = np.log(data['ETH_Close'] / data['ETH_Open']) if 'ETH_Open' in data.columns else 0
                
                gk_vol = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
                data[gk_col] = gk_vol.rolling(window=window, min_periods=window).mean() * np.sqrt(365)
                features.append(gk_col)
        
        # Volatility regime indicators
        if len(return_columns) > 0:
            main_return_col = return_columns[0]  # Use ETH returns
            
            # High/Low volatility regime - FIXED: Use proper min_periods
            vol_30d = data[main_return_col].rolling(window=30, min_periods=30).std()
            vol_90d = data[main_return_col].rolling(window=90, min_periods=90).std()
            
            data['vol_regime_high'] = (vol_30d > vol_90d * 1.5).astype(int)
            data['vol_regime_low'] = (vol_30d < vol_90d * 0.5).astype(int)
            features.extend(['vol_regime_high', 'vol_regime_low'])
        
        self.feature_groups['volatility'] = features
        logging.info(f"  Added {len(features)} volatility features")
        
        return data
    
    def _group5_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Group 5: Momentum Features (RSI, MACD, Bollinger Bands).
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with momentum features added
        """
        logging.info("Engineering Group 5: Momentum Features")
        
        features = []
        
        # Price columns for momentum indicators
        price_columns = [col for col in data.columns 
                        if 'close' in col.lower() and 'log' not in col.lower()]
        
        for col in price_columns:
            asset_name = col.split('_')[0]  # Extract ETH, BTC, etc.
            
            # RSI
            for period in [14, 21, 30]:
                if period in self.momentum_windows:
                    rsi_col = f'{asset_name}_RSI_{period}'
                    data[rsi_col] = ta.momentum.RSIIndicator(data[col], window=period).rsi()
                    features.append(rsi_col)
            
            # MACD
            macd_col = f'{asset_name}_MACD'
            macd_signal_col = f'{asset_name}_MACD_signal'
            macd_hist_col = f'{asset_name}_MACD_hist'
            
            macd_indicator = ta.trend.MACD(data[col])
            data[macd_col] = macd_indicator.macd()
            data[macd_signal_col] = macd_indicator.macd_signal()
            data[macd_hist_col] = macd_indicator.macd_diff()
            features.extend([macd_col, macd_signal_col, macd_hist_col])
            
            # Bollinger Bands
            for period in [20, 30]:
                if period in self.momentum_windows or period == 20:
                    bb_upper_col = f'{asset_name}_BB_upper_{period}'
                    bb_lower_col = f'{asset_name}_BB_lower_{period}'
                    bb_width_col = f'{asset_name}_BB_width_{period}'
                    bb_position_col = f'{asset_name}_BB_position_{period}'
                    
                    bb_indicator = ta.volatility.BollingerBands(data[col], window=period)
                    data[bb_upper_col] = bb_indicator.bollinger_hband()
                    data[bb_lower_col] = bb_indicator.bollinger_lband()
                    data[bb_width_col] = (data[bb_upper_col] - data[bb_lower_col]) / data[col]
                    data[bb_position_col] = (data[col] - data[bb_lower_col]) / (data[bb_upper_col] - data[bb_lower_col])
                    
                    features.extend([bb_upper_col, bb_lower_col, bb_width_col, bb_position_col])
            
            # Stochastic Oscillator
            if f'{asset_name}_High' in data.columns and f'{asset_name}_Low' in data.columns:
                stoch_col = f'{asset_name}_Stochastic'
                stoch_indicator = ta.momentum.StochasticOscillator(
                    high=data[f'{asset_name}_High'],
                    low=data[f'{asset_name}_Low'],
                    close=data[col]
                )
                data[stoch_col] = stoch_indicator.stoch()
                features.append(stoch_col)
            
            # Williams %R
            if f'{asset_name}_High' in data.columns and f'{asset_name}_Low' in data.columns:
                williams_col = f'{asset_name}_Williams_R'
                williams_indicator = ta.momentum.WilliamsRIndicator(
                    high=data[f'{asset_name}_High'],
                    low=data[f'{asset_name}_Low'],
                    close=data[col]
                )
                data[williams_col] = williams_indicator.williams_r()
                features.append(williams_col)
        
        # Momentum regime indicators
        return_columns = [col for col in data.columns if 'logreturn' in col.lower()]
        if len(return_columns) > 0:
            main_return_col = return_columns[0]
            
            # Trend strength
            for window in [7, 14, 30]:
                trend_col = f'trend_strength_{window}'
                cumulative_return = data[main_return_col].rolling(window=window).sum()
                volatility = data[main_return_col].rolling(window=window).std()
                data[trend_col] = cumulative_return / (volatility * np.sqrt(window))
                features.append(trend_col)
        
        self.feature_groups['momentum'] = features
        logging.info(f"  Added {len(features)} momentum features")
        
        return data
    
    def _group6_cross_asset_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Group 6: Cross-Asset Features (BTC correlation, ratios).
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with cross-asset features added
        """
        logging.info("Engineering Group 6: Cross-Asset Features")
        
        features = []
        
        # Check if we have both ETH and BTC data
        eth_cols = [col for col in data.columns if 'ETH' in col]
        btc_cols = [col for col in data.columns if 'BTC' in col]
        
        if len(eth_cols) > 0 and len(btc_cols) > 0:
            # Price ratios
            if 'ETH_Close' in data.columns and 'BTC_Close' in data.columns:
                data['ETH_BTC_ratio'] = data['ETH_Close'] / data['BTC_Close']
                features.append('ETH_BTC_ratio')
                
                # Ratio momentum
                data['ETH_BTC_ratio_change'] = data['ETH_BTC_ratio'].pct_change()
                features.append('ETH_BTC_ratio_change')
                
                # Ratio moving averages
                for window in [7, 14, 30]:
                    ratio_ma_col = f'ETH_BTC_ratio_ma_{window}'
                    data[ratio_ma_col] = data['ETH_BTC_ratio'].rolling(window=window).mean()
                    features.append(ratio_ma_col)
            
            # Return correlations
            eth_return_cols = [col for col in data.columns if 'ETH' in col and 'logreturn' in col.lower()]
            btc_return_cols = [col for col in data.columns if 'BTC' in col and 'logreturn' in col.lower()]
            
            if len(eth_return_cols) > 0 and len(btc_return_cols) > 0:
                eth_return = eth_return_cols[0]
                btc_return = btc_return_cols[0]
                
                # Rolling correlations
                for window in [7, 14, 30, 60]:
                    corr_col = f'ETH_BTC_correlation_{window}'
                    data[corr_col] = data[eth_return].rolling(window=window).corr(data[btc_return])
                    features.append(corr_col)
                
                # Beta (ETH vs BTC)
                for window in [30, 60]:
                    beta_col = f'ETH_BTC_beta_{window}'
                    covariance = data[eth_return].rolling(window=window).cov(data[btc_return])
                    btc_variance = data[btc_return].rolling(window=window).var()
                    data[beta_col] = covariance / btc_variance
                    features.append(beta_col)
            
            # Volatility ratios
            eth_vol_cols = [col for col in data.columns if 'ETH' in col and 'vol' in col.lower()]
            btc_vol_cols = [col for col in data.columns if 'BTC' in col and 'vol' in col.lower()]
            
            if len(eth_vol_cols) > 0 and len(btc_vol_cols) > 0:
                # Use realized volatility columns
                eth_vol = [col for col in eth_vol_cols if 'realized_vol' in col]
                btc_vol = [col for col in btc_vol_cols if 'realized_vol' in col]
                
                if len(eth_vol) > 0 and len(btc_vol) > 0:
                    vol_ratio_col = 'ETH_BTC_vol_ratio'
                    data[vol_ratio_col] = data[eth_vol[0]] / data[btc_vol[0]]
                    features.append(vol_ratio_col)
        
        # Market dominance (if volume data available)
        eth_volume_cols = [col for col in data.columns if 'ETH' in col and 'volume' in col.lower()]
        btc_volume_cols = [col for col in data.columns if 'BTC' in col and 'volume' in col.lower()]
        
        if len(eth_volume_cols) > 0 and len(btc_volume_cols) > 0:
            eth_volume = eth_volume_cols[0]
            btc_volume = btc_volume_cols[0]
            
            total_volume = data[eth_volume] + data[btc_volume]
            data['ETH_volume_dominance'] = data[eth_volume] / total_volume
            features.append('ETH_volume_dominance')
        
        self.feature_groups['cross_asset'] = features
        logging.info(f"  Added {len(features)} cross-asset features")
        
        return data
    
    def _group7_calendar_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Group 7: Calendar Features (day of week, month, holidays).
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with calendar features added
        """
        logging.info("Engineering Group 7: Calendar Features")
        
        features = []
        
        # Ensure we have datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            return data
        
        # Day of week (0=Monday, 6=Sunday)
        data['day_of_week'] = data.index.dayofweek
        features.append('day_of_week')
        
        # Weekend indicator
        data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
        features.append('is_weekend')
        
        # Month
        data['month'] = data.index.month
        features.append('month')
        
        # Quarter
        data['quarter'] = data.index.quarter
        features.append('quarter')
        
        # Day of month
        data['day_of_month'] = data.index.day
        features.append('day_of_month')
        
        # Week of year
        data['week_of_year'] = data.index.isocalendar().week
        features.append('week_of_year')
        
        # Cyclical encoding for periodic features
        # Day of week (cyclical)
        data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        features.extend(['day_of_week_sin', 'day_of_week_cos'])
        
        # Month (cyclical)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        features.extend(['month_sin', 'month_cos'])
        
        # Day of month (cyclical)
        data['day_of_month_sin'] = np.sin(2 * np.pi * data['day_of_month'] / 31)
        data['day_of_month_cos'] = np.cos(2 * np.pi * data['day_of_month'] / 31)
        features.extend(['day_of_month_sin', 'day_of_month_cos'])
        
        # Special periods
        # Month-end effect (last 3 days of month)
        data['is_month_end'] = (data.index.day >= data.index.days_in_month - 2).astype(int)
        features.append('is_month_end')
        
        # Month-start effect (first 3 days of month)
        data['is_month_start'] = (data.index.day <= 3).astype(int)
        features.append('is_month_start')
        
        # Quarter-end effect
        quarter_end_months = [3, 6, 9, 12]
        data['is_quarter_end'] = ((data.index.month.isin(quarter_end_months)) & 
                                 (data.index.day >= data.index.days_in_month - 2)).astype(int)
        features.append('is_quarter_end')
        
        # Year-end effect (December)
        data['is_year_end'] = (data.index.month == 12).astype(int)
        features.append('is_year_end')
        
        # Holiday effects (simplified - major US holidays that might affect crypto)
        # New Year's Day
        data['is_new_year'] = ((data.index.month == 1) & (data.index.day == 1)).astype(int)
        features.append('is_new_year')
        
        # Christmas
        data['is_christmas'] = ((data.index.month == 12) & (data.index.day == 25)).astype(int)
        features.append('is_christmas')
        
        # Holiday proximity (within 3 days of major holidays)
        # Ensure holiday dates have the same timezone as the data index
        tz = data.index.tz if hasattr(data.index, 'tz') and data.index.tz is not None else None
        
        new_year_dates = pd.to_datetime([f'{year}-01-01' for year in data.index.year.unique()], utc=True)
        christmas_dates = pd.to_datetime([f'{year}-12-25' for year in data.index.year.unique()], utc=True)
        
        # Convert to same timezone as data if needed
        if tz is not None:
            new_year_dates = new_year_dates.tz_convert(tz)
            christmas_dates = christmas_dates.tz_convert(tz)
        elif hasattr(data.index, 'tz_localize'):
            # If data index is timezone-naive, make holiday dates timezone-naive too
            new_year_dates = new_year_dates.tz_localize(None)
            christmas_dates = christmas_dates.tz_localize(None)
        
        data['near_holiday'] = 0
        for date in data.index:
            for holiday_set in [new_year_dates, christmas_dates]:
                if any(abs((date - holiday).days) <= 3 for holiday in holiday_set):
                    data.loc[date, 'near_holiday'] = 1
                    break
        features.append('near_holiday')
        
        self.feature_groups['calendar'] = features
        logging.info(f"  Added {len(features)} calendar features")
        
        return data
    
    def _group8_event_flags(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Group 8: Event Flags (extreme events, regime changes).
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with event flag features added
        """
        logging.info("Engineering Group 8: Event Flags")
        
        features = []
        
        # Return columns for event detection
        return_columns = [col for col in data.columns if 'logreturn' in col.lower()]
        
        for col in return_columns:
            asset_name = col.split('_')[0]
            
            # Extreme return events (already created in denoising, but ensure consistency)
            extreme_col = f'{col}_extreme_event'
            if extreme_col not in data.columns:
                rolling_std = data[col].rolling(window=30, min_periods=30).std()
                rolling_mean = data[col].rolling(window=30, min_periods=30).mean()
                threshold = 3 * rolling_std
                
                data[extreme_col] = (
                    (data[col] > rolling_mean + threshold) | 
                    (data[col] < rolling_mean - threshold)
                ).astype(int)
            features.append(extreme_col)
            
            # Large positive/negative moves
            large_positive_col = f'{asset_name}_large_positive'
            large_negative_col = f'{asset_name}_large_negative'
            
            # Define large moves as > 2 standard deviations - FIXED: Use proper min_periods
            rolling_std = data[col].rolling(window=30, min_periods=30).std()
            data[large_positive_col] = (data[col] > 2 * rolling_std).astype(int)
            data[large_negative_col] = (data[col] < -2 * rolling_std).astype(int)
            features.extend([large_positive_col, large_negative_col])
            
            # Consecutive move patterns
            consecutive_up_col = f'{asset_name}_consecutive_up'
            consecutive_down_col = f'{asset_name}_consecutive_down'
            
            # Count consecutive positive/negative returns
            positive_returns = (data[col] > 0).astype(int)
            negative_returns = (data[col] < 0).astype(int)
            
            # Calculate consecutive counts
            data[consecutive_up_col] = (positive_returns * 
                                      (positive_returns.groupby((positive_returns != positive_returns.shift()).cumsum()).cumcount() + 1))
            data[consecutive_down_col] = (negative_returns * 
                                        (negative_returns.groupby((negative_returns != negative_returns.shift()).cumsum()).cumcount() + 1))
            features.extend([consecutive_up_col, consecutive_down_col])
        
        # Volatility regime changes
        if len(return_columns) > 0:
            main_return_col = return_columns[0]
            
            # Volatility breakout
            vol_30d = data[main_return_col].rolling(window=30, min_periods=30).std()
            vol_90d = data[main_return_col].rolling(window=90, min_periods=90).std()
            
            data['vol_breakout'] = (vol_30d > vol_90d * 2).astype(int)
            features.append('vol_breakout')
            
            # Volatility compression
            data['vol_compression'] = (vol_30d < vol_90d * 0.5).astype(int)
            features.append('vol_compression')
        
        # Price level events
        price_columns = [col for col in data.columns 
                        if 'close' in col.lower() and 'log' not in col.lower()]
        
        for col in price_columns:
            asset_name = col.split('_')[0]
            
            # New highs/lows
            for window in [30, 60, 90]:
                new_high_col = f'{asset_name}_new_high_{window}d'
                new_low_col = f'{asset_name}_new_low_{window}d'
                
                rolling_max = data[col].rolling(window=window, min_periods=window).max()
                rolling_min = data[col].rolling(window=window, min_periods=window).min()
                
                data[new_high_col] = (data[col] >= rolling_max).astype(int)
                data[new_low_col] = (data[col] <= rolling_min).astype(int)
                features.extend([new_high_col, new_low_col])
        
        # Market structure events
        if 'ETH_Close' in data.columns:
            # Support/resistance breaks (simplified)
            eth_close = data['ETH_Close']
            
            # 20-day support/resistance levels - FIXED: Use proper min_periods
            support_20d = eth_close.rolling(window=20, min_periods=20).min()
            resistance_20d = eth_close.rolling(window=20, min_periods=20).max()
            
            data['support_break'] = (eth_close < support_20d.shift(1)).astype(int)
            data['resistance_break'] = (eth_close > resistance_20d.shift(1)).astype(int)
            features.extend(['support_break', 'resistance_break'])
        
        self.feature_groups['event_flags'] = features
        logging.info(f"  Added {len(features)} event flag features")
        
        return data
    
    def _apply_scaling(self, data: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """
        Apply scaling to features (fit only on training data).
        
        Args:
            data: Input DataFrame
            is_training: Whether this is training data
            
        Returns:
            Scaled DataFrame
        """
        logging.info(f"Applying scaling (training={is_training})")
        
        # Columns to exclude from scaling
        exclude_columns = []
        
        # Exclude binary/categorical features
        binary_patterns = ['is_', '_event', '_flag', 'day_of_week', 'month', 'quarter', 
                          'day_of_month', 'week_of_year', 'consecutive_', 'new_high', 
                          'new_low', '_break', 'regime_', 'near_holiday']
        
        for col in data.columns:
            if any(pattern in col.lower() for pattern in binary_patterns):
                exclude_columns.append(col)
        
        # Columns to scale
        scale_columns = [col for col in data.columns if col not in exclude_columns]
        
        if is_training:
            # Initialize scaler
            if self.scaler_type == 'standard':
                scaler = StandardScaler()
            elif self.scaler_type == 'robust':
                scaler = RobustScaler()
            elif self.scaler_type == 'minmax':
                scaler = MinMaxScaler()
            else:
                logging.warning(f"Unknown scaler type: {self.scaler_type}, using RobustScaler")
                scaler = RobustScaler()
            
            # Fit scaler on training data
            if len(scale_columns) > 0:
                scaler.fit(data[scale_columns])
                self.scalers['features'] = scaler
                
                # Transform training data
                data[scale_columns] = scaler.transform(data[scale_columns])
                
                logging.info(f"Fitted {self.scaler_type} scaler on {len(scale_columns)} features")
        else:
            # Transform using fitted scaler
            if 'features' in self.scalers and len(scale_columns) > 0:
                data[scale_columns] = self.scalers['features'].transform(data[scale_columns])
                logging.info(f"Applied fitted scaler to {len(scale_columns)} features")
            else:
                logging.warning("No fitted scaler found for feature scaling")
        
        return data
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Get feature groups for analysis and interpretation.
        
        Returns:
            Dictionary mapping group names to feature lists
        """
        if not hasattr(self, 'feature_groups') or not self.feature_groups:
            # If feature_groups is empty, create default groups based on common patterns
            logging.warning("Feature groups not populated. Creating default groups based on patterns.")
            return {
                'basic': ['Open', 'High', 'Low', 'Close', 'Volume'],
                'returns': ['log_return', 'return_1d', 'return_7d'],
                'volatility': ['volatility_7d', 'volatility_14d', 'volatility_30d'],
                'momentum': ['rsi_14', 'macd', 'bollinger_upper', 'bollinger_lower'],
                'rolling': ['rolling_mean_7d', 'rolling_std_7d', 'rolling_mean_30d'],
                'calendar': ['day_of_week', 'month', 'quarter'],
                'events': ['extreme_event', 'regime_change']
            }
        
        return self.feature_groups.copy()
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get summary of engineered features.
        
        Returns:
            Dictionary with feature summary
        """
        summary = {
            'total_features': sum(len(features) for features in self.feature_groups.values()),
            'feature_groups': {group: len(features) for group, features in self.feature_groups.items()},
            'scaler_type': self.scaler_type,
            'scalers_fitted': list(self.scalers.keys())
        }
        
        return summary
    
    def save_feature_importance(self, importance_dict: Dict[str, float], 
                               model_name: str, output_dir: str = "reports/metrics") -> None:
        """
        Save feature importance rankings.
        
        Args:
            importance_dict: Dictionary of feature importances
            model_name: Name of the model
            output_dir: Output directory
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to DataFrame and sort
        importance_df = pd.DataFrame(list(importance_dict.items()), 
                                   columns=['feature', 'importance'])
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Add feature group information
        importance_df['feature_group'] = 'unknown'
        for group, features in self.feature_groups.items():
            mask = importance_df['feature'].isin(features)
            importance_df.loc[mask, 'feature_group'] = group
        
        # Save to CSV
        filename = f'feature_importance_{model_name.lower()}.csv'
        filepath = os.path.join(output_dir, filename)
        importance_df.to_csv(filepath, index=False)
        
        # Store in memory
        self.feature_importance[model_name] = importance_df
        
        logging.info(f"Feature importance saved: {filepath}")


def main():
    """
    Main function for testing feature engineering.
    """
    # Load configuration
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.utils.helpers import load_config, setup_logging
    
    # Setup
    config = load_config()
    setup_logging()
    
    # Create synthetic data for testing
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # Generate synthetic OHLCV data
    base_price = 100
    returns = np.random.normal(0, 0.02, 500)
    prices = base_price * np.exp(np.cumsum(returns))
    
    synthetic_data = pd.DataFrame({
        'Date': dates,
        'ETH_Open': prices * (1 + np.random.normal(0, 0.001, 500)),
        'ETH_High': prices * (1 + np.abs(np.random.normal(0, 0.01, 500))),
        'ETH_Low': prices * (1 - np.abs(np.random.normal(0, 0.01, 500))),
        'ETH_Close': prices,
        'ETH_Volume': np.random.lognormal(10, 1, 500),
        'BTC_Close': prices * 100 * (1 + np.random.normal(0, 0.005, 500)),
        'ETH_LogReturn': returns,
        'BTC_LogReturn': returns * 0.8 + np.random.normal(0, 0.01, 500)
    })
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(config)
    
    # Engineer features
    features_df = feature_engineer.engineer_features(synthetic_data, is_training=True)
    
    # Print summary
    summary = feature_engineer.get_feature_summary()
    print("Feature Engineering Summary:")
    print(f"  Total features: {summary['total_features']}")
    print("  Feature groups:")
    for group, count in summary['feature_groups'].items():
        print(f"    {group}: {count}")
    
    print(f"\nFinal dataset shape: {features_df.shape}")
    print(f"Features: {list(features_df.columns)}")


if __name__ == "__main__":
    main()