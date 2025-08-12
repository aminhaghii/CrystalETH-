"""
Data denoising module for ETH forecasting project.

This module implements the sequential denoising pipeline as specified in Rule #8:
1. NaN handling (20% threshold)
2. Hampel filter (window=5, n_sigma=3)
3. Winsorization (0.5-1% tails)
4. Wavelet denoising (db4/db8)
5. Rolling median (3-day)
6. Extreme event flagging

Each step generates before/after plots for validation.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional, List
import pywt
from scipy import stats
from scipy.signal import medfilt
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class DataDenoiser:
    """
    Data denoising class implementing the sequential denoising pipeline.
    
    This class ensures proper execution order and generates diagnostic plots
    for each denoising step.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataDenoiser with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.preprocessing_config = config.get('preprocessing', {})
        
        # Denoising parameters
        self.nan_threshold = self.preprocessing_config.get('nan_threshold', 0.20)
        self.hampel_window = self.preprocessing_config.get('hampel_window', 5)
        self.hampel_n_sigma = self.preprocessing_config.get('hampel_n_sigma', 3)
        self.winsorize_lower = self.preprocessing_config.get('winsorize_lower', 0.005)
        self.winsorize_upper = self.preprocessing_config.get('winsorize_upper', 0.995)
        self.wavelet_type = self.preprocessing_config.get('wavelet_type', 'db4')
        self.rolling_median_window = self.preprocessing_config.get('rolling_median_window', 3)
        self.extreme_event_multiplier = self.preprocessing_config.get('extreme_event_multiplier', 8)
        self.rolling_std_window = self.preprocessing_config.get('rolling_std_window', 30)
        
        # Storage for before/after data
        self.denoising_history = {}
        
        logging.info("DataDenoiser initialized with sequential pipeline")
    
    def denoise_data(self, data: pd.DataFrame, 
                    target_columns: List[str] = None) -> pd.DataFrame:
        """
        Apply complete denoising pipeline to data.
        
        Args:
            data: Input DataFrame
            target_columns: Columns to denoise (if None, denoise price columns)
            
        Returns:
            Denoised DataFrame
        """
        if target_columns is None:
            # Default to price columns
            target_columns = [col for col in data.columns 
                            if any(price_type in col.lower() 
                                  for price_type in ['close', 'open', 'high', 'low'])]
        
        logging.info(f"Starting denoising pipeline for columns: {target_columns}")
        
        # Make a copy to avoid modifying original data
        denoised_data = data.copy()
        
        # Apply denoising steps in exact order (Rule #8)
        denoised_data = self._step1_nan_handling(denoised_data, target_columns)
        denoised_data = self._step2_hampel_filter(denoised_data, target_columns)
        denoised_data = self._step3_winsorization(denoised_data, target_columns)
        denoised_data = self._step4_wavelet_denoising(denoised_data, target_columns)
        denoised_data = self._step5_rolling_median(denoised_data, target_columns)
        denoised_data = self._step6_extreme_event_flagging(denoised_data, target_columns)
        
        logging.info("Denoising pipeline completed successfully")
        return denoised_data
    
    def _step1_nan_handling(self, data: pd.DataFrame, 
                           target_columns: List[str]) -> pd.DataFrame:
        """
        Step 1: NaN handling with 20% threshold rule.
        
        Args:
            data: Input DataFrame
            target_columns: Columns to process
            
        Returns:
            DataFrame with NaN handling applied
        """
        logging.info("Step 1: NaN handling (20% threshold)")
        
        for col in target_columns:
            if col not in data.columns:
                continue
                
            # Store original for comparison
            original_series = data[col].copy()
            
            # Calculate NaN percentage
            nan_percentage = data[col].isnull().sum() / len(data)
            
            if nan_percentage > self.nan_threshold:
                logging.warning(f"Column {col} has {nan_percentage:.2%} NaN values (> {self.nan_threshold:.1%})")
                # For high NaN percentage, use forward fill then backward fill
                data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
            else:
                # For low NaN percentage, use interpolation
                data[col] = data[col].interpolate(method='linear')
                # Fill any remaining NaNs at edges
                data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
            
            # Store for plotting
            self.denoising_history[f'{col}_step1'] = {
                'original': original_series,
                'processed': data[col].copy(),
                'step': 'NaN Handling'
            }
            
            logging.info(f"  {col}: {nan_percentage:.2%} NaN values handled")
        
        return data
    
    def _step2_hampel_filter(self, data: pd.DataFrame, 
                            target_columns: List[str]) -> pd.DataFrame:
        """
        Step 2: Hampel filter (window=5, n_sigma=3).
        
        Args:
            data: Input DataFrame
            target_columns: Columns to process
            
        Returns:
            DataFrame with Hampel filter applied
        """
        logging.info("Step 2: Hampel filter (window=5, n_sigma=3)")
        
        for col in target_columns:
            if col not in data.columns:
                continue
                
            original_series = data[col].copy()
            
            # Apply Hampel filter
            filtered_series = self._hampel_filter(
                data[col], 
                window_size=self.hampel_window, 
                n_sigma=self.hampel_n_sigma
            )
            
            data[col] = filtered_series
            
            # Store for plotting
            self.denoising_history[f'{col}_step2'] = {
                'original': original_series,
                'processed': data[col].copy(),
                'step': 'Hampel Filter'
            }
            
            # Count outliers removed
            outliers_removed = (original_series != filtered_series).sum()
            logging.info(f"  {col}: {outliers_removed} outliers removed by Hampel filter")
        
        return data
    
    def _hampel_filter(self, series: pd.Series, window_size: int = 5, 
                      n_sigma: float = 3) -> pd.Series:
        """
        Apply Hampel filter to remove outliers.
        
        Args:
            series: Input series
            window_size: Window size for median calculation
            n_sigma: Number of standard deviations for outlier detection
            
        Returns:
            Filtered series
        """
        filtered_series = series.copy()
        
        for i in range(len(series)):
            # Define window
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(series), i + window_size // 2 + 1)
            
            window = series.iloc[start_idx:end_idx]
            
            # Calculate median and MAD
            median = window.median()
            mad = np.median(np.abs(window - median))
            
            # Modified z-score
            if mad > 0:
                modified_z_score = 0.6745 * (series.iloc[i] - median) / mad
                
                # Replace outliers with median
                if np.abs(modified_z_score) > n_sigma:
                    filtered_series.iloc[i] = median
        
        return filtered_series
    
    def _step3_winsorization(self, data: pd.DataFrame, 
                            target_columns: List[str]) -> pd.DataFrame:
        """
        Step 3: Winsorization (0.5-1% tails).
        
        Args:
            data: Input DataFrame
            target_columns: Columns to process
            
        Returns:
            DataFrame with winsorization applied
        """
        logging.info("Step 3: Winsorization (0.5-1% tails)")
        
        for col in target_columns:
            if col not in data.columns:
                continue
                
            original_series = data[col].copy()
            
            # Calculate percentiles
            lower_percentile = data[col].quantile(self.winsorize_lower)
            upper_percentile = data[col].quantile(self.winsorize_upper)
            
            # Apply winsorization
            data[col] = data[col].clip(lower=lower_percentile, upper=upper_percentile)
            
            # Store for plotting
            self.denoising_history[f'{col}_step3'] = {
                'original': original_series,
                'processed': data[col].copy(),
                'step': 'Winsorization'
            }
            
            # Count values winsorized
            values_winsorized = ((original_series < lower_percentile) | 
                               (original_series > upper_percentile)).sum()
            logging.info(f"  {col}: {values_winsorized} values winsorized")
        
        return data
    
    def _step4_wavelet_denoising(self, data: pd.DataFrame, 
                                target_columns: List[str]) -> pd.DataFrame:
        """
        Step 4: Wavelet denoising (db4/db8).
        
        Args:
            data: Input DataFrame
            target_columns: Columns to process
            
        Returns:
            DataFrame with wavelet denoising applied
        """
        logging.info(f"Step 4: Wavelet denoising ({self.wavelet_type})")
        
        for col in target_columns:
            if col not in data.columns:
                continue
                
            original_series = data[col].copy()
            
            # Apply wavelet denoising
            denoised_values = self._wavelet_denoise(
                data[col].values, 
                wavelet=self.wavelet_type
            )
            
            data[col] = denoised_values
            
            # Store for plotting
            self.denoising_history[f'{col}_step4'] = {
                'original': original_series,
                'processed': data[col].copy(),
                'step': 'Wavelet Denoising'
            }
            
            # Calculate noise reduction
            noise_reduction = np.std(original_series - data[col])
            logging.info(f"  {col}: Noise reduction std = {noise_reduction:.6f}")
        
        return data
    
    def _wavelet_denoise(self, signal: np.ndarray, wavelet: str = 'db4', 
                        mode: str = 'symmetric') -> np.ndarray:
        """
        Apply wavelet denoising to signal.
        
        Args:
            signal: Input signal
            wavelet: Wavelet type
            mode: Boundary condition mode
            
        Returns:
            Denoised signal
        """
        # Decompose signal
        coeffs = pywt.wavedec(signal, wavelet, mode=mode)
        
        # Estimate noise level using median absolute deviation of finest detail coefficients
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        
        # Calculate threshold
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        
        # Apply soft thresholding
        coeffs_thresh = list(coeffs)
        coeffs_thresh[1:] = [pywt.threshold(detail, threshold, mode='soft') 
                            for detail in coeffs_thresh[1:]]
        
        # Reconstruct signal
        denoised_signal = pywt.waverec(coeffs_thresh, wavelet, mode=mode)
        
        # Ensure same length as original
        if len(denoised_signal) != len(signal):
            denoised_signal = denoised_signal[:len(signal)]
        
        return denoised_signal
    
    def _step5_rolling_median(self, data: pd.DataFrame, 
                             target_columns: List[str]) -> pd.DataFrame:
        """
        Step 5: Rolling median smoothing (3-day).
        
        Args:
            data: Input DataFrame
            target_columns: Columns to process
            
        Returns:
            DataFrame with rolling median applied
        """
        logging.info(f"Step 5: Rolling median smoothing ({self.rolling_median_window}-day)")
        
        for col in target_columns:
            if col not in data.columns:
                continue
                
            original_series = data[col].copy()
            
            # Apply rolling median
            data[col] = data[col].rolling(
                window=self.rolling_median_window, 
                center=True, 
                min_periods=1
            ).median()
            
            # Store for plotting
            self.denoising_history[f'{col}_step5'] = {
                'original': original_series,
                'processed': data[col].copy(),
                'step': 'Rolling Median'
            }
            
            # Calculate smoothing effect
            smoothing_effect = np.std(original_series - data[col])
            logging.info(f"  {col}: Smoothing effect std = {smoothing_effect:.6f}")
        
        return data
    
    def _step6_extreme_event_flagging(self, data: pd.DataFrame, 
                                     target_columns: List[str]) -> pd.DataFrame:
        """
        Step 6: Extreme event flagging (8*rolling_std_30d).
        
        Args:
            data: Input DataFrame
            target_columns: Columns to process
            
        Returns:
            DataFrame with extreme event flags
        """
        logging.info(f"Step 6: Extreme event flagging ({self.extreme_event_multiplier}*rolling_std_{self.rolling_std_window}d)")
        
        for col in target_columns:
            if col not in data.columns:
                continue
                
            # Calculate rolling standard deviation
            rolling_std = data[col].rolling(
                window=self.rolling_std_window, 
                min_periods=self.rolling_std_window
            ).std()
            
            # Calculate rolling mean
            rolling_mean = data[col].rolling(
                window=self.rolling_std_window, 
                min_periods=self.rolling_std_window
            ).mean()
            
            # Define extreme event threshold
            upper_threshold = rolling_mean + self.extreme_event_multiplier * rolling_std
            lower_threshold = rolling_mean - self.extreme_event_multiplier * rolling_std
            
            # Create extreme event flags
            extreme_events = (
                (data[col] > upper_threshold) | 
                (data[col] < lower_threshold)
            )
            
            # Add flag column
            flag_col = f'{col}_extreme_event'
            data[flag_col] = extreme_events.astype(int)
            
            # Count extreme events
            extreme_count = extreme_events.sum()
            logging.info(f"  {col}: {extreme_count} extreme events flagged ({extreme_count/len(data):.2%})")
        
        return data
    
    def generate_denoising_plots(self, output_dir: str = "reports/figures") -> None:
        """
        Generate before/after plots for each denoising step.
        
        Args:
            output_dir: Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logging.info("Generating denoising diagnostic plots...")
        
        # Group by column and step
        columns = set()
        for key in self.denoising_history.keys():
            col_name = key.split('_step')[0]
            columns.add(col_name)
        
        for col in columns:
            self._plot_denoising_steps(col, output_dir)
        
        logging.info(f"Denoising plots saved to {output_dir}")
    
    def _plot_denoising_steps(self, column: str, output_dir: str) -> None:
        """
        Plot denoising steps for a specific column.
        
        Args:
            column: Column name
            output_dir: Output directory
        """
        # Collect all steps for this column
        steps = []
        for i in range(1, 7):
            key = f'{column}_step{i}'
            if key in self.denoising_history:
                steps.append((i, self.denoising_history[key]))
        
        if not steps:
            return
        
        # Create subplot figure
        fig, axes = plt.subplots(len(steps), 1, figsize=(15, 3 * len(steps)))
        if len(steps) == 1:
            axes = [axes]
        
        fig.suptitle(f'Denoising Pipeline: {column}', fontsize=16, fontweight='bold')
        
        for idx, (step_num, step_data) in enumerate(steps):
            ax = axes[idx]
            
            # Plot original and processed
            ax.plot(step_data['original'].values, label='Before', alpha=0.7, linewidth=1)
            ax.plot(step_data['processed'].values, label='After', alpha=0.8, linewidth=1.5)
            
            ax.set_title(f"Step {step_num}: {step_data['step']}")
            ax.set_xlabel('Time Index')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            original_std = step_data['original'].std()
            processed_std = step_data['processed'].std()
            reduction = (original_std - processed_std) / original_std * 100
            
            ax.text(0.02, 0.98, f'Noise Reduction: {reduction:.1f}%', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        filename = f'denoising_{column.lower().replace("_", "-")}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Denoising plot saved: {filepath}")
    
    def get_denoising_summary(self) -> Dict[str, Any]:
        """
        Get summary of denoising effects.
        
        Returns:
            Dictionary with denoising summary statistics
        """
        summary = {}
        
        # Group by column
        columns = set()
        for key in self.denoising_history.keys():
            col_name = key.split('_step')[0]
            columns.add(col_name)
        
        for col in columns:
            col_summary = {}
            
            # Get original data (step 1)
            step1_key = f'{col}_step1'
            if step1_key in self.denoising_history:
                original = self.denoising_history[step1_key]['original']
                
                # Get final data (last step)
                final_step = None
                for i in range(6, 0, -1):
                    key = f'{col}_step{i}'
                    if key in self.denoising_history:
                        final_step = self.denoising_history[key]['processed']
                        break
                
                if final_step is not None:
                    col_summary = {
                        'original_std': float(original.std()),
                        'final_std': float(final_step.std()),
                        'noise_reduction_pct': float((original.std() - final_step.std()) / original.std() * 100),
                        'original_mean': float(original.mean()),
                        'final_mean': float(final_step.mean()),
                        'mean_shift': float(final_step.mean() - original.mean())
                    }
            
            summary[col] = col_summary
        
        return summary


def main():
    """
    Main function for testing denoising pipeline.
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
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # Generate synthetic price data with noise
    trend = np.linspace(100, 200, 1000)
    noise = np.random.normal(0, 5, 1000)
    outliers = np.random.choice([0, 20, -20], 1000, p=[0.95, 0.025, 0.025])
    
    synthetic_data = pd.DataFrame({
        'Date': dates,
        'ETH_Close': trend + noise + outliers,
        'BTC_Close': trend * 0.8 + noise * 0.5 + outliers * 0.7
    })
    
    # Add some NaN values
    synthetic_data.loc[50:55, 'ETH_Close'] = np.nan
    synthetic_data.loc[200:202, 'BTC_Close'] = np.nan
    
    # Initialize denoiser
    denoiser = DataDenoiser(config)
    
    # Apply denoising
    denoised_data = denoiser.denoise_data(synthetic_data, ['ETH_Close', 'BTC_Close'])
    
    # Generate plots
    denoiser.generate_denoising_plots()
    
    # Print summary
    summary = denoiser.get_denoising_summary()
    print("Denoising Summary:")
    for col, stats in summary.items():
        print(f"  {col}:")
        for key, value in stats.items():
            print(f"    {key}: {value:.4f}")


if __name__ == "__main__":
    main()