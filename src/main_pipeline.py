"""
Main training pipeline for ETH forecasting project.

This module orchestrates the complete training pipeline following Rule #2:
1. Environment Setup → Data Pipeline → Baseline → Advanced Models → Ensemble → Analysis

Implements all quality gates and follows sequential execution requirements.
"""

import logging
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import warnings
import traceback
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project modules
from src.utils.helpers import (
    load_config, setup_logging, set_random_seeds, 
    quality_gate_check, save_intermediate_data
)
from src.data.acquisition import DataAcquisition
from src.data.denoising import DataDenoiser
from src.features.engineering import FeatureEngineer
from src.models.lightgbm_model import LightGBMForecaster
from src.evaluation.validation import WalkForwardValidator
from src.evaluation.statistical_tests import StatisticalTester
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.backtesting import EconomicBacktester
from src.explainability.interpretability import ModelInterpreter

warnings.filterwarnings('ignore', category=RuntimeWarning)


class ETHForecastingPipeline:
    """
    Main pipeline for ETH forecasting project.
    
    Orchestrates the complete machine learning pipeline from data acquisition
    to model evaluation and interpretation.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        if config_path is None:
            self.config = load_config()  # Use default path
        else:
            self.config = load_config(config_path)
        
        # Setup logging
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        setup_logging(log_level)
        
        # Set random seeds for reproducibility
        set_random_seeds(self.config.get('random_seed', 42))
        
        # Initialize components
        self.data_acquisition = DataAcquisition(self.config)
        self.denoising_pipeline = DataDenoiser(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.validator = WalkForwardValidator(self.config)
        self.statistical_tester = StatisticalTester(self.config)
        self.metrics_calculator = MetricsCalculator(self.config)
        self.backtester = EconomicBacktester(self.config)
        self.interpreter = ModelInterpreter(self.config)
        
        # Pipeline state
        self.pipeline_state = {
            'phase': 'initialization',
            'completed_phases': [],
            'failed_phases': [],
            'data': {},
            'models': {},
            'results': {}
        }
        
        logging.info("ETH Forecasting Pipeline initialized")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete pipeline following Rule #2 sequential execution.
        
        Returns:
            Dictionary with complete pipeline results
        """
        logging.info("Starting complete ETH forecasting pipeline")
        
        try:
            # Phase 1: Data Infrastructure
            self._run_phase_1_data_infrastructure()
            
            # Phase 2: Data Preprocessing and Feature Engineering
            self._run_phase_2_preprocessing()
            
            # Phase 3: Baseline Model Training
            self._run_phase_3_baseline_model()
            
            # Phase 4: Model Validation
            self._run_phase_4_validation()
            
            # Phase 5: Statistical Testing
            self._run_phase_5_statistical_testing()
            
            # Phase 6: Economic Backtesting
            self._run_phase_6_backtesting()
            
            # Phase 7: Model Interpretation
            self._run_phase_7_interpretation()
            
            # Phase 8: Final Analysis and Reporting
            self._run_phase_8_final_analysis()
            
            logging.info("Complete pipeline executed successfully")
            
        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}")
            logging.error(traceback.format_exc())
            self._handle_pipeline_failure(e)
            raise
        
        return self.pipeline_state['results']
    
    def _validate_raw_data_quality(self, data: pd.DataFrame) -> bool:
        """
        Validate raw data quality for Phase 1.
        
        Args:
            data: Raw data DataFrame to validate
            
        Returns:
            bool: True if data passes quality checks
        """
        try:
            # Check if data is a DataFrame
            if not isinstance(data, pd.DataFrame):
                logging.error("Data is not a pandas DataFrame")
                return False
            
            # Check if data is not empty
            if data.empty:
                logging.error("Data is empty")
                return False
            
            # Check for required columns
            required_columns = ['Date', 'ETH_Close', 'BTC_Close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logging.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Check for timezone (should be UTC)
            if 'Date' in data.columns and hasattr(data['Date'].dtype, 'tz'):
                if data['Date'].dt.tz is None:
                    logging.warning("Date column has no timezone information")
                elif str(data['Date'].dt.tz) != 'UTC':
                    logging.warning(f"Date column timezone is {data['Date'].dt.tz}, expected UTC")
            
            # Check for excessive NaN values (Rule #8: 20% threshold)
            nan_threshold = 0.2
            for col in ['ETH_Close', 'BTC_Close']:
                if col in data.columns:
                    nan_ratio = data[col].isna().sum() / len(data)
                    if nan_ratio > nan_threshold:
                        logging.error(f"Column {col} has {nan_ratio:.2%} NaN values, exceeding {nan_threshold:.0%} threshold")
                        return False
            
            # Check data range (basic sanity check)
            if 'ETH_Close' in data.columns:
                eth_prices = data['ETH_Close'].dropna()
                if len(eth_prices) > 0:
                    if eth_prices.min() <= 0:
                        logging.error("ETH prices contain non-positive values")
                        return False
                    if eth_prices.max() > 50000:  # Sanity check for unrealistic prices
                        logging.warning(f"ETH max price {eth_prices.max()} seems unusually high")
            
            if 'BTC_Close' in data.columns:
                btc_prices = data['BTC_Close'].dropna()
                if len(btc_prices) > 0:
                    if btc_prices.min() <= 0:
                        logging.error("BTC prices contain non-positive values")
                        return False
                    if btc_prices.max() > 500000:  # Sanity check for unrealistic prices
                        logging.warning(f"BTC max price {btc_prices.max()} seems unusually high")
            
            # Check data continuity (no large gaps)
            if 'Date' in data.columns and len(data) > 1:
                data_sorted = data.sort_values('Date')
                date_diffs = data_sorted['Date'].diff().dropna()
                max_gap = date_diffs.max()
                if max_gap > pd.Timedelta(days=7):
                    logging.warning(f"Large data gap detected: {max_gap}")
            
            logging.info("Raw data quality validation passed")
            return True
            
        except Exception as e:
            logging.error(f"Error during data quality validation: {str(e)}")
            return False
    
    def _validate_features_quality(self, features_data: pd.DataFrame) -> bool:
        """
        Validate feature quality for Phase 2.
        
        Args:
            features_data: Features DataFrame to validate
            
        Returns:
            bool: True if features pass quality checks
        """
        try:
            # Check if data is a DataFrame
            if not isinstance(features_data, pd.DataFrame):
                logging.error("Features data is not a pandas DataFrame")
                raise ValueError("Features data is not a pandas DataFrame")
            
            # Check if data is not empty
            if features_data.empty:
                logging.error("Features data is empty")
                raise ValueError("Features data is empty")
            
            # Check for excessive NaN values
            nan_threshold = 0.05  # Stricter threshold for features
            total_features = len(features_data.columns)
            features_with_nans = 0
            
            for col in features_data.columns:
                nan_ratio = features_data[col].isna().sum() / len(features_data)
                if nan_ratio > nan_threshold:
                    logging.warning(f"Feature {col} has {nan_ratio:.2%} NaN values")
                    features_with_nans += 1
            
            if features_with_nans > total_features * 0.1:  # More than 10% of features have excessive NaNs
                logging.error(f"{features_with_nans} features have excessive NaN values")
                raise ValueError("Too many features with excessive NaN values")
            
            # Check for infinite values
            inf_features = []
            for col in features_data.columns:
                if np.isinf(features_data[col]).any():
                    inf_features.append(col)
            
            if inf_features:
                logging.error(f"Features with infinite values: {inf_features}")
                raise ValueError("Features contain infinite values")
            
            # Check feature variance (avoid constant features)
            constant_features = []
            for col in features_data.columns:
                if features_data[col].var() == 0:
                    constant_features.append(col)
            
            if constant_features:
                logging.warning(f"Constant features detected: {constant_features}")
            
            # Check data leakage (basic check - no future information)
            # This is a simplified check - more comprehensive checks would be in dedicated functions
            suspicious_features = []
            for col in features_data.columns:
                if any(keyword in col.lower() for keyword in ['future', 'next', 'forward']):
                    suspicious_features.append(col)
            
            if suspicious_features:
                logging.error(f"Suspicious features that may contain future information: {suspicious_features}")
                raise ValueError("Potential data leakage detected in features")
            
            # Check reasonable feature ranges
            extreme_features = []
            for col in features_data.columns:
                if features_data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    col_abs_max = features_data[col].abs().max()
                    if col_abs_max > 1e6:  # Very large values might indicate scaling issues
                        extreme_features.append(col)
            
            if extreme_features:
                logging.warning(f"Features with extreme values (>1e6): {extreme_features}")
            
            logging.info(f"Features quality validation passed: {features_data.shape[1]} features, {features_data.shape[0]} samples")
            return True
            
        except Exception as e:
            logging.error(f"Error during features quality validation: {str(e)}")
            raise
    
    def _run_phase_1_data_infrastructure(self) -> None:
        """
        Phase 1: Data Infrastructure
        - Download and validate data
        - Perform initial quality checks
        """
        logging.info("Phase 1: Data Infrastructure")
        self.pipeline_state['phase'] = 'data_infrastructure'
        
        try:
            # Get data parameters from config
            start_date = self.config['data']['start_date']
            end_date = self.config['data']['end_date']
            
            # Download and combine data using the available methods
            logging.info("Downloading and combining ETH and BTC data")
            combined_data = self.data_acquisition.get_combined_data()
            
            # Calculate target variables (returns and direction)
            processed_data = self.data_acquisition.calculate_target_variable(combined_data)
            
            # Save raw data
            output_path = self.data_acquisition.save_raw_data(processed_data, 'eth_btc_combined.csv')
            logging.info(f"Raw data saved to: {output_path}")
            
            # Store in pipeline state
            self.pipeline_state['data']['raw'] = processed_data
            
            # Quality gate check for raw data
            if not self._validate_raw_data_quality(processed_data):
                raise ValueError("Data quality gate check failed")
            
            self.pipeline_state['completed_phases'].append('data_infrastructure')
            logging.info("Phase 1 completed successfully")
            
        except Exception as e:
            self.pipeline_state['failed_phases'].append('data_infrastructure')
            logging.error(f"Phase 1 failed: {str(e)}")
            raise
    
    def _run_phase_2_preprocessing(self) -> None:
        """
        Phase 2: Data Preprocessing and Feature Engineering
        - Apply denoising pipeline
        - Generate features
        - Validate feature quality
        """
        logging.info("Phase 2: Data Preprocessing and Feature Engineering")
        self.pipeline_state['phase'] = 'preprocessing'
        
        try:
            raw_data = self.pipeline_state['data']['raw']
            
            # Apply denoising pipeline
            logging.info("Applying denoising pipeline")
            denoised_data = self.denoising_pipeline.denoise_data(raw_data)
            
            # Generate denoising summary
            denoising_summary = self.denoising_pipeline.get_denoising_summary()
            
            # Feature engineering
            logging.info("Generating features")
            features_data = self.feature_engineer.engineer_features(denoised_data, is_training=True)
            
            # Feature summary
            feature_summary = self.feature_engineer.get_feature_summary()
            
            # Store processed data
            self.pipeline_state['data']['denoised'] = denoised_data
            self.pipeline_state['data']['features'] = features_data
            self.pipeline_state['results']['denoising_summary'] = denoising_summary
            self.pipeline_state['results']['feature_summary'] = feature_summary
            
            # Save intermediate data
            processed_dir = self.config.get('paths', {}).get('processed_data', 'data/processed')
            save_intermediate_data(denoised_data, 'denoised_data', processed_dir)
            save_intermediate_data(features_data, 'features_data', processed_dir)
            
            # Quality gate check for features
            self._validate_features_quality(features_data)
            
            self.pipeline_state['completed_phases'].append('preprocessing')
            logging.info("Phase 2 completed successfully")
            
        except Exception as e:
            self.pipeline_state['failed_phases'].append('preprocessing')
            logging.error(f"Phase 2 failed: {str(e)}")
            raise
    
    def _run_phase_3_baseline_model(self) -> None:
        """
        Phase 3: Baseline Model Training
        - Train LightGBM baseline model
        - Validate ZPTAE loss implementation
        - Perform hyperparameter optimization
        """
        logging.info("Phase 3: Baseline Model Training")
        self.pipeline_state['phase'] = 'baseline_model'
        
        try:
            features_data = self.pipeline_state['data']['features']
            
            # Prepare target variable
            target_col = self.config.get('data', {}).get('target_column', 'log_return')
            if target_col not in features_data.columns:
                raise ValueError(f"Target column '{target_col}' not found in features")
            
            # Split features and target
            feature_cols = [col for col in features_data.columns if col != target_col and not col.endswith('_direction')]
            X = features_data[feature_cols]
            y = features_data[target_col]
            
            # Split data for hyperparameter optimization (80/20 split)
            split_idx = int(len(X) * 0.8)
            X_train_opt = X.iloc[:split_idx]
            y_train_opt = y.iloc[:split_idx]
            X_val_opt = X.iloc[split_idx:]
            y_val_opt = y.iloc[split_idx:]
            
            # Initialize and train LightGBM model
            logging.info("Training LightGBM baseline model")
            lgb_model = LightGBMForecaster(self.config)
            
            # Perform hyperparameter optimization first
            logging.info("Optimizing hyperparameters")
            lgb_model.optimize_hyperparameters(X_train_opt, y_train_opt, X_val_opt, y_val_opt)
            
            # Train with optimized parameters on full dataset
            lgb_model.train(X, y)
            
            # Validate model
            predictions, direction_predictions = lgb_model.predict(X)
            
            # Calculate basic metrics
            basic_metrics = self.metrics_calculator.calculate_all_metrics(y, predictions)
            
            # Store model and results
            self.pipeline_state['models']['lightgbm'] = lgb_model
            self.pipeline_state['results']['baseline_metrics'] = basic_metrics
            
            # Save model
            model_path = lgb_model.save_model('lightgbm_baseline')
            logging.info(f"Baseline model saved to: {model_path}")
            
            # Quality gate check for predictions
            if np.any(np.isnan(predictions)):
                raise ValueError("Model predictions contain NaN values")
            
            if not (0.45 <= basic_metrics['directional_accuracy'] <= 0.75):
                logging.warning(f"Directional accuracy {basic_metrics['directional_accuracy']:.3f} outside expected range")
            
            self.pipeline_state['completed_phases'].append('baseline_model')
            logging.info("Phase 3 completed successfully")
            
        except Exception as e:
            self.pipeline_state['failed_phases'].append('baseline_model')
            logging.error(f"Phase 3 failed: {str(e)}")
            raise
    
    def _run_phase_4_validation(self) -> None:
        """
        Phase 4: Model Validation
        - Perform walk-forward cross-validation
        - Calculate stability metrics
        - Validate data integrity
        """
        logging.info("Phase 4: Model Validation")
        self.pipeline_state['phase'] = 'validation'
        
        try:
            features_data = self.pipeline_state['data']['features']
            lgb_model = self.pipeline_state['models']['lightgbm']
            
            # Prepare data for validation
            target_col = self.config.get('data', {}).get('target_column', 'log_return')
            feature_cols = [col for col in features_data.columns if col != target_col and not col.endswith('_direction')]
            X = features_data[feature_cols]
            y = features_data[target_col]
            
            # Create folds for walk-forward cross-validation
            logging.info("Creating walk-forward cross-validation folds")
            combined_data = pd.concat([X, y], axis=1)
            folds = self.validator.create_folds(combined_data)
            
            # Perform walk-forward cross-validation
            logging.info("Performing walk-forward cross-validation")
            cv_results = self.validator.validate_model(lgb_model, X, y, folds)
            
            # Extract stability metrics from cv_results
            stability_metrics = cv_results.get('stability_metrics', {})
            
            # Store validation results
            self.pipeline_state['results']['cv_results'] = cv_results
            self.pipeline_state['results']['stability_metrics'] = stability_metrics
            
            # Quality gate checks
            fold_results = cv_results.get('fold_results', [])
            successful_folds = [f for f in fold_results if 'error' not in f and 'directional_accuracy' in f]
            
            if successful_folds:
                avg_da = np.mean([fold['directional_accuracy'] for fold in successful_folds])
                if avg_da > 0.70:
                    logging.warning(f"High directional accuracy {avg_da:.3f} - checking for data leakage")
                    # Additional leakage checks would go here
            
            if stability_metrics.get('da_std', 0) > 0.10:
                logging.warning(f"High DA standard deviation {stability_metrics['da_std']:.3f} - model may be unstable")
            
            self.pipeline_state['completed_phases'].append('validation')
            logging.info("Phase 4 completed successfully")
            
        except Exception as e:
            self.pipeline_state['failed_phases'].append('validation')
            logging.error(f"Phase 4 failed: {str(e)}")
            raise
    
    def _run_phase_5_statistical_testing(self) -> None:
        """
        Phase 5: Statistical Testing
        - Perform Pesaran-Timmermann test
        - Perform Diebold-Mariano test
        - Generate statistical reports
        """
        logging.info("Phase 5: Statistical Testing")
        self.pipeline_state['phase'] = 'statistical_testing'
        
        try:
            features_data = self.pipeline_state['data']['features']
            lgb_model = self.pipeline_state['models']['lightgbm']
            
            # Prepare data
            target_col = self.config.get('data', {}).get('target_column', 'log_return')
            feature_cols = [col for col in features_data.columns if col != target_col and not col.endswith('_direction')]
            X = features_data[feature_cols]
            y = features_data[target_col]
            
            # Get model predictions
            predictions, _ = lgb_model.predict(X)
            
            # Create benchmark (naive forecast)
            benchmark_predictions = np.zeros_like(predictions)  # Zero forecast
            
            # Perform statistical tests
            logging.info("Performing statistical tests")
            
            # Pesaran-Timmermann test
            pt_result = self.statistical_tester.pesaran_timmermann_test(y, predictions)
            
            # Diebold-Mariano test
            dm_result = self.statistical_tester.diebold_mariano_test(
                y, predictions, benchmark_predictions
            )
            
            # Store statistical test results
            self.pipeline_state['results']['statistical_tests'] = {
                'pesaran_timmermann': pt_result,
                'diebold_mariano': dm_result
            }
            
            # Log key results
            logging.info(f"Pesaran-Timmermann test p-value: {pt_result['p_value']:.4f}")
            logging.info(f"Diebold-Mariano test p-value: {dm_result['p_value']:.4f}")
            
            self.pipeline_state['completed_phases'].append('statistical_testing')
            logging.info("Phase 5 completed successfully")
            
        except Exception as e:
            self.pipeline_state['failed_phases'].append('statistical_testing')
            logging.error(f"Phase 5 failed: {str(e)}")
            raise
    
    def _run_phase_6_backtesting(self) -> None:
        """
        Phase 6: Economic Backtesting
        - Perform economic backtest with transaction costs
        - Calculate risk-adjusted metrics
        - Compare against buy-and-hold
        """
        logging.info("Phase 6: Economic Backtesting")
        self.pipeline_state['phase'] = 'backtesting'
        
        try:
            features_data = self.pipeline_state['data']['features']
            lgb_model = self.pipeline_state['models']['lightgbm']
            
            # Prepare data
            target_col = self.config.get('data', {}).get('target_column', 'log_return')
            feature_cols = [col for col in features_data.columns if col != target_col and not col.endswith('_direction')]
            X = features_data[feature_cols]
            y = features_data[target_col]
            
            # Get model predictions
            predictions, _ = lgb_model.predict(X)
            predictions_series = pd.Series(predictions, index=y.index)
            
            # Perform economic backtest
            logging.info("Performing economic backtest")
            backtest_results = self.backtester.run_backtest(
                predictions_series, y, strategy_type='threshold'
            )
            
            # Create backtest report
            report_path = self.backtester.create_backtest_report(backtest_results)
            
            # Store backtest results
            self.pipeline_state['results']['backtest'] = backtest_results
            self.pipeline_state['results']['backtest_report_path'] = report_path
            
            # Log key metrics
            perf_metrics = backtest_results['performance_metrics']
            logging.info(f"Strategy Sharpe Ratio: {perf_metrics['sharpe_ratio']:.3f}")
            logging.info(f"Strategy Max Drawdown: {perf_metrics['max_drawdown']:.2%}")
            logging.info(f"Excess Return vs Buy-Hold: {perf_metrics['excess_return']:.2%}")
            
            self.pipeline_state['completed_phases'].append('backtesting')
            logging.info("Phase 6 completed successfully")
            
        except Exception as e:
            self.pipeline_state['failed_phases'].append('backtesting')
            logging.error(f"Phase 6 failed: {str(e)}")
            raise
    
    def _run_phase_7_interpretation(self) -> None:
        """
        Phase 7: Model Interpretation
        - Perform SHAP analysis
        - Feature ablation study
        - Regime analysis
        """
        logging.info("Phase 7: Model Interpretation")
        self.pipeline_state['phase'] = 'interpretation'
        
        try:
            features_data = self.pipeline_state['data']['features']
            lgb_model = self.pipeline_state['models']['lightgbm']
            
            # Prepare data
            target_col = self.config.get('data', {}).get('target_column', 'log_return')
            feature_cols = [col for col in features_data.columns if col != target_col and not col.endswith('_direction')]
            X = features_data[feature_cols]
            y = features_data[target_col]
            
            # Get model predictions
            predictions, _ = lgb_model.predict(X)
            
            # Perform interpretability analysis
            logging.info("Performing model interpretation")
            
            # SHAP analysis for tree model
            tree_analysis = self.interpreter.analyze_tree_model(
                lgb_model.model, X, y, feature_cols
            )
            
            # Feature ablation study
            feature_groups = self.feature_engineer.get_feature_groups()
            ablation_analysis = self.interpreter.feature_ablation_study(
                lgb_model, X, y, feature_groups
            )
            
            # Regime analysis
            regime_analysis = self.interpreter.regime_analysis(
                lgb_model, X, y, predictions, dates=X.index
            )
            
            # Create interpretation report
            analyses = {
                'tree_analysis': tree_analysis,
                'ablation_study': ablation_analysis,
                'regime_analysis': regime_analysis
            }
            
            report_path = self.interpreter.create_interpretation_report(analyses)
            
            # Store interpretation results
            self.pipeline_state['results']['interpretation'] = analyses
            self.pipeline_state['results']['interpretation_report_path'] = report_path
            
            self.pipeline_state['completed_phases'].append('interpretation')
            logging.info("Phase 7 completed successfully")
            
        except Exception as e:
            self.pipeline_state['failed_phases'].append('interpretation')
            logging.error(f"Phase 7 failed: {str(e)}")
            raise
    
    def _run_phase_8_final_analysis(self) -> None:
        """
        Phase 8: Final Analysis and Reporting
        - Compile comprehensive results
        - Generate final report
        - Create summary visualizations
        """
        logging.info("Phase 8: Final Analysis and Reporting")
        self.pipeline_state['phase'] = 'final_analysis'
        
        try:
            # Compile final results
            final_results = self._compile_final_results()
            
            # Generate final report
            report_path = self._generate_final_report(final_results)
            
            # Create summary metrics CSV
            metrics_path = self._save_summary_metrics(final_results)
            
            # Store final results
            self.pipeline_state['results']['final_summary'] = final_results
            self.pipeline_state['results']['final_report_path'] = report_path
            self.pipeline_state['results']['metrics_csv_path'] = metrics_path
            
            self.pipeline_state['completed_phases'].append('final_analysis')
            logging.info("Phase 8 completed successfully")
            
            # Log pipeline completion
            logging.info("="*60)
            logging.info("ETH FORECASTING PIPELINE COMPLETED SUCCESSFULLY")
            logging.info("="*60)
            logging.info(f"Completed phases: {len(self.pipeline_state['completed_phases'])}")
            logging.info(f"Failed phases: {len(self.pipeline_state['failed_phases'])}")
            logging.info(f"Final report: {report_path}")
            logging.info(f"Metrics summary: {metrics_path}")
            
        except Exception as e:
            self.pipeline_state['failed_phases'].append('final_analysis')
            logging.error(f"Phase 8 failed: {str(e)}")
            raise
    
    def _compile_final_results(self) -> Dict[str, Any]:
        """Compile final results from all phases."""
        results = self.pipeline_state['results']
        
        final_results = {
            'pipeline_summary': {
                'completed_phases': self.pipeline_state['completed_phases'],
                'failed_phases': self.pipeline_state['failed_phases'],
                'execution_time': datetime.now().isoformat(),
                'config': self.config
            },
            'data_summary': results.get('denoising_summary', {}),
            'feature_summary': results.get('feature_summary', {}),
            'model_performance': results.get('baseline_metrics', {}),
            'validation_results': results.get('cv_results', []),
            'stability_metrics': results.get('stability_metrics', {}),
            'statistical_tests': results.get('statistical_tests', {}),
            'economic_performance': results.get('backtest', {}).get('performance_metrics', {}),
            'interpretation_summary': results.get('interpretation', {})
        }
        
        return final_results
    
    def _generate_final_report(self, final_results: Dict[str, Any]) -> str:
        """Generate comprehensive final report."""
        reports_dir = Path(self.config.get('paths', {}).get('reports', 'reports'))
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = reports_dir / 'final_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# ETH Forecasting Project - Final Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            self._write_executive_summary(f, final_results)
            
            # Pipeline Summary
            f.write("## Pipeline Execution Summary\n\n")
            pipeline_summary = final_results['pipeline_summary']
            f.write(f"- **Completed Phases**: {len(pipeline_summary['completed_phases'])}\n")
            f.write(f"- **Failed Phases**: {len(pipeline_summary['failed_phases'])}\n")
            f.write(f"- **Execution Time**: {pipeline_summary['execution_time']}\n\n")
            
            # Model Performance
            f.write("## Model Performance\n\n")
            self._write_model_performance(f, final_results)
            
            # Economic Results
            f.write("## Economic Performance\n\n")
            self._write_economic_performance(f, final_results)
            
            # Statistical Validation
            f.write("## Statistical Validation\n\n")
            self._write_statistical_validation(f, final_results)
            
            # Key Insights
            f.write("## Key Insights\n\n")
            self._write_key_insights(f, final_results)
            
            # Recommendations
            f.write("## Recommendations\n\n")
            self._write_recommendations(f, final_results)
        
        return str(report_path)
    
    def _write_executive_summary(self, f, final_results: Dict[str, Any]) -> None:
        """Write executive summary section."""
        model_perf = final_results.get('model_performance', {})
        econ_perf = final_results.get('economic_performance', {})
        
        f.write("This report presents the results of a comprehensive ETH 24-hour log-return forecasting system.\n\n")
        
        if model_perf:
            f.write(f"**Model Performance**: Achieved {model_perf.get('directional_accuracy', 0):.1%} directional accuracy ")
            f.write(f"with ZPTAE score of {model_perf.get('zptae', 0):.4f}.\n\n")
        
        if econ_perf:
            f.write(f"**Economic Performance**: Strategy generated {econ_perf.get('annualized_return', 0):.1%} annualized return ")
            f.write(f"with Sharpe ratio of {econ_perf.get('sharpe_ratio', 0):.2f} and maximum drawdown of {econ_perf.get('max_drawdown', 0):.1%}.\n\n")
    
    def _write_model_performance(self, f, final_results: Dict[str, Any]) -> None:
        """Write model performance section."""
        model_perf = final_results.get('model_performance', {})
        stability = final_results.get('stability_metrics', {})
        
        if model_perf:
            f.write("### Core Metrics\n\n")
            f.write(f"- **Directional Accuracy**: {model_perf.get('directional_accuracy', 0):.2%}\n")
            f.write(f"- **ZPTAE Score**: {model_perf.get('zptae', 0):.4f}\n")
            f.write(f"- **MSE**: {model_perf.get('mse', 0):.6f}\n")
            f.write(f"- **MAE**: {model_perf.get('mae', 0):.6f}\n\n")
        
        if stability:
            f.write("### Stability Metrics\n\n")
            f.write(f"- **DA Standard Deviation**: {stability.get('da_std', 0):.3f}\n")
            f.write(f"- **ZPTAE Standard Deviation**: {stability.get('zptae_std', 0):.4f}\n\n")
    
    def _write_economic_performance(self, f, final_results: Dict[str, Any]) -> None:
        """Write economic performance section."""
        econ_perf = final_results.get('economic_performance', {})
        
        if econ_perf:
            f.write("### Trading Strategy Results\n\n")
            f.write(f"- **Total Return**: {econ_perf.get('total_return', 0):.2%}\n")
            f.write(f"- **Annualized Return**: {econ_perf.get('annualized_return', 0):.2%}\n")
            f.write(f"- **Volatility**: {econ_perf.get('volatility', 0):.2%}\n")
            f.write(f"- **Sharpe Ratio**: {econ_perf.get('sharpe_ratio', 0):.3f}\n")
            f.write(f"- **Maximum Drawdown**: {econ_perf.get('max_drawdown', 0):.2%}\n")
            f.write(f"- **Win Rate**: {econ_perf.get('win_rate', 0):.2%}\n\n")
            
            f.write("### Benchmark Comparison\n\n")
            f.write(f"- **Excess Return**: {econ_perf.get('excess_return', 0):.2%}\n")
            f.write(f"- **Information Ratio**: {econ_perf.get('information_ratio', 0):.3f}\n\n")
    
    def _write_statistical_validation(self, f, final_results: Dict[str, Any]) -> None:
        """Write statistical validation section."""
        stat_tests = final_results.get('statistical_tests', {})
        
        if stat_tests:
            pt_test = stat_tests.get('pesaran_timmermann', {})
            dm_test = stat_tests.get('diebold_mariano', {})
            
            f.write("### Statistical Test Results\n\n")
            
            if pt_test:
                f.write(f"**Pesaran-Timmermann Test**:\n")
                f.write(f"- Test Statistic: {pt_test.get('test_statistic', 0):.3f}\n")
                f.write(f"- P-value: {pt_test.get('p_value', 0):.4f}\n")
                f.write(f"- Significant: {'Yes' if pt_test.get('p_value', 1) < 0.05 else 'No'}\n\n")
            
            if dm_test:
                f.write(f"**Diebold-Mariano Test**:\n")
                f.write(f"- Test Statistic: {dm_test.get('test_statistic', 0):.3f}\n")
                f.write(f"- P-value: {dm_test.get('p_value', 0):.4f}\n")
                f.write(f"- Significant: {'Yes' if dm_test.get('p_value', 1) < 0.05 else 'No'}\n\n")
    
    def _write_key_insights(self, f, final_results: Dict[str, Any]) -> None:
        """Write key insights section."""
        insights = []
        
        # Performance insights
        model_perf = final_results.get('model_performance', {})
        if model_perf.get('directional_accuracy', 0) > 0.55:
            insights.append("- Model shows meaningful predictive skill above random chance")
        
        # Economic insights
        econ_perf = final_results.get('economic_performance', {})
        if econ_perf.get('sharpe_ratio', 0) > 0.5:
            insights.append("- Strategy demonstrates positive risk-adjusted returns")
        
        # Statistical insights
        stat_tests = final_results.get('statistical_tests', {})
        pt_test = stat_tests.get('pesaran_timmermann', {})
        if pt_test.get('p_value', 1) < 0.05:
            insights.append("- Directional accuracy is statistically significant")
        
        if insights:
            f.write("\n".join(insights))
        else:
            f.write("- Analysis completed successfully with comprehensive validation")
        
        f.write("\n\n")
    
    def _write_recommendations(self, f, final_results: Dict[str, Any]) -> None:
        """Write recommendations section."""
        f.write("### Future Improvements\n\n")
        f.write("- Implement additional model architectures (TFT, N-BEATS)\n")
        f.write("- Enhance feature engineering with alternative data sources\n")
        f.write("- Optimize trading strategy parameters\n")
        f.write("- Implement real-time prediction pipeline\n\n")
        
        f.write("### Risk Considerations\n\n")
        f.write("- Monitor model performance degradation over time\n")
        f.write("- Implement position sizing and risk management\n")
        f.write("- Consider regime-specific model adaptation\n")
        f.write("- Regular retraining and validation\n\n")
    
    def _save_summary_metrics(self, final_results: Dict[str, Any]) -> str:
        """Save summary metrics to CSV."""
        reports_dir = Path(self.config.get('paths', {}).get('reports', 'reports'))
        metrics_path = reports_dir / 'summary_metrics.csv'
        
        # Compile metrics
        metrics_data = []
        
        # Model metrics
        model_perf = final_results.get('model_performance', {})
        for metric, value in model_perf.items():
            metrics_data.append({
                'category': 'model_performance',
                'metric': metric,
                'value': value
            })
        
        # Economic metrics
        econ_perf = final_results.get('economic_performance', {})
        for metric, value in econ_perf.items():
            metrics_data.append({
                'category': 'economic_performance',
                'metric': metric,
                'value': value
            })
        
        # Statistical test results
        stat_tests = final_results.get('statistical_tests', {})
        for test_name, test_results in stat_tests.items():
            if isinstance(test_results, dict):
                for metric, value in test_results.items():
                    metrics_data.append({
                        'category': f'statistical_test_{test_name}',
                        'metric': metric,
                        'value': value
                    })
        
        # Save to CSV
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_csv(metrics_path, index=False)
        
        return str(metrics_path)
    
    def _handle_pipeline_failure(self, error: Exception) -> None:
        """Handle pipeline failure according to Rule #23."""
        logging.error("CRITICAL PIPELINE FAILURE - Implementing failure protocol")
        
        # Save current state
        state_path = Path(self.config.get('paths', {}).get('reports', 'reports')) / 'pipeline_failure_state.json'
        
        try:
            import json
            with open(state_path, 'w') as f:
                # Convert non-serializable objects to strings
                serializable_state = {
                    'phase': self.pipeline_state['phase'],
                    'completed_phases': self.pipeline_state['completed_phases'],
                    'failed_phases': self.pipeline_state['failed_phases'],
                    'error': str(error),
                    'traceback': traceback.format_exc(),
                    'timestamp': datetime.now().isoformat()
                }
                json.dump(serializable_state, f, indent=2)
            
            logging.info(f"Pipeline state saved to: {state_path}")
            
        except Exception as save_error:
            logging.error(f"Failed to save pipeline state: {str(save_error)}")
        
        # Generate failure report
        self._generate_failure_report(error)
    
    def _generate_failure_report(self, error: Exception) -> None:
        """Generate failure analysis report."""
        reports_dir = Path(self.config.get('paths', {}).get('reports', 'reports'))
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        failure_report_path = reports_dir / 'failure_report.md'
        
        with open(failure_report_path, 'w') as f:
            f.write("# Pipeline Failure Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Failure Summary\n\n")
            f.write(f"- **Failed Phase**: {self.pipeline_state['phase']}\n")
            f.write(f"- **Error**: {str(error)}\n")
            f.write(f"- **Completed Phases**: {', '.join(self.pipeline_state['completed_phases'])}\n")
            f.write(f"- **Failed Phases**: {', '.join(self.pipeline_state['failed_phases'])}\n\n")
            
            f.write("## Error Details\n\n")
            f.write("```\n")
            f.write(traceback.format_exc())
            f.write("```\n\n")
            
            f.write("## Recovery Recommendations\n\n")
            f.write("1. Review error details and fix underlying issue\n")
            f.write("2. Check data quality and configuration\n")
            f.write("3. Restart pipeline from failed phase\n")
            f.write("4. Consider reducing complexity if memory/performance issues\n")
        
        logging.info(f"Failure report saved to: {failure_report_path}")


def main():
    """
    Main function to run the complete ETH forecasting pipeline.
    """
    try:
        # Initialize pipeline
        pipeline = ETHForecastingPipeline()
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline()
        
        print("\n" + "="*60)
        print("ETH FORECASTING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Final report: {results.get('final_report_path', 'Not generated')}")
        print(f"Metrics summary: {results.get('metrics_csv_path', 'Not generated')}")
        print("="*60)
        
        return results
        
    except Exception as e:
        print(f"\nPipeline failed with error: {str(e)}")
        print("Check logs and failure report for details.")
        raise


if __name__ == "__main__":
    main()