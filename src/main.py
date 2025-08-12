"""
Main Pipeline for ETH Forecasting Project

This module implements the complete end-to-end pipeline for ETH price forecasting,
following all project rules and ensuring data integrity, reproducibility, and quality.

The pipeline includes:
1. Environment Setup
2. Data Pipeline (acquisition, denoising, feature engineering)
3. Baseline Model Training
4. Advanced Model Training
5. Ensemble Creation
6. Evaluation and Analysis

Following Rule #2 (Sequential Execution) and Rule #25 (Deliverable Completeness Check).
"""

import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import yaml
import joblib
from datetime import datetime, timedelta

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import project modules
from utils.helpers import (
    load_config, setup_logging, set_random_seeds, create_directories,
    validate_timezone_utc, check_data_leakage, quality_gate_check,
    save_intermediate_data, memory_usage_check
)
from data.acquisition import DataAcquisition
from data.denoising import DataDenoiser
from features.engineering import FeatureEngineer
from models.lightgbm_model import LightGBMForecaster
from models.tft_model import TFTModel
from models.nbeats_model import NBeatsWrapper
from models.tcn_model import TCNWrapper
from models.ensemble import StackingEnsemble
from evaluation.validation import WalkForwardValidator
from evaluation.metrics import MetricsCalculator
from evaluation.backtesting import EconomicBacktester
from evaluation.statistical_tests import StatisticalTester
from explainability.interpretability import ModelInterpreter

logger = logging.getLogger(__name__)


class ETHForecastingPipeline:
    """
    Complete ETH forecasting pipeline implementation.
    
    This class orchestrates the entire forecasting workflow,
    ensuring data integrity and following all project rules.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the forecasting pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = None
        self.data = None
        self.features = None
        self.models = {}
        self.ensemble = None
        self.results = {}
        
        # Pipeline state tracking
        self.pipeline_state = {
            'environment_setup': False,
            'data_pipeline': False,
            'baseline_model': False,
            'advanced_models': False,
            'ensemble_creation': False,
            'evaluation_analysis': False
        }
        
        logger.info("ETH Forecasting Pipeline initialized")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete forecasting pipeline.
        
        Returns:
            Dictionary containing all results and metrics
        """
        try:
            logger.info("=" * 80)
            logger.info("STARTING ETH FORECASTING PIPELINE")
            logger.info("=" * 80)
            
            start_time = time.time()
            
            # Phase 1: Environment Setup
            self._phase_1_environment_setup()
            
            # Phase 2: Data Pipeline
            self._phase_2_data_pipeline()
            
            # Phase 3: Baseline Model
            self._phase_3_baseline_model()
            
            # Phase 4: Advanced Models
            self._phase_4_advanced_models()
            
            # Phase 5: Ensemble Creation
            self._phase_5_ensemble_creation()
            
            # Phase 6: Evaluation and Analysis
            self._phase_6_evaluation_analysis()
            
            # Final validation
            self._final_validation()
            
            total_time = time.time() - start_time
            logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            self._handle_pipeline_failure(e)
            raise
    
    def _phase_1_environment_setup(self) -> None:
        """
        Phase 1: Environment Setup
        
        Following Rule #6 (Reproducibility Requirements) and Rule #5 (No Hardcoded Values).
        """
        logger.info("Phase 1: Environment Setup")
        logger.info("-" * 40)
        
        try:
            # Load configuration
            self.config = load_config(self.config_path)
            logger.info(f"Configuration loaded from {self.config_path}")
            
            # Setup logging
            log_config = self.config.get('logging', {})
            setup_logging(level=log_config.get('level', 'INFO'))
            
            # Set random seeds for reproducibility
            seed = self.config.get('random_seed', 42)
            set_random_seeds(seed)
            logger.info("Random seeds set for reproducibility")
            
            # Create necessary directories
            directories = self.config.get('paths', {})
            for dir_type, dir_path in directories.items():
                if dir_path and isinstance(dir_path, str):
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                    logger.info(f"Directory created/verified: {dir_type} -> {dir_path}")
            
            logger.info("Directory structure created")
            
            # Log system information
            logger.info(f"Python version: {sys.version}")
            logger.info(f"Working directory: {os.getcwd()}")
            logger.info(f"Memory usage check: {memory_usage_check()}")
            
            self.pipeline_state['environment_setup'] = True
            logger.info("✓ Phase 1 completed successfully")
            
        except Exception as e:
            logger.error(f"Phase 1 failed: {str(e)}")
            raise
    
    def _phase_2_data_pipeline(self) -> None:
        """
        Phase 2: Data Pipeline (Acquisition, Denoising, Feature Engineering)
        
        Following Rule #1 (Data Integrity) and Rule #8 (Denoising Pipeline Compliance).
        """
        logger.info("Phase 2: Data Pipeline")
        logger.info("-" * 40)
        
        try:
            # Step 2.1: Data Acquisition
            logger.info("Step 2.1: Data Acquisition")
            
            data_config = self.config.get('data', {})
            collector = DataAcquisition(config=self.config)
            
            # Collect ETH data
            self.data = collector.get_combined_data()
            
            logger.info(f"Collected {len(self.data)} data points")
            
            # Validate timezone (Rule #7)
            data_for_tz_check = self.data.copy()
            data_for_tz_check.reset_index(inplace=True)  # Make Date a column
            if not validate_timezone_utc(data_for_tz_check, 'Date'):
                raise ValueError("Data timezone validation failed!")
            logger.info("✓ Timezone validation passed")
            
            # Save raw data
            raw_data_path = self.config['paths']['raw_data']
            save_intermediate_data(self.data, raw_data_path, 'raw_eth_data')
            
            # Step 2.2: Data Denoising
            logger.info("Step 2.2: Data Denoising")
            
            denoiser = DataDenoiser(config=self.config)
            
            # Apply denoising pipeline (Rule #8)
            self.data = denoiser.denoise_data(self.data)
            
            # Generate denoising plots
            plots_dir = self.config['paths']['reports'] + '/figures'
            Path(plots_dir).mkdir(parents=True, exist_ok=True)
            denoiser.generate_plots(save_dir=plots_dir)
            
            logger.info("✓ Data denoising completed with plots generated")
            
            # Save denoised data
            processed_data_path = self.config['paths']['processed_data']
            save_intermediate_data(self.data, processed_data_path, 'denoised_eth_data')
            
            # Step 2.3: Feature Engineering
            logger.info("Step 2.3: Feature Engineering")
            
            engineer = FeatureEngineer(config=self.config)
            
            # Create features (Rule #9)
            self.features = engineer.engineer_features(self.data)
            
            # Check for data leakage (Rule #1)
            # Separate features and targets for leakage detection
            target_cols = ['ETH_log_return_24h', 'ETH_direction']
            feature_cols = [col for col in self.features.columns if col not in target_cols]
            
            # Create separate DataFrames with Date column
            features_for_check = self.features[feature_cols].copy()
            features_for_check.reset_index(inplace=True)  # Date becomes a column
            
            targets_for_check = self.features[target_cols].copy()
            targets_for_check.reset_index(inplace=True)  # Date becomes a column
            
            leakage_detected = not check_data_leakage(features_for_check, targets_for_check)
            if leakage_detected:
                raise ValueError("Data leakage detected! Pipeline halted.")
            
            logger.info("✓ No data leakage detected")
            
            # Save features
            save_intermediate_data(self.features, processed_data_path, 'engineered_features')
            
            logger.info(f"Created {len(self.features.columns)} features")
            
            self.pipeline_state['data_pipeline'] = True
            logger.info("✓ Phase 2 completed successfully")
            
        except Exception as e:
            logger.error(f"Phase 2 failed: {str(e)}")
            raise
    
    def _phase_3_baseline_model(self) -> None:
        """
        Phase 3: Baseline Model Training (LightGBM)
        
        Following Rule #10 (Multi-Model Mandate) and Rule #11 (Hyperparameter Optimization).
        """
        logger.info("Phase 3: Baseline Model Training")
        logger.info("-" * 40)
        
        try:
            # Prepare data for modeling
            target_cols = ['ETH_log_return_24h', 'ETH_direction']
            feature_cols = [col for col in self.features.columns if col not in target_cols]
            X = self.features[feature_cols]
            y = self.features['ETH_log_return_24h']  # Use continuous target for regression
            
            # Create validator
            validator = WalkForwardValidator(config=self.config)
            
            # Create train/validation split
            train_idx, val_idx = validator.create_holdout_split(len(X))
            
            # Extract features and targets from split data
            X_train = train_data[feature_cols]
            X_val = val_data[feature_cols]
            y_train = train_data['ETH_log_return_24h']
            y_val = val_data['ETH_log_return_24h']
            
            logger.info(f"Training set: {len(X_train)} samples")
            logger.info(f"Validation set: {len(X_val)} samples")
            
            # Initialize LightGBM model
            lgb_model = LightGBMForecaster(config=self.config)
            
            # Hyperparameter optimization (Rule #11)
            logger.info("Optimizing LightGBM hyperparameters...")
            
            lgb_config = self.config.get('lightgbm', {})
            n_trials = lgb_config.get('n_trials', 200)
            
            # Set the number of trials for optimization
            lgb_model.optuna_trials = n_trials
            
            # Get direction targets for multi-task learning
            y_direction_train = self.features['ETH_direction'].iloc[train_idx]
            y_direction_val = self.features['ETH_direction'].iloc[val_idx]
            
            best_params = lgb_model.optimize_hyperparameters(
                X_train, y_train, X_val, y_val, y_direction_train, y_direction_val
            )
            
            logger.info(f"Best parameters: {best_params}")
            
            # Train model
            logger.info("Training LightGBM model...")
            lgb_model.train(X_train, y_train, X_val, y_val, y_direction_train, y_direction_val)
            
            # Validate model
            log_return_pred, direction_pred = lgb_model.predict(X_val)
            predictions = log_return_pred  # For compatibility with quality gate check
            
            # Quality gate check (Rule #16)
            # Prepare data for quality gate check
            X_val_with_date = X_val.copy()
            X_val_with_date.reset_index(inplace=True)  # Date becomes a column
            
            y_val_df = pd.DataFrame({'ETH_log_return_24h': y_val})
            y_val_df.reset_index(inplace=True)  # Date becomes a column
            
            quality_passed = quality_gate_check(X_val_with_date, y_val_df, predictions)
            
            if not quality_passed:
                logger.warning("Quality gate issues detected")
            else:
                logger.info("✓ Quality gate check passed")
            
            # Save model
            model_name = 'lightgbm_baseline'
            model_path = lgb_model.save_model(model_name)
            
            # Store model
            self.models['lightgbm'] = lgb_model
            
            # Store baseline results
            self.results['baseline'] = {
                'model': 'lightgbm',
                'best_params': best_params,
                'validation_log_return_predictions': log_return_pred,
                'validation_direction_predictions': direction_pred,
                'model_path': model_path
            }
            
            self.pipeline_state['baseline_model'] = True
            logger.info("✓ Phase 3 completed successfully")
            
        except Exception as e:
            logger.error(f"Phase 3 failed: {str(e)}")
            raise
    
    def _phase_4_advanced_models(self) -> None:
        """
        Phase 4: Advanced Model Training (TFT, N-BEATS, TCN)
        
        Following Rule #10 (Multi-Model Mandate) and Rule #12 (Multi-Task Loss Implementation).
        """
        logger.info("Phase 4: Advanced Model Training")
        logger.info("-" * 40)
        
        try:
            # Prepare data
            target_cols = ['ETH_log_return_24h', 'ETH_direction']
            feature_cols = [col for col in self.features.columns if col not in target_cols]
            X = self.features[feature_cols]
            y = self.features['ETH_log_return_24h']  # Use continuous target for regression
            
            # Use same train/validation split as baseline
            validator = WalkForwardValidator(config=self.config)
            train_data, val_data = validator.create_holdout_split(self.features)
            
            # Extract features and targets from split data
            X_train = train_data[feature_cols]
            X_val = val_data[feature_cols]
            y_train = train_data['ETH_log_return_24h']
            y_val = val_data['ETH_log_return_24h']
            
            logger.info(f"Training set: {len(X_train)} samples")
            logger.info(f"Validation set: {len(X_val)} samples")
            
            # Model 1: Temporal Fusion Transformer
            logger.info("Training TFT model...")
            try:
                tft_model = TFTModel(config=self.config)
                
                # Hyperparameter optimization
                tft_config = self.config.get('tft', {})
                n_trials = tft_config.get('n_trials', 100)
                
                best_tft_params = tft_model.optimize_hyperparameters(
                    X_train, y_train, X_val, y_val, n_trials=n_trials
                )
                
                tft_model.update_params(best_tft_params)
                tft_model.fit(X_train, y_train, X_val, y_val)
                
                # Save model
                tft_path = os.path.join(self.config['paths']['models'], 'tft_model.pkl')
                tft_model.save_model(tft_path)
                
                self.models['tft'] = tft_model
                logger.info("✓ TFT model trained successfully")
                
            except Exception as e:
                logger.warning(f"TFT model training failed: {str(e)}")
                self.models['tft'] = None
            
            # Model 2: N-BEATS
            logger.info("Training N-BEATS model...")
            try:
                nbeats_model = NBeatsWrapper(config=self.config)
                
                # Hyperparameter optimization
                nbeats_config = self.config.get('nbeats', {})
                n_trials = nbeats_config.get('n_trials', 80)
                
                best_nbeats_params = nbeats_model.optimize_hyperparameters(
                    X_train, y_train, X_val, y_val, n_trials=n_trials
                )
                
                nbeats_model.update_params(best_nbeats_params)
                nbeats_model.fit(X_train, y_train, X_val, y_val)
                
                # Save model
                nbeats_path = os.path.join(self.config['paths']['models'], 'nbeats_model.pkl')
                nbeats_model.save_model(nbeats_path)
                
                self.models['nbeats'] = nbeats_model
                logger.info("✓ N-BEATS model trained successfully")
                
            except Exception as e:
                logger.warning(f"N-BEATS model training failed: {str(e)}")
                self.models['nbeats'] = None
            
            # Model 3: TCN
            logger.info("Training TCN model...")
            try:
                tcn_model = TCNWrapper(model_type='tcn', config=self.config)
                
                # Hyperparameter optimization
                tcn_config = self.config.get('tcn', {})
                n_trials = tcn_config.get('n_trials', 50)
                
                best_tcn_params = tcn_model.optimize_hyperparameters(
                    X_train, y_train, X_val, y_val, n_trials=n_trials
                )
                
                # Update config and retrain
                temp_config = self.config.copy()
                temp_config['tcn'].update(best_tcn_params)
                
                tcn_model = TCNWrapper(model_type='tcn', config=temp_config)
                tcn_model.fit(X_train, y_train, X_val, y_val)
                
                # Save model
                tcn_path = os.path.join(self.config['paths']['models'], 'tcn_model.pkl')
                tcn_model.save_model(tcn_path)
                
                self.models['tcn'] = tcn_model
                logger.info("✓ TCN model trained successfully")
                
            except Exception as e:
                logger.warning(f"TCN model training failed: {str(e)}")
                self.models['tcn'] = None
            
            # Model 4: CNN-LSTM
            logger.info("Training CNN-LSTM model...")
            try:
                cnn_lstm_model = TCNWrapper(model_type='cnn_lstm', config=self.config)
                cnn_lstm_model.fit(X_train, y_train, X_val, y_val)
                
                # Save model
                cnn_lstm_path = os.path.join(self.config['paths']['models'], 'cnn_lstm_model.pkl')
                cnn_lstm_model.save_model(cnn_lstm_path)
                
                self.models['cnn_lstm'] = cnn_lstm_model
                logger.info("✓ CNN-LSTM model trained successfully")
                
            except Exception as e:
                logger.warning(f"CNN-LSTM model training failed: {str(e)}")
                self.models['cnn_lstm'] = None
            
            # Filter out failed models
            successful_models = {k: v for k, v in self.models.items() if v is not None}
            logger.info(f"Successfully trained {len(successful_models)} models")
            
            if len(successful_models) == 0:
                raise ValueError("All advanced models failed to train")
            
            self.pipeline_state['advanced_models'] = True
            logger.info("✓ Phase 4 completed successfully")
            
        except Exception as e:
            logger.error(f"Phase 4 failed: {str(e)}")
            raise
    
    def _phase_5_ensemble_creation(self) -> None:
        """
        Phase 5: Ensemble Creation
        
        Following Rule #10 (Multi-Model Mandate) and ensemble best practices.
        """
        logger.info("Phase 5: Ensemble Creation")
        logger.info("-" * 40)
        
        try:
            # Filter successful models
            successful_models = {k: v for k, v in self.models.items() if v is not None}
            
            if len(successful_models) < 2:
                logger.warning("Not enough models for ensemble creation")
                self.ensemble = None
                self.pipeline_state['ensemble_creation'] = True
                return
            
            # Create ensemble
            logger.info(f"Creating ensemble with {len(successful_models)} models")
            
            self.ensemble = StackingEnsemble(
                base_models=successful_models,
                config=self.config
            )
            
            # Prepare data for ensemble training
            feature_cols = [col for col in self.features.columns if col not in ['timestamp', 'ETH_log_return_24h', 'ETH_direction']]
            
            # Use same train/validation split
            validator = WalkForwardValidator(config=self.config)
            train_data, val_data = validator.create_holdout_split(self.features)
            
            X_train = train_data[feature_cols]
            X_val = val_data[feature_cols]
            y_train = train_data['ETH_log_return_24h']
            y_val = val_data['ETH_log_return_24h']
            
            # Train ensemble
            self.ensemble.fit(X_train, y_train, X_val, y_val)
            
            # Save ensemble
            ensemble_path = os.path.join(self.config['paths']['models'], 'ensemble_model.pkl')
            self.ensemble.save_ensemble(ensemble_path)
            
            # Get model weights
            model_weights = self.ensemble.get_base_model_weights()
            logger.info(f"Ensemble model weights: {model_weights}")
            
            self.results['ensemble'] = {
                'model_weights': model_weights,
                'base_models': list(successful_models.keys())
            }
            
            self.pipeline_state['ensemble_creation'] = True
            logger.info("✓ Phase 5 completed successfully")
            
        except Exception as e:
            logger.error(f"Phase 5 failed: {str(e)}")
            raise
    
    def _phase_6_evaluation_analysis(self) -> None:
        """
        Phase 6: Evaluation and Analysis
        
        Following Rule #13 (Walk-Forward CV), Rule #14 (Statistical Testing),
        and Rule #21 (Model Interpretability).
        """
        logger.info("Phase 6: Evaluation and Analysis")
        logger.info("-" * 40)
        
        try:
            # Prepare data
            feature_cols = [col for col in self.features.columns if col not in ['timestamp', 'ETH_log_return_24h', 'ETH_direction']]
            X = self.features[feature_cols]
            y = self.features['ETH_log_return_24h']
            
            # Step 6.1: Walk-Forward Cross-Validation
            logger.info("Step 6.1: Walk-Forward Cross-Validation")
            
            validator = WalkForwardValidator(config=self.config)
            
            # Evaluate all models
            cv_results = {}
            successful_models = {k: v for k, v in self.models.items() if v is not None}
            
            for model_name, model in successful_models.items():
                logger.info(f"Evaluating {model_name} with walk-forward CV...")
                
                try:
                    model_results = validator.validate_model(model, X, y)
                    cv_results[model_name] = model_results
                    logger.info(f"✓ {model_name} evaluation completed")
                    
                except Exception as e:
                    logger.warning(f"CV evaluation failed for {model_name}: {str(e)}")
                    cv_results[model_name] = None
            
            # Evaluate ensemble if available
            if self.ensemble is not None:
                logger.info("Evaluating ensemble with walk-forward CV...")
                try:
                    ensemble_results = validator.validate_model(self.ensemble, X, y)
                    cv_results['ensemble'] = ensemble_results
                    logger.info("✓ Ensemble evaluation completed")
                except Exception as e:
                    logger.warning(f"Ensemble CV evaluation failed: {str(e)}")
                    cv_results['ensemble'] = None
            
            # Step 6.2: Statistical Testing
            logger.info("Step 6.2: Statistical Testing")
            
            tester = StatisticalTester(config=self.config)
            
            # Collect predictions for statistical tests
            test_predictions = {}
            
            # Use holdout set for final testing
            train_data, test_data = validator.create_holdout_split(self.features)
            X_test = test_data[feature_cols]
            y_test = test_data['ETH_log_return_24h']
            
            for model_name, model in successful_models.items():
                if model is not None:
                    try:
                        pred = model.predict(X_test)
                        test_predictions[model_name] = pred
                    except Exception as e:
                        logger.warning(f"Failed to get test predictions for {model_name}: {str(e)}")
            
            if self.ensemble is not None:
                try:
                    ensemble_pred = self.ensemble.predict(X_test)
                    test_predictions['ensemble'] = ensemble_pred
                except Exception as e:
                    logger.warning(f"Failed to get ensemble test predictions: {str(e)}")
            
            # Perform statistical tests
            statistical_results = {}
            
            for model_name, predictions in test_predictions.items():
                try:
                    # Pesaran-Timmermann test
                    pt_result = tester.pesaran_timmermann_test(y_test.values, predictions)
                    
                    # Store results
                    statistical_results[model_name] = {
                        'pesaran_timmermann': pt_result
                    }
                    
                    logger.info(f"✓ Statistical tests completed for {model_name}")
                    
                except Exception as e:
                    logger.warning(f"Statistical tests failed for {model_name}: {str(e)}")
            
            # Diebold-Mariano tests between models
            dm_results = {}
            model_names = list(test_predictions.keys())
            
            for i, model1 in enumerate(model_names):
                for model2 in model_names[i+1:]:
                    try:
                        dm_result = tester.diebold_mariano_test(
                            y_test.values, 
                            test_predictions[model1], 
                            test_predictions[model2]
                        )
                        dm_results[f"{model1}_vs_{model2}"] = dm_result
                        
                    except Exception as e:
                        logger.warning(f"DM test failed for {model1} vs {model2}: {str(e)}")
            
            # Step 6.3: Economic Backtesting
            logger.info("Step 6.3: Economic Backtesting")
            
            backtester = EconomicBacktester(config=self.config)
            
            backtest_results = {}
            for model_name, predictions in test_predictions.items():
                try:
                    bt_result = backtester.backtest_strategy(
                        predictions=predictions,
                        actual_returns=y_test.values,
                        timestamps=test_data['timestamp'] if 'timestamp' in test_data.columns else None
                    )
                    backtest_results[model_name] = bt_result
                    logger.info(f"✓ Backtesting completed for {model_name}")
                    
                except Exception as e:
                    logger.warning(f"Backtesting failed for {model_name}: {str(e)}")
            
            # Step 6.4: Model Explainability
            logger.info("Step 6.4: Model Explainability")
            
            explainer = ModelInterpreter(config=self.config)
            explainability_results = {}
            
            for model_name, model in successful_models.items():
                if model is not None:
                    try:
                        explanation = explainer.explain_model(
                            model=model,
                            X_test=X_test,
                            feature_names=feature_cols
                        )
                        explainability_results[model_name] = explanation
                        logger.info(f"✓ Explainability analysis completed for {model_name}")
                        
                    except Exception as e:
                        logger.warning(f"Explainability analysis failed for {model_name}: {str(e)}")
            
            # Compile all results
            self.results.update({
                'cross_validation': cv_results,
                'statistical_tests': statistical_results,
                'diebold_mariano': dm_results,
                'economic_backtesting': backtest_results,
                'explainability': explainability_results,
                'test_predictions': test_predictions
            })
            
            # Step 6.5: Generate Summary Report
            logger.info("Step 6.5: Generating Summary Report")
            self._generate_summary_report()
            
            self.pipeline_state['evaluation_analysis'] = True
            logger.info("✓ Phase 6 completed successfully")
            
        except Exception as e:
            logger.error(f"Phase 6 failed: {str(e)}")
            raise
    
    def _generate_summary_report(self) -> None:
        """
        Generate comprehensive summary report.
        
        Following Rule #25 (Deliverable Completeness Check).
        """
        try:
            logger.info("Generating summary report...")
            
            # Create summary metrics
            summary = {
                'pipeline_completion_time': datetime.now().isoformat(),
                'data_points': len(self.data) if self.data is not None else 0,
                'features_created': len(self.features.columns) - 2 if self.features is not None else 0,  # Exclude timestamp and target
                'models_trained': len([m for m in self.models.values() if m is not None]),
                'ensemble_created': self.ensemble is not None,
                'pipeline_state': self.pipeline_state
            }
            
            # Add model performance summary
            if 'cross_validation' in self.results:
                cv_summary = {}
                for model_name, cv_result in self.results['cross_validation'].items():
                    if cv_result is not None:
                        cv_summary[model_name] = {
                            'mean_zptae': np.mean(cv_result.get('zptae_scores', [])),
                            'mean_directional_accuracy': np.mean(cv_result.get('directional_accuracy', [])),
                            'stability': cv_result.get('stability_metrics', {})
                        }
                summary['model_performance'] = cv_summary
            
            # Save summary
            summary_path = os.path.join(self.config['paths']['reports'], 'pipeline_summary.yaml')
            with open(summary_path, 'w') as f:
                yaml.dump(summary, f, default_flow_style=False)
            
            # Save detailed results
            results_path = os.path.join(self.config['paths']['reports'], 'detailed_results.pkl')
            joblib.dump(self.results, results_path)
            
            logger.info(f"Summary report saved to {summary_path}")
            logger.info(f"Detailed results saved to {results_path}")
            
        except Exception as e:
            logger.warning(f"Failed to generate summary report: {str(e)}")
    
    def _final_validation(self) -> None:
        """
        Final validation of pipeline completion.
        
        Following Rule #29 (End-to-End Testing) and Rule #25 (Deliverable Completeness Check).
        """
        logger.info("Final Validation")
        logger.info("-" * 40)
        
        try:
            # Check pipeline state
            incomplete_phases = [phase for phase, completed in self.pipeline_state.items() if not completed]
            
            if incomplete_phases:
                logger.warning(f"Incomplete phases: {incomplete_phases}")
            else:
                logger.info("✓ All pipeline phases completed")
            
            # Check deliverables (Rule #25)
            deliverables = {
                'config_file': os.path.exists(self.config_path),
                'raw_data': self.data is not None,
                'processed_features': self.features is not None,
                'trained_models': len([m for m in self.models.values() if m is not None]) > 0,
                'evaluation_results': 'cross_validation' in self.results,
                'summary_report': os.path.exists(os.path.join(self.config['paths']['reports'], 'pipeline_summary.yaml'))
            }
            
            missing_deliverables = [item for item, exists in deliverables.items() if not exists]
            
            if missing_deliverables:
                logger.warning(f"Missing deliverables: {missing_deliverables}")
            else:
                logger.info("✓ All deliverables present")
            
            # Performance expectations check (Rule #15)
            if 'cross_validation' in self.results:
                for model_name, cv_result in self.results['cross_validation'].items():
                    if cv_result is not None:
                        da_scores = cv_result.get('directional_accuracy', [])
                        if da_scores:
                            mean_da = np.mean(da_scores)
                            if mean_da > 0.70:
                                logger.warning(f"Suspicious directional accuracy for {model_name}: {mean_da:.3f} (>70%)")
                            elif 0.52 <= mean_da <= 0.58:
                                logger.info(f"✓ {model_name} directional accuracy in expected range: {mean_da:.3f}")
                            else:
                                logger.info(f"{model_name} directional accuracy: {mean_da:.3f}")
            
            logger.info("✓ Final validation completed")
            
        except Exception as e:
            logger.error(f"Final validation failed: {str(e)}")
            raise
    
    def _handle_pipeline_failure(self, error: Exception) -> None:
        """
        Handle pipeline failure according to Rule #23 (Failure Escalation Protocol).
        
        Args:
            error: The exception that caused the failure
        """
        logger.error("=" * 80)
        logger.error("PIPELINE FAILURE DETECTED")
        logger.error("=" * 80)
        
        try:
            # Log detailed failure context
            failure_context = {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'pipeline_state': self.pipeline_state,
                'timestamp': datetime.now().isoformat(),
                'memory_usage': memory_usage_check()
            }
            
            # Save current state
            failure_dir = os.path.join(self.config['paths']['reports'], 'failure_analysis')
            Path(failure_dir).mkdir(parents=True, exist_ok=True)
            
            failure_path = os.path.join(failure_dir, f"failure_context_{int(time.time())}.yaml")
            with open(failure_path, 'w') as f:
                yaml.dump(failure_context, f, default_flow_style=False)
            
            # Save intermediate results if available
            if self.results:
                results_path = os.path.join(failure_dir, f"partial_results_{int(time.time())}.pkl")
                joblib.dump(self.results, results_path)
                logger.info(f"Partial results saved to {results_path}")
            
            logger.error(f"Failure context saved to {failure_path}")
            logger.error("Please review the failure context for debugging")
            
        except Exception as save_error:
            logger.error(f"Failed to save failure context: {str(save_error)}")


def main():
    """
    Main entry point for the ETH forecasting pipeline.
    """
    try:
        # Initialize pipeline
        pipeline = ETHForecastingPipeline()
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline()
        
        print("\n" + "=" * 80)
        print("ETH FORECASTING PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Models trained: {len([m for m in pipeline.models.values() if m is not None])}")
        print(f"Ensemble created: {pipeline.ensemble is not None}")
        print(f"Results available in: {pipeline.config['paths']['reports']}")
        print("=" * 80)
        
        return results
        
    except Exception as e:
        print(f"\nPipeline failed with error: {str(e)}")
        print("Check logs for detailed error information")
        return None


if __name__ == "__main__":
    main()