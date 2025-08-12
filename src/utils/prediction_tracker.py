"""
Prediction Tracking System for ETH Forecasting

This module provides comprehensive tracking of model predictions, ground truth comparison,
loss calculation, and chain response logging for production monitoring.
"""

import os
import json
import csv
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PredictionTracker:
    """
    Comprehensive prediction tracking system for ETH forecasting model.
    
    Features:
    - Store predictions with timestamps and topic IDs
    - Calculate losses when ground truth becomes available
    - Log chain responses (txHash, scores, etc.)
    - Performance monitoring and analytics
    - Data integrity validation
    """
    
    def __init__(self, 
                 predictions_file: str = "predictions_log.csv",
                 chain_responses_file: str = "chain_responses.jsonl",
                 performance_file: str = "performance_metrics.json",
                 base_dir: str = "logs/predictions"):
        """
        Initialize the prediction tracker.
        
        Args:
            predictions_file: CSV file for storing predictions and losses
            chain_responses_file: JSONL file for chain responses
            performance_file: JSON file for performance metrics
            base_dir: Base directory for all tracking files
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.predictions_file = self.base_dir / predictions_file
        self.chain_responses_file = self.base_dir / chain_responses_file
        self.performance_file = self.base_dir / performance_file
        
        # Initialize files if they don't exist
        self._initialize_files()
        
        # In-memory cache for fast access
        self._predictions_cache = {}
        self._performance_cache = {}
        
        logger.info(f"PredictionTracker initialized with base_dir: {self.base_dir}")
    
    def _initialize_files(self):
        """Initialize tracking files with proper headers."""
        # Initialize predictions CSV
        if not self.predictions_file.exists():
            with open(self.predictions_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'topic_id', 'prediction_value', 'prediction_direction',
                    'confidence_score', 'model_version', 'features_hash',
                    'ground_truth_value', 'ground_truth_direction', 'ground_truth_timestamp',
                    'mse_loss', 'mae_loss', 'directional_accuracy', 'zptae_loss',
                    'days_to_resolution', 'prediction_status'
                ])
        
        # Initialize performance metrics JSON
        if not self.performance_file.exists():
            initial_metrics = {
                'total_predictions': 0,
                'resolved_predictions': 0,
                'pending_predictions': 0,
                'overall_mse': None,
                'overall_mae': None,
                'overall_directional_accuracy': None,
                'overall_zptae_loss': None,
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'model_performance_by_version': {},
                'daily_performance': {},
                'topic_performance': {}
            }
            with open(self.performance_file, 'w') as f:
                json.dump(initial_metrics, f, indent=2)
    
    def store_prediction(self,
                        topic_id: str,
                        prediction_value: float,
                        prediction_direction: int,
                        confidence_score: float = None,
                        model_version: str = "v1.0",
                        features_hash: str = None,
                        timestamp: datetime = None) -> str:
        """
        Store a new prediction.
        
        Args:
            topic_id: Unique identifier for the prediction topic
            prediction_value: Predicted value (e.g., price change, return)
            prediction_direction: Predicted direction (0=down, 1=up)
            confidence_score: Model confidence in the prediction
            model_version: Version of the model making the prediction
            features_hash: Hash of input features for reproducibility
            timestamp: Prediction timestamp (defaults to current UTC time)
        
        Returns:
            prediction_id: Unique identifier for this prediction
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        prediction_id = f"{topic_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Store in CSV
        with open(self.predictions_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp.isoformat(),
                topic_id,
                prediction_value,
                prediction_direction,
                confidence_score,
                model_version,
                features_hash,
                None,  # ground_truth_value
                None,  # ground_truth_direction
                None,  # ground_truth_timestamp
                None,  # mse_loss
                None,  # mae_loss
                None,  # directional_accuracy
                None,  # zptae_loss
                None,  # days_to_resolution
                'pending'  # prediction_status
            ])
        
        # Update cache
        self._predictions_cache[prediction_id] = {
            'timestamp': timestamp,
            'topic_id': topic_id,
            'prediction_value': prediction_value,
            'prediction_direction': prediction_direction,
            'confidence_score': confidence_score,
            'model_version': model_version,
            'features_hash': features_hash,
            'status': 'pending'
        }
        
        # Update performance metrics
        self._update_performance_metrics()
        
        logger.info(f"Stored prediction {prediction_id} for topic {topic_id}")
        return prediction_id
    
    def update_ground_truth(self,
                           topic_id: str,
                           ground_truth_value: float,
                           ground_truth_direction: int = None,
                           ground_truth_timestamp: datetime = None) -> List[str]:
        """
        Update ground truth for predictions and calculate losses.
        
        Args:
            topic_id: Topic identifier to update
            ground_truth_value: Actual observed value
            ground_truth_direction: Actual direction (0=down, 1=up)
            ground_truth_timestamp: When the ground truth was observed
        
        Returns:
            List of updated prediction IDs
        """
        if ground_truth_timestamp is None:
            ground_truth_timestamp = datetime.now(timezone.utc)
        
        if ground_truth_direction is None:
            ground_truth_direction = 1 if ground_truth_value > 0 else 0
        
        # Read current predictions
        df = pd.read_csv(self.predictions_file)
        
        # Find pending predictions for this topic
        mask = (df['topic_id'] == topic_id) & (df['prediction_status'] == 'pending')
        updated_predictions = []
        
        if mask.sum() == 0:
            logger.warning(f"No pending predictions found for topic {topic_id}")
            return updated_predictions
        
        # Calculate losses for each matching prediction
        for idx in df[mask].index:
            prediction_value = df.loc[idx, 'prediction_value']
            prediction_direction = df.loc[idx, 'prediction_direction']
            prediction_timestamp = pd.to_datetime(df.loc[idx, 'timestamp'])
            
            # Calculate losses
            mse_loss = (prediction_value - ground_truth_value) ** 2
            mae_loss = abs(prediction_value - ground_truth_value)
            directional_accuracy = 1 if prediction_direction == ground_truth_direction else 0
            zptae_loss = self._calculate_zptae_loss(prediction_value, ground_truth_value)
            
            # Calculate days to resolution
            days_to_resolution = (ground_truth_timestamp - prediction_timestamp).total_seconds() / 86400
            
            # Update the row
            df.loc[idx, 'ground_truth_value'] = ground_truth_value
            df.loc[idx, 'ground_truth_direction'] = ground_truth_direction
            df.loc[idx, 'ground_truth_timestamp'] = ground_truth_timestamp.isoformat()
            df.loc[idx, 'mse_loss'] = mse_loss
            df.loc[idx, 'mae_loss'] = mae_loss
            df.loc[idx, 'directional_accuracy'] = directional_accuracy
            df.loc[idx, 'zptae_loss'] = zptae_loss
            df.loc[idx, 'days_to_resolution'] = days_to_resolution
            df.loc[idx, 'prediction_status'] = 'resolved'
            
            prediction_id = f"{topic_id}_{prediction_timestamp.strftime('%Y%m%d_%H%M%S')}"
            updated_predictions.append(prediction_id)
        
        # Save updated dataframe
        df.to_csv(self.predictions_file, index=False)
        
        # Update performance metrics
        self._update_performance_metrics()
        
        logger.info(f"Updated ground truth for {len(updated_predictions)} predictions on topic {topic_id}")
        return updated_predictions
    
    def log_chain_response(self,
                          prediction_id: str,
                          response_data: Dict[str, Any],
                          timestamp: datetime = None):
        """
        Log chain response data (txHash, scores, etc.).
        
        Args:
            prediction_id: Associated prediction ID
            response_data: Chain response data (txHash, score, etc.)
            timestamp: Response timestamp
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        log_entry = {
            'timestamp': timestamp.isoformat(),
            'prediction_id': prediction_id,
            'response_data': response_data
        }
        
        # Append to JSONL file
        with open(self.chain_responses_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        logger.info(f"Logged chain response for prediction {prediction_id}")
    
    def get_performance_metrics(self, 
                               model_version: str = None,
                               topic_id: str = None,
                               start_date: datetime = None,
                               end_date: datetime = None) -> Dict[str, Any]:
        """
        Get performance metrics with optional filtering.
        
        Args:
            model_version: Filter by model version
            topic_id: Filter by topic ID
            start_date: Filter predictions after this date
            end_date: Filter predictions before this date
        
        Returns:
            Dictionary of performance metrics
        """
        df = pd.read_csv(self.predictions_file)
        
        # Apply filters
        if model_version:
            df = df[df['model_version'] == model_version]
        
        if topic_id:
            df = df[df['topic_id'] == topic_id]
        
        if start_date:
            # Convert to timezone-aware datetime for comparison
            try:
                df_timestamps = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')
            except:
                # If already timezone-aware, convert to UTC
                df_timestamps = pd.to_datetime(df['timestamp']).dt.tz_convert('UTC')
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
            df = df[df_timestamps >= start_date]
        
        if end_date:
            # Convert to timezone-aware datetime for comparison
            try:
                df_timestamps = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')
            except:
                # If already timezone-aware, convert to UTC
                df_timestamps = pd.to_datetime(df['timestamp']).dt.tz_convert('UTC')
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)
            df = df[df_timestamps <= end_date]
        
        # Calculate metrics for resolved predictions only
        resolved_df = df[df['prediction_status'] == 'resolved']
        
        if len(resolved_df) == 0:
            return {
                'total_predictions': len(df),
                'resolved_predictions': 0,
                'pending_predictions': len(df),
                'metrics': None
            }
        
        metrics = {
            'total_predictions': len(df),
            'resolved_predictions': len(resolved_df),
            'pending_predictions': len(df) - len(resolved_df),
            'metrics': {
                'mse': resolved_df['mse_loss'].mean(),
                'mae': resolved_df['mae_loss'].mean(),
                'directional_accuracy': resolved_df['directional_accuracy'].mean(),
                'zptae_loss': resolved_df['zptae_loss'].mean(),
                'avg_days_to_resolution': resolved_df['days_to_resolution'].mean(),
                'confidence_correlation': self._calculate_confidence_correlation(resolved_df)
            }
        }
        
        return metrics
    
    def get_pending_predictions(self, topic_id: str = None) -> pd.DataFrame:
        """
        Get all pending predictions, optionally filtered by topic.
        
        Args:
            topic_id: Optional topic filter
        
        Returns:
            DataFrame of pending predictions
        """
        df = pd.read_csv(self.predictions_file)
        pending_df = df[df['prediction_status'] == 'pending']
        
        if topic_id:
            pending_df = pending_df[pending_df['topic_id'] == topic_id]
        
        return pending_df
    
    def export_performance_report(self, output_file: str = None) -> str:
        """
        Export comprehensive performance report.
        
        Args:
            output_file: Output file path (defaults to timestamped file)
        
        Returns:
            Path to the exported report
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.base_dir / f"performance_report_{timestamp}.json"
        
        # Get overall metrics
        overall_metrics = self.get_performance_metrics()
        
        # Get metrics by model version
        df = pd.read_csv(self.predictions_file)
        model_versions = df['model_version'].unique()
        version_metrics = {}
        for version in model_versions:
            version_metrics[version] = self.get_performance_metrics(model_version=version)
        
        # Get daily performance
        try:
            df['date'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC').dt.date
        except:
            # If already timezone-aware, convert to UTC
            df['date'] = pd.to_datetime(df['timestamp']).dt.tz_convert('UTC').dt.date
        daily_metrics = {}
        for date in df['date'].dropna().unique():
            date_str = str(date)
            start_date = datetime.combine(date, datetime.min.time()).replace(tzinfo=timezone.utc)
            end_date = datetime.combine(date, datetime.max.time()).replace(tzinfo=timezone.utc)
            daily_metrics[date_str] = self.get_performance_metrics(
                start_date=start_date, end_date=end_date
            )
        
        report = {
            'report_timestamp': datetime.now(timezone.utc).isoformat(),
            'overall_performance': overall_metrics,
            'performance_by_model_version': version_metrics,
            'daily_performance': daily_metrics,
            'data_summary': {
                'total_predictions': len(df),
                'unique_topics': df['topic_id'].nunique(),
                'date_range': {
                    'start': df['timestamp'].min(),
                    'end': df['timestamp'].max()
                }
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report exported to {output_file}")
        return str(output_file)
    
    def _calculate_zptae_loss(self, prediction: float, ground_truth: float, 
                             a: float = 1.0, p: float = 1.5) -> float:
        """Calculate Zero-inflated Penalized Threshold Asymmetric Error."""
        error = ground_truth - prediction
        if error >= 0:
            return a * (abs(error) ** p)
        else:
            return abs(error) ** p
    
    def _calculate_confidence_correlation(self, df: pd.DataFrame) -> float:
        """Calculate correlation between confidence scores and accuracy."""
        if 'confidence_score' not in df.columns or df['confidence_score'].isna().all():
            return None
        
        # Calculate accuracy as inverse of absolute error
        df_clean = df.dropna(subset=['confidence_score', 'mae_loss'])
        if len(df_clean) < 2:
            return None
        
        accuracy = 1 / (1 + df_clean['mae_loss'])  # Higher accuracy for lower error
        correlation = np.corrcoef(df_clean['confidence_score'], accuracy)[0, 1]
        return correlation if not np.isnan(correlation) else None
    
    def _update_performance_metrics(self):
        """Update the performance metrics file."""
        try:
            metrics = self.get_performance_metrics()
            metrics['last_updated'] = datetime.now(timezone.utc).isoformat()
            
            with open(self.performance_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")


class ProductionPredictionLogger:
    """
    Simplified logger for production use with minimal overhead.
    """
    
    def __init__(self, log_dir: str = "logs/production"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = PredictionTracker(base_dir=str(self.log_dir))
    
    def log_prediction(self, topic_id: str, prediction: Dict[str, Any]) -> str:
        """Log a prediction with minimal overhead."""
        return self.tracker.store_prediction(
            topic_id=topic_id,
            prediction_value=prediction.get('value', 0.0),
            prediction_direction=prediction.get('direction', 0),
            confidence_score=prediction.get('confidence'),
            model_version=prediction.get('model_version', 'production'),
            features_hash=prediction.get('features_hash')
        )
    
    def log_chain_response(self, prediction_id: str, tx_hash: str, 
                          score: float = None, **kwargs):
        """Log chain response with transaction hash."""
        response_data = {
            'tx_hash': tx_hash,
            'score': score,
            **kwargs
        }
        self.tracker.log_chain_response(prediction_id, response_data)
    
    def update_result(self, topic_id: str, actual_value: float):
        """Update with actual result when available."""
        return self.tracker.update_ground_truth(topic_id, actual_value)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get quick performance statistics."""
        return self.tracker.get_performance_metrics()