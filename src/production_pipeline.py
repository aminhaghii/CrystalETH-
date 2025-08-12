"""
Production Pipeline with Prediction Tracking

This module integrates the ETH forecasting ensemble with comprehensive prediction tracking,
chain response logging, and performance monitoring for production deployment.
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.utils.prediction_tracker import PredictionTracker
from src.production.ensemble_manager import EnsembleManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductionPipeline:
    """
    Production pipeline for ETH price prediction with comprehensive tracking.
    
    This class integrates the ensemble model with the prediction tracking system
    to provide real-time prediction generation, automatic tracking, and performance monitoring.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the production pipeline.
        
        Args:
            config_path: Path to main configuration file
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize tracking system
        self.tracker = PredictionTracker(
            predictions_file='predictions_log.csv',
            chain_responses_file='chain_responses.jsonl',
            performance_file='performance_metrics.json',
            base_dir='logs/predictions'
        )
        
        # Production logger is handled by the tracker
        self.prod_logger = self.tracker
        
        # Initialize ensemble manager with real models
        try:
            self.ensemble_manager = EnsembleManager(self.config)
            self.logger.info("✅ EnsembleManager initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize EnsembleManager: {e}")
            self.ensemble_manager = None
        
        self.logger.info("✅ ProductionPipeline initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                import yaml
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"Could not load config from {config_path}: {e}")
            return {}
    
    def _get_model_version(self) -> str:
        """Get the current model version."""
        if self.ensemble_manager is not None:
            return "ensemble_v2.1.0"
        else:
            return "mock_v1.0"
    

    
    def predict_and_track(self, 
                         topic_id: str,
                         current_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate prediction and automatically track it.
        
        Args:
            topic_id: Unique identifier for the prediction topic
            current_data: Current market data (if available)
            
        Returns:
            Dictionary containing prediction results and tracking information
        """
        try:
            # Use EnsembleManager for real predictions
            if self.ensemble_manager is not None:
                try:
                    # Get live prediction from ensemble
                    prediction_result = self.ensemble_manager.predict()
                    
                    prediction_value = prediction_result.prediction
                    direction = prediction_result.direction
                    confidence = prediction_result.confidence
                    model_version = self._get_model_version()
                    features_hash = hashlib.md5(str(prediction_result.feature_importance).encode()).hexdigest()[:8]
                    
                    self.logger.info(f"🎯 Real ensemble prediction: {prediction_value:.6f} (direction: {direction}, confidence: {confidence:.3f})")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Ensemble prediction failed, using fallback: {e}")
                    # Fallback to mock prediction
                    prediction_value = np.random.normal(0, 0.05)
                    confidence = np.random.uniform(0.5, 0.9)
                    direction = 1 if prediction_value > 0 else 0
                    model_version = "fallback_mock_v1.0"
                    features_hash = "fallback_hash"
            else:
                # Mock prediction for testing
                prediction_value = np.random.normal(0, 0.05)  # Random price change
                confidence = np.random.uniform(0.5, 0.9)  # Random confidence
                direction = 1 if prediction_value > 0 else 0
                model_version = "mock_v1.0"
                features_hash = "test_hash"
                
                self.logger.info(f"🧪 Mock prediction: {prediction_value:.6f} (direction: {direction}, confidence: {confidence:.3f})")
            
            # Store prediction with tracking
            tracking_id = self.tracker.store_prediction(
                topic_id=topic_id,
                prediction_value=prediction_value,
                prediction_direction=direction,
                confidence_score=confidence,
                model_version=model_version,
                features_hash=features_hash
            )
            
            # Log to production system
            self.prod_logger.store_prediction(
                topic_id=topic_id,
                prediction_value=prediction_value,
                prediction_direction=direction,
                confidence_score=confidence,
                model_version=model_version,
                features_hash=features_hash
            )
            
            # Return comprehensive result
            return {
                'prediction': prediction_value,
                'direction': direction,
                'confidence': confidence,
                'tracking_id': tracking_id,
                'topic_id': topic_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'model_version': model_version,
                'test_mode': self.ensemble_manager is None
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in predict_and_track: {e}")
            raise
    
    def log_chain_response(self, 
                          prediction_id: str, 
                          response_data: Dict[str, Any]) -> None:
        """
        Log blockchain response data.
        
        Args:
            prediction_id: Prediction identifier
            response_data: Response data from blockchain
        """
        try:
            self.tracker.log_chain_response(prediction_id, response_data)
            self.logger.info(f"Logged chain response for {prediction_id}")
            
        except Exception as e:
            self.logger.error(f"Error logging chain response: {e}")
            raise
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        try:
            return self.tracker.get_performance_metrics()
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def log_chain_submission(self, 
                           prediction_id: str,
                           tx_hash: str,
                           submission_score: float = None,
                           gas_used: int = None,
                           **additional_data):
        """
        Log chain submission details.
        
        Args:
            prediction_id: ID of the prediction being submitted
            tx_hash: Transaction hash from blockchain submission
            submission_score: Score received from submission
            gas_used: Gas used for the transaction
            **additional_data: Any additional chain response data
        """
        try:
            self.tracker.log_chain_response(
                prediction_id=prediction_id,
                tx_hash=tx_hash,
                score=submission_score,
                gas_used=gas_used,
                **additional_data
            )
            self.logger.info(f"Logged chain submission for prediction {prediction_id}")
        except Exception as e:
            self.logger.error(f"Error logging chain submission: {e}")
    
    def update_ground_truth(self, 
                          topic_id: str,
                          actual_value: float,
                          actual_timestamp: datetime = None) -> List[str]:
        """
        Update ground truth when actual values become available.
        
        Args:
            topic_id: Topic ID to update
            actual_value: Actual observed value
            actual_timestamp: When the actual value was observed
        
        Returns:
            List of updated prediction IDs
        """
        try:
            if actual_timestamp is None:
                actual_timestamp = datetime.now(timezone.utc)
            
            updated_predictions = self.tracker.update_ground_truth(
                topic_id=topic_id,
                ground_truth_value=actual_value,
                ground_truth_direction=1 if actual_value > 0 else 0,
                ground_truth_timestamp=actual_timestamp
            )
            
            self.logger.info(f"Updated ground truth for topic {topic_id}, "
                           f"affected {len(updated_predictions)} predictions")
            
            # Trigger performance analysis if significant updates
            if len(updated_predictions) > 0:
                self._analyze_recent_performance()
            
            return updated_predictions
            
        except Exception as e:
            self.logger.error(f"Error updating ground truth for topic {topic_id}: {e}")
            return []
    
    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get performance summary for the last N days.
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Performance summary dictionary
        """
        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            metrics = self.tracker.get_performance_metrics(
                start_date=start_date,
                end_date=end_date
            )
            
            # Add additional analysis
            pending_predictions = self.tracker.get_pending_predictions()
            
            summary = {
                'period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'days': days
                },
                'performance': metrics,
                'pending_predictions': len(pending_predictions),
                'model_status': {
                    'ensemble_loaded': self.ensemble_model is not None
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating performance summary: {e}")
            return {}
    
    def export_performance_report(self) -> str:
        """Export comprehensive performance report."""
        try:
            report_path = self.tracker.export_performance_report()
            self.logger.info(f"Performance report exported to {report_path}")
            return report_path
        except Exception as e:
            self.logger.error(f"Error exporting performance report: {e}")
            return ""
    
    def _prepare_features(self) -> Optional[np.ndarray]:
        """Prepare features for prediction."""
        try:
            # For testing purposes, return mock features
            # In production, this would load and process real data
            return np.random.randn(1, 10)  # Mock feature vector
                
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return None
    

    
    def _hash_features(self, features: np.ndarray) -> str:
        """Create hash of features for reproducibility tracking."""
        try:
            # Convert features to string and hash
            features_str = np.array2string(features, precision=6)
            return hashlib.md5(features_str.encode()).hexdigest()[:16]
        except Exception as e:
            self.logger.error(f"Error hashing features: {e}")
            return "unknown"
    
    def _analyze_recent_performance(self):
        """Analyze recent performance and log insights."""
        try:
            # Get recent performance
            recent_metrics = self.get_performance_summary(days=7)
            
            if recent_metrics.get('performance', {}).get('metrics'):
                metrics = recent_metrics['performance']['metrics']
                
                # Log key insights
                self.logger.info(f"Recent Performance (7 days): "
                               f"DA={metrics.get('directional_accuracy', 0):.3f}, "
                               f"MAE={metrics.get('mae', 0):.4f}, "
                               f"ZPTAE={metrics.get('zptae_loss', 0):.4f}")
                
                # Check for performance degradation
                if metrics.get('directional_accuracy', 0) < 0.45:
                    self.logger.warning("Directional accuracy below 45% - model may need retraining")
                
                if metrics.get('mae', float('inf')) > 0.1:
                    self.logger.warning("High MAE detected - check for data quality issues")
            
        except Exception as e:
            self.logger.error(f"Error analyzing recent performance: {e}")


def main():
    """Example usage of the production pipeline."""
    try:
        # Initialize pipeline
        pipeline = ProductionPipeline()
        
        # Generate a prediction
        topic_id = f"eth_24h_{datetime.now().strftime('%Y%m%d_%H%M')}"
        prediction = pipeline.predict_and_track(topic_id)
        
        print(f"Generated prediction: {prediction}")
        
        # Simulate chain submission
        pipeline.log_chain_submission(
            prediction_id=prediction['tracking_id'],
            tx_hash="0x1234567890abcdef",
            submission_score=0.75,
            gas_used=21000
        )
        
        # Get performance summary
        performance = pipeline.get_performance_summary()
        print(f"Performance summary: {performance}")
        
        # Export report
        report_path = pipeline.export_performance_report()
        print(f"Report exported to: {report_path}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()