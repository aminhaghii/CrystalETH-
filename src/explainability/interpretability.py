"""
Model interpretability module for ETH forecasting project.

This module implements model explainability following Rule #21:
- Generate SHAP analysis (tree models)
- Extract attention weights (transformer models)
- Perform feature ablation studies
- Analyze regime-specific performance
- Create visualization summaries

Provides comprehensive model interpretation capabilities.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class ModelInterpreter:
    """
    Model interpretability and explainability class.
    
    Provides various methods to understand model behavior,
    feature importance, and prediction rationale.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelInterpreter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.explainability_config = config.get('explainability', {})
        
        # Output paths
        self.output_dir = Path(config.get('paths', {}).get('reports', 'reports')) / 'explainability'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualization settings
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        logging.info("ModelInterpreter initialized")
    
    def analyze_tree_model(self, model: Any, X: pd.DataFrame, 
                          y: pd.Series, feature_names: List[str] = None,
                          sample_size: int = 1000) -> Dict[str, Any]:
        """
        Analyze tree-based model using SHAP.
        
        Args:
            model: Trained tree model (LightGBM, XGBoost, etc.)
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
            sample_size: Sample size for SHAP analysis
            
        Returns:
            Dictionary with analysis results
        """
        logging.info("Analyzing tree model with SHAP")
        
        try:
            import shap
        except ImportError:
            logging.error("SHAP not available. Install with: pip install shap")
            return {'error': 'SHAP not available'}
        
        if feature_names is None:
            feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        
        # Sample data for efficiency
        if len(X) > sample_size:
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[sample_indices] if hasattr(X, 'iloc') else X[sample_indices]
            y_sample = y.iloc[sample_indices] if hasattr(y, 'iloc') else y[sample_indices]
        else:
            X_sample = X
            y_sample = y
        
        results = {}
        
        try:
            # Create SHAP explainer
            if hasattr(model, 'predict'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
                # Handle multi-output case
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]  # Use first output
                
                results['shap_values'] = shap_values
                results['feature_names'] = feature_names
                
                # Feature importance
                feature_importance = np.abs(shap_values).mean(axis=0)
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
                
                results['feature_importance'] = importance_df
                
                # Global feature importance plot
                self._plot_feature_importance(importance_df, 'Tree Model Feature Importance (SHAP)')
                
                # SHAP summary plot
                self._plot_shap_summary(shap_values, X_sample, feature_names)
                
                # SHAP waterfall plot for sample predictions
                self._plot_shap_waterfall(explainer, X_sample, feature_names, n_samples=3)
                
                # Feature interactions
                if len(feature_names) <= 20:  # Limit for computational efficiency
                    interaction_values = explainer.shap_interaction_values(X_sample[:100])
                    results['interaction_values'] = interaction_values
                    self._plot_feature_interactions(interaction_values, feature_names)
                
                logging.info("Tree model SHAP analysis completed")
                
        except Exception as e:
            logging.error(f"Error in SHAP analysis: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def analyze_neural_network(self, model: Any, X: pd.DataFrame,
                             feature_names: List[str] = None,
                             sample_size: int = 500) -> Dict[str, Any]:
        """
        Analyze neural network model.
        
        Args:
            model: Trained neural network model
            X: Feature matrix
            feature_names: List of feature names
            sample_size: Sample size for analysis
            
        Returns:
            Dictionary with analysis results
        """
        logging.info("Analyzing neural network model")
        
        if feature_names is None:
            feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        
        results = {}
        
        # Sample data
        if len(X) > sample_size:
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[sample_indices] if hasattr(X, 'iloc') else X[sample_indices]
        else:
            X_sample = X
        
        try:
            # Gradient-based feature importance
            if hasattr(model, 'predict') and hasattr(model, 'parameters'):
                # PyTorch model
                import torch
                
                if isinstance(X_sample, pd.DataFrame):
                    X_tensor = torch.tensor(X_sample.values, dtype=torch.float32, requires_grad=True)
                else:
                    X_tensor = torch.tensor(X_sample, dtype=torch.float32, requires_grad=True)
                
                model.eval()
                predictions = model(X_tensor)
                
                # Calculate gradients
                gradients = torch.autograd.grad(
                    outputs=predictions.sum(),
                    inputs=X_tensor,
                    create_graph=True
                )[0]
                
                # Feature importance based on gradient magnitude
                feature_importance = torch.abs(gradients).mean(dim=0).detach().numpy()
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
                
                results['feature_importance'] = importance_df
                results['gradients'] = gradients.detach().numpy()
                
                # Plot feature importance
                self._plot_feature_importance(importance_df, 'Neural Network Feature Importance (Gradients)')
                
        except Exception as e:
            logging.error(f"Error in neural network analysis: {str(e)}")
            results['error'] = str(e)
        
        # Permutation importance as fallback
        try:
            from sklearn.inspection import permutation_importance
            
            # Convert to numpy if needed
            X_np = X_sample.values if hasattr(X_sample, 'values') else X_sample
            
            # Create dummy target for permutation importance
            y_dummy = np.random.randn(len(X_sample))
            
            perm_importance = permutation_importance(
                model, X_np, y_dummy, n_repeats=10, random_state=42
            )
            
            perm_df = pd.DataFrame({
                'feature': feature_names,
                'importance': perm_importance.importances_mean,
                'std': perm_importance.importances_std
            }).sort_values('importance', ascending=False)
            
            results['permutation_importance'] = perm_df
            
            # Plot permutation importance
            self._plot_permutation_importance(perm_df, 'Neural Network Permutation Importance')
            
        except Exception as e:
            logging.warning(f"Permutation importance failed: {str(e)}")
        
        return results
    
    def feature_ablation_study(self, model: Any, X: pd.DataFrame, y: pd.Series,
                             feature_groups: Dict[str, List[str]] = None,
                             metric: str = 'mse') -> Dict[str, Any]:
        """
        Perform feature ablation study.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target variable
            feature_groups: Dictionary of feature groups
            metric: Evaluation metric
            
        Returns:
            Dictionary with ablation results
        """
        logging.info("Performing feature ablation study")
        
        if feature_groups is None:
            # Create individual feature groups
            feature_groups = {col: [col] for col in X.columns}
        
        results = {}
        baseline_score = self._calculate_metric(y, model.predict(X), metric)
        results['baseline_score'] = baseline_score
        
        ablation_results = []
        
        for group_name, features in feature_groups.items():
            try:
                # Remove feature group
                X_ablated = X.drop(columns=features, errors='ignore')
                
                if X_ablated.shape[1] == 0:
                    continue
                
                # Make predictions without this feature group
                predictions = model.predict(X_ablated)
                score = self._calculate_metric(y, predictions, metric)
                
                # Calculate importance as performance drop
                importance = baseline_score - score if metric in ['r2', 'accuracy'] else score - baseline_score
                
                ablation_results.append({
                    'feature_group': group_name,
                    'features': features,
                    'score_without': score,
                    'importance': importance,
                    'relative_importance': importance / abs(baseline_score) if baseline_score != 0 else 0
                })
                
            except Exception as e:
                logging.warning(f"Ablation failed for {group_name}: {str(e)}")
                continue
        
        ablation_df = pd.DataFrame(ablation_results).sort_values('importance', ascending=False)
        results['ablation_results'] = ablation_df
        
        # Plot ablation results
        self._plot_ablation_results(ablation_df, metric)
        
        logging.info("Feature ablation study completed")
        
        return results
    
    def regime_analysis(self, model: Any, X: pd.DataFrame, y: pd.Series,
                       predictions: np.ndarray, dates: pd.Series = None) -> Dict[str, Any]:
        """
        Analyze model performance across different market regimes.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target variable
            predictions: Model predictions
            dates: Date series for temporal analysis
            
        Returns:
            Dictionary with regime analysis results
        """
        logging.info("Performing regime analysis")
        
        results = {}
        
        # Define regimes based on volatility and returns
        regimes = self._identify_regimes(y, dates)
        results['regimes'] = regimes
        
        # Analyze performance by regime
        regime_performance = []
        
        for regime_name, regime_mask in regimes.items():
            if np.sum(regime_mask) < 10:  # Skip regimes with too few observations
                continue
            
            y_regime = y[regime_mask]
            pred_regime = predictions[regime_mask]
            
            # Calculate metrics for this regime
            metrics = {
                'regime': regime_name,
                'n_observations': np.sum(regime_mask),
                'mse': np.mean((y_regime - pred_regime) ** 2),
                'mae': np.mean(np.abs(y_regime - pred_regime)),
                'directional_accuracy': np.mean((y_regime > 0) == (pred_regime > 0)),
                'correlation': np.corrcoef(y_regime, pred_regime)[0, 1] if len(y_regime) > 1 else 0
            }
            
            regime_performance.append(metrics)
        
        regime_df = pd.DataFrame(regime_performance)
        results['regime_performance'] = regime_df
        
        # Plot regime performance
        self._plot_regime_performance(regime_df)
        
        # Feature importance by regime
        if hasattr(model, 'feature_importances_'):
            regime_importance = self._analyze_feature_importance_by_regime(
                model, X, y, regimes
            )
            results['regime_feature_importance'] = regime_importance
        
        logging.info("Regime analysis completed")
        
        return results
    
    def _identify_regimes(self, returns: pd.Series, dates: pd.Series = None) -> Dict[str, np.ndarray]:
        """
        Identify market regimes based on volatility and returns.
        
        Args:
            returns: Return series
            dates: Date series
            
        Returns:
            Dictionary of regime masks
        """
        regimes = {}
        
        # Rolling volatility (20-day window)
        rolling_vol = returns.rolling(window=20).std()
        vol_median = rolling_vol.median()
        
        # Rolling returns (20-day window)
        rolling_ret = returns.rolling(window=20).mean()
        ret_median = rolling_ret.median()
        
        # Define regimes
        regimes['high_vol_positive'] = (rolling_vol > vol_median) & (rolling_ret > ret_median)
        regimes['high_vol_negative'] = (rolling_vol > vol_median) & (rolling_ret <= ret_median)
        regimes['low_vol_positive'] = (rolling_vol <= vol_median) & (rolling_ret > ret_median)
        regimes['low_vol_negative'] = (rolling_vol <= vol_median) & (rolling_ret <= ret_median)
        
        # Extreme regimes
        vol_95 = rolling_vol.quantile(0.95)
        ret_5 = rolling_ret.quantile(0.05)
        ret_95 = rolling_ret.quantile(0.95)
        
        regimes['extreme_volatility'] = rolling_vol > vol_95
        regimes['extreme_negative'] = rolling_ret < ret_5
        regimes['extreme_positive'] = rolling_ret > ret_95
        
        # Convert to numpy arrays
        regimes = {k: v.values for k, v in regimes.items()}
        
        return regimes
    
    def _analyze_feature_importance_by_regime(self, model: Any, X: pd.DataFrame,
                                            y: pd.Series, regimes: Dict[str, np.ndarray]) -> Dict[str, pd.DataFrame]:
        """
        Analyze feature importance by regime.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target variable
            regimes: Dictionary of regime masks
            
        Returns:
            Dictionary of feature importance by regime
        """
        regime_importance = {}
        
        for regime_name, regime_mask in regimes.items():
            if np.sum(regime_mask) < 50:  # Skip regimes with too few observations
                continue
            
            try:
                # Get regime data
                X_regime = X[regime_mask]
                y_regime = y[regime_mask]
                
                # Retrain model on regime data (simplified)
                from sklearn.ensemble import RandomForestRegressor
                regime_model = RandomForestRegressor(n_estimators=100, random_state=42)
                regime_model.fit(X_regime, y_regime)
                
                # Get feature importance
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': regime_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                regime_importance[regime_name] = importance_df
                
            except Exception as e:
                logging.warning(f"Regime importance analysis failed for {regime_name}: {str(e)}")
                continue
        
        return regime_importance
    
    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
        """Calculate specified metric."""
        if metric == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            return np.mean(np.abs(y_true - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        elif metric == 'accuracy':
            return np.mean((y_true > 0) == (y_pred > 0))
        else:
            return np.mean((y_true - y_pred) ** 2)  # Default to MSE
    
    def _plot_feature_importance(self, importance_df: pd.DataFrame, title: str) -> None:
        """Plot feature importance."""
        plt.figure(figsize=(12, 8))
        
        # Plot top 20 features
        top_features = importance_df.head(20)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.output_dir / f'feature_importance_{title.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_shap_summary(self, shap_values: np.ndarray, X: pd.DataFrame, 
                          feature_names: List[str]) -> None:
        """Plot SHAP summary."""
        try:
            import shap
            
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
            plt.tight_layout()
            
            plt.savefig(self.output_dir / 'shap_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.warning(f"SHAP summary plot failed: {str(e)}")
    
    def _plot_shap_waterfall(self, explainer: Any, X: pd.DataFrame, 
                           feature_names: List[str], n_samples: int = 3) -> None:
        """Plot SHAP waterfall plots for sample predictions."""
        try:
            import shap
            
            for i in range(min(n_samples, len(X))):
                plt.figure(figsize=(12, 8))
                # Compute shap values for the single sample
                shap_vals = explainer.shap_values(X.iloc[i:i+1])
                try:
                    # New API: shap.Explanation
                    import numpy as np
                    base_value = explainer.expected_value
                    if isinstance(base_value, (list, np.ndarray)):
                        base_val = np.array(base_value).mean()
                    else:
                        base_val = float(base_value)
                    values = shap_vals[0] if isinstance(shap_vals, (list, tuple)) else shap_vals
                    if hasattr(values, 'values') and hasattr(values, 'base_values'):
                        exp = values
                    else:
                        exp = shap.Explanation(values=values.reshape(-1),
                                               base_values=base_val,
                                               data=X.iloc[i].values,
                                               feature_names=feature_names)
                    shap.plots.waterfall(exp, show=False)
                except Exception:
                    # Fallback for older API
                    shap.waterfall_plot(explainer.expected_value, 
                                        shap_vals[0] if isinstance(shap_vals, (list, tuple)) else shap_vals,
                                        X.iloc[i], show=False)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / f'shap_waterfall_sample_{i}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            logging.warning(f"SHAP waterfall plot failed: {str(e)}")
    
    def _plot_feature_interactions(self, interaction_values: np.ndarray, 
                                 feature_names: List[str]) -> None:
        """Plot feature interactions heatmap."""
        try:
            # Average interaction values across samples
            avg_interactions = np.abs(interaction_values).mean(axis=0)
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(avg_interactions, 
                       xticklabels=feature_names, 
                       yticklabels=feature_names,
                       annot=False, cmap='viridis')
            plt.title('Feature Interactions (SHAP)')
            plt.tight_layout()
            
            plt.savefig(self.output_dir / 'feature_interactions.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.warning(f"Feature interactions plot failed: {str(e)}")
    
    def _plot_permutation_importance(self, importance_df: pd.DataFrame, title: str) -> None:
        """Plot permutation importance with error bars."""
        plt.figure(figsize=(12, 8))
        
        top_features = importance_df.head(20)
        
        plt.barh(range(len(top_features)), top_features['importance'],
                xerr=top_features['std'], capsize=3)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plt.savefig(self.output_dir / f'permutation_importance_{title.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_ablation_results(self, ablation_df: pd.DataFrame, metric: str) -> None:
        """Plot feature ablation results."""
        plt.figure(figsize=(12, 8))
        
        plt.barh(range(len(ablation_df)), ablation_df['importance'])
        plt.yticks(range(len(ablation_df)), ablation_df['feature_group'])
        plt.xlabel(f'Performance Drop ({metric})')
        plt.title('Feature Ablation Study')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'feature_ablation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_regime_performance(self, regime_df: pd.DataFrame) -> None:
        """Plot performance by regime."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = ['mse', 'mae', 'directional_accuracy', 'correlation']
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            
            ax.bar(regime_df['regime'], regime_df[metric])
            ax.set_title(f'{metric.upper()} by Regime')
            ax.set_xlabel('Regime')
            ax.set_ylabel(metric.upper())
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'regime_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_interpretation_report(self, analyses: Dict[str, Any]) -> str:
        """
        Create comprehensive interpretation report.
        
        Args:
            analyses: Dictionary of analysis results
            
        Returns:
            Path to generated report
        """
        logging.info("Creating interpretation report")
        
        report_path = self.output_dir / 'interpretation_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Model Interpretation Report\n\n")
            f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
            
            # Feature Importance Section
            if 'feature_importance' in analyses:
                f.write("## Feature Importance\n\n")
                importance_df = analyses['feature_importance']
                f.write("Top 10 Most Important Features:\n\n")
                f.write(importance_df.head(10).to_markdown(index=False))
                f.write("\n\n")
            
            # Ablation Study Section
            if 'ablation_study' in analyses:
                f.write("## Feature Ablation Study\n\n")
                ablation_df = analyses['ablation_study']['ablation_results']
                f.write("Feature Group Importance (by performance drop):\n\n")
                f.write(ablation_df.to_markdown(index=False))
                f.write("\n\n")
            
            # Regime Analysis Section
            if 'regime_analysis' in analyses:
                f.write("## Regime Analysis\n\n")
                regime_df = analyses['regime_analysis']['regime_performance']
                f.write("Performance by Market Regime:\n\n")
                f.write(regime_df.to_markdown(index=False))
                f.write("\n\n")
            
            # Key Insights
            f.write("## Key Insights\n\n")
            f.write(self._generate_insights(analyses))
        
        logging.info(f"Interpretation report saved to: {report_path}")
        
        return str(report_path)
    
    def _generate_insights(self, analyses: Dict[str, Any]) -> str:
        """Generate key insights from analyses."""
        insights = []
        
        # Feature importance insights
        if 'feature_importance' in analyses:
            top_feature = analyses['feature_importance'].iloc[0]['feature']
            insights.append(f"- The most important feature is '{top_feature}'")
        
        # Regime insights
        if 'regime_analysis' in analyses:
            regime_df = analyses['regime_analysis']['regime_performance']
            best_regime = regime_df.loc[regime_df['directional_accuracy'].idxmax(), 'regime']
            insights.append(f"- Model performs best in '{best_regime}' regime")
        
        # Ablation insights
        if 'ablation_study' in analyses:
            ablation_df = analyses['ablation_study']['ablation_results']
            critical_group = ablation_df.iloc[0]['feature_group']
            insights.append(f"- Most critical feature group is '{critical_group}'")
        
        return "\n".join(insights) if insights else "No specific insights generated."


def main():
    """
    Main function for testing interpretability module.
    """
    # Load configuration
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.utils.helpers import load_config, setup_logging
    
    # Setup
    config = load_config()
    setup_logging()
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features),
                    columns=[f'feature_{i}' for i in range(n_features)])
    
    # Create target with known relationships
    y = (0.5 * X['feature_0'] + 0.3 * X['feature_1'] + 
         0.2 * X['feature_2'] + np.random.randn(n_samples) * 0.1)
    
    # Train simple model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Initialize interpreter
    interpreter = ModelInterpreter(config)
    
    # Analyze model
    tree_analysis = interpreter.analyze_tree_model(model, X, y)
    
    # Feature ablation
    feature_groups = {
        'group_1': ['feature_0', 'feature_1'],
        'group_2': ['feature_2', 'feature_3'],
        'group_3': ['feature_4', 'feature_5']
    }
    
    ablation_analysis = interpreter.feature_ablation_study(
        model, X, y, feature_groups
    )
    
    # Regime analysis
    predictions = model.predict(X)
    regime_analysis = interpreter.regime_analysis(model, X, y, predictions)
    
    # Create report
    analyses = {
        'feature_importance': tree_analysis.get('feature_importance'),
        'ablation_study': ablation_analysis,
        'regime_analysis': regime_analysis
    }
    
    report_path = interpreter.create_interpretation_report(analyses)
    
    print(f"Interpretation analysis completed!")
    print(f"Report saved to: {report_path}")
    print(f"Plots saved to: {interpreter.output_dir}")


if __name__ == "__main__":
    main()