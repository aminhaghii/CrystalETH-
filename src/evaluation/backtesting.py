"""
Economic backtesting module for ETH forecasting project.

This module implements economic backtesting following Rule #22:
- Include transaction costs (0.05% default)
- Include slippage (0.05% default)
- Implement realistic entry/exit logic
- Calculate risk-adjusted metrics (Sharpe, max drawdown)
- Compare against buy-and-hold baseline

Provides comprehensive economic evaluation of trading strategies.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class EconomicBacktester:
    """
    Economic backtesting class for evaluating trading strategies.
    
    Implements realistic trading simulation with transaction costs,
    slippage, and comprehensive performance metrics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize EconomicBacktester.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.backtesting_config = config.get('backtesting', {})
        
        # Trading parameters
        self.transaction_cost = self.backtesting_config.get('transaction_cost', 0.0005)  # 0.05%
        self.slippage = self.backtesting_config.get('slippage', 0.0005)  # 0.05%
        self.initial_capital = self.backtesting_config.get('initial_capital', 100000)
        self.max_position_size = self.backtesting_config.get('max_position_size', 1.0)
        
        # Risk management
        self.stop_loss = self.backtesting_config.get('stop_loss', None)
        self.take_profit = self.backtesting_config.get('take_profit', None)
        self.max_drawdown_limit = self.backtesting_config.get('max_drawdown_limit', None)
        
        # Output paths
        self.output_dir = Path(config.get('paths', {}).get('reports', 'reports')) / 'backtesting'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualization settings
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        logging.info("EconomicBacktester initialized")
    
    def run_backtest(self, predictions: pd.Series, returns: pd.Series,
                    prices: pd.Series = None, strategy_type: str = 'threshold') -> Dict[str, Any]:
        """
        Run comprehensive backtest.
        
        Args:
            predictions: Model predictions (log returns or probabilities)
            returns: Actual returns
            prices: Price series (optional, for position sizing)
            strategy_type: Type of strategy ('threshold', 'quantile', 'kelly')
            
        Returns:
            Dictionary with backtest results
        """
        logging.info(f"Running backtest with {strategy_type} strategy")
        
        # Align data
        common_index = predictions.index.intersection(returns.index)
        predictions = predictions.loc[common_index]
        returns = returns.loc[common_index]
        
        if prices is not None:
            prices = prices.loc[common_index]
        
        # Generate trading signals
        signals = self._generate_signals(predictions, strategy_type)
        
        # Calculate positions
        positions = self._calculate_positions(signals, returns, prices)
        
        # Simulate trading
        portfolio_returns, trades = self._simulate_trading(positions, returns)
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(portfolio_returns, returns)
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(portfolio_returns, positions)
        
        # Trade analysis
        trade_analysis = self._analyze_trades(trades)
        
        # Benchmark comparison
        benchmark_metrics = self._calculate_benchmark_metrics(returns)
        
        results = {
            'portfolio_returns': portfolio_returns,
            'positions': positions,
            'trades': trades,
            'signals': signals,
            'performance_metrics': metrics,
            'risk_metrics': risk_metrics,
            'trade_analysis': trade_analysis,
            'benchmark_metrics': benchmark_metrics,
            'strategy_type': strategy_type
        }
        
        # Generate plots
        self._create_backtest_plots(results, returns)
        
        logging.info("Backtest completed")
        
        return results
    
    def _generate_signals(self, predictions: pd.Series, strategy_type: str) -> pd.Series:
        """
        Generate trading signals from predictions.
        
        Args:
            predictions: Model predictions
            strategy_type: Strategy type
            
        Returns:
            Trading signals (-1, 0, 1)
        """
        signals = pd.Series(0, index=predictions.index)
        
        if strategy_type == 'threshold':
            # Simple threshold strategy
            threshold = self.backtesting_config.get('signal_threshold', 0.001)
            signals[predictions > threshold] = 1
            signals[predictions < -threshold] = -1
            
        elif strategy_type == 'quantile':
            # Quantile-based strategy
            upper_quantile = self.backtesting_config.get('upper_quantile', 0.7)
            lower_quantile = self.backtesting_config.get('lower_quantile', 0.3)
            
            upper_threshold = predictions.quantile(upper_quantile)
            lower_threshold = predictions.quantile(lower_quantile)
            
            signals[predictions > upper_threshold] = 1
            signals[predictions < lower_threshold] = -1
            
        elif strategy_type == 'kelly':
            # Kelly criterion-based position sizing
            # Simplified implementation
            win_rate = 0.55  # Assumed win rate
            avg_win = 0.02   # Assumed average win
            avg_loss = 0.015 # Assumed average loss
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = np.clip(kelly_fraction, 0, 0.25)  # Cap at 25%
            
            signals = np.sign(predictions) * kelly_fraction
            
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        return signals
    
    def _calculate_positions(self, signals: pd.Series, returns: pd.Series,
                           prices: pd.Series = None) -> pd.Series:
        """
        Calculate position sizes from signals.
        
        Args:
            signals: Trading signals
            returns: Return series
            prices: Price series
            
        Returns:
            Position sizes
        """
        positions = pd.Series(0.0, index=signals.index)
        
        # Simple position sizing (can be enhanced)
        for i, (date, signal) in enumerate(signals.items()):
            if i == 0:
                positions.iloc[i] = signal * self.max_position_size
            else:
                # Consider position limits and risk management
                new_position = signal * self.max_position_size
                
                # Apply position limits
                new_position = np.clip(new_position, -self.max_position_size, self.max_position_size)
                
                positions.iloc[i] = new_position
        
        return positions
    
    def _simulate_trading(self, positions: pd.Series, returns: pd.Series) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Simulate trading with transaction costs and slippage.
        
        Args:
            positions: Position sizes
            returns: Return series
            
        Returns:
            Portfolio returns and trade log
        """
        portfolio_returns = pd.Series(0.0, index=positions.index)
        trades = []
        
        current_position = 0.0
        cash = self.initial_capital
        portfolio_value = self.initial_capital
        
        for i, (date, target_position) in enumerate(positions.items()):
            if i == 0:
                current_position = target_position
                continue
            
            # Calculate position change
            position_change = target_position - current_position
            
            # Calculate trading costs
            trading_cost = 0.0
            if abs(position_change) > 1e-6:  # Only if there's a meaningful change
                trading_cost = abs(position_change) * portfolio_value * (self.transaction_cost + self.slippage)
            
            # Calculate return for this period
            period_return = returns.iloc[i]
            
            # Portfolio return (before costs)
            gross_return = current_position * period_return
            
            # Net return (after costs)
            net_return = gross_return - trading_cost / portfolio_value
            
            portfolio_returns.iloc[i] = net_return
            
            # Update portfolio value
            portfolio_value *= (1 + net_return)
            
            # Log trade if position changed
            if abs(position_change) > 1e-6:
                trades.append({
                    'date': date,
                    'position_change': position_change,
                    'new_position': target_position,
                    'trading_cost': trading_cost,
                    'portfolio_value': portfolio_value
                })
            
            # Update current position
            current_position = target_position
        
        trades_df = pd.DataFrame(trades)
        
        return portfolio_returns, trades_df
    
    def _calculate_performance_metrics(self, portfolio_returns: pd.Series,
                                     benchmark_returns: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            
        Returns:
            Dictionary of performance metrics
        """
        # Basic metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Benchmark comparison
        benchmark_total_return = (1 + benchmark_returns).prod() - 1
        benchmark_annualized_return = (1 + benchmark_total_return) ** (252 / len(benchmark_returns)) - 1
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
        benchmark_sharpe = benchmark_annualized_return / benchmark_volatility if benchmark_volatility > 0 else 0
        
        excess_return = annualized_return - benchmark_annualized_return
        information_ratio = excess_return / (portfolio_returns - benchmark_returns).std() / np.sqrt(252) if (portfolio_returns - benchmark_returns).std() > 0 else 0
        
        # Drawdown metrics
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        win_rate = (portfolio_returns > 0).mean()
        
        # Average win/loss
        wins = portfolio_returns[portfolio_returns > 0]
        losses = portfolio_returns[portfolio_returns < 0]
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        # Profit factor
        total_wins = wins.sum() if len(wins) > 0 else 0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'information_ratio': information_ratio,
            'excess_return': excess_return,
            'benchmark_return': benchmark_annualized_return,
            'benchmark_sharpe': benchmark_sharpe
        }
    
    def _calculate_risk_metrics(self, portfolio_returns: pd.Series,
                              positions: pd.Series) -> Dict[str, float]:
        """
        Calculate risk metrics.
        
        Args:
            portfolio_returns: Portfolio return series
            positions: Position series
            
        Returns:
            Dictionary of risk metrics
        """
        # Value at Risk (VaR)
        var_95 = portfolio_returns.quantile(0.05)
        var_99 = portfolio_returns.quantile(0.01)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()
        
        # Maximum position size
        max_position = positions.abs().max()
        avg_position = positions.abs().mean()
        
        # Skewness and Kurtosis
        skewness = portfolio_returns.skew()
        kurtosis = portfolio_returns.kurtosis()
        
        # Sortino ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (portfolio_returns.mean() * 252) / downside_deviation if downside_deviation > 0 else 0
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'max_position': max_position,
            'avg_position': avg_position,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'sortino_ratio': sortino_ratio
        }
    
    def _analyze_trades(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze trading activity.
        
        Args:
            trades_df: DataFrame of trades
            
        Returns:
            Dictionary of trade analysis
        """
        if len(trades_df) == 0:
            return {'num_trades': 0}
        
        # Basic trade statistics
        num_trades = len(trades_df)
        avg_trade_size = trades_df['position_change'].abs().mean()
        total_trading_costs = trades_df['trading_cost'].sum()
        
        # Trade frequency
        if len(trades_df) > 1:
            trade_dates = pd.to_datetime(trades_df['date'])
            avg_days_between_trades = (trade_dates.diff().dt.days.mean())
        else:
            avg_days_between_trades = np.nan
        
        return {
            'num_trades': num_trades,
            'avg_trade_size': avg_trade_size,
            'total_trading_costs': total_trading_costs,
            'avg_days_between_trades': avg_days_between_trades,
            'trading_cost_ratio': total_trading_costs / self.initial_capital
        }
    
    def _calculate_benchmark_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate buy-and-hold benchmark metrics.
        
        Args:
            returns: Return series
            
        Returns:
            Dictionary of benchmark metrics
        """
        # Buy and hold strategy
        bh_total_return = (1 + returns).prod() - 1
        bh_annualized_return = (1 + bh_total_return) ** (252 / len(returns)) - 1
        bh_volatility = returns.std() * np.sqrt(252)
        bh_sharpe = bh_annualized_return / bh_volatility if bh_volatility > 0 else 0
        
        # Buy and hold drawdown
        bh_cumulative = (1 + returns).cumprod()
        bh_running_max = bh_cumulative.expanding().max()
        bh_drawdown = (bh_cumulative - bh_running_max) / bh_running_max
        bh_max_drawdown = bh_drawdown.min()
        
        return {
            'bh_total_return': bh_total_return,
            'bh_annualized_return': bh_annualized_return,
            'bh_volatility': bh_volatility,
            'bh_sharpe_ratio': bh_sharpe,
            'bh_max_drawdown': bh_max_drawdown
        }
    
    def _create_backtest_plots(self, results: Dict[str, Any], benchmark_returns: pd.Series) -> None:
        """
        Create comprehensive backtest visualization plots.
        
        Args:
            results: Backtest results
            benchmark_returns: Benchmark return series
        """
        portfolio_returns = results['portfolio_returns']
        positions = results['positions']
        
        # 1. Cumulative returns comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Cumulative returns
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        
        axes[0, 0].plot(portfolio_cumulative.index, portfolio_cumulative.values, 
                       label='Strategy', linewidth=2)
        axes[0, 0].plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                       label='Buy & Hold', linewidth=2)
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Drawdown
        portfolio_running_max = portfolio_cumulative.expanding().max()
        portfolio_drawdown = (portfolio_cumulative - portfolio_running_max) / portfolio_running_max
        
        benchmark_running_max = benchmark_cumulative.expanding().max()
        benchmark_drawdown = (benchmark_cumulative - benchmark_running_max) / benchmark_running_max
        
        axes[0, 1].fill_between(portfolio_drawdown.index, portfolio_drawdown.values, 0, 
                               alpha=0.7, label='Strategy')
        axes[0, 1].fill_between(benchmark_drawdown.index, benchmark_drawdown.values, 0, 
                               alpha=0.7, label='Buy & Hold')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Position sizes
        axes[1, 0].plot(positions.index, positions.values, linewidth=1)
        axes[1, 0].set_title('Position Sizes')
        axes[1, 0].set_ylabel('Position Size')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio
        rolling_window = 60  # 60-day rolling window
        if len(portfolio_returns) > rolling_window:
            portfolio_rolling_sharpe = portfolio_returns.rolling(rolling_window).mean() / portfolio_returns.rolling(rolling_window).std() * np.sqrt(252)
            benchmark_rolling_sharpe = benchmark_returns.rolling(rolling_window).mean() / benchmark_returns.rolling(rolling_window).std() * np.sqrt(252)
            
            axes[1, 1].plot(portfolio_rolling_sharpe.index, portfolio_rolling_sharpe.values, 
                           label='Strategy', linewidth=2)
            axes[1, 1].plot(benchmark_rolling_sharpe.index, benchmark_rolling_sharpe.values, 
                           label='Buy & Hold', linewidth=2)
            axes[1, 1].set_title(f'{rolling_window}-Day Rolling Sharpe Ratio')
            axes[1, 1].set_ylabel('Sharpe Ratio')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'backtest_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Return distribution comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        axes[0].hist(portfolio_returns.dropna(), bins=50, alpha=0.7, label='Strategy', density=True)
        axes[0].hist(benchmark_returns.dropna(), bins=50, alpha=0.7, label='Buy & Hold', density=True)
        axes[0].set_title('Return Distribution')
        axes[0].set_xlabel('Daily Returns')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        portfolio_sorted = np.sort(portfolio_returns.dropna())
        benchmark_sorted = np.sort(benchmark_returns.dropna())
        
        # Align lengths
        min_len = min(len(portfolio_sorted), len(benchmark_sorted))
        portfolio_sorted = portfolio_sorted[:min_len]
        benchmark_sorted = benchmark_sorted[:min_len]
        
        axes[1].scatter(benchmark_sorted, portfolio_sorted, alpha=0.6)
        axes[1].plot([benchmark_sorted.min(), benchmark_sorted.max()], 
                    [benchmark_sorted.min(), benchmark_sorted.max()], 'r--', linewidth=2)
        axes[1].set_title('Q-Q Plot: Strategy vs Buy & Hold')
        axes[1].set_xlabel('Buy & Hold Returns')
        axes[1].set_ylabel('Strategy Returns')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'return_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Risk-Return scatter
        plt.figure(figsize=(10, 8))
        
        # Calculate rolling metrics for scatter plot
        window = 60
        if len(portfolio_returns) > window:
            portfolio_rolling_return = portfolio_returns.rolling(window).mean() * 252
            portfolio_rolling_vol = portfolio_returns.rolling(window).std() * np.sqrt(252)
            
            benchmark_rolling_return = benchmark_returns.rolling(window).mean() * 252
            benchmark_rolling_vol = benchmark_returns.rolling(window).std() * np.sqrt(252)
            
            plt.scatter(portfolio_rolling_vol, portfolio_rolling_return, 
                       alpha=0.6, label='Strategy', s=20)
            plt.scatter(benchmark_rolling_vol, benchmark_rolling_return, 
                       alpha=0.6, label='Buy & Hold', s=20)
            
            plt.xlabel('Annualized Volatility')
            plt.ylabel('Annualized Return')
            plt.title(f'Risk-Return Profile ({window}-Day Rolling)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(self.output_dir / 'risk_return_scatter.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_backtest_report(self, results: Dict[str, Any]) -> str:
        """
        Create comprehensive backtest report.
        
        Args:
            results: Backtest results
            
        Returns:
            Path to generated report
        """
        logging.info("Creating backtest report")
        
        report_path = self.output_dir / 'backtest_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Economic Backtest Report\n\n")
            f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
            f.write(f"Strategy Type: {results['strategy_type']}\n\n")
            
            # Performance Summary
            f.write("## Performance Summary\n\n")
            metrics = results['performance_metrics']
            
            f.write("### Key Metrics\n\n")
            f.write(f"- **Total Return**: {metrics['total_return']:.2%}\n")
            f.write(f"- **Annualized Return**: {metrics['annualized_return']:.2%}\n")
            f.write(f"- **Volatility**: {metrics['volatility']:.2%}\n")
            f.write(f"- **Sharpe Ratio**: {metrics['sharpe_ratio']:.3f}\n")
            f.write(f"- **Maximum Drawdown**: {metrics['max_drawdown']:.2%}\n")
            f.write(f"- **Calmar Ratio**: {metrics['calmar_ratio']:.3f}\n")
            f.write(f"- **Win Rate**: {metrics['win_rate']:.2%}\n")
            f.write(f"- **Profit Factor**: {metrics['profit_factor']:.3f}\n\n")
            
            # Benchmark Comparison
            f.write("### Benchmark Comparison\n\n")
            f.write(f"- **Excess Return**: {metrics['excess_return']:.2%}\n")
            f.write(f"- **Information Ratio**: {metrics['information_ratio']:.3f}\n")
            f.write(f"- **Benchmark Return**: {metrics['benchmark_return']:.2%}\n")
            f.write(f"- **Benchmark Sharpe**: {metrics['benchmark_sharpe']:.3f}\n\n")
            
            # Risk Metrics
            f.write("## Risk Analysis\n\n")
            risk_metrics = results['risk_metrics']
            
            f.write("### Value at Risk\n\n")
            f.write(f"- **VaR (95%)**: {risk_metrics['var_95']:.2%}\n")
            f.write(f"- **VaR (99%)**: {risk_metrics['var_99']:.2%}\n")
            f.write(f"- **CVaR (95%)**: {risk_metrics['cvar_95']:.2%}\n")
            f.write(f"- **CVaR (99%)**: {risk_metrics['cvar_99']:.2%}\n\n")
            
            f.write("### Distribution Metrics\n\n")
            f.write(f"- **Skewness**: {risk_metrics['skewness']:.3f}\n")
            f.write(f"- **Kurtosis**: {risk_metrics['kurtosis']:.3f}\n")
            f.write(f"- **Sortino Ratio**: {risk_metrics['sortino_ratio']:.3f}\n\n")
            
            # Trading Analysis
            f.write("## Trading Analysis\n\n")
            trade_analysis = results['trade_analysis']
            
            if trade_analysis['num_trades'] > 0:
                f.write(f"- **Number of Trades**: {trade_analysis['num_trades']}\n")
                f.write(f"- **Average Trade Size**: {trade_analysis['avg_trade_size']:.3f}\n")
                f.write(f"- **Total Trading Costs**: ${trade_analysis['total_trading_costs']:.2f}\n")
                f.write(f"- **Trading Cost Ratio**: {trade_analysis['trading_cost_ratio']:.2%}\n")
                if not np.isnan(trade_analysis['avg_days_between_trades']):
                    f.write(f"- **Average Days Between Trades**: {trade_analysis['avg_days_between_trades']:.1f}\n")
            else:
                f.write("- No trades executed during backtest period\n")
            
            f.write("\n")
            
            # Configuration
            f.write("## Configuration\n\n")
            f.write(f"- **Transaction Cost**: {self.transaction_cost:.2%}\n")
            f.write(f"- **Slippage**: {self.slippage:.2%}\n")
            f.write(f"- **Initial Capital**: ${self.initial_capital:,.2f}\n")
            f.write(f"- **Max Position Size**: {self.max_position_size:.2%}\n\n")
            
            # Conclusions
            f.write("## Conclusions\n\n")
            f.write(self._generate_conclusions(results))
        
        logging.info(f"Backtest report saved to: {report_path}")
        
        return str(report_path)
    
    def _generate_conclusions(self, results: Dict[str, Any]) -> str:
        """Generate conclusions from backtest results."""
        conclusions = []
        
        metrics = results['performance_metrics']
        
        # Performance assessment
        if metrics['sharpe_ratio'] > 1.0:
            conclusions.append("- Strategy shows strong risk-adjusted performance (Sharpe > 1.0)")
        elif metrics['sharpe_ratio'] > 0.5:
            conclusions.append("- Strategy shows moderate risk-adjusted performance")
        else:
            conclusions.append("- Strategy shows weak risk-adjusted performance")
        
        # Benchmark comparison
        if metrics['excess_return'] > 0:
            conclusions.append("- Strategy outperforms buy-and-hold benchmark")
        else:
            conclusions.append("- Strategy underperforms buy-and-hold benchmark")
        
        # Risk assessment
        if abs(metrics['max_drawdown']) < 0.1:
            conclusions.append("- Strategy maintains low drawdown risk")
        elif abs(metrics['max_drawdown']) < 0.2:
            conclusions.append("- Strategy has moderate drawdown risk")
        else:
            conclusions.append("- Strategy has high drawdown risk")
        
        # Trading efficiency
        trade_analysis = results['trade_analysis']
        if trade_analysis['num_trades'] > 0:
            if trade_analysis['trading_cost_ratio'] < 0.01:
                conclusions.append("- Trading costs are well-controlled")
            else:
                conclusions.append("- Trading costs may be impacting performance")
        
        return "\n".join(conclusions) if conclusions else "No specific conclusions generated."


def main():
    """
    Main function for testing backtesting module.
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
    n_days = 1000
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    # Synthetic returns with some predictable patterns
    returns = np.random.randn(n_days) * 0.02
    trend = np.sin(np.arange(n_days) * 2 * np.pi / 252) * 0.001
    returns += trend
    
    returns = pd.Series(returns, index=dates)
    
    # Synthetic predictions (with some skill)
    predictions = returns.shift(-1) + np.random.randn(n_days) * 0.01
    predictions = predictions.dropna()
    
    # Align data
    common_index = returns.index.intersection(predictions.index)
    returns = returns.loc[common_index]
    predictions = predictions.loc[common_index]
    
    # Initialize backtester
    backtester = EconomicBacktester(config)
    
    # Run backtest
    results = backtester.run_backtest(predictions, returns, strategy_type='threshold')
    
    # Create report
    report_path = backtester.create_backtest_report(results)
    
    print(f"Backtest completed!")
    print(f"Report saved to: {report_path}")
    print(f"Plots saved to: {backtester.output_dir}")
    
    # Print key metrics
    metrics = results['performance_metrics']
    print(f"\nKey Results:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")


if __name__ == "__main__":
    main()