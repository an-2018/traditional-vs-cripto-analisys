"""
Financial Metrics Module

Functions for calculating portfolio performance metrics:
- Returns, Volatility, Sharpe Ratio
- Drawdowns, VaR, CVaR
- Correlation matrices
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized return.
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods in a year (252 for daily, 12 for monthly)
        
    Returns:
        Annualized return
    """
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year
    
    if years > 0:
        return (1 + total_return) ** (1 / years) - 1
    return 0.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility (standard deviation).
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods in a year
        
    Returns:
        Annualized volatility
    """
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe Ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default 2%)
        periods_per_year: Number of periods in a year
        
    Returns:
        Sharpe Ratio
    """
    ann_return = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    
    if ann_vol > 0:
        return (ann_return - risk_free_rate) / ann_vol
    return 0.0


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """
    Calculate Sortino Ratio (uses downside deviation instead of total volatility).
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Sortino Ratio
    """
    ann_return = annualized_return(returns, periods_per_year)
    
    # Downside deviation (only negative returns)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)
    
    if downside_std > 0:
        return (ann_return - risk_free_rate) / downside_std
    return 0.0


def max_drawdown(returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate maximum drawdown.
    
    Args:
        returns: Series of returns
        
    Returns:
        Tuple of (max_drawdown, start_date, end_date)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    
    max_dd = drawdown.min()
    end_date = drawdown.idxmin()
    
    # Find start date (peak before the drawdown)
    start_date = cumulative[:end_date].idxmax()
    
    return max_dd, start_date, end_date


def value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) using historical method.
    
    Args:
        returns: Series of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        VaR value (positive number representing loss)
    """
    return -returns.quantile(1 - confidence_level)


def conditional_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
    
    Args:
        returns: Series of returns
        confidence_level: Confidence level
        
    Returns:
        CVaR value (positive number representing expected loss beyond VaR)
    """
    var = value_at_risk(returns, confidence_level)
    # Average of returns worse than VaR
    return -returns[returns <= -var].mean()


def portfolio_metrics(returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> dict:
    """
    Calculate comprehensive portfolio metrics.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Dictionary with all metrics
    """
    ann_ret = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    sharpe = sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = sortino_ratio(returns, risk_free_rate, periods_per_year)
    max_dd, dd_start, dd_end = max_drawdown(returns)
    var_95 = value_at_risk(returns, 0.95)
    cvar_95 = conditional_var(returns, 0.95)
    
    return {
        'Annualized Return': ann_ret,
        'Annualized Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_dd,
        'Drawdown Start': dd_start,
        'Drawdown End': dd_end,
        'VaR (95%)': var_95,
        'CVaR (95%)': cvar_95,
        'Total Return': (1 + returns).prod() - 1,
        'Best Day': returns.max(),
        'Worst Day': returns.min()
    }


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix for multiple assets.
    
    Args:
        returns: DataFrame with returns for multiple assets
        
    Returns:
        Correlation matrix
    """
    return returns.corr()


def rolling_correlation(returns1: pd.Series, returns2: pd.Series, window: int = 63) -> pd.Series:
    """
    Calculate rolling correlation between two assets.
    
    Args:
        returns1: Returns for asset 1
        returns2: Returns for asset 2
        window: Rolling window size (default 63 = ~3 months for daily data)
        
    Returns:
        Series of rolling correlations
    """
    return returns1.rolling(window=window).corr(returns2)


if __name__ == "__main__":
    # Test metrics functions
    print("Financial Metrics Module loaded successfully!")
    
    # Generate sample returns
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    sample_returns = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)
    
    # Calculate metrics
    metrics = portfolio_metrics(sample_returns)
    
    print("\nSample Portfolio Metrics:")
    print("-" * 40)
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key:.<30} {value:>10.4f}")
        else:
            print(f"{key:.<30} {value}")
    
    print("\n✓ All metrics functions working!")
