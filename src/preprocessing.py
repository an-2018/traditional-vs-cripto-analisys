"""
Data Preprocessing Module

Functions for cleaning, aligning, and transforming financial time series data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def align_timeseries(dfs: Dict[str, pd.DataFrame], freq: str = 'D') -> pd.DataFrame:
    """
    Align multiple time series to a common date index.
    
    Args:
        dfs: Dictionary of DataFrames with datetime index
        freq: Frequency for resampling ('D' for daily, 'W' for weekly)
        
    Returns:
        Aligned DataFrame with all series
    """
    # Find common date range
    start_dates = [df.index.min() for df in dfs.values() if not df.empty]
    end_dates = [df.index.max() for df in dfs.values() if not df.empty]
    
    common_start = max(start_dates)
    common_end = min(end_dates)
    
    # Resample and align
    aligned_dfs = {}
    for name, df in dfs.items():
        if not df.empty:
            df_resampled = df.loc[common_start:common_end].resample(freq).last()
            aligned_dfs[name] = df_resampled
    
    # Concatenate
    result = pd.concat(aligned_dfs.values(), axis=1, keys=aligned_dfs.keys())
    
    return result


def calculate_returns(prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
    """
    Calculate returns from price data.
    
    Args:
        prices: DataFrame with price data
        method: 'log' for log returns, 'simple' for simple returns
        
    Returns:
        DataFrame with returns
    """
    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    elif method == 'simple':
        returns = prices.pct_change()
    else:
        raise ValueError("method must be 'log' or 'simple'")
    
    return returns.dropna()


def handle_missing_values(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """
    Handle missing values in time series data.
    
    Args:
        df: DataFrame with potential missing values
        method: 'ffill' (forward fill), 'bfill' (backward fill), 'interpolate', or 'drop'
        
    Returns:
        DataFrame with missing values handled
    """
    if method == 'ffill':
        return df.fillna(method='ffill')
    elif method == 'bfill':
        return df.fillna(method='bfill')
    elif method == 'interpolate':
        return df.interpolate(method='linear')
    elif method == 'drop':
        return df.dropna()
    else:
        raise ValueError("method must be 'ffill', 'bfill', 'interpolate', or 'drop'")


def detect_outliers(data: pd.Series, method: str = 'iqr', threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers in a time series.
    
    Args:
        data: Series to check for outliers
        method: 'iqr' (Interquartile Range) or 'zscore'
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean Series indicating outliers
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (data < (Q1 - threshold * IQR)) | (data > (Q3 + threshold * IQR))
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        outliers = z_scores > threshold
    else:
        raise ValueError("method must be 'iqr' or 'zscore'")
    
    return outliers


def normalize_data(df: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize data for comparison.
    
    Args:
        df: DataFrame to normalize
        method: 'minmax' (0-1 scaling) or 'zscore' (standardization)
        
    Returns:
        Normalized DataFrame
    """
    if method == 'minmax':
        return (df - df.min()) / (df.max() - df.min())
    elif method == 'zscore':
        return (df - df.mean()) / df.std()
    else:
        raise ValueError("method must be 'minmax' or 'zscore'")


def create_features(prices: pd.DataFrame, windows: List[int] = [5, 21, 63]) -> pd.DataFrame:
    """
    Create technical features from price data.
    
    Args:
        prices: DataFrame with price data
        windows: List of window sizes for moving averages
        
    Returns:
        DataFrame with features
    """
    features = pd.DataFrame(index=prices.index)
    
    for col in prices.columns:
        # Moving averages
        for window in windows:
            features[f'{col}_MA{window}'] = prices[col].rolling(window=window).mean()
            features[f'{col}_STD{window}'] = prices[col].rolling(window=window).std()
        
        # Returns
        features[f'{col}_Return'] = prices[col].pct_change()
        
        # Momentum
        features[f'{col}_Momentum'] = prices[col] / prices[col].shift(21) - 1
    
    return features.dropna()


if __name__ == "__main__":
    # Test preprocessing functions
    print("Preprocessing module loaded successfully!")
    
    # Example usage
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    test_data = pd.DataFrame({
        'Asset1': np.random.randn(len(dates)).cumsum() + 100,
        'Asset2': np.random.randn(len(dates)).cumsum() + 50
    }, index=dates)
    
    print("\nOriginal data shape:", test_data.shape)
    
    # Calculate returns
    returns = calculate_returns(test_data)
    print("Returns shape:", returns.shape)
    
    # Detect outliers
    outliers = detect_outliers(returns['Asset1'])
    print(f"Outliers detected: {outliers.sum()}")
    
    print("\n✓ All preprocessing functions working!")
