# %% [markdown]
# # Portfolio Analysis Pipeline: Traditional vs Crypto Assets
# 
# **Comparative Analysis of Traditional and Cryptocurrency Portfolio Management**
# 
# This notebook implements a complete financial analysis pipeline, from data gathering 
# to portfolio management theory application. Each section explains the theory and 
# shows the Python implementation.
# 
# ## Learning Objectives
# - Understand how to collect and preprocess financial data
# - Calculate key portfolio metrics (Sharpe, Sortino, VaR, CVaR)
# - Apply Modern Portfolio Theory (Markowitz) to portfolio construction
# - Compare traditional portfolios with crypto-enhanced portfolios
# 
# ## Time Range Configuration
# This notebook is designed to be easily replicated for different time ranges.

# %%
# =============================================================================
# SECTION 1: CONFIGURATION & IMPORTS
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

# Core libraries
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Data fetching
import yfinance as yf

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Optimization (for portfolio construction)
from scipy.optimize import minimize

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("✓ Libraries imported successfully!")

# %%
# =============================================================================
# CONFIGURATION: Easy to modify for different time ranges
# =============================================================================

# Choose your time range: '6_months', '1_year', '2_years', '5_years'
SELECTED_RANGE = '6_months'

# Time range definitions
TIME_RANGES = {
    '6_months': {
        'start': '2024-06-01',
        'end': '2024-12-01',
        'description': '6 Month Analysis'
    },
    '1_year': {
        'start': '2023-12-01',
        'end': '2024-12-01',
        'description': '1 Year Analysis'
    },
    '2_years': {
        'start': '2022-12-01',
        'end': '2024-12-01',
        'description': '2 Year Analysis'
    },
    '5_years': {
        'start': '2019-12-01',
        'end': '2024-12-01',
        'description': '5 Year Analysis'
    }
}

# Get selected range
config = TIME_RANGES[SELECTED_RANGE]
START_DATE = config['start']
END_DATE = config['end']

print(f"📊 Analysis Period: {config['description']}")
print(f"   From: {START_DATE} to {END_DATE}")

# %%
# Asset definitions
ASSETS = {
    'stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ'],
    'crypto': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD'],
    'bonds_etfs': ['AGG', 'TLT', 'SPY']
}

# Risk-free rate (annual) - US Treasury proxy
RISK_FREE_RATE = 0.05  # 5% (2024 rates)

# Trading days per year
TRADING_DAYS = 252

print("\n📈 Assets to analyze:")
for category, tickers in ASSETS.items():
    print(f"   {category.upper()}: {', '.join(tickers)}")

# %% [markdown]
# ---
# # SECTION 2: DATA GATHERING
# 
# ## Theory
# Financial analysis starts with reliable data. We use Yahoo Finance as our data source
# because it provides:
# - **OHLCV data**: Open, High, Low, Close, Volume
# - **Adjusted prices**: Accounts for splits and dividends
# - **Free access**: No API key required
# 
# ## Implementation
# We'll fetch data for all assets and save to parquet for faster reloads.

# %%
def fetch_market_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical market data from Yahoo Finance.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with OHLCV data for all tickers
    """
    print(f"Fetching data for {len(tickers)} tickers...")
    
    # Download data - yfinance handles multi-ticker downloads
    data = yf.download(
        tickers, 
        start=start_date, 
        end=end_date, 
        group_by='ticker',
        auto_adjust=True,  # Adjust for splits/dividends
        progress=False
    )
    
    print(f"✓ Fetched {len(data)} trading days of data")
    return data

# %%
# Fetch all data
all_tickers = ASSETS['stocks'] + ASSETS['crypto'] + ASSETS['bonds_etfs']
raw_data = fetch_market_data(all_tickers, START_DATE, END_DATE)

# Display sample
print("\n📊 Raw data shape:", raw_data.shape)
print("\nSample data (first 5 rows):")
raw_data.head()

# %%
# Extract closing prices for each ticker
def extract_close_prices(data: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Extract closing prices for specified tickers."""
    prices = pd.DataFrame()
    
    for ticker in tickers:
        try:
            if ticker in data.columns.get_level_values(0):
                prices[ticker] = data[ticker]['Close']
            else:
                print(f"Warning: {ticker} not found in data")
        except Exception as e:
            print(f"Error extracting {ticker}: {e}")
    
    return prices

# Get close prices
prices = extract_close_prices(raw_data, all_tickers)
print(f"\n✓ Extracted closing prices for {len(prices.columns)} assets")
prices.head()

# %% [markdown]
# ---
# # SECTION 3: DATA PREPROCESSING
# 
# ## Theory
# Clean data is essential for accurate analysis. Common preprocessing steps:
# 
# 1. **Handle Missing Values**: Financial data may have gaps (holidays, trading halts)
#    - Forward fill (`ffill`): Use last known value
#    - This is standard for price data
# 
# 2. **Calculate Returns**: We use **log returns** because:
#    - Additive over time: $r_{t_1 \to t_n} = \sum r_t$
#    - Better statistical properties
#    - Formula: $r_t = \ln(P_t / P_{t-1})$
# 
# 3. **Detect Outliers**: Identify extreme values using IQR method

# %%
# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================

def handle_missing_values(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """
    Handle missing values in time series data.
    
    Args:
        df: DataFrame with potential missing values
        method: 'ffill' (forward fill), 'bfill' (backward fill), 
                'interpolate', or 'drop'
    
    Returns:
        DataFrame with missing values handled
    """
    missing_before = df.isnull().sum().sum()
    
    if method == 'ffill':
        df_clean = df.ffill()
    elif method == 'bfill':
        df_clean = df.bfill()
    elif method == 'interpolate':
        df_clean = df.interpolate(method='linear')
    elif method == 'drop':
        df_clean = df.dropna()
    else:
        raise ValueError("method must be 'ffill', 'bfill', 'interpolate', or 'drop'")
    
    # Also backward fill any remaining NaN at the start
    df_clean = df_clean.bfill()
    
    missing_after = df_clean.isnull().sum().sum()
    print(f"Missing values: {missing_before} → {missing_after}")
    
    return df_clean


def calculate_returns(prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
    """
    Calculate returns from price data.
    
    Theory:
    - Log returns: r = ln(P_t / P_{t-1})
      Advantage: Additive over time, better for modeling
    - Simple returns: r = (P_t - P_{t-1}) / P_{t-1}
      Advantage: Easier to interpret
    
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


def detect_outliers(data: pd.Series, method: str = 'iqr', 
                    threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers in a time series using IQR or Z-score method.
    
    Theory:
    - IQR Method: Outlier if x < Q1 - k*IQR or x > Q3 + k*IQR
    - Z-score: Outlier if |z| > threshold
    
    Args:
        data: Series to check for outliers
        method: 'iqr' or 'zscore'
        threshold: Multiplier for detection
    
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

# %%
# Apply preprocessing
print("=" * 50)
print("PREPROCESSING PIPELINE")
print("=" * 50)

# Step 1: Handle missing values
print("\n1. Handling missing values...")
prices_clean = handle_missing_values(prices, method='ffill')

# Step 2: Calculate returns
print("\n2. Calculating log returns...")
returns = calculate_returns(prices_clean, method='log')
print(f"   Returns shape: {returns.shape}")

# Step 3: Detect outliers
print("\n3. Detecting outliers...")
outlier_counts = {}
for col in returns.columns:
    outliers = detect_outliers(returns[col])
    outlier_counts[col] = outliers.sum()

print("   Outliers detected per asset:")
for asset, count in sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"   - {asset}: {count} outliers")

print("\n✓ Preprocessing complete!")

# %%
# Display processed data summary
print("\n📊 Processed Data Summary")
print("=" * 50)
print(f"Date range: {returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')}")
print(f"Trading days: {len(returns)}")
print(f"Assets: {len(returns.columns)}")

returns.describe().round(4)

# %% [markdown]
# ---
# # SECTION 4: EXPLORATORY DATA ANALYSIS (EDA)
# 
# Understanding the data before applying complex models is crucial.
# We'll examine:
# 1. Price movements (normalized to base 100)
# 2. Return distributions
# 3. Descriptive statistics

# %%
# =============================================================================
# VISUALIZATION 1: Normalized Price Chart
# =============================================================================

def plot_normalized_prices(prices: pd.DataFrame, title: str = "Normalized Prices"):
    """Plot prices normalized to base 100 for comparison."""
    # Normalize to base 100
    normalized = (prices / prices.iloc[0]) * 100
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot each asset
    for col in normalized.columns:
        ax.plot(normalized.index, normalized[col], label=col, linewidth=1.5)
    
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Price (Base = 100)')
    ax.set_title(title)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), ncol=1)
    plt.tight_layout()
    plt.show()

plot_normalized_prices(prices_clean, f"Asset Performance ({config['description']})")

# %%
# =============================================================================
# VISUALIZATION 2: Returns Distribution
# =============================================================================

def plot_returns_distribution(returns: pd.DataFrame, assets: list = None):
    """Plot return distributions for selected assets."""
    if assets is None:
        assets = returns.columns[:6]  # First 6 assets
    
    n_assets = len(assets)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    for i, asset in enumerate(assets):
        if i < len(axes):
            ax = axes[i]
            data = returns[asset].dropna()
            
            # Histogram with KDE
            ax.hist(data, bins=50, density=True, alpha=0.7, color='steelblue')
            
            # Add statistics
            mean = data.mean()
            std = data.std()
            ax.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.4f}')
            ax.axvline(mean + 2*std, color='orange', linestyle=':', alpha=0.7)
            ax.axvline(mean - 2*std, color='orange', linestyle=':', alpha=0.7)
            
            ax.set_title(f'{asset}')
            ax.set_xlabel('Daily Return')
            ax.legend(fontsize=8)
    
    # Hide unused subplots
    for i in range(len(assets), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Daily Returns Distribution', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

# Plot for a mix of asset types
sample_assets = ['AAPL', 'MSFT', 'BTC-USD', 'ETH-USD', 'SPY', 'AGG']
sample_assets = [a for a in sample_assets if a in returns.columns]
plot_returns_distribution(returns, sample_assets)

# %%
# =============================================================================
# DESCRIPTIVE STATISTICS
# =============================================================================

def calculate_descriptive_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive descriptive statistics."""
    from scipy import stats
    
    stats_dict = {}
    for col in returns.columns:
        data = returns[col].dropna()
        stats_dict[col] = {
            'Mean (Daily)': data.mean(),
            'Std (Daily)': data.std(),
            'Skewness': stats.skew(data),
            'Kurtosis': stats.kurtosis(data),
            'Min': data.min(),
            'Max': data.max(),
            'Positive Days %': (data > 0).mean() * 100
        }
    
    return pd.DataFrame(stats_dict).T

desc_stats = calculate_descriptive_stats(returns)
print("📊 Descriptive Statistics (Daily Returns)")
print("=" * 60)
desc_stats.round(4)

# %% [markdown]
# ---
# # SECTION 5: FINANCIAL METRICS CALCULATION
# 
# ## Theory to Code Conversion
# 
# | Metric | Formula | Interpretation |
# |--------|---------|----------------|
# | **Annualized Return** | $(1 + r_{total})^{252/n} - 1$ | Expected yearly return |
# | **Annualized Volatility** | $\sigma_{daily} \times \sqrt{252}$ | Yearly risk measure |
# | **Sharpe Ratio** | $(R_p - R_f) / \sigma_p$ | Risk-adjusted return |
# | **Sortino Ratio** | $(R_p - R_f) / \sigma_{down}$ | Downside risk-adjusted return |
# | **Max Drawdown** | $\min(P_t / P_{max,t} - 1)$ | Largest peak-to-trough decline |
# | **VaR (95%)** | 5th percentile | Max expected loss (95% confidence) |
# | **CVaR (95%)** | Mean of returns < VaR | Expected loss when VaR is breached |

# %%
# =============================================================================
# FINANCIAL METRICS FUNCTIONS
# =============================================================================

def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized return.
    
    Formula: (1 + total_return)^(252/n) - 1
    
    Args:
        returns: Series of daily returns
        periods_per_year: Trading days (252 for stocks, 365 for crypto)
    
    Returns:
        Annualized return as decimal
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
    
    Formula: daily_std * sqrt(252)
    
    The sqrt(252) scaling comes from the assumption that returns are 
    independent and identically distributed (i.i.d.).
    """
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05, 
                 periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe Ratio.
    
    Formula: (R_p - R_f) / σ_p
    
    Measures excess return per unit of total risk.
    - > 1.0: Good
    - > 2.0: Very good
    - > 3.0: Excellent
    """
    ann_return = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    
    if ann_vol > 0:
        return (ann_return - risk_free_rate) / ann_vol
    return 0.0


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.05,
                  periods_per_year: int = 252) -> float:
    """
    Calculate Sortino Ratio.
    
    Similar to Sharpe but uses only downside deviation.
    Better for asymmetric return distributions (like crypto).
    """
    ann_return = annualized_return(returns, periods_per_year)
    
    # Downside deviation (only negative returns)
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return np.inf
    
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)
    
    if downside_std > 0:
        return (ann_return - risk_free_rate) / downside_std
    return 0.0


def max_drawdown(returns: pd.Series) -> tuple:
    """
    Calculate maximum drawdown.
    
    Maximum peak-to-trough decline during the period.
    
    Returns:
        Tuple of (max_drawdown, start_date, end_date)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    
    max_dd = drawdown.min()
    end_date = drawdown.idxmin()
    start_date = cumulative[:end_date].idxmax()
    
    return max_dd, start_date, end_date


def value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) using historical method.
    
    VaR at 95% means: "In 95% of days, losses won't exceed this value"
    
    Returns:
        VaR as positive number (representing potential loss)
    """
    return -returns.quantile(1 - confidence_level)


def conditional_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Conditional VaR (Expected Shortfall).
    
    CVaR answers: "When we do breach VaR, what's the expected loss?"
    More conservative than VaR.
    """
    var = value_at_risk(returns, confidence_level)
    tail_losses = returns[returns <= -var]
    if len(tail_losses) > 0:
        return -tail_losses.mean()
    return var


def calculate_all_metrics(returns: pd.Series, risk_free_rate: float = 0.05) -> dict:
    """Calculate all portfolio metrics at once."""
    max_dd, dd_start, dd_end = max_drawdown(returns)
    
    return {
        'Annualized Return': annualized_return(returns),
        'Annualized Volatility': annualized_volatility(returns),
        'Sharpe Ratio': sharpe_ratio(returns, risk_free_rate),
        'Sortino Ratio': sortino_ratio(returns, risk_free_rate),
        'Max Drawdown': max_dd,
        'VaR (95%)': value_at_risk(returns, 0.95),
        'CVaR (95%)': conditional_var(returns, 0.95),
        'Total Return': (1 + returns).prod() - 1,
        'Win Rate': (returns > 0).mean()
    }

# %%
# Calculate metrics for all assets
print("📊 FINANCIAL METRICS BY ASSET")
print("=" * 70)

metrics_df = pd.DataFrame()
for asset in returns.columns:
    metrics = calculate_all_metrics(returns[asset], RISK_FREE_RATE)
    metrics_df[asset] = pd.Series(metrics)

# Format for display
metrics_display = metrics_df.T.copy()
metrics_display['Annualized Return'] = metrics_display['Annualized Return'].apply(lambda x: f"{x:.2%}")
metrics_display['Annualized Volatility'] = metrics_display['Annualized Volatility'].apply(lambda x: f"{x:.2%}")
metrics_display['Max Drawdown'] = metrics_display['Max Drawdown'].apply(lambda x: f"{x:.2%}")
metrics_display['Total Return'] = metrics_display['Total Return'].apply(lambda x: f"{x:.2%}")
metrics_display['VaR (95%)'] = metrics_display['VaR (95%)'].apply(lambda x: f"{x:.2%}")
metrics_display['CVaR (95%)'] = metrics_display['CVaR (95%)'].apply(lambda x: f"{x:.2%}")
metrics_display['Win Rate'] = metrics_display['Win Rate'].apply(lambda x: f"{x:.1%}")
metrics_display['Sharpe Ratio'] = metrics_display['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
metrics_display['Sortino Ratio'] = metrics_display['Sortino Ratio'].apply(lambda x: f"{x:.2f}")

metrics_display

# %%
# =============================================================================
# VISUALIZATION: Risk vs Return Plot
# =============================================================================

def plot_risk_return(returns: pd.DataFrame, title: str = "Risk vs Return"):
    """Plot risk-return scatter with Sharpe ratio as color."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for col in returns.columns:
        ann_ret = annualized_return(returns[col]) * 100
        ann_vol = annualized_volatility(returns[col]) * 100
        sharpe = sharpe_ratio(returns[col], RISK_FREE_RATE)
        
        # Color by asset type
        if 'USD' in col:
            color = 'orange'
            marker = 's'
        elif col in ['AGG', 'TLT', 'SPY']:
            color = 'green'
            marker = '^'
        else:
            color = 'blue'
            marker = 'o'
        
        ax.scatter(ann_vol, ann_ret, s=100, c=color, marker=marker, alpha=0.7)
        ax.annotate(col, (ann_vol, ann_ret), xytext=(5, 5), 
                   textcoords='offset points', fontsize=9)
    
    # Add reference lines
    ax.axhline(y=RISK_FREE_RATE * 100, color='gray', linestyle='--', 
               label=f'Risk-Free Rate ({RISK_FREE_RATE:.0%})')
    
    ax.set_xlabel('Annualized Volatility (%)')
    ax.set_ylabel('Annualized Return (%)')
    ax.set_title(title)
    ax.legend(['Stocks', 'Crypto', 'Bonds/ETFs', f'Risk-Free ({RISK_FREE_RATE:.0%})'],
              loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_risk_return(returns, f"Risk vs Return ({config['description']})")

# %% [markdown]
# ---
# # SECTION 6: CORRELATION & DIVERSIFICATION ANALYSIS
# 
# ## Theory
# **Diversification** is "the only free lunch in investing" (Harry Markowitz).
# 
# - Low correlation between assets → Better diversification
# - Portfolio variance: $\sigma_p^2 = \sum_i \sum_j w_i w_j \sigma_i \sigma_j \rho_{ij}$
# - If $\rho < 1$, portfolio volatility is less than weighted average of individual volatilities

# %%
# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def plot_correlation_matrix(returns: pd.DataFrame, title: str = "Correlation Matrix"):
    """Plot correlation heatmap."""
    corr = returns.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', 
                cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax)
    
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    
    return corr

# Calculate and display correlation
print("📊 ASSET CORRELATIONS")
print("=" * 50)
correlation_matrix = plot_correlation_matrix(returns, f"Asset Correlations ({config['description']})")

# %%
# Analyze crypto vs traditional correlations
print("\n📈 Crypto vs Traditional Asset Correlations")
print("=" * 50)

cryptos = [c for c in ASSETS['crypto'] if c in returns.columns]
traditional = [t for t in ASSETS['stocks'] + ASSETS['bonds_etfs'] if t in returns.columns]

if cryptos and traditional:
    crypto_trad_corr = returns[cryptos + traditional].corr()
    
    print("\nAverage correlation of each crypto with traditional assets:")
    for crypto in cryptos:
        avg_corr = crypto_trad_corr.loc[crypto, traditional].mean()
        print(f"  {crypto}: {avg_corr:.3f}")

# %%
# =============================================================================
# ROLLING CORRELATION
# =============================================================================

def plot_rolling_correlation(returns: pd.DataFrame, asset1: str, asset2: str, 
                             window: int = 21):
    """Plot rolling correlation between two assets."""
    if asset1 not in returns.columns or asset2 not in returns.columns:
        print(f"Assets not found: {asset1} or {asset2}")
        return
    
    rolling_corr = returns[asset1].rolling(window=window).corr(returns[asset2])
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    ax.plot(rolling_corr.index, rolling_corr.values, linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=rolling_corr.mean(), color='red', linestyle='--', 
               label=f'Mean: {rolling_corr.mean():.2f}')
    
    ax.fill_between(rolling_corr.index, rolling_corr.values, alpha=0.3)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Correlation')
    ax.set_title(f'{window}-Day Rolling Correlation: {asset1} vs {asset2}')
    ax.legend()
    ax.set_ylim(-1, 1)
    plt.tight_layout()
    plt.show()

# Plot BTC vs SPY rolling correlation
if 'BTC-USD' in returns.columns and 'SPY' in returns.columns:
    plot_rolling_correlation(returns, 'BTC-USD', 'SPY', window=21)

# %% [markdown]
# ---
# # SECTION 7: PORTFOLIO CONSTRUCTION (Modern Portfolio Theory)
# 
# ## Markowitz Mean-Variance Optimization
# 
# **Objective**: Find optimal portfolio weights that:
# 1. Minimize risk for a given return target, OR
# 2. Maximize return for a given risk level, OR
# 3. Maximize Sharpe ratio (best risk-adjusted return)
# 
# ### Mathematical Formulation
# 
# **Minimize** portfolio variance:
# $$\sigma_p^2 = w^T \Sigma w$$
# 
# **Subject to**:
# - $\sum_i w_i = 1$ (weights sum to 1)
# - $w_i \geq 0$ (no short selling, optional)
# - $w^T \mu = r_{target}$ (target return, if specified)
# 
# Where:
# - $w$ = vector of weights
# - $\Sigma$ = covariance matrix
# - $\mu$ = expected returns vector

# %%
# =============================================================================
# PORTFOLIO OPTIMIZATION FUNCTIONS
# =============================================================================

def portfolio_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    """Calculate portfolio expected return."""
    return np.dot(weights, mean_returns)


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Calculate portfolio volatility (standard deviation)."""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def portfolio_sharpe(weights: np.ndarray, mean_returns: np.ndarray, 
                     cov_matrix: np.ndarray, risk_free_rate: float) -> float:
    """Calculate portfolio Sharpe ratio."""
    ret = portfolio_return(weights, mean_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    return (ret - risk_free_rate) / vol if vol > 0 else 0


def negative_sharpe(weights: np.ndarray, mean_returns: np.ndarray, 
                    cov_matrix: np.ndarray, risk_free_rate: float) -> float:
    """Negative Sharpe for minimization."""
    return -portfolio_sharpe(weights, mean_returns, cov_matrix, risk_free_rate)


def optimize_portfolio(returns: pd.DataFrame, 
                       optimization_target: str = 'max_sharpe',
                       risk_free_rate: float = 0.05) -> dict:
    """
    Optimize portfolio allocation.
    
    Args:
        returns: DataFrame of asset returns
        optimization_target: 'max_sharpe', 'min_variance', or 'equal_weight'
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Dictionary with optimal weights and portfolio metrics
    """
    # Calculate inputs
    mean_returns = returns.mean() * 252  # Annualize
    cov_matrix = returns.cov() * 252     # Annualize
    n_assets = len(returns.columns)
    
    # Initial guess: equal weights
    init_weights = np.array([1/n_assets] * n_assets)
    
    # Constraints
    bounds = tuple((0, 1) for _ in range(n_assets))  # Long only
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Sum to 1
    
    if optimization_target == 'equal_weight':
        optimal_weights = init_weights
    
    elif optimization_target == 'min_variance':
        result = minimize(
            portfolio_volatility,
            init_weights,
            args=(cov_matrix.values,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        optimal_weights = result.x
    
    elif optimization_target == 'max_sharpe':
        result = minimize(
            negative_sharpe,
            init_weights,
            args=(mean_returns.values, cov_matrix.values, risk_free_rate / 252 * 252),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        optimal_weights = result.x
    
    else:
        raise ValueError("Invalid optimization_target")
    
    # Calculate portfolio metrics
    port_return = portfolio_return(optimal_weights, mean_returns.values)
    port_vol = portfolio_volatility(optimal_weights, cov_matrix.values)
    port_sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
    
    return {
        'weights': pd.Series(optimal_weights, index=returns.columns),
        'return': port_return,
        'volatility': port_vol,
        'sharpe': port_sharpe
    }

# %%
# =============================================================================
# BUILD PORTFOLIOS
# =============================================================================

print("📊 PORTFOLIO OPTIMIZATION RESULTS")
print("=" * 70)

# Define portfolio universes
traditional_assets = [a for a in ASSETS['stocks'] + ASSETS['bonds_etfs'] 
                      if a in returns.columns]
all_assets_list = [a for a in all_tickers if a in returns.columns]

portfolios = {}

# 1. Traditional Portfolio (Stocks + Bonds)
if len(traditional_assets) > 1:
    print("\n1. TRADITIONAL PORTFOLIO (Stocks + Bonds)")
    print("-" * 50)
    portfolios['traditional'] = optimize_portfolio(
        returns[traditional_assets], 
        'max_sharpe', 
        RISK_FREE_RATE
    )
    
    print(f"   Expected Return: {portfolios['traditional']['return']:.2%}")
    print(f"   Volatility: {portfolios['traditional']['volatility']:.2%}")
    print(f"   Sharpe Ratio: {portfolios['traditional']['sharpe']:.2f}")
    print("\n   Top Allocations:")
    top_weights = portfolios['traditional']['weights'].sort_values(ascending=False).head(5)
    for asset, weight in top_weights.items():
        if weight > 0.01:
            print(f"     {asset}: {weight:.1%}")

# 2. Crypto-Enhanced Portfolio
if len(all_assets_list) > len(traditional_assets):
    print("\n2. CRYPTO-ENHANCED PORTFOLIO (Stocks + Bonds + Crypto)")
    print("-" * 50)
    portfolios['crypto_enhanced'] = optimize_portfolio(
        returns[all_assets_list], 
        'max_sharpe', 
        RISK_FREE_RATE
    )
    
    print(f"   Expected Return: {portfolios['crypto_enhanced']['return']:.2%}")
    print(f"   Volatility: {portfolios['crypto_enhanced']['volatility']:.2%}")
    print(f"   Sharpe Ratio: {portfolios['crypto_enhanced']['sharpe']:.2f}")
    print("\n   Top Allocations:")
    top_weights = portfolios['crypto_enhanced']['weights'].sort_values(ascending=False).head(5)
    for asset, weight in top_weights.items():
        if weight > 0.01:
            print(f"     {asset}: {weight:.1%}")

# 3. Equal Weight Portfolio
print("\n3. EQUAL WEIGHT PORTFOLIO (Benchmark)")
print("-" * 50)
portfolios['equal_weight'] = optimize_portfolio(
    returns[all_assets_list], 
    'equal_weight', 
    RISK_FREE_RATE
)
print(f"   Expected Return: {portfolios['equal_weight']['return']:.2%}")
print(f"   Volatility: {portfolios['equal_weight']['volatility']:.2%}")
print(f"   Sharpe Ratio: {portfolios['equal_weight']['sharpe']:.2f}")

# %%
# =============================================================================
# VISUALIZE OPTIMAL ALLOCATIONS
# =============================================================================

def plot_portfolio_weights(portfolio: dict, title: str):
    """Plot portfolio allocation pie chart."""
    weights = portfolio['weights']
    weights = weights[weights > 0.01]  # Filter small allocations
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(weights)))
    wedges, texts, autotexts = ax.pie(
        weights.values, 
        labels=weights.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    
    ax.set_title(f'{title}\nReturn: {portfolio["return"]:.1%} | '
                 f'Vol: {portfolio["volatility"]:.1%} | '
                 f'Sharpe: {portfolio["sharpe"]:.2f}')
    
    plt.tight_layout()
    plt.show()

# Plot portfolios
if 'crypto_enhanced' in portfolios:
    plot_portfolio_weights(portfolios['crypto_enhanced'], 'Crypto-Enhanced Portfolio')

if 'traditional' in portfolios:
    plot_portfolio_weights(portfolios['traditional'], 'Traditional Portfolio')

# %% [markdown]
# ---
# # SECTION 8: PORTFOLIO COMPARISON & BACKTESTING
# 
# Compare the performance of different portfolio strategies over the analysis period.

# %%
# =============================================================================
# BACKTEST PORTFOLIOS
# =============================================================================

def backtest_portfolio(returns: pd.DataFrame, weights: pd.Series, 
                       initial_value: float = 10000) -> pd.Series:
    """
    Backtest a portfolio with given weights.
    
    Args:
        returns: Asset returns
        weights: Portfolio weights
        initial_value: Starting portfolio value (e.g., 10000€)
    
    Returns:
        Series of portfolio values over time
    """
    # Align weights with returns columns
    aligned_weights = weights.reindex(returns.columns).fillna(0)
    
    # Calculate portfolio returns
    portfolio_returns = (returns * aligned_weights.values).sum(axis=1)
    
    # Calculate cumulative value
    portfolio_value = initial_value * (1 + portfolio_returns).cumprod()
    
    return portfolio_value

# %%
# Backtest all portfolios
print("📊 PORTFOLIO BACKTEST RESULTS")
print("=" * 70)

initial_investment = 10000  # €10,000 initial investment

backtest_results = {}
for name, portfolio in portfolios.items():
    if 'weights' in portfolio:
        backtest_results[name] = backtest_portfolio(
            returns[portfolio['weights'].index], 
            portfolio['weights'],
            initial_investment
        )

# %%
# Visualize cumulative returns
fig, ax = plt.subplots(figsize=(14, 7))

colors = {'traditional': 'blue', 'crypto_enhanced': 'orange', 'equal_weight': 'green'}

for name, values in backtest_results.items():
    ax.plot(values.index, values.values, label=name.replace('_', ' ').title(), 
            linewidth=2, color=colors.get(name, 'gray'))

ax.axhline(y=initial_investment, color='gray', linestyle='--', alpha=0.5, 
           label='Initial Investment')

ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Value (€)')
ax.set_title(f'Portfolio Performance Comparison ({config["description"]})\nInitial Investment: €{initial_investment:,}')
ax.legend()
ax.grid(True, alpha=0.3)

# Format y-axis
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x:,.0f}'))

plt.tight_layout()
plt.show()

# %%
# =============================================================================
# FINAL COMPARISON TABLE
# =============================================================================

print("\n" + "=" * 80)
print("📊 FINAL PORTFOLIO COMPARISON")
print("=" * 80)

comparison_data = []

for name, portfolio in portfolios.items():
    if name in backtest_results:
        values = backtest_results[name]
        final_value = values.iloc[-1]
        total_return = (final_value / initial_investment - 1)
        
        # Calculate actual backtest metrics
        portfolio_returns = values.pct_change().dropna()
        
        comparison_data.append({
            'Portfolio': name.replace('_', ' ').title(),
            'Final Value': f'€{final_value:,.0f}',
            'Total Return': f'{total_return:.2%}',
            'Ann. Return': f'{portfolio["return"]:.2%}',
            'Ann. Volatility': f'{portfolio["volatility"]:.2%}',
            'Sharpe Ratio': f'{portfolio["sharpe"]:.2f}',
            'Max Drawdown': f'{max_drawdown(portfolio_returns)[0]:.2%}'
        })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.set_index('Portfolio', inplace=True)
print(comparison_df.to_string())

# %%
# =============================================================================
# KEY INSIGHTS
# =============================================================================

print("\n" + "=" * 80)
print("🔍 KEY INSIGHTS")
print("=" * 80)

if 'traditional' in portfolios and 'crypto_enhanced' in portfolios:
    trad = portfolios['traditional']
    crypto = portfolios['crypto_enhanced']
    
    return_diff = crypto['return'] - trad['return']
    vol_diff = crypto['volatility'] - trad['volatility']
    sharpe_diff = crypto['sharpe'] - trad['sharpe']
    
    print(f"""
1. RETURN ENHANCEMENT:
   Adding crypto {'increased' if return_diff > 0 else 'decreased'} expected returns by {abs(return_diff):.2%}

2. RISK IMPACT:
   Portfolio volatility {'increased' if vol_diff > 0 else 'decreased'} by {abs(vol_diff):.2%}

3. RISK-ADJUSTED PERFORMANCE:
   Sharpe Ratio {'improved' if sharpe_diff > 0 else 'declined'} by {abs(sharpe_diff):.2f}

4. DIVERSIFICATION BENEFIT:
   Crypto assets show {'low' if correlation_matrix.loc[cryptos, traditional].mean().mean() < 0.3 else 'moderate' if correlation_matrix.loc[cryptos, traditional].mean().mean() < 0.6 else 'high'} correlation with traditional assets, 
   {'providing' if correlation_matrix.loc[cryptos, traditional].mean().mean() < 0.5 else 'limiting'} diversification benefits.
""")

print("\n✅ Analysis Complete!")
print(f"Period analyzed: {config['description']} ({START_DATE} to {END_DATE})")

