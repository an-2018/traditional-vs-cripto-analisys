# Portfolio Analysis Pipeline Notebook

## 📊 Overview

This Jupyter notebook implements a **complete financial analysis pipeline** for comparing traditional investment portfolios with cryptocurrency-enhanced portfolios. It covers the full workflow from data gathering to portfolio optimization using Modern Portfolio Theory.

## 🎯 Learning Objectives

After completing this notebook, you will understand:

- How to gather financial data using Python (yfinance)
- Data preprocessing techniques for time series
- Calculation of key portfolio metrics (Sharpe, Sortino, VaR, CVaR)
- Correlation analysis for diversification assessment
- Markowitz mean-variance portfolio optimization
- Comparison of different portfolio strategies

---

## 📁 Files

| File | Description |
|------|-------------|
| `01_portfolio_analysis_6months.ipynb` | Main Jupyter notebook (ready to run) |
| `01_portfolio_analysis_pipeline.py` | Python source (percent format, for version control) |

---

## 🚀 Quick Start

### Prerequisites

```bash
# Ensure you have uv installed
pip install uv

# Install project dependencies
cd source
uv sync
```

### Running the Notebook

```bash
# Start Jupyter
uv run jupyter notebook

# Open: notebooks/01_portfolio_analysis_6months.ipynb
# Run all cells (Kernel → Restart & Run All)
```

---

## ⚙️ Configuration

### Changing the Analysis Period

Modify the `SELECTED_RANGE` variable at the top of the notebook:

```python
# Options: '6_months', '1_year', '2_years', '5_years'
SELECTED_RANGE = '1_year'  # Change this line
```

### Available Time Ranges

| Range | Period | Use Case |
|-------|--------|----------|
| `6_months` | Last 6 months | Quick analysis, recent trends |
| `1_year` | Last 12 months | Standard annual review |
| `2_years` | Last 24 months | Medium-term patterns |
| `5_years` | Last 60 months | Long-term analysis, full market cycles |

### Modifying Assets

Edit the `ASSETS` dictionary to include different tickers:

```python
ASSETS = {
    'stocks': ['AAPL', 'MSFT', 'GOOGL', ...],     # Stock tickers
    'crypto': ['BTC-USD', 'ETH-USD', ...],        # Crypto (add -USD suffix)
    'bonds_etfs': ['AGG', 'TLT', 'SPY', ...]      # ETFs and bond funds
}
```

---

## 📖 Notebook Sections

### Section 1: Configuration & Imports
- Library imports and warnings suppression
- Time range configuration
- Asset definitions

### Section 2: Data Gathering
- **Function**: `fetch_market_data(tickers, start, end)`
- Uses Yahoo Finance API via `yfinance`
- Downloads OHLCV data for all assets

### Section 3: Data Preprocessing
- **Functions**:
  - `handle_missing_values(df, method)` - Forward/backward fill
  - `calculate_returns(prices, method)` - Log or simple returns
  - `detect_outliers(data, method, threshold)` - IQR or Z-score

### Section 4: Exploratory Data Analysis
- **Visualizations**:
  - Normalized price chart (base 100)
  - Returns distribution histograms
- **Statistics**: Mean, std, skewness, kurtosis

### Section 5: Financial Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Annualized Return | $(1 + r)^{252/n} - 1$ | Expected yearly return |
| Annualized Volatility | $\sigma \times \sqrt{252}$ | Yearly risk |
| Sharpe Ratio | $(R_p - R_f) / \sigma$ | Risk-adjusted return |
| Sortino Ratio | $(R_p - R_f) / \sigma_{down}$ | Downside-adjusted return |
| Max Drawdown | Peak-to-trough decline | Worst-case loss |
| VaR (95%) | 5th percentile | Maximum expected daily loss |
| CVaR (95%) | Expected shortfall | Average loss when VaR breached |

### Section 6: Correlation Analysis
- Correlation matrix heatmap
- Rolling correlation (21-day window)
- Crypto vs traditional assets comparison

### Section 7: Portfolio Optimization (Markowitz)
- **Theory**: Mean-variance optimization
- **Portfolios built**:
  1. Equal-weight (benchmark)
  2. Minimum variance
  3. Maximum Sharpe ratio
- **Optimization target**: Maximize `(Return - RiskFreeRate) / Volatility`

### Section 8: Portfolio Comparison
- Backtest with €10,000 initial investment
- Performance comparison chart
- Final metrics table
- Key insights summary

---

## 📊 Key Formulas

### Log Returns
```
r_t = ln(P_t / P_{t-1})
```
*Advantage: Additive over time, better for statistical modeling*

### Portfolio Variance
```
σ_p² = Σᵢ Σⱼ wᵢ wⱼ σᵢ σⱼ ρᵢⱼ
```
*Where: w = weights, σ = volatility, ρ = correlation*

### Sharpe Ratio
```
Sharpe = (R_p - R_f) / σ_p
```
*Interpretation: Return per unit of risk. Higher is better.*

---

## 🛠️ Troubleshooting

### "No data fetched" error
- Check internet connection
- Verify ticker symbols are correct
- Some tickers may not have data for the full period

### Missing values in results
- The notebook handles missing data with forward-fill
- Check if assets have overlapping trading days

### Optimization doesn't converge
- Try reducing the number of assets
- Check for assets with extreme volatility

---

## 📚 References

- Markowitz, H. (1952). "Portfolio Selection". *Journal of Finance*
- Sharpe, W. (1966). "Mutual Fund Performance". *Journal of Business*
- Yahoo Finance API: https://pypi.org/project/yfinance/

---

## 📝 License

This notebook is for educational purposes as part of the postgraduate program in Financial Markets and Portfolio Management.
