"""
Prefect Workflow for Portfolio Data Pipeline

This workflow orchestrates:
1. Multi-source data extraction (stocks, crypto, bonds)
2. Data validation and quality checks
3. Data transformation and cleaning
4. Final dataset preparation

Run with: uv run python src/workflows/data_pipeline.py
View UI: prefect server start (in separate terminal)
"""

from prefect import flow, task
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from data_loader import DataLoader


@task(name="Extract Stock Data", retries=2, retry_delay_seconds=10)
def extract_stocks(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Extract stock data from Yahoo Finance.
    
    Args:
        tickers: List of stock tickers
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with stock data
    """
    loader = DataLoader()
    print(f"📊 Extracting {len(tickers)} stocks...")
    data = loader.fetch_data(tickers, start_date, end_date)
    print(f"✓ Extracted {data.shape[0]} rows for stocks")
    return data


@task(name="Extract Crypto Data", retries=2, retry_delay_seconds=10)
def extract_crypto(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Extract cryptocurrency data from Yahoo Finance.
    
    Args:
        tickers: List of crypto tickers (e.g., 'BTC-USD')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with crypto data
    """
    loader = DataLoader()
    print(f"₿ Extracting {len(tickers)} cryptocurrencies...")
    data = loader.fetch_data(tickers, start_date, end_date)
    print(f"✓ Extracted {data.shape[0]} rows for crypto")
    return data


@task(name="Extract Bond/ETF Data", retries=2, retry_delay_seconds=10)
def extract_bonds(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Extract bond/ETF data from Yahoo Finance.
    
    Args:
        tickers: List of bond/ETF tickers
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with bond data
    """
    loader = DataLoader()
    print(f"📈 Extracting {len(tickers)} bonds/ETFs...")
    data = loader.fetch_data(tickers, start_date, end_date)
    print(f"✓ Extracted {data.shape[0]} rows for bonds/ETFs")
    return data


@task(name="Validate Data Quality")
def validate_data(data: pd.DataFrame, asset_type: str) -> pd.DataFrame:
    """
    Validate data quality and check for issues.
    
    Args:
        data: DataFrame to validate
        asset_type: Type of asset (for logging)
        
    Returns:
        Validated DataFrame
    """
    print(f"🔍 Validating {asset_type} data...")
    
    # Check for missing values
    missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
    print(f"  Missing values: {missing_pct:.2f}%")
    
    # Check date range
    if not data.empty:
        print(f"  Date range: {data.index.min()} to {data.index.max()}")
        print(f"  Total rows: {len(data)}")
    
    return data


@task(name="Save Raw Data")
def save_raw_data(data: pd.DataFrame, filename: str) -> str:
    """
    Save raw data to Parquet format.
    
    Args:
        data: DataFrame to save
        filename: Output filename
        
    Returns:
        Path to saved file
    """
    loader = DataLoader()
    loader.save_data(data, filename)
    return f"data/raw/{filename}"


@task(name="Merge Datasets")
def merge_datasets(stocks: pd.DataFrame, crypto: pd.DataFrame, bonds: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all datasets into a single DataFrame.
    
    Args:
        stocks: Stock data
        crypto: Crypto data
        bonds: Bond data
        
    Returns:
        Merged DataFrame
    """
    print("🔗 Merging datasets...")
    
    # Combine all data
    # Note: yfinance returns MultiIndex columns when multiple tickers
    # We'll keep them separate for now and merge on date index
    
    merged = pd.concat([stocks, crypto, bonds], axis=1)
    print(f"✓ Merged dataset shape: {merged.shape}")
    
    return merged


@task(name="Clean and Transform Data")
def clean_transform_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform the merged dataset.
    
    Args:
        data: Raw merged data
        
    Returns:
        Cleaned and transformed data
    """
    print("🧹 Cleaning and transforming data...")
    
    # Remove rows with all NaN
    data = data.dropna(how='all')
    
    # Forward fill missing values (common for financial data)
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    print(f"✓ Cleaned dataset shape: {data.shape}")
    
    return data


@task(name="Save Processed Data")
def save_processed_data(data: pd.DataFrame, filename: str = "processed_portfolio_data.parquet") -> str:
    """
    Save processed data to data/processed directory.
    
    Args:
        data: Processed DataFrame
        filename: Output filename
        
    Returns:
        Path to saved file
    """
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    data.to_parquet(output_path)
    
    print(f"💾 Saved processed data to {output_path}")
    return str(output_path)


@flow(
    name="Portfolio Data Pipeline",
    description="Extract, validate, and process portfolio data from multiple sources"
)
def portfolio_data_pipeline(
    start_date: str = "2019-01-01",
    end_date: str = "2024-01-01",
    stock_tickers: list = None,
    crypto_tickers: list = None,
    bond_tickers: list = None
):
    """
    Main Prefect flow for portfolio data pipeline.
    
    Args:
        start_date: Start date for data extraction
        end_date: End date for data extraction
        stock_tickers: List of stock tickers
        crypto_tickers: List of crypto tickers
        bond_tickers: List of bond/ETF tickers
    """
    print("=" * 60)
    print("🚀 Starting Portfolio Data Pipeline")
    print("=" * 60)
    
    # Default tickers if not provided
    if stock_tickers is None:
        stock_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ']
    
    if crypto_tickers is None:
        crypto_tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD']
    
    if bond_tickers is None:
        bond_tickers = ['AGG', 'TLT', 'SPY']  # Bond ETFs and S&P500
    
    # PHASE 1: EXTRACTION (Parallel execution)
    print("\n📥 PHASE 1: Data Extraction (Parallel)")
    stocks_raw = extract_stocks(stock_tickers, start_date, end_date)
    crypto_raw = extract_crypto(crypto_tickers, start_date, end_date)
    bonds_raw = extract_bonds(bond_tickers, start_date, end_date)
    
    # PHASE 2: VALIDATION (Parallel)
    print("\n✅ PHASE 2: Data Validation (Parallel)")
    stocks_validated = validate_data(stocks_raw, "Stocks")
    crypto_validated = validate_data(crypto_raw, "Crypto")
    bonds_validated = validate_data(bonds_raw, "Bonds")
    
    # PHASE 3: SAVE RAW DATA (Parallel)
    print("\n💾 PHASE 3: Saving Raw Data (Parallel)")
    save_raw_data(stocks_validated, "stocks_raw.parquet")
    save_raw_data(crypto_validated, "crypto_raw.parquet")
    save_raw_data(bonds_validated, "bonds_raw.parquet")
    
    # PHASE 4: MERGE AND TRANSFORM (Sequential)
    print("\n🔄 PHASE 4: Merge and Transform")
    merged_data = merge_datasets(stocks_validated, crypto_validated, bonds_validated)
    cleaned_data = clean_transform_data(merged_data)
    
    # PHASE 5: SAVE FINAL DATASET
    print("\n💾 PHASE 5: Save Processed Dataset")
    final_path = save_processed_data(cleaned_data)
    
    print("\n" + "=" * 60)
    print(f"✅ Pipeline Complete! Final dataset: {final_path}")
    print(f"   Shape: {cleaned_data.shape}")
    print(f"   Date range: {cleaned_data.index.min()} to {cleaned_data.index.max()}")
    print("=" * 60)
    
    return final_path


if __name__ == "__main__":
    # Run the pipeline
    # For 6-month prototype
    result = portfolio_data_pipeline(
        start_date="2023-06-01",
        end_date="2023-12-31"
    )
    
    print(f"\n🎉 Pipeline executed successfully!")
    print(f"📊 Data ready for analysis in: {result}")
