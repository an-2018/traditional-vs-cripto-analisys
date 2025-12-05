"""
Standalone Data Pipeline Script

Alternative to Prefect workflow for simpler execution.
Orchestrates data extraction, validation, and processing.

Run with: uv run python src/run_pipeline.py
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_loader import DataLoader
from preprocessing import calculate_returns, handle_missing_values


def run_data_pipeline(start_date="2023-06-01", end_date="2023-12-31"):
    """
    Run the complete data pipeline.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    """
    print("=" * 60)
    print("🚀 Starting Portfolio Data Pipeline")
    print("=" * 60)
    
    # Initialize loader
    loader = DataLoader()
    
    # Define assets
    stock_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ']
    crypto_tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD']
    bond_tickers = ['AGG', 'TLT', 'SPY']
    
    # PHASE 1: EXTRACTION
    print("\n📥 PHASE 1: Data Extraction")
    print("-" * 60)
    
    print("📊 Extracting stocks...")
    stocks_raw = loader.fetch_data(stock_tickers, start_date, end_date)
    
    print("₿ Extracting cryptocurrencies...")
    crypto_raw = loader.fetch_data(crypto_tickers, start_date, end_date)
    
    print("📈 Extracting bonds/ETFs...")
    bonds_raw = loader.fetch_data(bond_tickers, start_date, end_date)
    
    # PHASE 2: VALIDATION
    print("\n✅ PHASE 2: Data Validation")
    print("-" * 60)
    
    for name, data in [("Stocks", stocks_raw), ("Crypto", crypto_raw), ("Bonds", bonds_raw)]:
        missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
        print(f"{name:.<20} Shape: {str(data.shape):.<15} Missing: {missing_pct:.2f}%")
    
    # PHASE 3: SAVE RAW DATA
    print("\n💾 PHASE 3: Saving Raw Data")
    print("-" * 60)
    
    loader.save_data(stocks_raw, "stocks_raw.parquet")
    loader.save_data(crypto_raw, "crypto_raw.parquet")
    loader.save_data(bonds_raw, "bonds_raw.parquet")
    
    # PHASE 4: MERGE AND TRANSFORM
    print("\n🔄 PHASE 4: Merge and Transform")
    print("-" * 60)
    
    print("🔗 Merging datasets...")
    merged_data = pd.concat([stocks_raw, crypto_raw, bonds_raw], axis=1)
    print(f"✓ Merged dataset shape: {merged_data.shape}")
    
    print("🧹 Cleaning data...")
    # Remove rows with all NaN
    cleaned_data = merged_data.dropna(how='all')
    # Forward fill missing values
    cleaned_data = cleaned_data.fillna(method='ffill').fillna(method='bfill')
    print(f"✓ Cleaned dataset shape: {cleaned_data.shape}")
    
    # PHASE 5: SAVE FINAL DATASET
    print("\n💾 PHASE 5: Save Processed Dataset")
    print("-" * 60)
    
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "processed_portfolio_data.parquet"
    cleaned_data.to_parquet(output_path)
    
    print(f"💾 Saved processed data to {output_path}")
    
    # SUMMARY
    print("\n" + "=" * 60)
    print(f"✅ Pipeline Complete!")
    print(f"   Final dataset: {output_path}")
    print(f"   Shape: {cleaned_data.shape}")
    print(f"   Date range: {cleaned_data.index.min()} to {cleaned_data.index.max()}")
    print("=" * 60)
    
    return str(output_path)


if __name__ == "__main__":
    # Run for 6-month period
    result = run_data_pipeline(
        start_date="2023-06-01",
        end_date="2023-12-31"
    )
    
    print(f"\n🎉 Pipeline executed successfully!")
    print(f"📊 Data ready for analysis in: {result}")
    print(f"\n📓 Next step: Open notebooks/01_portfolio_analysis_6months.ipynb")
