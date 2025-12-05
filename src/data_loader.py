import yfinance as yf
import pandas as pd
import os
from typing import List, Optional

class DataLoader:
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir (str): Directory to save fetched data.
        """
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance.
        
        Args:
            tickers (List[str]): List of ticker symbols (e.g., ['AAPL', 'BTC-USD']).
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            
        Returns:
            pd.DataFrame: MultiIndex DataFrame with historical data.
        """
        print(f"Fetching data for {tickers} from {start_date} to {end_date}...")
        
        # yfinance download
        # auto_adjust=True adjusts for splits and dividends
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
        
        if data.empty:
            print("Warning: No data fetched.")
        else:
            print(f"Fetched data shape: {data.shape}")
            
        return data

    def save_data(self, data: pd.DataFrame, filename: str):
        """
        Save DataFrame to Parquet.
        
        Args:
            data (pd.DataFrame): Data to save.
            filename (str): Filename (e.g., 'market_data.parquet').
        """
        path = os.path.join(self.data_dir, filename)
        try:
            data.to_parquet(path)
            print(f"Data saved to {path}")
        except Exception as e:
            print(f"Error saving data: {e}")

    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load data from Parquet.
        
        Args:
            filename (str): Filename to load.
            
        Returns:
            pd.DataFrame: Loaded data.
        """
        path = os.path.join(self.data_dir, filename)
        if os.path.exists(path):
            return pd.read_parquet(path)
        else:
            print(f"File {path} not found.")
            return pd.DataFrame()

if __name__ == "__main__":
    # Test execution
    loader = DataLoader()
    
    # Define assets
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY', 'AGG'] # Tech + Market + Bond ETF
    cryptos = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD']
    
    # Fetch data (Last 5 years approx)
    start = "2018-01-01"
    end = "2024-01-01"
    
    print("Fetching Stocks...")
    df_stocks = loader.fetch_data(stocks, start, end)
    loader.save_data(df_stocks, "stocks_data.parquet")
    
    print("Fetching Cryptos...")
    df_cryptos = loader.fetch_data(cryptos, start, end)
    loader.save_data(df_cryptos, "crypto_data.parquet")
