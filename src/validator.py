"""
Data Validator Module

Handles validation of financial data from multiple sources.
Compares datasets for consistency and checks descriptive statistics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path
import json

class DataValidator:
    """Validates financial data consistency across sources."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize validator with configuration.
        
        Args:
            config: Validation configuration dictionary
        """
        self.config = config
        self.enabled = config.get('enabled', False)
        self.price_tolerance = config.get('price_tolerance', 0.05)
        self.min_correlation = config.get('min_correlation', 0.95)
        self.report = {}

    def validate_sources(self, primary_data: pd.DataFrame, secondary_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Compare primary data source against secondary sources.
        
        Args:
            primary_data: DataFrame from primary source (e.g., Yahoo Finance)
            secondary_data: Dictionary of DataFrames from other sources
            
        Returns:
            Validation report dictionary
        """
        if not self.enabled or not secondary_data:
            return {"status": "skipped", "message": "Validation disabled or no secondary sources"}

        report = {
            "status": "success",
            "sources_compared": list(secondary_data.keys()),
            "metrics": {}
        }

        for source_name, comparison_df in secondary_data.items():
            source_metrics = self._compare_datasets(primary_data, comparison_df)
            report["metrics"][source_name] = source_metrics
            
            if not source_metrics["passed"]:
                report["status"] = "failed"
                report["failure_reason"] = f"Validation failed for {source_name}"

        self.report = report
        return report

    def _compare_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
        """Compare two datasets and return metrics."""
        # Align datasets by index (dates) and columns (tickers)
        common_dates = df1.index.intersection(df2.index)
        common_cols = df1.columns.intersection(df2.columns)
        
        if len(common_dates) == 0 or len(common_cols) == 0:
            return {"passed": False, "reason": "No overlapping data"}

        d1 = df1.loc[common_dates, common_cols]
        d2 = df2.loc[common_dates, common_cols]

        # 1. Price Difference Check
        # Calculate percentage difference
        diff_pct = np.abs((d1 - d2) / d1)
        mean_diff = diff_pct.mean().mean()
        max_diff = diff_pct.max().max()
        
        price_check = mean_diff < self.price_tolerance

        # 2. Correlation Check
        # Calculate correlation between matching columns
        correlations = []
        for col in common_cols:
            corr = d1[col].corr(d2[col])
            correlations.append(corr)
        
        avg_correlation = np.mean(correlations)
        correlation_check = avg_correlation > self.min_correlation

        # 3. Descriptive Stats Comparison
        stats1 = d1.describe()
        stats2 = d2.describe()
        # Simple check: compare means of the means
        stats_diff = np.abs((stats1.loc['mean'] - stats2.loc['mean']) / stats1.loc['mean']).mean()
        stats_check = stats_diff < self.price_tolerance

        passed = price_check and correlation_check and stats_check

        return {
            "passed": passed,
            "mean_price_diff_pct": float(mean_diff),
            "max_price_diff_pct": float(max_diff),
            "avg_correlation": float(avg_correlation),
            "stats_diff_pct": float(stats_diff),
            "details": {
                "price_check": bool(price_check),
                "correlation_check": bool(correlation_check),
                "stats_check": bool(stats_check)
            }
        }

    def validate_schema(self, df: pd.DataFrame, expected_columns: List[str] = None) -> bool:
        """Check if DataFrame has expected structure."""
        if df.empty:
            return False
        
        # Check for required index type (DatetimeIndex)
        if not isinstance(df.index, pd.DatetimeIndex):
            return False

        # Check for expected columns if provided
        if expected_columns:
            missing = [col for col in expected_columns if col not in df.columns]
            if missing:
                return False
        
        return True

    def save_report(self, output_dir: str):
        """Save validation report to JSON."""
        path = Path(output_dir) / "validation_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.report, f, indent=4)

if __name__ == "__main__":
    # Test stub
    print("Validator module test")
    # Create dummy data
    dates = pd.date_range(start="2023-01-01", periods=10)
    df1 = pd.DataFrame({"AAPL": np.random.rand(10) * 100}, index=dates)
    df2 = df1 * 1.02 # 2% difference
    
    config = {"enabled": True, "price_tolerance": 0.05, "min_correlation": 0.95}
    validator = DataValidator(config)
    result = validator.validate_sources(df1, {"source_b": df2})
    print(json.dumps(result, indent=2))
