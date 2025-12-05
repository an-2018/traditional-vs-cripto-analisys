"""
Configuration Loader Module

Loads and validates pipeline configuration from YAML file.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Loads and manages pipeline configuration."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to YAML config file
        """
        if config_path is None:
            # Default to config/pipeline_config.yaml
            config_path = Path(__file__).parent.parent / "config" / "pipeline_config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}")
            return self._get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'date_range': {
                'start_date': '2023-06-01',
                'end_date': '2023-12-31'
            },
            'assets': {
                'stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ'],
                'crypto': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD'],
                'bonds_etfs': ['AGG', 'TLT', 'SPY']
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_assets(self) -> Dict[str, list]:
        """Get all configured assets."""
        return self.get('assets', {})
    
    def get_date_range(self) -> Dict[str, str]:
        """Get configured date range."""
        return self.get('date_range', {})
    
    def get_portfolios(self) -> Dict[str, Dict]:
        """Get portfolio configurations."""
        return self.get('portfolios', {})
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration."""
        return self.get('validation', {})
    
    def is_validation_enabled(self) -> bool:
        """Check if multi-source validation is enabled."""
        return self.get('validation.enabled', False)


if __name__ == "__main__":
    # Test configuration loader
    config = ConfigLoader()
    
    print("Assets:", config.get_assets())
    print("Date Range:", config.get_date_range())
    print("Validation Enabled:", config.is_validation_enabled())
