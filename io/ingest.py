from .interface import DataProvider
from .yahoo_feed import YahooProvider
from .alpaca_feed import AlpacaProvider
from ..config import Config

def fetch_data(ticker: str, lookback_years: int = 6):
    """
    Factory function to fetch data from the configured provider.
    Returns: (daily_df, intraday_df, options_df, spot_price)
    """
    cfg = Config()
    provider_name = cfg.api_provider.lower()
    
    provider: DataProvider = None
    
    if provider_name == "alpaca":
        try:
            print(f"--- Initializing Alpaca Provider for {ticker} ---")
            provider = AlpacaProvider()
        except ImportError:
            print("Error: alpaca-py not installed. Falling back to Yahoo.")
            provider = YahooProvider()
        except ValueError as e:
            print(f"Error: {e}. Falling back to Yahoo.")
            provider = YahooProvider()
            
    elif provider_name == "yahoo":
        print(f"--- Initializing Yahoo Provider for {ticker} ---")
        provider = YahooProvider()
        
    else:
        print(f"Unknown provider '{provider_name}'. Defaulting to Yahoo.")
        provider = YahooProvider()
        
    return provider.fetch_data(ticker, lookback_years)