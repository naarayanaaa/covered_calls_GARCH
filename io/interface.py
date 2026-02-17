from abc import ABC, abstractmethod
import pandas as pd

class DataProvider(ABC):
    @abstractmethod
    def fetch_data(self, ticker: str, lookback_years: int):
        """
        Fetches market data for the given ticker.
        
        Returns:
            tuple: (daily_df, intraday_df, options_df, spot_price)
            
            daily_df columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            intraday_df columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            options_df columns: ['strike', 'expiration', 'bid', 'ask', 'mid', 
                               'openInterest', 'impliedVolatility', 'lastPrice', 
                               'side', 'dte', 'underlying_price']
            spot_price: float
        """
        pass