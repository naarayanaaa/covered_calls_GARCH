import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .interface import DataProvider
from ..config import Config

# --- ALPACA IMPORTS ---
try:
    from alpaca.data.historical import StockHistoricalDataClient, OptionHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, OptionChainRequest
    from alpaca.data.timeframe import TimeFrame
    # Removed 'StockFeed' import to avoid version conflicts
except ImportError:
    pass 

# Fallback import
from .yahoo_feed import YahooProvider

class AlpacaProvider(DataProvider):
    def __init__(self):
        self.cfg = Config()
        if not self.cfg.alpaca_key or not self.cfg.alpaca_secret:
            raise ValueError("Alpaca API keys not found in environment variables.")
            
        self.stock_client = StockHistoricalDataClient(self.cfg.alpaca_key, self.cfg.alpaca_secret)
        self.option_client = OptionHistoricalDataClient(self.cfg.alpaca_key, self.cfg.alpaca_secret)

    def fetch_data(self, ticker: str, lookback_years: int = 6):
        print(f"[AlpacaProvider] Fetching data for {ticker}...")
        
        # 1. Setup Dates
        end_dt = datetime.now()
        start_dt_daily = end_dt - timedelta(days=lookback_years*365)
        
        # 2. Daily History (Force IEX Feed for Free Data)
        try:
            req_daily = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=start_dt_daily,
                end=end_dt,
                feed="iex"  # <--- CRITICAL FIX: Use string "iex" instead of Enum
            )
            bars_daily = self.stock_client.get_stock_bars(req_daily)
            daily_df = bars_daily.df.reset_index()
        except Exception as e:
            raise ValueError(f"Failed to fetch Daily Data from Alpaca. Check API Keys. Error: {e}")
        
        # Map Columns
        daily_df = daily_df.rename(columns={
            'timestamp': 'timestamp', 
            'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'
        })
        daily_df['timestamp'] = daily_df['timestamp'].dt.tz_localize(None)
        spot = daily_df['close'].iloc[-1]

        # 3. Intraday History (Force IEX Feed)
        start_dt_intra = end_dt - timedelta(days=59)
        req_intra = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Minute, 
            start=start_dt_intra,
            end=end_dt,
            feed="iex"  # <--- CRITICAL FIX: Use string "iex"
        )
        bars_intra = self.stock_client.get_stock_bars(req_intra)
        intraday_df = bars_intra.df.reset_index()
        intraday_df = intraday_df.rename(columns={
            'timestamp': 'timestamp', 
            'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'
        })
        intraday_df['timestamp'] = intraday_df['timestamp'].dt.tz_localize(None)

        # 4. Options Chain
        print("Fetching Options Chain...")
        
        try:
            # Try Alpaca Options First
            options = self._fetch_alpaca_options(ticker, spot)
        except Exception as e:
            # Check if it's a permission error or data error
            print(f"  [WARNING] Alpaca Options API failed (likely requires paid sub). Error: {e}")
            print("  -> Falling back to Yahoo Finance for Options Data only...")
            
            # HYBRID FALLBACK: Use Yahoo just for options
            yahoo_provider = YahooProvider()
            # We only need the options part from Yahoo, but the interface returns all 4.
            # We'll discard the yahoo price data and keep our clean Alpaca price data.
            _, _, options, _ = yahoo_provider.fetch_data(ticker, lookback_years=1)
            
            # Ensure the yahoo options dataframe has the 'underlying_price' from Alpaca
            options['underlying_price'] = spot

        return daily_df, intraday_df, options, spot

    def _fetch_alpaca_options(self, ticker, spot):
        """Helper to try fetching options from Alpaca."""
        req_chain = OptionChainRequest(underlying_symbol=ticker)
        chain_res = self.option_client.get_option_chain(req_chain)
        
        processed_rows = []
        
        for symbol, snapshot in chain_res.items():
            try:
                # Basic OSI Parse: Root + YYMMDD + Type + Strike
                root_len = len(ticker)
                remainder = symbol[root_len:]
                yymmdd = remainder[:6]
                type_char = remainder[6]
                strike_str = remainder[7:]
                
                exp_date = datetime.strptime(yymmdd, "%y%m%d")
                strike = float(strike_str) / 1000.0
                side = 'call' if type_char == 'C' else 'put'
                
                dte = (exp_date - datetime.now()).days
                
                if not (0 <= dte <= 45):
                    continue
                
                # Extract Data
                bid = snapshot.latest_quote.bid_price if snapshot.latest_quote else 0.0
                ask = snapshot.latest_quote.ask_price if snapshot.latest_quote else 0.0
                last = snapshot.latest_trade.price if snapshot.latest_trade else 0.0
                oi = snapshot.open_interest if snapshot.open_interest else 0
                iv = snapshot.greeks.iv if snapshot.greeks else 0.0
                
                processed_rows.append({
                    'strike': strike,
                    'expiration': exp_date,
                    'side': side,
                    'bid': bid,
                    'ask': ask,
                    'mid': (bid + ask) / 2 if (bid and ask) else 0.0,
                    'lastPrice': last,
                    'openInterest': oi,
                    'impliedVolatility': iv,
                    'dte': dte,
                    'underlying_price': spot
                })
                
            except Exception:
                continue

        if not processed_rows:
            raise ValueError(f"No valid options data retrieved for {ticker}.")

        return pd.DataFrame(processed_rows)