import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .interface import DataProvider

class YahooProvider(DataProvider):
    def fetch_data(self, ticker: str, lookback_years: int = 6):
        """Fetches daily, intraday, and options chain using yfinance."""
        print(f"[YahooProvider] Fetching data for {ticker}...")
        tk = yf.Ticker(ticker)
        
        # 1. Daily History (for GARCH)
        daily = tk.history(period=f"{lookback_years}y", interval="1d", auto_adjust=False)
        if daily.empty:
            raise ValueError(f"No daily data for {ticker}")
        
        daily = daily.reset_index()
        # Robust Column Renaming
        daily.columns = [str(c).lower() for c in daily.columns]
        rename_map = {'date': 'timestamp', 'datetime': 'timestamp', 'index': 'timestamp'}
        daily = daily.rename(columns=rename_map)
        
        if 'timestamp' in daily.columns:
            daily['timestamp'] = pd.to_datetime(daily['timestamp']).dt.tz_localize(None)

        # 2. Intraday (for VWAP/0-7 DTE context)
        intraday = tk.history(period="60d", interval="5m", auto_adjust=False)
        if not intraday.empty:
            intraday = intraday.reset_index()
            # Robust Column Renaming for Intraday
            intraday.columns = [str(c).lower() for c in intraday.columns]
            intraday = intraday.rename(columns=rename_map)
            
            # Fallback: if still no timestamp, assume the first column is it
            if 'timestamp' not in intraday.columns and not intraday.empty:
                first_col = intraday.columns[0]
                intraday = intraday.rename(columns={first_col: 'timestamp'})
            
            if 'timestamp' in intraday.columns:
                intraday['timestamp'] = pd.to_datetime(intraday['timestamp']).dt.tz_localize(None)
        
        # 3. Spot Price (Initial Guess)
        try:
            spot = tk.fast_info['last_price']
        except:
            spot = daily['close'].iloc[-1]

        # 4. Options Chain
        expiries = tk.options
        today = datetime.now()
        options_dfs = []
        
        latest_valid_spot = None # For synchronization logic
        
        for exp_str in expiries:
            exp_date = pd.to_datetime(exp_str)
            dte = (exp_date - today).days
            if 0 <= dte <= 45:
                try:
                    chain = tk.option_chain(exp_str)
                    
                    # --- Task 2: Synchronize Spot Price ---
                    # Try to get the underlying price from the chain metadata to match the quotes
                    if hasattr(chain, 'underlying') and chain.underlying:
                        if 'regularMarketPrice' in chain.underlying:
                            latest_valid_spot = chain.underlying['regularMarketPrice']
                    # ---------------------------------------
                    
                    calls = chain.calls
                    puts = chain.puts
                    
                    if calls.empty and puts.empty:
                        continue
                        
                    calls['side'] = 'call'
                    puts['side'] = 'put'
                    
                    df = pd.concat([calls, puts])
                    df['expiration'] = exp_date
                    df['dte'] = dte
                    df['underlying_price'] = spot # Will update later if sync found
                    
                    options_dfs.append(df)
                except Exception as e:
                    # Silent skip for individual bad expiries
                    continue

        if not options_dfs:
            options = pd.DataFrame()
        else:
            options = pd.concat(options_dfs, ignore_index=True)
            
            # Apply the synchronized spot price if found
            if latest_valid_spot:
                print(f"  [Sync] Updating Spot Price from Options Chain: {spot:.2f} -> {latest_valid_spot:.2f}")
                spot = latest_valid_spot
                options['underlying_price'] = spot
        
        return daily, intraday, options, spot