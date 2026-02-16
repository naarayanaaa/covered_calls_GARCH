import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_data(ticker: str, lookback_years: int = 6):
    """Fetches daily, intraday, and options chain."""
    tk = yf.Ticker(ticker)
    
    # 1. Daily History (for GARCH)
    daily = tk.history(period=f"{lookback_years}y", interval="1d", auto_adjust=False)
    if daily.empty:
        raise ValueError(f"No daily data for {ticker}")
    daily = daily.reset_index()
    # Normalize columns
    cols = {c: c.lower() for c in daily.columns}
    daily = daily.rename(columns=cols)
    if 'date' in daily.columns: daily = daily.rename(columns={'date': 'timestamp'})
    daily['timestamp'] = pd.to_datetime(daily['timestamp']).dt.tz_localize(None)

    # 2. Intraday (for VWAP/0-7 DTE context)
    # yfinance limits: 7d for 1m, 60d for 5m. We try 5m.
    intraday = tk.history(period="60d", interval="5m", auto_adjust=False)
    if not intraday.empty:
        intraday = intraday.reset_index().rename(columns={c: c.lower() for c in intraday.columns})
        if 'datetime' in intraday.columns: intraday = intraday.rename(columns={'datetime': 'timestamp'})
    
    # 3. Spot Price
    try:
        spot = tk.fast_info['last_price']
    except:
        spot = daily['close'].iloc[-1]

    # 4. Options Chain
    # We fetch all expiries within 0-45 days
    expiries = tk.options
    today = datetime.now()
    valid_exps = []
    
    options_dfs = []
    
    for exp_str in expiries:
        exp_date = pd.to_datetime(exp_str)
        dte = (exp_date - today).days
        if 0 <= dte <= 45:
            try:
                # Robust fetch
                chain = tk.option_chain(exp_str)
                calls = chain.calls
                puts = chain.puts
                
                if calls.empty and puts.empty:
                    continue
                    
                calls['side'] = 'call'
                puts['side'] = 'put'
                
                df = pd.concat([calls, puts])
                df['expiration'] = exp_date
                df['dte'] = dte
                
                # Stamp underlying price
                df['underlying_price'] = spot
                
                options_dfs.append(df)
            except Exception as e:
                # Log warning but continue
                print(f"Skipping expiry {exp_str}: Data unavailable ({e})")
                continue

    if not options_dfs:
        raise ValueError("No options data found within DTE window.")
        
    options = pd.concat(options_dfs, ignore_index=True)
    
    return daily, intraday, options, spot
