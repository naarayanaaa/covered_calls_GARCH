import pandas as pd
import numpy as np

def clean_options(df: pd.DataFrame, min_oi=10, min_mid=0.05):
    """
    Computes mid, spread, cleans crossed quotes.
    Includes fallback to 'lastPrice' if Bid/Ask are zero (common with Yahoo).
    """
    df = df.copy()
    
    # 1. Handle NaNs from data feed
    df['openInterest'] = df['openInterest'].fillna(0)
    df['bid'] = df['bid'].fillna(0)
    df['ask'] = df['ask'].fillna(0)
    if 'lastPrice' not in df.columns:
        df['lastPrice'] = 0.0
    else:
        df['lastPrice'] = df['lastPrice'].fillna(0)
    
    # 2. Fallback Logic: Use LastPrice if Quotes are Missing
    # Check for rows where Bid & Ask are 0 but LastPrice exists
    missing_quotes = (df['bid'] <= 0) & (df['ask'] <= 0) & (df['lastPrice'] > 0)
    num_fallback = missing_quotes.sum()
    
    if num_fallback > 0:
        # Force Bid/Ask to match LastPrice so we can run analysis
        print(f"    [DATA FIX] Using 'lastPrice' for {num_fallback} contracts with missing quotes.")
        df.loc[missing_quotes, 'bid'] = df.loc[missing_quotes, 'lastPrice']
        df.loc[missing_quotes, 'ask'] = df.loc[missing_quotes, 'lastPrice']
        # Set spread to 0 for these implied quotes
    
    # 3. Compute Mid
    df['mid'] = (df['bid'] + df['ask']) / 2
    df['spread'] = df['ask'] - df['bid']
    df['rel_spread'] = np.where(df['mid'] > 0, df['spread'] / df['mid'], 0.0)
    
    # 4. Define Filters
    mask_bid_exists = (df['bid'] > 0)
    mask_ask_exists = (df['ask'] > 0) # Should be covered by fallback
    mask_not_crossed = (df['bid'] <= df['ask'])
    mask_min_mid = (df['mid'] >= min_mid)
    mask_min_oi = (df['openInterest'] >= min_oi)
    
    # Combined mask
    mask = (
        mask_bid_exists & 
        mask_ask_exists & 
        mask_not_crossed & 
        mask_min_mid & 
        mask_min_oi
    )
    
    cleaned = df[mask].reset_index(drop=True)
    
    # --- DIAGNOSTICS ---
    if cleaned.empty and not df.empty:
        print(f"    >>> DIAGNOSTIC REPORT ({len(df)} Raw Contracts) <<<")
        print(f"    - Zero Bid:       {len(df) - mask_bid_exists.sum()} rejected")
        print(f"    - Zero Ask:       {len(df) - mask_ask_exists.sum()} rejected")
        print(f"    - Low Price:      {len(df) - mask_min_mid.sum()} rejected (< ${min_mid})")
        print(f"    - Low Liquidity:  {len(df) - mask_min_oi.sum()} rejected (< {min_oi} OI)")
        
        print(f"    - SAMPLE RAW DATA (First 1 row):")
        cols = ['strike', 'bid', 'ask', 'lastPrice', 'openInterest']
        print(df[cols].iloc[0].to_dict())
        print("    " + "-"*40)
        
    return cleaned