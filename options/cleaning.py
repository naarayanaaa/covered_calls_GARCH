import pandas as pd
import numpy as np

def clean_options(df: pd.DataFrame, min_oi=10, min_mid=0.05):
    """Computes mid, spread, cleans crossed quotes."""
    df = df.copy()
    
    # 1. Compute Mid
    df['mid'] = (df['bid'] + df['ask']) / 2
    df['spread'] = df['ask'] - df['bid']
    df['rel_spread'] = df['spread'] / df['mid']
    
    # 2. Filters
    mask = (
        (df['bid'] > 0) & 
        (df['ask'] > 0) &
        (df['bid'] <= df['ask']) &  # Not crossed
        (df['mid'] >= min_mid) &
        (df['openInterest'] >= min_oi)
    )
    return df[mask].reset_index(drop=True)