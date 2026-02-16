import numpy as np
import pandas as pd
from scipy.stats import norm

def select_strike(
    df_opts: pd.DataFrame, 
    p_otm_dict: dict, 
    resistance_df: pd.DataFrame, 
    config
):
    """
    df_opts: options chain for ONE expiry
    p_otm_dict: {strike: (p_otm, lcb, p_touch)}
    """
    if df_opts.empty: return None

    # Filter to Calls
    calls = df_opts[df_opts['side'] == 'call'].copy()
    
    candidates = []
    
    spot = calls['underlying_price'].iloc[0]
    
    # Get Immediate Resistance (First one above spot)
    res_candidates = resistance_df[resistance_df['level'] > spot]
    if res_candidates.empty:
        # No resistance found? fallback
        res_level = spot * 1.10
    else:
        res_level = res_candidates.iloc[0]['level']

    for idx, row in calls.iterrows():
        K = row['strike']
        if K <= spot: continue
        
        # Get Probabilities
        if K not in p_otm_dict: continue
        p_otm, p_lcb, p_touch = p_otm_dict[K]
        
        # Constraint: LCB >= Target
        if p_lcb < config.p_target_min:
            continue
            
        # Constraint: P_Touch <= Cap
        if config.touch_cap and p_touch > config.touch_cap:
            continue
            
        # Metric: Distance to Resistance
        dist_res = abs(K - res_level)
        
        # Metric: Premium Yield (Annualized)
        # Yield = (Premium / Spot) * (365 / DTE)
        dte = max(row['dte'], 1)
        yld = (row['mid'] / spot) * (365 / dte)
        
        # Objective Function
        # Maximize Yield - Penalty * Distance
        # We normalize distance by spot
        score = yld - config.lambda_res * (dist_res / spot)
        
        candidates.append({
            'strike': K,
            'expiration': row['expiration'],
            'type': 'call',
            'bid': row['bid'],
            'ask': row['ask'],
            'mid': row['mid'],
            'p_otm': p_otm,
            'p_lcb': p_lcb,
            'p_touch': p_touch,
            'resistance': res_level,
            'dist_res': dist_res,
            'yield': yld,
            'score': score
        })
        
    if not candidates:
        return None
        
    # Sort by score descending
    df_cand = pd.DataFrame(candidates)
    best = df_cand.sort_values('score', ascending=False).iloc[0]
    return best.to_dict()