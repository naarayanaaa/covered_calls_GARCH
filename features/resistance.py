import pandas as pd
import numpy as np

def detect_resistance(daily: pd.DataFrame, intraday: pd.DataFrame, current_price: float, zone_width=0.01):
    """
    Returns a DataFrame of resistance levels with 'level', 'type', 'strength'.
    Implements clustering to merge nearby levels.
    """
    raw_levels = []
    
    # 1. Swing Highs (Donchian Channel Top)
    for window in [20, 50, 100]:
        val = daily['high'].rolling(window).max().iloc[-1]
        if val > current_price:
            raw_levels.append({'level': val, 'type': f'{window}d_high', 'strength': 1.0})
            
    # 2. Simple Moving Averages
    for window in [20, 50, 200]:
        val = daily['close'].rolling(window).mean().iloc[-1]
        if val > current_price:
            raw_levels.append({'level': val, 'type': f'SMA_{window}', 'strength': 0.8})
            
    # 3. Anchored VWAP (Approximate from Intraday)
    if not intraday.empty:
        intraday = intraday.copy()
        intraday['tp'] = (intraday['high'] + intraday['low'] + intraday['close']) / 3
        intraday['cum_vol'] = intraday['volume'].cumsum()
        intraday['cum_vol_price'] = (intraday['tp'] * intraday['volume']).cumsum()
        vwap = intraday['cum_vol_price'].iloc[-1] / intraday['cum_vol'].iloc[-1]
        
        if vwap > current_price:
            raw_levels.append({'level': vwap, 'type': 'VWAP_session', 'strength': 0.6})
            
    if not raw_levels:
        # Fallback if at ATH
        fallback = np.ceil(current_price / 10) * 10
        if fallback == current_price: fallback += 10
        raw_levels.append({'level': fallback, 'type': 'psychological', 'strength': 0.5})

    # Sort by level
    raw_levels.sort(key=lambda x: x['level'])
    
    # Clustering
    clustered = []
    if raw_levels:
        current_cluster = [raw_levels[0]]
        
        for i in range(1, len(raw_levels)):
            prev = current_cluster[-1]
            curr = raw_levels[i]
            
            # Check if within zone_width (percentage difference)
            if (curr['level'] - prev['level']) / prev['level'] <= zone_width:
                current_cluster.append(curr)
            else:
                # Process completed cluster
                clustered.append(_merge_cluster(current_cluster))
                current_cluster = [curr]
        
        # Process last cluster
        if current_cluster:
            clustered.append(_merge_cluster(current_cluster))
            
    df = pd.DataFrame(clustered)
    df = df.sort_values('level').reset_index(drop=True)
    return df

def _merge_cluster(cluster):
    if len(cluster) == 1:
        return cluster[0]
        
    total_strength = sum(item['strength'] for item in cluster)
    weighted_level = sum(item['level'] * item['strength'] for item in cluster) / total_strength
    
    types = [item['type'] for item in cluster]
    return {
        'level': weighted_level,
        'type': '+'.join(types),
        'strength': total_strength
    }
