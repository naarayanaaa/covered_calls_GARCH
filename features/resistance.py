import pandas as pd
import numpy as np

def detect_resistance(daily: pd.DataFrame, intraday: pd.DataFrame, current_price: float):
    """
    Returns a DataFrame of resistance levels with 'level', 'type', 'strength'.
    """
    levels = []
    
    # 1. Swing Highs (Donchian Channel Top)
    # We look at the max of the last 20 and 50 days
    for window in [20, 50, 100]:
        val = daily['high'].rolling(window).max().iloc[-1]
        if val > current_price:
            levels.append({'level': val, 'type': f'{window}d_high', 'strength': 1.0})
            
    # 2. Simple Moving Averages
    for window in [20, 50, 200]:
        val = daily['close'].rolling(window).mean().iloc[-1]
        if val > current_price:
            levels.append({'level': val, 'type': f'SMA_{window}', 'strength': 0.8})
            
    # 3. Anchored VWAP (Approximate from Intraday)
    if not intraday.empty:
        # Calculate VWAP of the available intraday window
        intraday['tp'] = (intraday['high'] + intraday['low'] + intraday['close']) / 3
        intraday['cum_vol'] = intraday['volume'].cumsum()
        intraday['cum_vol_price'] = (intraday['tp'] * intraday['volume']).cumsum()
        vwap = intraday['cum_vol_price'].iloc[-1] / intraday['cum_vol'].iloc[-1]
        
        if vwap > current_price:
            levels.append({'level': vwap, 'type': 'VWAP_session', 'strength': 0.6})
            
    df = pd.DataFrame(levels)
    if df.empty:
        # Fallback if at ATH: use psychological levels (e.g. nearest 10 or 50)
        fallback = np.ceil(current_price / 10) * 10
        if fallback == current_price: fallback += 10
        df = pd.DataFrame([{'level': fallback, 'type': 'psychological', 'strength': 0.5}])
        
    df = df.sort_values('level').reset_index(drop=True)
    return df