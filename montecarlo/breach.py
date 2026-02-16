import numpy as np
from numba import jit

@jit(nopython=True)
def brownian_bridge_touch(S_start, S_end, Barrier, sigma, dt):
    """
    Probability of touching Barrier in (0, dt) given start and end,
    assuming GBM locally.
    """
    if S_start >= Barrier or S_end >= Barrier:
        return 1.0
        
    # Standard Bridge Formula
    # P(hit) = exp( -2 * (B - S_0)*(B - S_T) / (sigma^2 * T) )
    # Note: working with Prices directly (approx) or Logs.
    # Logs is more accurate for GBM:
    x0 = np.log(S_start)
    xT = np.log(S_end)
    b = np.log(Barrier)
    
    if sigma < 1e-9: return 0.0
    
    arg = -2 * (b - x0) * (b - xT) / (sigma**2 * dt)
    return np.exp(arg)

@jit(nopython=True)
def _bridge_loop(prices, strike, T_years, vol_annual):
    n_paths = prices.shape[0]
    total_touch_prob = 0.0
    
    for i in range(n_paths):
        # 1. Discrete check (Max of path)
        path_max = -np.inf
        for p in prices[i]:
            if p > path_max:
                path_max = p
        
        if path_max >= strike:
            total_touch_prob += 1.0
        else:
            # 2. Bridge check (Conditional probability)
            # using start and end of the full path
            s_start = prices[i, 0]
            s_end = prices[i, -1]
            
            # vol is annualized, dt is T_years
            prob = brownian_bridge_touch(s_start, s_end, strike, vol_annual, T_years)
            total_touch_prob += prob
            
    return total_touch_prob / n_paths

def calculate_probabilities(prices, strike, T_years, vol_daily):
    """
    prices: (n_paths, n_steps + 1)
    strike: float
    T_years: float
    vol_daily: scalar volatility (daily)
    """
    n_paths, n_steps = prices.shape
    
    # 1. Terminal Breach (Expires ITM)
    final_prices = prices[:, -1]
    is_itm = final_prices > strike
    p_itm = np.mean(is_itm)
    p_otm = 1.0 - p_itm
    
    # 2. Standard Error for P_OTM
    se_otm = np.sqrt(p_otm * (1 - p_otm) / n_paths)
    
    # 3. Path Breach (Touch) with Bridge Correction
    # Convert daily vol to annual for the bridge formula (assuming T_years is annual)
    vol_annual = vol_daily * np.sqrt(365.0)
    
    p_touch = _bridge_loop(prices, float(strike), float(T_years), float(vol_annual))
    
    return p_otm, se_otm, p_touch
