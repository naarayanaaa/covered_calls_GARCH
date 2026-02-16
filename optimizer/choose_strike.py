import numpy as np
import pandas as pd
from options.iv_surface import get_iv_from_surface
from options.greeks import calculate_delta

def select_strike(
    df_opts: pd.DataFrame,
    p_otm_dict: dict,
    resistance_df: pd.DataFrame,
    config,
    svi_params=None
):
    """
    df_opts: options chain for ONE expiry
    p_otm_dict: {strike: (p_otm, lcb, p_touch)}
    svi_params: Fitted SVI parameters for this expiry (optional)
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
        res_strength = 1.0
    else:
        res_level = res_candidates.iloc[0]['level']
        res_strength = res_candidates.iloc[0]['strength']

    for idx, row in calls.iterrows():
        K = row['strike']
        if K <= spot: continue

        # --- Task 2: Liquidity Filter ---
        oi = row.get('openInterest', 0)
        if pd.isna(oi): oi = 0
        if oi < config.min_oi:
            continue

        # Get Probabilities
        if K not in p_otm_dict: continue
        p_otm, p_lcb, p_touch = p_otm_dict[K]

        # 1. Constraint: LCB >= Target
        if p_lcb < config.p_target_min:
            continue

        # 2. Constraint: P_Touch <= Cap
        if config.touch_cap and p_touch > config.touch_cap:
            continue

        # 3. Model IV & Delta
        T_years = max(row['dte'], 1) / 365.0

        if svi_params is not None:
            model_iv = get_iv_from_surface(K, T_years, spot, svi_params)
        else:
            model_iv = row['impliedVolatility']

        delta = calculate_delta(spot, K, T_years, risk_free=0.04, vol=model_iv)

        # 4. Constraint: Max Delta
        if delta > config.max_delta:
            continue

        # --- Task 2: Effective Premium Calculation ---
        # Don't trust Mid if spread is wide
        bid = row['bid']
        ask = row['ask']
        mid = row['mid']
        spread = row['spread']
        rel_spread = row['rel_spread'] if not pd.isna(row['rel_spread']) else 0.0

        if rel_spread < config.liquidity_spread_thresh:
            effective_price = mid
        else:
            # Assume we must cross the spread somewhat to get filled on illiquid options
            effective_price = bid + (config.liquidity_crossing_factor * spread)

        # Metric: Premium Yield (Annualized) using Effective Price
        dte = max(row['dte'], 1)
        yld = (effective_price / spot) * (365 / dte)

        # Metric: Distance to Resistance
        dist_res = abs(K - res_level)

        # Objective Function
        score = yld \
                - config.lambda_res * (dist_res / spot) \
                - config.lambda_risk * delta

        candidates.append({
            'strike': K,
            'expiration': row['expiration'],
            'type': 'call',
            'bid': bid,
            'ask': ask,
            'mid': mid,
            'effective_price': effective_price,
            'p_otm': p_otm,
            'p_lcb': p_lcb,
            'p_touch': p_touch,
            'model_iv': model_iv,
            'delta': delta,
            'resistance': res_level,
            'res_strength': res_strength,
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
