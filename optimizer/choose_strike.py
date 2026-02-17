import numpy as np
import pandas as pd
from ..options.iv_surface import get_iv_from_surface
from ..options.greeks import calculate_delta

def select_strike(
    df_opts: pd.DataFrame, 
    p_otm_dict: dict, 
    resistance_df: pd.DataFrame, 
    config,
    svi_params=None
):
    if df_opts.empty: return None

    # Filter to Calls
    calls = df_opts[df_opts['side'] == 'call'].copy()
    candidates = []
    spot = calls['underlying_price'].iloc[0]
    
    # Get Immediate Resistance
    res_candidates = resistance_df[resistance_df['level'] > spot]
    if res_candidates.empty:
        res_level = spot * 1.10
        res_strength = 1.0
    else:
        res_level = res_candidates.iloc[0]['level']
        res_strength = res_candidates.iloc[0]['strength']

    print(f"\n--- DEBUG: Analyzing {len(calls)} Calls for Expiry {calls.iloc[0]['expiration']} ---")
    print(f"Spot: {spot:.2f} | Resistance: {res_level:.2f}")

    for idx, row in calls.iterrows():
        K = row['strike']
        if K <= spot: continue
        
        # --- Task 2: Liquidity Filter ---
        oi = row.get('openInterest', 0)
        if pd.isna(oi): oi = 0
        if oi < config.min_oi:
            print(f"Strike {K}: REJECTED (Low OI: {oi} < {config.min_oi})")
            continue
        
        # --- Task 3: Zombie & Spread Filter ---
        if row['bid'] < 0.05:
            print(f"Strike {K}: REJECTED (Zombie Quote: Bid {row['bid']:.2f})")
            continue
            
        if row['bid'] > 0:
            spread_pct = (row['ask'] - row['bid']) / row['bid']
            if spread_pct > 0.5:
                print(f"Strike {K}: REJECTED (Wide Spread: {spread_pct:.1%})")
                continue
        
        # --- Task 4: Transaction Costs & Net Premium ---
        # Calculate Effective Price first
        bid = row['bid']
        ask = row['ask']
        mid = row['mid']
        spread = row['spread']
        rel_spread = row['rel_spread'] if not pd.isna(row['rel_spread']) else 0.0
        
        if rel_spread < config.liquidity_spread_thresh:
            effective_price = mid
        else:
            effective_price = bid + (config.liquidity_crossing_factor * spread)
            
        # Commission Logic (Interactive Brokers)
        gross_premium = effective_price * 100
        net_premium = gross_premium - config.commission_fee
        
        if net_premium < config.min_premium_abs:
            print(f"Strike {K}: REJECTED (Net Premium ${net_premium:.2f} < ${config.min_premium_abs})")
            continue
            
        # Re-calculate yield based on Net Premium (per share basis)
        net_price_per_share = net_premium / 100.0
        dte = max(row['dte'], 1)
        yld = (net_price_per_share / spot) * (365 / dte)

        # Get Probabilities
        if K not in p_otm_dict: 
            print(f"Strike {K}: REJECTED (No Prob Data)")
            continue
        p_otm, p_lcb, p_touch = p_otm_dict[K]
        
        # Constraints
        if p_lcb < config.p_target_min:
            print(f"Strike {K}: REJECTED (Unsafe: LCB {p_lcb:.2%} < {config.p_target_min:.1%})")
            continue
            
        if config.touch_cap and p_touch > config.touch_cap:
            print(f"Strike {K}: REJECTED (Touch Risk: {p_touch:.2%} > {config.touch_cap:.1%})")
            continue
            
        # Model IV & Delta
        T_years = max(row['dte'], 1) / 365.0
        if svi_params is not None:
            model_iv = get_iv_from_surface(K, T_years, spot, svi_params)
        else:
            model_iv = row['impliedVolatility']
            
        delta = calculate_delta(spot, K, T_years, risk_free=0.04, vol=model_iv)
        
        if delta > config.max_delta:
            print(f"Strike {K}: REJECTED (High Delta: {delta:.2f} > {config.max_delta})")
            continue
            
        dist_res = abs(K - res_level)
        
        score = yld \
                - config.lambda_res * (dist_res / spot) \
                - config.lambda_risk * delta
        
        print(f"Strike {K}: ACCEPTED (Score: {score:.4f}, Prob: {p_lcb:.1%}, Net: ${net_premium:.2f})")

        candidates.append({
            'strike': K,
            'expiration': row['expiration'],
            'type': 'call',
            'bid': bid,
            'ask': ask,
            'mid': mid,
            'effective_price': effective_price,
            'net_premium': net_premium,  # Store net
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
        
    df_cand = pd.DataFrame(candidates)
    best = df_cand.sort_values('score', ascending=False).iloc[0]
    return best.to_dict()