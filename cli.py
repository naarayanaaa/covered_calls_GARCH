import argparse
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

from .config import Config
from .io.ingest import fetch_data
from .features.resistance import detect_resistance
from .options.cleaning import clean_options
from .options.iv_surface import fit_svi, get_iv_from_surface
from .models.garch import GarchModel
from .montecarlo.breach import calculate_probabilities
from .optimizer.choose_strike import select_strike

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="GME")
    parser.add_argument("--out_dir", type=str, default="./output")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config(ticker=args.ticker)
    
    print(f"--- Starting Analysis for {cfg.ticker} ---")
    
# 1. Ingest
    print("Fetching data...")
    daily, intraday, opts, spot = fetch_data(cfg.ticker)

    # --- FIX START ---
    # Calculate Log Returns safely
    daily['prev_close'] = daily['close'].shift(1)
    
    # 1. Drop rows where price is 0 or NaN to avoid divide-by-zero or log(0) errors
    daily = daily[(daily['close'] > 0) & (daily['prev_close'] > 0)]
    
    # 2. Calculate log returns
    daily['log_ret'] = np.log(daily['close'] / daily['prev_close'])
    
    # 3. Explicitly drop NaN and Infinite values
    daily = daily.dropna(subset=['log_ret'])
    daily = daily[np.isfinite(daily['log_ret'])]
    
    if daily.empty:
        print("Error: No valid daily return data after cleaning.")
        return
    # --- FIX END ---

    # 2. Resistance
    print("Detecting resistance...")
    # ... rest of the code ...
    res_df = detect_resistance(daily, intraday, spot)
    print(f"Immediate Resistance: {res_df.iloc[0]['level']:.2f} ({res_df.iloc[0]['type']})")
    
    # 3. Model Fitting
    print("Fitting GARCH Model...")
    garch = GarchModel(daily['log_ret'])
    
    # 4. Process Expiries
    recommendations = []
    
    # Filter opts for DTE Grid
    target_dtes = cfg.dte_grid
    
    # Group by Expiry
    for exp_date, group in opts.groupby('expiration'):
        dte = group.iloc[0]['dte']
        
        # Is this a DTE we care about? (Approximate match)
        if not any(abs(dte - t) <= 2 for t in target_dtes):
            continue
            
        print(f"Processing DTE {dte} ({exp_date.date()})...")
        
        # Clean
        clean_group = clean_options(group)
        if clean_group.empty: continue
        
        # Fit SVI Surface (Calls + Puts)
        # Using OTM options for fitting is standard
        # Simple implementation: use mids
        strikes = clean_group['strike'].values
        ivs = clean_group['impliedVolatility'].values # from Yahoo
        # If Yahoo IVs are bad/missing, we should solve BS. 
        # For this script we assume Yahoo IVs are populated or use fallback.
        # Fallback: fillna with mean
        clean_group['impliedVolatility'] = clean_group['impliedVolatility'].fillna(0.5)
        
        # Run Monte Carlo for each strike in Calls
        T_years = dte / 365.0
        if T_years == 0: T_years = 1/365.0
        
        # Simulate Paths
        n_days = max(int(dte), 1) # Trading days approx
        sim_prices, last_vol = garch.simulate_paths(n_days, cfg.mc_paths, spot)
        
        p_otm_dict = {}
        calls = clean_group[clean_group['side'] == 'call']
        
        for k in calls['strike'].unique():
            p_otm, se_otm, p_touch = calculate_probabilities(
                sim_prices, k, T_years, last_vol
            )
            # LCB (Lower Confidence Bound)
            z_score = 1.645 # 95% one-sided
            lcb = p_otm - z_score * se_otm
            p_otm_dict[k] = (p_otm, lcb, p_touch)
            
        # Optimize
        rec = select_strike(clean_group, p_otm_dict, res_df, cfg)
        if rec:
            rec['dte'] = dte
            recommendations.append(rec)
            print(f"  -> Recommended: Strike {rec['strike']} (Prob: {rec['p_otm']:.2%}, Yield: {rec['yield']:.2%})")
        else:
            print("  -> No feasible strike found.")

    # 5. Output
    out_file = os.path.join(args.out_dir, f"{cfg.ticker}_recommendations.json")
    
    # Convert dates to str for JSON
    for r in recommendations:
        r['expiration'] = str(r['expiration'])
        
    with open(out_file, 'w') as f:
        json.dump(recommendations, f, indent=2, cls=NumpyEncoder)
        
    print(f"Done. Results saved to {out_file}")

if __name__ == "__main__":
    main()