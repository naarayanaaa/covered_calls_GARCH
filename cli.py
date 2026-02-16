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
from .options.iv_surface import fit_svi
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

    # Calculate Log Returns safely
    daily['prev_close'] = daily['close'].shift(1)
    
    # Drop rows where price is 0 or NaN
    daily = daily[(daily['close'] > 0) & (daily['prev_close'] > 0)]
    
    # Calculate log returns
    daily['log_ret'] = np.log(daily['close'] / daily['prev_close'])
    
    # Explicitly drop NaN and Infinite values
    daily = daily.dropna(subset=['log_ret'])
    daily = daily[np.isfinite(daily['log_ret'])]
    
    # Verification: Check sufficient data for GARCH
    if len(daily) < 252: # Prefer at least 1 year, but hard stop at something small
        print(f"Error: Insufficient daily data ({len(daily)} rows). Need at least 252.")
        return
    
    if daily.empty:
        print("Error: No valid daily return data after cleaning.")
        return

    # 2. Resistance (Clustered)
    print("Detecting resistance...")
    res_df = detect_resistance(daily, intraday, spot, cfg.res_zone_width)
    if not res_df.empty:
        print(f"Immediate Resistance Zone: {res_df.iloc[0]['level']:.2f} (Str: {res_df.iloc[0]['strength']:.1f})")
    
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
        
        # Fit SVI Surface
        strikes = clean_group['strike'].values
        # Fill missing IVs with naive mean if necessary for fitting attempt
        clean_group['impliedVolatility'] = clean_group['impliedVolatility'].fillna(0.5)
        ivs = clean_group['impliedVolatility'].values
        
        T_years = dte / 365.0
        if T_years == 0: T_years = 1/365.0
        
        svi_params = fit_svi(strikes, ivs, T_years, spot)
        
        # Run Monte Carlo for each strike in Calls
        # Simulate Paths
        n_days = max(int(dte), 1) # Trading days approx
        sim_prices, last_vol = garch.simulate_paths(n_days, cfg.mc_paths, spot)
        
        p_otm_dict = {}
        calls = clean_group[clean_group['side'] == 'call']
        
        for k in calls['strike'].unique():
            # Pass scalar last_vol (daily)
            p_otm, se_otm, p_touch = calculate_probabilities(
                sim_prices, k, T_years, last_vol
            )
            # LCB (Lower Confidence Bound)
            z_score = 1.645 # 95% one-sided
            lcb = p_otm - z_score * se_otm
            p_otm_dict[k] = (p_otm, lcb, p_touch)
            
        # Optimize with SVI params
        rec = select_strike(clean_group, p_otm_dict, res_df, cfg, svi_params)
        if rec:
            rec['dte'] = dte
            recommendations.append(rec)
            print(f"  -> Recommended: Strike {rec['strike']} (Prob: {rec['p_otm']:.2%}, Delta: {rec['delta']:.2f})")
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
