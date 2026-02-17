import argparse
import pandas as pd
import numpy as np
import json
import os
import glob
from datetime import datetime

# Relative imports for package execution
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

def load_earnings_dates(ticker, base_dir):
    search_path = os.path.join(base_dir, ticker, "**", "events_earnings.*")
    if not glob.glob(search_path):
        search_path = os.path.join("data", ticker, "**", "events_earnings.*")
    files = glob.glob(search_path, recursive=True)
    if not files: return pd.DataFrame()
    latest_file = max(files, key=os.path.getmtime)
    try:
        if latest_file.endswith('.parquet'): return pd.read_parquet(latest_file)
        else: return pd.read_csv(latest_file)
    except: return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="GME")
    parser.add_argument("--out_dir", type=str, default="./output")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config(ticker=args.ticker)
    
    print(f"--- Starting Analysis for {cfg.ticker} ---")
    
    print("Fetching data...")
    daily, intraday, opts, spot = fetch_data(cfg.ticker)

    daily['prev_close'] = daily['close'].shift(1)
    daily = daily[(daily['close'] > 0) & (daily['prev_close'] > 0)]
    daily['log_ret'] = np.log(daily['close'] / daily['prev_close'])
    daily = daily.dropna(subset=['log_ret'])
    daily = daily[np.isfinite(daily['log_ret'])]
    
    if len(daily) < 252:
        print(f"Error: Insufficient daily data ({len(daily)} rows).")
        return
    if daily.empty:
        print("Error: No valid daily return data.")
        return

    earnings_df = load_earnings_dates(cfg.ticker, args.out_dir)
    if not earnings_df.empty:
        print("Earnings data loaded. Event Shield Active.")
        if 'earnings_date' in earnings_df.columns:
            earnings_df['earnings_date'] = pd.to_datetime(earnings_df['earnings_date']).dt.tz_localize(None)
    else:
        print("Warning: Earnings data not found. Event Shield DISABLED.")

    print("Detecting resistance...")
    res_df = detect_resistance(daily, intraday, spot, cfg.res_zone_width)
    if not res_df.empty:
        print(f"Immediate Resistance Zone: {res_df.iloc[0]['level']:.2f} (Str: {res_df.iloc[0]['strength']:.1f})")
    
    print("Fitting GARCH Model...")
    garch = GarchModel(daily['log_ret'])
    
    recommendations = []
    target_dtes = cfg.dte_grid
    today = datetime.now()
    
    for exp_date, group in opts.groupby('expiration'):
        dte = group.iloc[0]['dte']
        
        if not any(abs(dte - t) <= 2 for t in target_dtes):
            continue
            
        print(f"Processing DTE {dte} ({exp_date.date()})...")
        
        if not earnings_df.empty:
            future_earnings = earnings_df[earnings_df['earnings_date'] > today]
            if not future_earnings.empty:
                next_earn = future_earnings.iloc[0]['earnings_date']
                days_to_earn = (next_earn - today).days
                if 0 <= days_to_earn <= dte + 1:
                    print(f"  [SKIP] Earnings in {days_to_earn} days.")
                    continue

        # --- FIX: Pass Config & Add Diagnostics ---
        clean_group = clean_options(group, min_oi=cfg.min_oi)
        
        if clean_group.empty:
            print(f"  [WARNING] All {len(group)} options dropped by cleaner (Check Bid/Ask/OI).")
            continue
        # ------------------------------------------
        
        strikes = clean_group['strike'].values
        clean_group['impliedVolatility'] = clean_group['impliedVolatility'].fillna(0.5)
        ivs = clean_group['impliedVolatility'].values
        
        T_years = dte / 365.0
        if T_years == 0: T_years = 1/365.0
        
        svi_params = fit_svi(strikes, ivs, T_years, spot)
        
        last_garch_sigma = garch.res.conditional_volatility.iloc[-1] / 100.0
        garch_vol_annual = last_garch_sigma * np.sqrt(252)
        
        calls = clean_group[clean_group['side'] == 'call']
        if not calls.empty:
            idx_atm = (calls['strike'] - spot).abs().idxmin()
            market_iv = calls.loc[idx_atm, 'impliedVolatility']
        else:
            market_iv = 0.0
            
        simulation_vol = last_garch_sigma
        
        if market_iv > 2.0 * garch_vol_annual:
            print(f"  [WARNING] Market IV ({market_iv:.2%}) > 2.0x GARCH. Using Market IV.")
            simulation_vol = market_iv / np.sqrt(252)
        
        n_days = max(int(dte), 1)
        sim_prices, _ = garch.simulate_paths(n_days, cfg.mc_paths, spot)
        
        p_otm_dict = {}
        for k in calls['strike'].unique():
            p_otm, se_otm, p_touch = calculate_probabilities(
                sim_prices, k, T_years, simulation_vol
            )
            z_score = 1.645
            lcb = p_otm - z_score * se_otm
            p_otm_dict[k] = (p_otm, lcb, p_touch)
            
        rec = select_strike(clean_group, p_otm_dict, res_df, cfg, svi_params)
        if rec:
            rec['dte'] = dte
            recommendations.append(rec)
            print(f"  -> Recommended: Strike {rec['strike']} (Prob: {rec['p_otm']:.2%}, Eff.Yield: {rec['yield']:.2%})")
        else:
            print("  -> No feasible strike found.")

    out_file = os.path.join(args.out_dir, f"{cfg.ticker}_recommendations.json")
    for r in recommendations:
        r['expiration'] = str(r['expiration'])
        
    with open(out_file, 'w') as f:
        json.dump(recommendations, f, indent=2, cls=NumpyEncoder)
        
    print(f"Done. Results saved to {out_file}")

if __name__ == "__main__":
    main()