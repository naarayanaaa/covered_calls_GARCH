import pandas as pd
import numpy as np
import warnings
import sys
import os
from datetime import timedelta

# Fix for 'io' module conflict
sys.path.append(os.path.join(os.path.dirname(__file__), 'covered_calls')) # Adjust import path if needed based on root
from covered_calls.io.ingest import fetch_data
from covered_calls.models.garch import GarchModel

warnings.filterwarnings("ignore")

def run_backtest(ticker="GME", window_years=2, horizon_days=7):
    print(f"--- Starting Probability Calibration Backtest for {ticker} ---")
    
    # 1. Get History
    print("Fetching historical data...")
    daily, _, _, _ = fetch_data(ticker, lookback_years=window_years + 2)
    
    # Prep data
    daily['prev_close'] = daily['close'].shift(1)
    daily = daily[(daily['close'] > 0) & (daily['prev_close'] > 0)]
    daily['log_ret'] = np.log(daily['close'] / daily['prev_close'])
    daily = daily.dropna(subset=['log_ret'])
    daily = daily[np.isfinite(daily['log_ret'])].reset_index(drop=True)
    
    # 2. Define Test Points
    min_history = 252
    if len(daily) < min_history + horizon_days:
        print("Not enough history for backtest.")
        return

    test_indices = range(min_history, len(daily) - horizon_days, 5)
    
    # Parameter Sweep Settings
    target_probs = [0.60, 0.65, 0.70, 0.75, 0.80]
    results = {tp: {'hits': 0, 'total': 0, 'brier': []} for tp in target_probs}
    
    print(f"Running backtest on {len(test_indices)} periods across {len(target_probs)} aggression levels...")
    
    for i, t in enumerate(test_indices):
        if i % 10 == 0: print(f"Processing step {i}/{len(test_indices)}...")
        
        train_data = daily.iloc[:t]
        current_price = train_data['close'].iloc[-1]
        
        future_data = daily.iloc[t : t + horizon_days]
        final_price = future_data['close'].iloc[-1]
        
        try:
            garch = GarchModel(train_data['log_ret'])
            # Use price clamping from recent fix in model
            sim_prices, _ = garch.simulate_paths(n_days=horizon_days, n_paths=10000, current_price=current_price)
            terminal_prices = sim_prices[:, -1]
            
            # Check each target probability against this simulation
            for tp in target_probs:
                k_target = np.percentile(terminal_prices, tp * 100)
                is_otm = final_price <= k_target
                
                results[tp]['total'] += 1
                if is_otm:
                    results[tp]['hits'] += 1
                
                outcome = 1.0 if is_otm else 0.0
                results[tp]['brier'].append((outcome - tp) ** 2)
                
        except Exception as e:
            continue

    if results[0.60]['total'] == 0:
        print("No valid backtest steps completed.")
        return

    # 3. Output Table
    print("\n" + "="*60)
    print(f"BACKTEST RESULTS: {ticker} ({horizon_days} Days Horizon)")
    print("="*60)
    print(f"{'Target Prob':<12} | {'Realized Win':<14} | {'Brier':<8} | {'Status'}")
    print("-" * 60)
    
    for tp in sorted(target_probs):
        res = results[tp]
        if res['total'] == 0: continue
        
        realized = res['hits'] / res['total']
        brier = np.mean(res['brier'])
        
        status = "OK"
        if realized < tp - 0.05: status = "RISKY"
        elif realized > tp + 0.10: status = "SAFE++"
        elif realized > tp + 0.05: status = "SAFE"
        
        print(f"{tp:<12.0%} | {realized:<14.1%} | {brier:<8.4f} | {status}")
        
    print("="*60)
    print("Aggression Tuning Guide:")
    print(" - SAFE++: You are leaving money on the table. Lower target.")
    print(" - RISKY:  Model is underestimating risk. Raise target.")
    print(" - OK:     Model is calibrated.")

if __name__ == "__main__":
    run_backtest()