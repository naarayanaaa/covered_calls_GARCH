import pandas as pd
import numpy as np
import warnings
from datetime import timedelta

# --- CORRECTED IMPORTS (Package Execution) ---
from covered_calls.io.ingest import fetch_data
from covered_calls.models.garch import GarchModel

warnings.filterwarnings("ignore")

def run_backtest(ticker="GME", window_years=2, target_prob=0.80, horizon_days=7):
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
    
    hits = 0
    total = 0
    brier_scores = []
    
    print(f"Running backtest on {len(test_indices)} periods...")
    
    for t in test_indices:
        train_data = daily.iloc[:t]
        current_price = train_data['close'].iloc[-1]
        
        future_data = daily.iloc[t : t + horizon_days]
        final_price = future_data['close'].iloc[-1]
        
        try:
            garch = GarchModel(train_data['log_ret'])
            sim_prices, _ = garch.simulate_paths(n_days=horizon_days, n_paths=10000, current_price=current_price)
            
            terminal_prices = sim_prices[:, -1]
            k_80 = np.percentile(terminal_prices, target_prob * 100)
            
            is_otm = final_price <= k_80
            
            hits += 1 if is_otm else 0
            total += 1
            
            outcome = 1.0 if is_otm else 0.0
            brier_scores.append((outcome - target_prob) ** 2)
            
        except Exception as e:
            continue

    if total == 0:
        print("No valid backtest steps completed.")
        return

    hit_rate = hits / total
    avg_brier = np.mean(brier_scores)
    
    print("\n" + "="*40)
    print(f"BACKTEST RESULTS: {ticker}")
    print("="*40)
    print(f"Total Weeks Tested: {total}")
    print(f"Target Probability: {target_prob:.1%}")
    print(f"Realized Hit Rate:  {hit_rate:.1%}")
    print(f"Brier Score:        {avg_brier:.4f}")
    print("="*40)

if __name__ == "__main__":
    run_backtest()