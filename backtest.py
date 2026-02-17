import pandas as pd
import numpy as np
import warnings
import sys
import os
from datetime import timedelta

# Fix for 'io' module conflict
sys.path.append(os.path.join(os.path.dirname(__file__), 'io'))
from ingest import fetch_data

from models.garch import GarchModel

# Suppress arch warnings
warnings.filterwarnings("ignore")

def run_backtest(ticker="GME", window_years=2, target_prob=0.80, horizon_days=7):
    print(f"--- Starting Probability Calibration Backtest for {ticker} ---")
    print(f"Target OTM Probability: {target_prob:.1%}")
    print(f"Horizon: {horizon_days} days")

    # 1. Get History (using existing ingestion logic)
    print("Fetching historical data...")
    daily, _, _, _ = fetch_data(ticker, lookback_years=window_years + 2)

    # Prep data
    daily['prev_close'] = daily['close'].shift(1)
    daily = daily[(daily['close'] > 0) & (daily['prev_close'] > 0)]
    daily['log_ret'] = np.log(daily['close'] / daily['prev_close'])
    daily = daily.dropna(subset=['log_ret'])
    daily = daily[np.isfinite(daily['log_ret'])].reset_index(drop=True)

    # 2. Define Test Points (Fridays)
    # We need at least 1 year of data for GARCH before the test window
    min_history = 252
    if len(daily) < min_history + horizon_days:
        print("Not enough history for backtest.")
        return

    # Identify test indices (every 5th day, roughly weekly, skipping first year)
    test_indices = range(min_history, len(daily) - horizon_days, 5)

    hits = 0
    total = 0
    brier_scores = []

    print(f"Running backtest on {len(test_indices)} periods...")

    for t in test_indices:
        # Train window
        train_data = daily.iloc[:t]
        current_price = train_data['close'].iloc[-1]
        current_date = train_data['timestamp'].iloc[-1]

        # Future outcome
        future_data = daily.iloc[t : t + horizon_days]
        final_price = future_data['close'].iloc[-1]

        # 3. Fit Model & Simulate
        try:
            # Fit GARCH on history up to t
            garch = GarchModel(train_data['log_ret'])

            # Simulate paths (10k is enough for backtest speed)
            sim_prices, _ = garch.simulate_paths(n_days=horizon_days, n_paths=10000, current_price=current_price)

            # 4. Calculate Theoretical Strike for 80% OTM
            # The strike K where P(S_T <= K) = 0.80 is the 80th percentile of terminal prices
            terminal_prices = sim_prices[:, -1]
            k_80 = np.percentile(terminal_prices, target_prob * 100)

            # 5. Evaluate
            is_otm = final_price <= k_80

            hits += 1 if is_otm else 0
            total += 1

            # Brier Score for this single event: (Outcome - Prob)^2
            # Outcome is 1 if OTM, 0 if ITM. Prob predicted was 0.80.
            outcome = 1.0 if is_otm else 0.0
            brier_scores.append((outcome - target_prob) ** 2)

            if total % 10 == 0:
                print(f"Step {total}/{len(test_indices)}: {current_date.date()} | Price: {current_price:.2f} -> Strike: {k_80:.2f} -> Final: {final_price:.2f} | {'WIN' if is_otm else 'LOSS'}")

        except Exception as e:
            print(f"Error at step {t}: {e}")
            continue

    if total == 0:
        print("No valid backtest steps completed.")
        return

    # 6. Summary Stats
    hit_rate = hits / total
    avg_brier = np.mean(brier_scores)

    print("\n" + "="*40)
    print(f"BACKTEST RESULTS: {ticker}")
    print("="*40)
    print(f"Total Weeks Tested: {total}")
    print(f"Target Probability: {target_prob:.1%}")
    print(f"Realized Hit Rate:  {hit_rate:.1%}")
    print(f"Brier Score:        {avg_brier:.4f} (Lower is better)")

    diff = hit_rate - target_prob
    if diff < -0.05:
        print("VERDICT: Model is UNDER-CALIBRATED (Too Risky).")
        print("Suggestion: Increase volatility scaling or use fatter-tailed distribution.")
    elif diff > 0.05:
        print("VERDICT: Model is CONSERVATIVE (Leaving premium on table).")
    else:
        print("VERDICT: Model is WELL-CALIBRATED.")
    print("="*40)

if __name__ == "__main__":
    run_backtest()
