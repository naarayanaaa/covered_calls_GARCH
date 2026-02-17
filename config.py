import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    ticker: str = "GME"
    
    # Data Provider Configuration
    api_provider: str = os.getenv("API_PROVIDER", "alpaca")
    alpaca_key: str = os.getenv("ALPACA_KEY", "")
    alpaca_secret: str = os.getenv("ALPACA_SECRET", "")
    alpaca_endpoint: str = os.getenv("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets")

    # Strategy Parameters
    p_target_min: float = 0.58
    p_target_max: float = 0.60
    alpha_lcb: float = 0.05      # 5% one-sided confidence
    mc_paths: int = 50000        # Monte Carlo paths
    
    # DTE Grid: Focused on Short-Term Weekly Options
    dte_grid: List[int] = field(default_factory=lambda: [0, 5, 7, 14])
    
    # Transaction Costs (Interactive Brokers)
    commission_fee: float = 2.0  # $2.00 per contract (flat fee)
    min_premium_abs: float = 4.0 # Minimum net profit per contract ($5.00)
    
    # Safety: Manual Earnings Date Fallback (YYYY-MM-DD)
    manual_next_earnings_date: str = "2026-03-26" 
    
    # Resistance Parameters
    res_window_high: int = 20    
    res_ma_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
    res_zone_width: float = 0.01 
    
    # Risk Constraints
    max_delta: float = 0.5
    touch_cap: float = 0.5       
    
    # Liquidity & Execution Parameters
    min_oi: int = 10              
    liquidity_spread_thresh: float = 0.05 
    liquidity_crossing_factor: float = 0.2 
    
    # Optimization Weights
    lambda_res: float = 0.5      
    lambda_risk: float = 1.0
