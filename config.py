from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    ticker: str = "GME"
    p_target_min: float = 0.75
    p_target_max: float = 0.80
    alpha_lcb: float = 0.05      # 5% one-sided confidence
    mc_paths: int = 50000        # Monte Carlo paths
    
    # DTE Grid: We look for expiries closest to these days
    dte_grid: List[int] = field(default_factory=lambda: [0, 5, 7, 14, 21, 28, 35, 42])
    
    # Resistance Parameters
    res_window_high: int = 20    # Lookback for swing highs
    res_ma_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
    res_zone_width: float = 0.01 # +/- 1% around resistance level
    
    # Risk Constraints
    max_delta: float = 0.30
    touch_cap: float = 0.40      # Max allowed touch probability
    
    # Optimization Weights
    lambda_res: float = 0.5      # Penalty for distance from resistance
    lambda_risk: float = 1.0     # Penalty for risk (delta/gamma)