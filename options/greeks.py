import numpy as np
from scipy.stats import norm

def calculate_delta(spot, strike, time_to_maturity, risk_free=0.04, vol=0.5):
    """
    Calculates Black-Scholes Delta for a Call option.
    """
    if time_to_maturity <= 0:
        return 0.0 # Standard convention for expired OTM, checks usually occur before
    
    if vol <= 0:
        return 1.0 if spot > strike else 0.0
        
    d1 = (np.log(spot / strike) + (risk_free + 0.5 * vol**2) * time_to_maturity) / (vol * np.sqrt(time_to_maturity))
    
    return float(norm.cdf(d1))
