import numpy as np
from scipy.optimize import least_squares

def svi_raw(k, a, b, rho, m, sigma):
    # k = log(K/S)
    # w = total variance = sigma_BS^2 * T
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def fit_svi(strikes, ivs, T, spot):
    """
    Fits SVI parameters for a single expiry.
    strikes: array of K
    ivs: array of implied volatilities (decimal)
    T: time to expiry (years)
    spot: current underlying price
    """
    if len(ivs) < 5:
        return None  # Not enough data

    k = np.log(strikes / spot)
    w = (ivs ** 2) * T
    
    # Objective function
    def residuals(params):
        a, b, rho, m, sigma_p = params
        w_model = svi_raw(k, a, b, rho, m, sigma_p)
        return w_model - w

    # Initial guess & Bounds
    # a, b, rho, m, sigma
    x0 = [0.04, 0.1, -0.5, 0.0, 0.1]
    bounds = (
        [0, 0, -0.99, -5.0, 0.001], # Lower
        [2.0, 2.0, 0.99, 5.0, 2.0]  # Upper
    )
    
    try:
        res = least_squares(residuals, x0, bounds=bounds, loss='soft_l1')
        return res.x # params
    except:
        return None

def get_iv_from_surface(strike, T, spot, params):
    if params is None: return None
    k = np.log(strike / spot)
    w = svi_raw(k, *params)
    if w < 0: return 0.0
    return np.sqrt(w / T)