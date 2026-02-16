from arch import arch_model
import numpy as np
import pandas as pd

class GarchModel:
    def __init__(self, returns):
        # returns: pd.Series of log returns
        self.returns = returns * 100 # Rescale for convergence
        self.model = arch_model(self.returns, vol='Garch', p=1, o=1, q=1, dist='t')
        self.res = self.model.fit(disp='off')
        
    def simulate_paths(self, n_days, n_paths, current_price):
        """
        Simulates future price paths.
        Returns: (n_paths, n_days+1) array of prices.
        """
        # Forecast variance
        # GJR-GARCH simulation logic:
        # sigma_t^2 = omega + alpha*e_{t-1}^2 + gamma*e_{t-1}^2*I + beta*sigma_{t-1}^2
        
        params = self.res.params
        omega = params['omega']
        alpha = params['alpha[1]']
        gamma = params['gamma[1]']
        beta = params['beta[1]']
        nu = params['nu'] # Student-t df
        
        # Last known volatility
        last_vol = self.res.conditional_volatility.iloc[-1]
        
        # Generate innovations (Student-t)
        # Shape: (n_paths, n_days)
        z = np.random.standard_t(nu, size=(n_paths, n_days))
        
        log_ret = np.zeros((n_paths, n_days))
        vol_sq = np.zeros((n_paths, n_days))
        
        # Step 0
        # We need e_0 (last residual). Approx as 0 for simplicity or use last model resid
        h_t = last_vol**2
        
        current_vol_sq = np.full(n_paths, h_t)
        
        for t in range(n_days):
            # epsilon = z * sigma
            eps = z[:, t] * np.sqrt(current_vol_sq)
            
            # Record returns (divide by 100 because we scaled up)
            log_ret[:, t] = eps / 100.0
            
            # Update variance for next step
            # I indicator
            I = (eps < 0).astype(float)
            current_vol_sq = omega + (alpha + gamma * I) * (eps**2) + beta * current_vol_sq
            
        # Cumsum to get prices
        cum_ret = np.cumsum(log_ret, axis=1)
        prices = current_price * np.exp(cum_ret)
        
        # Prepend current price
        prices = np.hstack([np.full((n_paths, 1), current_price), prices])
        
        # Also return average volatility path (for Bridge)
        # simplified: return constant last vol for bridge approximation
        return prices, np.sqrt(current_vol_sq)/100.0