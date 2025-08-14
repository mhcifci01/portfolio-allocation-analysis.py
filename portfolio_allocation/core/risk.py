import numpy as np
import pandas as pd

def annualize_rets(returns: pd.Series, periods_per_year: int = 252):
    return returns.mean() * periods_per_year

def annualize_vol(returns: pd.Series, periods_per_year: int = 252):
    return returns.std(ddof=1) * np.sqrt(periods_per_year)

def drawdowns(returns: pd.Series):
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    dd = (wealth - peak) / peak
    return pd.DataFrame({"wealth": wealth, "peak": peak, "drawdown": dd})

def hist_var(returns: pd.Series, level: float = 5.0):
    # Historical Value-at-Risk (one-period, percent)
    return -np.percentile(returns.dropna(), level)

def hist_es(returns: pd.Series, level: float = 5.0):
    # Historical Expected Shortfall (conditional VaR)
    x = returns.dropna()
    cutoff = np.percentile(x, level)
    return -x[x <= cutoff].mean()

def portfolio_stats(weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame):
    port_mu = float(weights @ mu)
    port_vol = float(np.sqrt(weights @ cov @ weights))
    return {"return": port_mu, "volatility": port_vol}
