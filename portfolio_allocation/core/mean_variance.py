import numpy as np
import pandas as pd
from scipy.optimize import minimize

def _annualize_mean_cov(returns: pd.DataFrame, periods_per_year: int = 252):
    mu = returns.mean() * periods_per_year
    cov = returns.cov() * periods_per_year
    return mu, cov

def random_weights(n_assets: int, n_portfolios: int = 10_000, seed: int = 42):
    rng = np.random.default_rng(seed)
    w = rng.random((n_portfolios, n_assets))
    return (w.T / w.sum(axis=1)).T

def portfolio_performance(weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame, riskfree: float = 0.0):
    ret = float(np.dot(weights, mu))
    vol = float(np.sqrt(np.dot(weights, cov @ weights)))
    sharpe = (ret - riskfree) / vol if vol > 0 else np.nan
    return ret, vol, sharpe

def _minimize_volatility_for_return(target_return: float, mu: pd.Series, cov: pd.DataFrame):
    n = len(mu)
    init_w = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    cons = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: float(w @ mu) - target_return},
    )
    obj = lambda w: float(np.sqrt(w @ cov @ w))
    res = minimize(obj, init_w, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 10_000})
    return res.x

def min_variance_weights(cov: pd.DataFrame):
    n = len(cov)
    init_w = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    obj = lambda w: float(np.sqrt(w @ cov @ w))
    res = minimize(obj, init_w, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 10_000})
    return res.x

def max_sharpe_weights(mu: pd.Series, cov: pd.DataFrame, riskfree: float = 0.0):
    n = len(mu)
    init_w = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    def neg_sharpe(w):
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ cov @ w))
        if vol == 0:
            return 1e6
        return - (ret - riskfree) / vol
    res = minimize(neg_sharpe, init_w, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 10_000})
    return res.x

def efficient_frontier(mu: pd.Series, cov: pd.DataFrame, n: int = 50):
    min_var_w = min_variance_weights(cov)
    min_ret = float(min_var_w @ mu)
    max_ret = float(mu.max())
    targets = np.linspace(min_ret, max_ret, n)
    weights = np.vstack([_minimize_volatility_for_return(t, mu, cov) for t in targets])
    return targets, weights

def prepare_inputs(returns: pd.DataFrame, periods_per_year: int = 252):
    mu, cov = _annualize_mean_cov(returns, periods_per_year=periods_per_year)
    return mu, cov
