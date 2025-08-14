import pandas as pd
import numpy as np
from portfolio_allocation.core import mean_variance as mv

def test_random_weights_sum_to_one():
    w = mv.random_weights(5, n_portfolios=100, seed=123)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-8)

def test_min_variance_output_shape():
    # simple identity covariance should lead to equal weights
    cov = pd.DataFrame(np.eye(4))
    w = mv.min_variance_weights(cov)
    assert w.shape == (4,)
    assert np.allclose(w.sum(), 1.0, atol=1e-8)
