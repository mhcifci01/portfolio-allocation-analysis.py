import pandas as pd
import numpy as np
from portfolio_allocation.core import risk

def test_drawdowns_basic():
    s = pd.Series([0.1, -0.1, 0.0, 0.05])
    ddf = risk.drawdowns(s)
    assert "drawdown" in ddf.columns
    assert ddf["drawdown"].min() <= 0
