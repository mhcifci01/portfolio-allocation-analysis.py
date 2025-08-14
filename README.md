# Portfolio Allocation Analysis

A concise, reproducible toolkit for mean-variance portfolio allocation, risk metrics, and scenario analysis with a simple CLI and examples.

## Features
- Mean-variance optimization (minimum-variance, maximum Sharpe, target return)
- Risk metrics: volatility, downside deviation, VaR/ES (historical), drawdowns
- Portfolio simulation (random weights, efficient frontier)
- Clean plotting helpers (matplotlib)
- Simple CLI: load returns, optimize, and export results to CSV/PNG
- Tested with `pytest`; packaged with `pyproject.toml`

## Quickstart

```bash
# 1) Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install
pip install -e .

# 3) Run the CLI on sample data
portfolio-analyzer --data data/sample_returns.csv --opt max-sharpe --riskfree 0.02 --export outputs
```

This command will:
- compute core risk statistics,
- run the selected optimization,
- export a report CSV and plots under `outputs/`.

## CLI usage

```bash
portfolio-analyzer --help
```

**Examples**

- Minimum variance portfolio:
```bash
portfolio-analyzer --data data/sample_returns.csv --opt min-var --export outputs
```

- Target return portfolio (e.g., 12% annualized target):
```bash
portfolio-analyzer --data data/sample_returns.csv --opt target --target 0.12 --export outputs
```

## Library usage

```python
import pandas as pd
from portfolio_allocation.core import mean_variance, risk

rets = pd.read_csv("data/sample_returns.csv", index_col=0, parse_dates=True)
mu = rets.mean() * 252
cov = rets.cov() * 252

w_mv = mean_variance.min_variance_weights(cov)
stats = risk.portfolio_stats(w_mv, mu, cov)
print(stats)
```

## Project structure

```
portfolio-allocation-analysis/
├── data/                      # synthetic sample returns
├── portfolio_allocation/      # core library
│   ├── core/
│   │   ├── mean_variance.py
│   │   ├── risk.py
│   │   └── plotting.py
│   ├── __init__.py
│   └── cli.py                 # console entry point
├── tests/
│   ├── test_mean_variance.py
│   └── test_risk.py
├── notebooks/
│   └── 01_quick_demo.ipynb
├── outputs/                   # created by CLI (ignored by git)
├── pyproject.toml
├── README.md
└── LICENSE
```

## GitHub instructions

1. Initialize and commit:
```bash
git init
git add .
git commit -m "feat: initial project scaffold for Portfolio Allocation Analysis (CLI, core library, tests, sample data, docs)"
git branch -M main
```

2. Create a new repository on GitHub named `portfolio-allocation-analysis` and add it as remote:
```bash
git remote add origin https://github.com/<YOUR_USERNAME>/portfolio-allocation-analysis.git
git push -u origin main
```

## Notes
- Sample data are synthetic yet realistic daily log returns for 5 assets over ~3 years.
- All plots are generated with matplotlib and avoid style/colour overrides.
- No external data sources required.
