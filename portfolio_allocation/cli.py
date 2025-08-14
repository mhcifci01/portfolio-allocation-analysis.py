import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from .core import mean_variance as mv
from .core import risk
from .core import plotting as pltg

def main():
    p = argparse.ArgumentParser(description="Portfolio Allocation Analysis CLI")
    p.add_argument("--data", type=str, required=True, help="Path to CSV of daily simple returns (index=Date)")
    p.add_argument("--opt", type=str, choices=["min-var", "max-sharpe", "target"], default="max-sharpe")
    p.add_argument("--riskfree", type=float, default=0.0, help="Annual risk-free rate (e.g., 0.02 for 2%)")
    p.add_argument("--target", type=float, default=None, help="Annual target return for 'target' optimization")
    p.add_argument("--export", type=str, default=None, help="Export directory for reports/plots")
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Data file not found: {data_path}")

    rets = pd.read_csv(data_path, index_col=0, parse_dates=True)
    mu, cov = mv.prepare_inputs(rets)

    if args.opt == "min-var":
        w = mv.min_variance_weights(cov)
    elif args.opt == "max-sharpe":
        w = mv.max_sharpe_weights(mu, cov, riskfree=args.riskfree)
    else:
        if args.target is None:
            raise SystemExit("--target is required for 'target' optimization")
        w = mv._minimize_volatility_for_return(args.target, mu, cov)

    assets = rets.columns.tolist()
    weights = pd.Series(w, index=assets, name="weight")
    stats = risk.portfolio_stats(w, mu, cov)
    frontier_targets, frontier_weights = mv.efficient_frontier(mu, cov, n=50)

    print("Selected optimization:", args.opt)
    print("Weights:")
    print(weights.round(4))
    print("Stats:", {k: round(v, 4) for k, v in stats.items()})

    if args.export:
        out = Path(args.export)
        out.mkdir(parents=True, exist_ok=True)
        weights.to_csv(out / "portfolio_weights.csv")
        pd.Series(stats).to_csv(out / "portfolio_stats.csv")
        # Efficient frontier plot
        _, _ = pltg.plot_efficient_frontier(frontier_targets, frontier_weights, mu, cov, path=out / "efficient_frontier.png")
        # Drawdown plot of equal-weight portfolio as reference
        eq = np.repeat(1/len(assets), len(assets))
        eq_rets = (rets * eq).sum(axis=1)
        ddf = risk.drawdowns(eq_rets)
        _, _ = pltg.plot_drawdowns(ddf, path=out / "drawdowns_equal_weight.png")
        print(f"Exported to: {out.resolve()}")
