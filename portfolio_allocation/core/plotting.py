import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_efficient_frontier(targets, weights, mu, cov, path=None):
    vols = [float(np.sqrt(w @ cov @ w)) for w in weights]
    fig, ax = plt.subplots()
    ax.plot(vols, targets, marker='o', linewidth=1)
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Return")
    ax.set_title("Efficient Frontier")
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig, ax

def plot_drawdowns(ddf, title="Drawdowns", path=None):
    fig, ax = plt.subplots()
    ddf["drawdown"].plot(ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig, ax
