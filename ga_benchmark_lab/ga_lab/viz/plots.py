import numpy as np
import matplotlib.pyplot as plt

def plot_convergence_curve(best_cost: np.ndarray, title: str):
    fig = plt.figure()
    plt.plot(best_cost)
    plt.xlabel("Generation")
    plt.ylabel("Best cost so far")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    return fig

def plot_band_curves(curves: np.ndarray, title: str):
    # curves: (k_seeds, G)
    fig = plt.figure()
    median = np.median(curves, axis=0)
    p25 = np.percentile(curves, 25, axis=0)
    p75 = np.percentile(curves, 75, axis=0)
    best = np.min(curves, axis=0)

    plt.plot(best, label="best (across seeds)")
    plt.plot(median, label="median")
    plt.fill_between(np.arange(curves.shape[1]), p25, p75, alpha=0.2, label="p25–p75")
    plt.xlabel("Generation")
    plt.ylabel("Best cost so far")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return fig

def plot_box(values: np.ndarray, title: str, ylabel: str):
    fig = plt.figure()
    plt.boxplot(values, vert=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y")
    plt.tight_layout()
    return fig

def plot_series(y: np.ndarray, title: str, ylabel: str):
    fig = plt.figure()
    plt.plot(y)
    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    return fig