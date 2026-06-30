"""
visualize_v2.py
===============
Plots for the improved pipeline:
  • plot_gnn_vs_dht()    – grouped bar chart of P@1 / P@3 / NDCG@3
  • plot_holdout_radar() – metric summary on the holdout set
  • plot_training_v2()   – validation NDCG@3 / P@3 over epochs
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "gnn": "#5CB85C", "dht": "#E8A838", "bg": "#1A1A2E",
    "text": "#E0E0E0", "grid": "#2A2A4A", "accent": "#4A90D9",
}


def _style(ax):
    ax.set_facecolor(PALETTE["bg"])
    ax.tick_params(colors=PALETTE["text"])
    for sp in ax.spines.values():
        sp.set_edgecolor(PALETTE["grid"])
    ax.grid(color=PALETTE["grid"], linestyle="--", linewidth=0.5, axis="y")
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])


def plot_gnn_vs_dht(gnn: dict, dht: dict, save_path: str = None):
    metrics = ["precision@1", "precision@3", "ndcg@3"]
    g_vals = [gnn.get(m, 0) for m in metrics]
    d_vals = [dht.get(m, 0) for m in metrics]
    x = np.arange(len(metrics)); w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(PALETTE["bg"]); _style(ax)
    b1 = ax.bar(x - w/2, g_vals, w, label="GNN (ours)", color=PALETTE["gnn"])
    b2 = ax.bar(x + w/2, d_vals, w, label="DHT (hop-count)", color=PALETTE["dht"])
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f"{b.get_height():.2f}", ha="center", color=PALETTE["text"], fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
    ax.set_title("Surrogate Discovery — GNN vs DHT (holdout topologies)")
    ax.legend(facecolor=PALETTE["bg"], labelcolor=PALETTE["text"])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=PALETTE["bg"], bbox_inches="tight")
    return fig


def plot_training_v2(history: dict, save_path: str = None):
    val = history["val"]
    ndcg = [m.get("ndcg@3", 0) for m in val]
    p3   = [m.get("precision@3", 0) for m in val]
    p1   = [m.get("precision@1", 0) for m in val]
    loss = history.get("train_loss", [])
    ep = range(1, len(val) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in axes: _style(ax)

    axes[0].plot(ep, loss, color=PALETTE["accent"], label="Train loss")
    axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(facecolor=PALETTE["bg"], labelcolor=PALETTE["text"])

    axes[1].plot(ep, ndcg, color=PALETTE["gnn"], label="NDCG@3")
    axes[1].plot(ep, p3,   color=PALETTE["accent"], label="Precision@3")
    axes[1].plot(ep, p1,   color=PALETTE["dht"], label="Precision@1")
    axes[1].set_title("Validation Ranking Metrics")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score"); axes[1].set_ylim(0, 1.05)
    axes[1].legend(facecolor=PALETTE["bg"], labelcolor=PALETTE["text"])

    plt.suptitle("Improved GNN — Training Curves", color=PALETTE["text"], y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=PALETTE["bg"], bbox_inches="tight")
    return fig
