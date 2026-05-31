"""
visualize.py
============
Plotting helpers:
  • draw_network()       – colour-coded NetworkX graph
  • plot_training()      – loss curves from history dict
  • plot_score_dist()    – surrogate score distributions (GNN vs label)
  • plot_embeddings()    – t-SNE / UMAP of node embeddings
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import torch
from sklearn.manifold import TSNE


# ── colour palette ────────────────────────────────────────────────────
PALETTE = {
    "edge_device": "#4A90D9",
    "broker":      "#E8A838",
    "surrogate":   "#5CB85C",
    "background":  "#1A1A2E",
    "text":        "#E0E0E0",
    "grid":        "#2A2A4A",
}


def draw_network(G: nx.Graph, title: str = "Edge Network Graph",
                 save_path: str = None):
    """
    Draw the network graph with colour-coded node types.
    Blue = Edge Device | Orange = Broker | Green = Surrogate
    """
    type_map  = nx.get_node_attributes(G, "node_type")
    color_map = {0: PALETTE["edge_device"], 1: PALETTE["broker"], 2: PALETTE["surrogate"]}
    node_cols = [color_map.get(type_map[n], "#888888") for n in G.nodes()]

    size_map = {0: 200, 1: 500, 2: 350}
    node_sizes = [size_map.get(type_map[n], 200) for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(PALETTE["background"])
    ax.set_facecolor(PALETTE["background"])

    pos = nx.spring_layout(G, seed=42, k=1.5)
    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_color=node_cols, node_size=node_sizes,
        font_size=7, font_color=PALETTE["text"],
        edge_color="#555577", alpha=0.9,
        with_labels=True,
    )

    legend = [
        mpatches.Patch(color=PALETTE["edge_device"], label="Edge Device"),
        mpatches.Patch(color=PALETTE["broker"],      label="Broker"),
        mpatches.Patch(color=PALETTE["surrogate"],   label="Surrogate"),
    ]
    ax.legend(handles=legend, loc="upper left",
              facecolor=PALETTE["background"], labelcolor=PALETTE["text"],
              fontsize=10)
    ax.set_title(title, color=PALETTE["text"], fontsize=14, pad=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["background"])
    return fig


def plot_training(history: dict, save_path: str = None):
    """Loss curves for train and validation."""
    train_total = [m["loss_total"]     for m in history["train"]]
    val_total   = [m["loss_total"]     for m in history["val"]]
    val_sur     = [m["loss_surrogate"] for m in history["val"]]
    val_load    = [m["loss_load"]      for m in history["val"]]

    epochs = range(1, len(train_total) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(PALETTE["background"])

    for ax in axes:
        ax.set_facecolor(PALETTE["background"])
        ax.tick_params(colors=PALETTE["text"])
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["grid"])
        ax.grid(color=PALETTE["grid"], linestyle="--", linewidth=0.5)

    axes[0].plot(epochs, train_total, color="#4A90D9", label="Train Total")
    axes[0].plot(epochs, val_total,   color="#E8A838", label="Val Total")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(facecolor=PALETTE["background"], labelcolor=PALETTE["text"])

    axes[1].plot(epochs, val_sur,  color="#5CB85C", label="Surrogate Loss")
    axes[1].plot(epochs, val_load, color="#E74C3C", label="Load Loss")
    axes[1].set_title("Validation – Task Breakdown")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(facecolor=PALETTE["background"], labelcolor=PALETTE["text"])

    plt.suptitle("Training Curves – GNN Surrogate Discovery",
                 color=PALETTE["text"], fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["background"])
    return fig


def plot_score_distribution(pred_scores: np.ndarray,
                            true_scores: np.ndarray,
                            save_path: str = None):
    """Overlay histogram: predicted vs ground-truth surrogate scores."""
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE["background"])
    ax.set_facecolor(PALETTE["background"])

    bins = np.linspace(0, 1, 25)
    ax.hist(true_scores,  bins=bins, alpha=0.65, color="#5CB85C", label="Ground Truth")
    ax.hist(pred_scores,  bins=bins, alpha=0.65, color="#4A90D9", label="GNN Predicted")

    ax.set_xlabel("Surrogate Quality Score", color=PALETTE["text"])
    ax.set_ylabel("Count", color=PALETTE["text"])
    ax.set_title("Predicted vs. True Surrogate Scores", color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"])
    ax.legend(facecolor=PALETTE["background"], labelcolor=PALETTE["text"])
    ax.grid(color=PALETTE["grid"], linestyle="--", linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_edgecolor(PALETTE["grid"])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["background"])
    return fig


def plot_embeddings(embeddings: torch.Tensor,
                    node_types:  torch.Tensor,
                    save_path: str = None):
    """t-SNE projection of learned node embeddings, coloured by node type."""
    emb_np  = embeddings.detach().cpu().numpy()
    types_np = node_types.detach().cpu().numpy()

    tsne  = TSNE(n_components=2, perplexity=min(30, len(emb_np) - 1), random_state=42)
    proj  = tsne.fit_transform(emb_np)

    label_map = {0: "Edge Device", 1: "Broker", 2: "Surrogate"}
    color_map = {0: PALETTE["edge_device"], 1: PALETTE["broker"], 2: PALETTE["surrogate"]}

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(PALETTE["background"])
    ax.set_facecolor(PALETTE["background"])

    for t in [0, 1, 2]:
        mask = (types_np == t)
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=color_map[t], label=label_map[t],
                   alpha=0.8, s=60, edgecolors="none")

    ax.set_title("t-SNE of Node Embeddings", color=PALETTE["text"], fontsize=13)
    ax.tick_params(colors=PALETTE["text"])
    ax.legend(facecolor=PALETTE["background"], labelcolor=PALETTE["text"])
    for sp in ax.spines.values():
        sp.set_edgecolor(PALETTE["grid"])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["background"])
    return fig
